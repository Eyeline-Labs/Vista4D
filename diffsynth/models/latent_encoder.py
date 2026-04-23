from typing import Callable, Optional, Tuple

from einops import rearrange
import numpy as np
import torch
from torch import nn


def patchify(x, patch_embedding, check_patchify_match=None, check_patchify_match_prefix="Patchify"):  # Vista4D
    x = patch_embedding(x)
    b, c, f, h, w = x.shape
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    if check_patchify_match is not None:
        assert (f, h, w) == check_patchify_match,\
            f"{check_patchify_match_prefix}: x={(f, h, w)} and patchify={check_patchify_match} don't match."
    return x, (f, h, w)


class PatchEmbedding(nn.Module):  # Vista4D

    def __init__(
        self,
        init_mode: str = "zero_init",  # zero_init, wan_patch_embed, wan_patch_embed_frozen
        in_channels: Optional[int] = None,
        wan_patch_embedding: nn.Conv3d = None,
    ):
        super().__init__()
        # Three options (shapes initialized from wan_patch_embedding):
        # 1. `zero_init` -> Zero-initialized patch embedding layers for masks
        # 2. `wan_patch_embed` -> Patch embedding layers initialized from Wan, trainable, for
        # 3. `wan_patch_embed_frozen` -> Patch embedding layers fixed from Wan (not trainable)
        assert init_mode in ("zero_init", "wan_patch_embed", "wan_patch_embed_frozen")
        assert wan_patch_embedding is not None

        out_channels, in_channels_, p1, p2, p3 = wan_patch_embedding.weight.shape
        if in_channels is None:
            in_channels = in_channels_
        elif in_channels != in_channels_:  # Cannot initialize from wan_patch_embedding (different in_channels)
            init_mode = "zero_init"

        if init_mode == "wan_patch_embed_frozen":  # Use wan_patch_embedding Callable directly (no extra params)
            self.patch_embedding = None
        else:
            self.patch_embedding =\
                nn.Conv3d(in_channels, out_channels, kernel_size=(p1, p2, p3), stride=(p1, p2, p3), bias=True)
            if init_mode == "zero_init":
                nn.init.zeros_(self.patch_embedding.weight)
                nn.init.zeros_(self.patch_embedding.bias)
            elif init_mode == "wan_patch_embed":
                self.patch_embedding.weight = nn.Parameter(wan_patch_embedding.weight.clone().detach())
                self.patch_embedding.bias = nn.Parameter(wan_patch_embedding.bias.clone().detach())

        self.init_mode = init_mode
        self.out_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,
        wan_patch_embedding: Optional[Callable] = None,
        check_patchify_match: Optional[Tuple[int]] = None,
        check_patchify_match_prefix: Optional[str] = "Patchify",
    ):
        patch_embedding = self.patch_embedding
        if patch_embedding is None:
            assert wan_patch_embedding is not None,\
                "wan_patch_embedding cannot be None with init_mode=`wan_patch_embed_frozen`"
            patch_embedding = wan_patch_embedding
        x, (f, h, w) = patchify(
            x,
            patch_embedding,
            check_patchify_match=check_patchify_match,
            check_patchify_match_prefix=check_patchify_match_prefix,
        )
        return x, (f, h, w)


def format_to_channels(format, vae_in_channels, num_input_channels):
    # Assume VAE temporal, height, and width compression of 4, 8, 8
    return {
        "vae": vae_in_channels,
        "downsample": num_input_channels,
        "downsample_channelpack": num_input_channels * 4,
        "downsample_allpack": num_input_channels * 4 * 8 * 8,
    }[format]


class RGBMaskPatchEmbedding(nn.Module):  # Vista4D

    def __init__(
        self,
        rgb_init_mode: str = "wan_patch_embed",  # zero_init, wan_patch_embed, wan_patch_embed_frozen
        mask_init_mode: str = None,  # zero_init, wan_patch_embed, wan_patch_embed_frozen
        wan_patch_embedding: nn.Conv3d = None,
        rgb_in_channels: Optional[int] = None,
        mask_in_channels: Optional[int] = None,
    ):
        super().__init__()

        if rgb_init_mode is not None:
            self.rgb_patchify = PatchEmbedding(
                init_mode=rgb_init_mode, wan_patch_embedding=wan_patch_embedding, in_channels=rgb_in_channels,
            )
        if mask_init_mode is not None:
            self.mask_patchify = PatchEmbedding(
                init_mode=mask_init_mode, wan_patch_embedding=wan_patch_embedding, in_channels=mask_in_channels
            )
            if self.mask_patchify.init_mode != "zero_init":  # If (somehow) not zero-init, need zero-init projector
                out_channels = wan_patch_embedding.weight.shape[0]
                self.projector = nn.Linear(self.mask_patchify.out_channels, out_channels, bias=True)

    def forward(
        self,
        rgb_latents: Optional[torch.Tensor] = None,
        mask_latents: Optional[torch.Tensor] = None,
        wan_patch_embedding: Optional[Callable] = None,
        check_patchify_match: Optional[Tuple[int]] = None,
        check_patchify_match_prefix: Optional[str] = "Patch embedding",  # For error/debug messages
    ):
        is_arraylike = lambda x: isinstance(x, (list, tuple, np.ndarray, torch.Tensor))
        is_batch_none = lambda x: x is None or (is_arraylike(x) and any([x_ is None for x_ in x]))

        output_latents = 0.0

        if hasattr(self, "rgb_patchify") and not is_batch_none(rgb_latents):
            rgb_latents, (f, h, w) = self.rgb_patchify(
                rgb_latents,
                wan_patch_embedding=wan_patch_embedding,
                check_patchify_match=check_patchify_match,
                check_patchify_match_prefix=f"{check_patchify_match_prefix}, RGB",
            )
            output_latents = output_latents + rgb_latents

        if hasattr(self, "mask_patchify") and not is_batch_none(mask_latents):
            mask_latents, _ = self.mask_patchify(
                mask_latents,
                wan_patch_embedding=wan_patch_embedding,
                check_patchify_match=check_patchify_match,
                check_patchify_match_prefix=f"{check_patchify_match_prefix}, mask",
            )
            if hasattr(self, "projector"):
                mask_latents = self.projector(mask_latents)
            output_latents = output_latents + mask_latents

        return output_latents, (f, h, w)


class LatentEncoder(nn.Module):

    def __init__(
        self,
        source_init_mode: str = "wan_patch_embed",  # zero_init, wan_patch_embed, wan_patch_embed_frozen
        point_cloud_init_mode: str = "wan_patch_embed",  # zero_init, wan_patch_embed, wan_patch_embed_frozen
        mask_init_mode: str = "zero_init",
        use_source_masks: bool = True,
        use_point_cloud_masks: bool = True,
        wan_patch_embedding: nn.Conv3d = None,
        rgb_in_channels: Optional[int] = None,
        mask_in_channels: int = 2 * 4 * 8 * 8,  # Alpha mask and motion mask for (2), then 4 * 8 * 8 is VAE compression
    ):
        super().__init__()

        self.output_patch_embedding = RGBMaskPatchEmbedding(
            rgb_init_mode="wan_patch_embed_frozen",
            mask_init_mode=None,
            wan_patch_embedding=wan_patch_embedding,
            rgb_in_channels=rgb_in_channels,
            mask_in_channels=None,
        )
        self.source_patch_embedding = RGBMaskPatchEmbedding(
            rgb_init_mode=source_init_mode,
            mask_init_mode=mask_init_mode if use_source_masks else None,
            wan_patch_embedding=wan_patch_embedding,
            rgb_in_channels=rgb_in_channels,
            mask_in_channels=mask_in_channels,
        )
        self.point_cloud_patch_embedding = RGBMaskPatchEmbedding(
            rgb_init_mode=point_cloud_init_mode,
            mask_init_mode=mask_init_mode if use_point_cloud_masks else None,
            wan_patch_embedding=wan_patch_embedding,
            rgb_in_channels=rgb_in_channels,
            mask_in_channels=mask_in_channels,
        )

    def forward(
        self,
        wan_patch_embedding_fn: Callable,
        x: torch.Tensor,
        source_video_latents: Optional[torch.Tensor] = None,
        source_mask_latents: Optional[torch.Tensor] = None,
        point_cloud_video_latents: Optional[torch.Tensor] = None,
        point_cloud_mask_latents: Optional[torch.Tensor] = None,
    ):
        x, (f, h, w) = self.output_patch_embedding(
            rgb_latents=x,
            mask_latents=None,
            wan_patch_embedding=wan_patch_embedding_fn,
            check_patchify_match=None,
        )
        source_latents, _ = self.source_patch_embedding(
            rgb_latents=source_video_latents,
            mask_latents=source_mask_latents,
            wan_patch_embedding=wan_patch_embedding_fn,
            check_patchify_match=(f, h, w),
            check_patchify_match_prefix="Source patch embedding",
        )
        point_cloud_latents, _ = self.point_cloud_patch_embedding(
            rgb_latents=point_cloud_video_latents,
            mask_latents=point_cloud_mask_latents,
            wan_patch_embedding=wan_patch_embedding_fn,
            check_patchify_match=(f, h, w),
            check_patchify_match_prefix="Point cloud patch embedding",
        )
        return x, source_latents, point_cloud_latents, (f, h, w)
