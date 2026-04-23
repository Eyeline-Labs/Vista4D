from types import MethodType
from typing import List, Optional, Union
from typing_extensions import Literal

from einops import rearrange
import numpy as np
import numpy.typing as npt
from PIL import Image
import torch
from tqdm.auto import tqdm

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.latent_encoder import LatentEncoder
from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder

from ..utils.vista4d.camera import get_plucker_embedding
from ..utils.vista4d.media import apply_num_frames, crop_and_resize_pil, crop_and_resize_tensor


class Vista4DPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.in_iteration_models = ("dit",)
        self.in_iteration_models_2 = ("dit2",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_CameraEmbedder(),
            WanVideoUnit_Vista4DVideoInput(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = []
        self.model_fn = model_fn_vista4d

    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = MethodType(usp_dit_forward, self.dit)  # Not actually used
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = MethodType(usp_dit_forward, self.dit2)  # Not actually used
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="google/*"),
        vista4d_config: dict = None,
        vista4d_checkpoint: Union[str, List[str]] = None,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = Vista4DPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp(device)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean="whitespace")

        # Apply Vista4D configs
        assert vista4d_config is not None, "Vista4D config must be provided."

        if pipe.dit is not None:
            dit_config = vista4d_config["dit"]
            dim = pipe.dit.blocks[0].self_attn.dim
            vae_channels = pipe.dit.patch_embedding.weight.shape[1]  # in_channels of patch_embedding

            pipe.dit.positional_embedding_offset = dit_config["positional_embedding_offset"]

            pipe.dit.latent_encoder = LatentEncoder(
                source_init_mode = dit_config["latent_encoder"]["source_init_mode"],
                point_cloud_init_mode = dit_config["latent_encoder"]["point_cloud_init_mode"],
                mask_init_mode = dit_config["latent_encoder"]["mask_init_mode"],
                use_source_masks = dit_config["latent_encoder"]["use_source_masks"],
                use_point_cloud_masks = dit_config["latent_encoder"]["use_point_cloud_masks"],
                wan_patch_embedding = pipe.dit.patch_embedding,
                rgb_in_channels = vae_channels,
                mask_in_channels = 2 * 4 * pow(pipe.vae.upsampling_factor, 2),  # 4 is VAE temporal compression factor
            )

            for block in pipe.dit.blocks:
                # Camera encoder (zero-initialized)
                block.cam_encoder = torch.nn.Linear(6, dim)
                torch.nn.init.zeros_(block.cam_encoder.weight)
                torch.nn.init.zeros_(block.cam_encoder.bias)
                # Projection after self-attention (identity-initialized)
                block.projector = torch.nn.Linear(dim, dim)
                block.projector.weight = torch.nn.Parameter(torch.eye(dim))
                block.projector.bias = torch.nn.Parameter(torch.zeros(dim))

            if vista4d_checkpoint is not None:
                missing_keys, unexpected_keys = pipe.dit.load_state_dict(
                    torch.load(vista4d_checkpoint, map_location="cpu", weights_only=True), strict=False,
                )
                assert len(unexpected_keys) == 0,\
                    f"Encountered the following unexpected keys from Vista4D checkpoint: {unexpected_keys}."
                print(f"Loaded checkpoint from: {vista4d_checkpoint}")

        # Change image_encoder.model.log_scale from scalar to 1D tensor, otherwise FSDP2 reports an error
        if pipe.image_encoder is not None:
            pipe.image_encoder.model.log_scale = torch.nn.Parameter(pipe.image_encoder.model.log_scale.data.view(1))

        pipe = pipe.to(dtype=torch_dtype, device=device)  # TODO: Will this be a problem for training?

        # Unified Sequence Parallel
        if use_usp:
            pipe.enable_usp()

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    def vae_output_to_video(self, vae_output, min_value=-1, max_value=1):
        vae_output = rearrange(vae_output, "b c f h w -> b f h w c")
        video = [
            [
                self.vae_output_to_image(vae_output[i, j], pattern="H W C", min_value=min_value, max_value=max_value)
                for j in range(vae_output.shape[1])
            ]
            for i in range(vae_output.shape[0])
        ]
        return video

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: List[str] = None,
        negative_prompt: Optional[List[str]] = None,
        # First-frame(-last-frame)-to-video
        input_image: Optional[List[Image.Image]] = None,
        end_image: Optional[List[Image.Image]] = None,
        # Vista4D: Videos and masks
        source_video: List[List[Image.Image]] = None,
        point_cloud_video: List[List[Image.Image]] = None,
        source_alpha_mask: npt.NDArray = None,  # b f h w
        point_cloud_alpha_mask: npt.NDArray = None,  # b f h w
        source_motion_mask: npt.NDArray = None,  # b f h w
        point_cloud_motion_mask: npt.NDArray = None,  # b f h w
        # Vista4D: Target cameras
        target_cam_c2w: npt.NDArray = None,  # b f 4 4
        target_intrinsics: npt.NDArray = None,  # b f 4
        # Noise augmentation
        source_noise_level: Optional[float] = 0.0,
        point_cloud_noise_level: Optional[float] = 0.0,
        image_noise_level: Optional[float] = 0.0,
        # Randomness
        seed: Optional[List[int]] = None,
        rand_device: Optional[str] = "cpu",
        # Shape and frames
        height: Optional[int] = 384,
        width: Optional[int] = 672,
        num_frames: Optional[int] = 49,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary (for Wan 2.2 MOE)
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Progress bar
        progress_bar_cmd = tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
    ):
        # Seed (multiple-seed inference, length of seed array determines inference batch size)
        batch_size = len(seed)

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=sigma_shift)

        # Inputs
        negative_prompt = [""] * batch_size if negative_prompt is None else negative_prompt
        inputs_posi = {"prompt": prompt, "num_inference_steps": num_inference_steps}
        inputs_nega = {"negative_prompt": negative_prompt, "num_inference_steps": num_inference_steps}
        inputs_shared = {
            # First-frame(-last-frame)-to-video
            "input_image": input_image, "end_image": end_image,
            # Vista4D: Videos and masks
            "source_video": source_video, "point_cloud_video": point_cloud_video,
            "source_alpha_mask": source_alpha_mask, "point_cloud_alpha_mask": point_cloud_alpha_mask,
            "source_motion_mask": source_motion_mask, "point_cloud_motion_mask": point_cloud_motion_mask,
            # Vista4D: Target cameras
            "cam_c2w": target_cam_c2w, "intrinsics": target_intrinsics,
            # Noise augmentation
            "source_noise_level": source_noise_level, "point_cloud_noise_level": point_cloud_noise_level,
            "image_noise_level": image_noise_level,
            # Drop augmentation
            "drop_source": [False] * batch_size, "drop_point_cloud": [False] * batch_size,
            "drop_prompt": [False] * batch_size, "drop_camera": [False] * batch_size,
            # Randomness
            "seed": seed, "rand_device": rand_device,
            # Shape and frames
            "batch_size": batch_size, "height": height, "width": width, "num_frames": num_frames,
            # Classifier-free guidance
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            # Scheduler
            "sigma_shift": sigma_shift,
            # VAE tiling
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega =\
                self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if (
                timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps
                and self.dit2 is not None and not models["dit"] is self.dit2
            ):
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2

            # Timestep
            timestep = timestep.repeat(batch_size).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"],
            )
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        # Post-denoising, pre-decoding processing logic (self.post_units is empty for Vista4D)
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Decode
        self.load_models_to_device(["vae"])
        video = self.vae.decode(
            inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        )
        if output_type == "quantized":  # [0, 1] float -> [0, 255] uint8
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        self.load_models_to_device([])
        return video


def is_none(t):
    if isinstance(t, (list, tuple)):
        return all([t_ is None for t_ in t])
    return t is None


def repeat_none(t, batch_size=1):
    if t is None:
        t = [None] * batch_size
    return t


def apply_augmentation_noise(
    latents: torch.Tensor, noise: Optional[torch.Tensor] = None, noise_level: Union[float, List[float]] = 0.0,
):
    if isinstance(noise_level, (list, tuple, np.ndarray, torch.Tensor)):  # Per-batch-instance noise level
        if noise is None or all([l <= 0.0 for l in noise_level]):
            return latents
        assert latents.shape[0] == len(noise_level),\
            f"Batch size of latents ({latents.shape[0]}) must match that of noise_level ({len(noise_level)})."
        assert all([0.0 <= l <= 1.0 for l in noise_level]),\
            f"Must have 0 <= noise_level <= 1, but got noise_level={noise_level}."
        noise_level = torch.tensor(noise_level).to(noise).reshape(-1, *[1] * (latents.ndim - 1))
    else:
        if noise is None or noise_level <= 0.0:
            return latents
        assert 0.0 <= noise_level <= 1.0, f"Must have 0 <= noise_level <= 1, but got noise_level={noise_level}."
    assert latents.shape == noise.shape,\
        f"Shape of latents {tuple(latents.shape)} and noise {tuple(noise.shape)} do not match"
    latents = (1.0 - noise_level) * latents + noise_level * noise
    return latents


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: Vista4DPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("noise", "height", "width", "num_frames", "batch_size", "seed", "rand_device"),
            output_params=("noise", "augmentation_noise")
        )

    @staticmethod
    def generate_noise(pipe: Vista4DPipeline, shape, batch_size, seeds, rand_device):
        if seeds is None:  # Training and inference w/o seed
            noise = pipe.generate_noise((batch_size, *shape), rand_device=rand_device)
        else:  # Inference w/ seed(s)
            assert batch_size == len(seeds), f"batch_size={batch_size} and len(seeds)={len(seeds)} do not match."
            noise = [pipe.generate_noise(shape, seed=seed, rand_device=rand_device) for seed in seeds]
            noise = torch.stack(noise, dim=0)
        return noise

    def process(self, pipe: Vista4DPipeline, noise, height, width, num_frames, batch_size, seed, rand_device):
        length = (num_frames - 1) // 4 + 1  # Hardcoded for Wan (always 4x temporal compression)
        shape =\
            (pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        if noise is None:
            all_noise = self.generate_noise(pipe, (4, *shape), batch_size, seed, rand_device)
            noise = all_noise[:, 0]
            augmentation_noise = [all_noise[:, i] for i in range(1, 4)]  # 3: Source, point cloud, image(s)
        else:
            augmentation_noise = self.generate_noise(pipe, (3, *shape), batch_size, seed, rand_device)
            augmentation_noise = [augmentation_noise[:, i] for i in range(3)]  # 3: Source, point cloud, image(s)
        return {"noise": noise, "augmentation_noise": augmentation_noise}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("noise",),
            output_params=("latents",),
        )

    def process(self, pipe: Vista4DPipeline, noise):
        return {"latents": noise}  # TODO: Implement input_video with SDEdit-style noise adding


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params=("drop_prompt",),
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("context",),
            onload_model_names=("text_encoder",),
        )

    def encode_prompt(self, pipe: Vista4DPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: Vista4DPipeline, prompt, drop_prompt) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        assert isinstance(prompt, (list, tuple)), f"`prompt` argument must have type list/tuple, got {type(prompt)}."
        prompt = [("" if drop_prompt_ else prompt_) for prompt_, drop_prompt_ in zip(prompt, drop_prompt)]
        prompt_emb = [self.encode_prompt(pipe, prompt_) for prompt_ in prompt]
        prompt_emb = torch.cat(prompt_emb, dim=0)
        return {"context": prompt_emb}


class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",),
        )

    @staticmethod
    def encode_image(pipe: Vista4DPipeline, start_image, end_image, height, width):
        start_image = pipe.preprocess_image(start_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([start_image])[0]  # Assume single image
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context_end = pipe.image_encoder.encode_image([end_image])[0]
                clip_context = torch.concat([clip_context, clip_context_end], dim=0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return clip_context

    def process(self, pipe: Vista4DPipeline, input_image, end_image, height, width):
        if is_none(input_image) or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        clip_context = [
            self.encode_image(pipe, input_image_, end_image_, height, width)
            for input_image_, end_image_ in zip(input_image, repeat_none(end_image, batch_size=len(input_image)))
        ]
        clip_context = torch.stack(clip_context, dim=0)
        return {"clip_feature": clip_context}


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_image", "end_image", "num_frames", "height", "width",
                "augmentation_noise", "image_noise_level", "tiled", "tile_size", "tile_stride",
            ),
            output_params=("y", "y_empty"),
            onload_model_names=("vae",)
        )

    @staticmethod
    def encode_image(
        pipe: Vista4DPipeline, start_image, end_image, num_frames, height, width,
        tiled, tile_size, tile_stride,
    ):  # Encode single start (+ end) image, without image noise augmentation
        vae_input = torch.zeros(3, num_frames, height, width).to(pipe.device)
        mask = torch.zeros(
            num_frames, height // pipe.height_division_factor, width // pipe.width_division_factor,
            device=pipe.device,
        )

        if start_image is not None:
            start_image = pipe.preprocess_image(start_image.resize((width, height))).to(pipe.device)
            vae_input[:, 0] = start_image.transpose(0, 1)
            mask[0] = 1.0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input[:, -1] = end_image.transpose(0, 1)
            mask[-1] = 1.0

        y = pipe.vae.encode(
            [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
            device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        mask = torch.concat([torch.repeat_interleave(mask[0:1], repeats=4, dim=1), mask[1:]], dim=0)
        mask = mask.view(
            mask.shape[1] // 4, 4, height // pipe.height_division_factor, width // pipe.width_division_factor,
        )
        mask = mask.transpose(0, 1).to(dtype=pipe.torch_dtype, device=pipe.device)

        return y, mask

    def encode_images(
        self, pipe: Vista4DPipeline,
        start_image: Optional[List[Image.Image]], end_image: Optional[List[Image.Image]],
        image_noise: torch.Tensor, image_noise_level: List[float],
        num_frames: int, height: int, width: int, batch_size: int,
        tiled: bool, tile_size: int, tile_stride: int,
    ):  # Encode all start (+ end) images with image noise augmentation, handles batching
        if len(start_image) is not None:
            batch_size_image = len(start_image)
        elif len(end_image) is not None:
            batch_size_image = len(end_image)
        else:
            batch_size_image = batch_size
        start_image = repeat_none(start_image, batch_size=batch_size_image)
        end_image = repeat_none(end_image, batch_size=batch_size_image)

        y = []
        mask = []
        for start_image_, end_image_ in zip(start_image, end_image):
            y_, mask_ = self.encode_image(
                pipe, start_image_, end_image_, num_frames, height, width, tiled, tile_size, tile_stride,
            )
            y.append(y_)
            mask.append(mask_)
        y = torch.stack(y, dim=0)
        mask = torch.stack(mask, dim=0)

        y = apply_augmentation_noise(y, noise=image_noise, noise_level=image_noise_level)
        y = torch.cat([mask, y], dim=1)  # b (c_m + c) f h w
        return y

    def process(
        self, pipe: Vista4DPipeline,
        input_image: Optional[List[Image.Image]], end_image: Optional[List[Image.Image]],
        augmentation_noise: torch.Tensor, image_noise_level: List[float],
        num_frames: int, height: int, width: int,
        tiled: bool, tile_size: int, tile_stride: int,
    ):
        if is_none(input_image) or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        y = self.encode_images(
            pipe, input_image, end_image, augmentation_noise[2], image_noise_level,
            num_frames, height, width, len(input_image), tiled, tile_size, tile_stride,
        )
        y_empty = self.encode_images(
            pipe, None, None, augmentation_noise[2], image_noise_level,
            num_frames, height, width, len(input_image), tiled, tile_size, tile_stride,
        )
        return {"y": y, "y_empty": y_empty}


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: Vista4DPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if is_none(input_image) or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        raise NotImplementedError("Image embedding is currently not supported for Wan2.2-TI2V.")
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}


class WanVideoUnit_CameraEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("cam_c2w", "intrinsics", "height", "width", "num_frames", "drop_camera"),
            output_params=("cam_emb",),
        )

    def process(
        self, pipe: Vista4DPipeline, cam_c2w, intrinsics,
        height, width, num_frames, drop_camera,
    ):

        def check_num_frames_and_convert_torch(t, name):
            t = apply_num_frames(t, num_frames=num_frames, batched=True)
            assert t.shape[1] == num_frames, (
                f"`{name}` of length={t.shape[1]} (after frame interval skipping) "
                f"must be the same as the given num_frames={num_frames}."
            )
            t = torch.from_numpy(t).to(dtype=pipe.torch_dtype, device=pipe.device)
            return t

        cam_c2w = check_num_frames_and_convert_torch(cam_c2w, "cam_c2w")  # b f 4 4

        # `intrinsics` should be b f 4, i.e., b f [ fx fy cx cy ]
        intrinsics = check_num_frames_and_convert_torch(intrinsics, "intrinsics")

        cam_emb = get_plucker_embedding(
            intrinsics, cam_c2w, height, width,
            height_dit=height // pipe.height_division_factor, width_dit=width // pipe.width_division_factor,
        )  # b f h w 6
        cam_emb = cam_emb[:, ::pipe.time_division_factor]

        indices_to_drop = [i for i in range(cam_emb.shape[0]) if drop_camera[i]]
        cam_emb[indices_to_drop] = 0.0  # Drop camera: Set camera embedding to zero

        return {"cam_emb": cam_emb}


class WanVideoUnit_Vista4DVideoInput(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "source_video", "point_cloud_video",
                "source_alpha_mask", "source_motion_mask", "point_cloud_alpha_mask", "point_cloud_motion_mask",
                "height", "width", "num_frames",
                "drop_source", "drop_point_cloud",
                "augmentation_noise", "source_noise_level", "point_cloud_noise_level",
                "tiled", "tile_size", "tile_stride",
            ),
            output_params=(
                "source_video_latents", "point_cloud_video_latents", "source_mask_latents", "point_cloud_mask_latents",
            ),
            onload_model_names=("vae",)
        )

    def drop_latent_as_noise(self, latent: torch.Tensor, noise: torch.Tensor, drop: List[bool]):
        assert latent.shape[0] == len(drop),\
            f"Latent batch size {latent.shape[0]} does not match drop batch size ({len(drop)})."
        latent = latent.detach().clone()
        for i in range(latent.shape[0]):
            if drop[i]:
                latent[i] = noise[i]
        return latent

    def drop_mask_as_zeros(self, mask: npt.NDArray, drop: List[bool]):
        assert mask.shape[0] == len(drop),\
            f"Mask batch size {mask.shape[0]} does not match drop batch size ({len(drop)})."
        mask = mask.copy()
        for i in range(len(drop)):
            if drop[i]:  # Drop the corresponding batch index/element
                mask[i] = False
        return mask

    def encode_videos(
        self, pipe: Vista4DPipeline, video: List[List[Image.Image]], drop_video: List[bool],
        noise: torch.Tensor, noise_level: List[float],
        height: int, width: int, num_frames: int, tiled: bool, tile_size: int, tile_stride: int,
    ):
        video = apply_num_frames(video, num_frames=num_frames, batched=True)
        assert len(video[0]) == num_frames, f"Length of video ({len(video[0])}) must match `num_frames` ({num_frames})."
        if height is not None and width is not None:
            video = crop_and_resize_pil(video, height=height, width=width, batched=True)

        pipe.load_models_to_device(self.onload_model_names)

        latents = torch.cat([
            pipe.vae.encode(
                pipe.preprocess_video(video[i]), device=pipe.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            if not drop_video[i] or noise is None else noise[i]
            for i in range(len(video))
        ], dim=0)
        if noise is not None:
            latents = self.drop_latent_as_noise(latents, noise=noise, drop=drop_video)
            latents = apply_augmentation_noise(latents, noise=noise, noise_level=noise_level)
        return latents

    def shuffle_mask(
        self, pipe: Vista4DPipeline, masks: List[torch.Tensor], drop_mask: List[bool],
        height: int, width: int, num_frames: int,
    ):
        # assert (
        #     not hasattr(pipe.dit, "warped_target_mask_patch_embedding") or
        #     len(masks) == pipe.dit.warped_target_mask_patch_embedding.weight.shape[1]
        # ), (
        #     f"Number of masks ({len(masks)}) does not match warped_target_mask_patch_embedding "
        #     f"input number of channels ({pipe.dit.warped_target_mask_patch_embedding.weight.shape[1]})"
        # )  # TODO: Reimplement this check!
        masks = np.stack(masks, axis=-1)  # b f h w c, where c is number of masks
        masks = apply_num_frames(masks, num_frames=num_frames, batched=True)
        masks = self.drop_mask_as_zeros(masks, drop=drop_mask)
        masks = torch.from_numpy(masks).to(dtype=pipe.torch_dtype, device=pipe.device).moveaxis(-1, -4)  # b c f h w

        # f' is latent num_frames, where f = 1 + 4 (f' - 1)
        masks = crop_and_resize_tensor(masks, height, width, mode="trilinear")  # b c 1+4(f'-1) 8h 8w
        masks = torch.cat((torch.repeat_interleave(masks[:, :, 0:1], repeats=4, dim=2), masks[:, :, 1:]), dim=2)  # b c 4+4(f-1) 8h 8w
        masks = rearrange(masks, "b c (f sf) (h sh) (w sw) -> b (c sf sh sw) f h w", sf=4, sh=8, sw=8)  # b (c*4*8*8) f h w

        return masks

    def process(
        self,
        pipe: Vista4DPipeline,
        source_video: List[List[Image.Image]],  # List[List] is b f
        point_cloud_video: List[List[Image.Image]],
        source_alpha_mask: npt.NDArray,  # b f h w, np.bool_
        point_cloud_alpha_mask: npt.NDArray,  # b f h w, np.bool_
        source_motion_mask: npt.NDArray,  # b f h w, np.bool_
        point_cloud_motion_mask: npt.NDArray,  # b f h w, np.bool_
        height: int,
        width: int,
        num_frames: int,
        drop_source: List[bool],
        drop_point_cloud: List[bool],
        augmentation_noise: List[torch.Tensor],  # Source, point cloud, image(s)
        source_noise_level: List[float],
        point_cloud_noise_level: List[float],
        tiled: bool,
        tile_size: int,
        tile_stride: int,
    ):
        source_video_latents = self.encode_videos(
            pipe, source_video, drop_source, augmentation_noise[0], source_noise_level,
            height, width, num_frames, tiled, tile_size, tile_stride,
        )
        point_cloud_video_latents = self.encode_videos(
            pipe, point_cloud_video, drop_point_cloud, augmentation_noise[1], point_cloud_noise_level,
            height, width, num_frames, tiled, tile_size, tile_stride,
        )

        source_mask_latents = self.shuffle_mask(
            pipe, [source_alpha_mask, source_motion_mask], drop_source, height, width, num_frames,
        )
        point_cloud_mask_latents = self.shuffle_mask(
            pipe, [point_cloud_alpha_mask, point_cloud_motion_mask], drop_point_cloud, height, width, num_frames,
        )

        return {
            "source_video_latents": source_video_latents,
            "point_cloud_video_latents": point_cloud_video_latents,
            "source_mask_latents": source_mask_latents,
            "point_cloud_mask_latents": point_cloud_mask_latents,
        }


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("use_unified_sequence_parallel",))

    def process(self, pipe: Vista4DPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = [
            "context", "cam_emb", "clip_feature", "y", "y_empty",
            "source_latents", "point_cloud_latents", "source_mask_latents", "point_cloud_mask_latents",
        ]

    def process(self, pipe: Vista4DPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


def get_freqs(dit, patchify_shape, device):
    f, h, w = patchify_shape
    fhw = f * h * w

    offset = dit.positional_embedding_offset
    F = offset * 2 + f  # Total frame dimension (concatenating output, point cloud, and source)
    ohw = offset * h * w

    freqs_all = torch.cat([
        dit.freqs[0][:F].view(F, 1, 1, -1).expand(F, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(F, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(F, h, w, -1)
    ], dim=-1).reshape(F * h * w, 1, -1).to(device)

    #                  Output            point cloud               source
    freqs = torch.cat((freqs_all[0:fhw], freqs_all[ohw:ohw + fhw], freqs_all[2 * ohw: 2 * ohw + fhw]), dim=0)
    return freqs


def model_fn_vista4d(
    dit: WanModel,
    latents: torch.Tensor = None,
    source_video_latents: torch.Tensor = None,
    point_cloud_video_latents: torch.Tensor = None,
    source_mask_latents: torch.Tensor = None,
    point_cloud_mask_latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    cam_emb: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    y_empty: Optional[torch.Tensor] = None,
    use_unified_sequence_parallel: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

    def chunk_usp(t, dim=1):
        chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=dim)
        pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
        chunks = [
            torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0)
            for chunk in chunks
        ]
        t = chunks[get_sequence_parallel_rank()]
        return t, pad_shape

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones(
                (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                dtype=latents.dtype, device=latents.device,
            ) * timestep,
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t, _ = chunk_usp(t, dim=1)
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    t = t[:, None]

    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image latents and embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat((x, y), dim=1)
        source_video_latents = torch.cat((source_video_latents, y_empty), dim=1)
        point_cloud_video_latents = torch.cat((point_cloud_video_latents, y), dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # Latent encoder (patchify)
    x, source_latents, point_cloud_latents, (f, h, w) = dit.latent_encoder(
        dit.patch_embedding, x,
        source_video_latents, source_mask_latents,
        point_cloud_video_latents, point_cloud_mask_latents,
    )
    x = torch.cat((x, point_cloud_latents, source_latents), dim=1)  # Concatenate output, point cloud, and source
    freqs = get_freqs(dit, (f, h, w), device=x.device)

    # Camera embedding (already downsampled to post-patchify, just need to group dims)
    if cam_emb is not None:
        cam_emb = rearrange(cam_emb, "b f h w d -> b (f h w) d")
        cam_emb = cam_emb.repeat(1, 3, 1)  # 3: Output, point cloud, source (matching x)

    # Unified sequence parallel (also necessary to chunk cam_emb)
    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x, pad_shape = chunk_usp(x, dim=1)
        if cam_emb is not None:
            cam_emb, _ = chunk_usp(cam_emb, dim=1)  # pad_shape is only necessary for output slicing

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    # DiT blocks
    for block in dit.blocks:
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), x, context, t_mod, freqs, cam_emb, use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block), x, context, t_mod, freqs, cam_emb, use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs, cam_emb)

    x = dit.head(x, t)
    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)
        x = x[:, :-pad_shape] if pad_shape > 0 else x

    x = torch.chunk(x, 3, dim=1)[0]  # Extract output latents
    x = dit.unpatchify(x, (f, h, w))
    return x
