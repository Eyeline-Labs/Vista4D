from functools import partial
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from utils.media import resize_2d, resize_intrinsics
from utils.point_cloud.filter import contract_mask, get_depths_outliers


SKY_DEPTH = 1e3


def preprocess_scene(
    scene: dict,
    indices: npt.NDArray[np.int_],
    height: int,
    width: int,
    depth_outliers: str = "gaussian",
    ignore_sky_mask: bool = False,
    contract_masks: bool = True,
    filter_depth_outliers: bool = True,
    progress_fn: Optional[Callable[[str], None]] = None,
    # `progress_fn` (optional): Called with a short label at each major stage so UI callers can surface what's happening
    # during the otherwise-opaque CPU-bound block. Inner per-frame loops also show tqdm bars when progress_fn is set
):
    # Slice a loaded recon_and_seg scene by `indices`, resize to (height, width), and run the standard filtering
    # pipeline: Static-mask contraction, depth-outlier removal on both masks, and sky-mask handling (subtract sky from
    # dynamic, contract sky, union into static, overwrite sky depths).
    def _progress(msg):
        if progress_fn is not None:
            progress_fn(msg)

    height_in, width_in = scene["video"].shape[1:3]

    _progress("Preprocessing: slicing scene arrays")
    video = scene["video"][indices]
    depths = scene["depths"][indices]
    cam_c2w = scene["cam_c2w"][indices]
    intrinsics = scene["intrinsics"][indices]
    dynamic_mask = scene["dynamic_mask"][indices]
    static_mask = scene["static_mask"][indices]
    sky_mask = scene["sky_mask"][indices]

    resize_2d_fn = partial(resize_2d, height=height, width=width, crop=True)
    resize_intrinsics_fn = partial(
        resize_intrinsics, height=height, width=width,
        height_input=height_in, width_input=width_in, crop=True,
    )
    _progress(f"Preprocessing: resizing to {height}x{width}")
    video = np.moveaxis(resize_2d_fn(np.moveaxis(video, -1, -3), mode="bicubic", inverse=False), -3, -1)
    depths = resize_2d_fn(depths, mode="bilinear", inverse=True)
    intrinsics = resize_intrinsics_fn(intrinsics)
    dynamic_mask = resize_2d_fn(dynamic_mask, mode="bilinear", inverse=False)
    static_mask = resize_2d_fn(static_mask, mode="bilinear", inverse=False)
    sky_mask = resize_2d_fn(sky_mask, mode="bilinear", inverse=False)

    if contract_masks:
        _progress("Preprocessing: contracting static mask")
        static_mask = contract_mask(
            static_mask, radius=6, iterations=3,
            desc="Contracting static mask" if progress_fn is not None else None,
        )
    if filter_depth_outliers:
        _progress("Preprocessing: filtering depth outliers (dynamic)")
        dynamic_mask = dynamic_mask & ~get_depths_outliers(
            depths, dynamic_mask, mode=depth_outliers,
            desc="Depth outliers (dynamic)" if progress_fn is not None else None,
        )
        _progress("Preprocessing: filtering depth outliers (static)")
        static_mask = static_mask & ~get_depths_outliers(
            depths, static_mask, mode=depth_outliers,
            desc="Depth outliers (static)" if progress_fn is not None else None,
        )

    if not ignore_sky_mask:
        _progress("Preprocessing: applying sky mask")
        dynamic_mask = dynamic_mask & ~sky_mask  # Remove sky mask, uncontracted to be conservative
        if contract_masks:
            sky_mask = contract_mask(
                sky_mask, radius=6, iterations=3,
                desc="Contracting sky mask" if progress_fn is not None else None,
            )
        static_mask = static_mask | sky_mask  # Ensure the sky is also static
        depths[sky_mask] = SKY_DEPTH

    return {
        "video": video, "depths": depths, "cam_c2w": cam_c2w, "intrinsics": intrinsics,
        "dynamic_mask": dynamic_mask, "static_mask": static_mask, "sky_mask": sky_mask,
    }
