from typing import Union

import numpy as np
import numpy.typing as npt
from PIL import Image
import torch

from depth_anything_3.api import DepthAnything3
from utils.media import K_to_intrinsics, resize_2d
from utils.misc import cleanup


def init_da3(model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1", device: Union[torch.device, str] = "cuda"):
    print(f"Loading Depth Anything 3 model with model_id=`{model_id}`")
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return model


def run_da3(
    video: npt.NDArray[np.uint8],
    model: DepthAnything3,
    process_res: int = 504,
    resize_output: bool = True,
    verbose: bool = True
):
    cleanup()
    num_frames, height, width, _ = video.shape

    def print_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Run DA3 inference
    print_verbose(f"Running DA3 inference at process_res={process_res} ...")
    prediction = model.inference(
        [Image.fromarray(video[i]) for i in range(num_frames)], process_res=process_res, export_format="npz",
    )

    depths = prediction.depth  # f h w
    mask_sky = prediction.sky  # f h w
    cam_w2c = prediction.extrinsics  # f 3 4, world-to-camera (not camera-to-world!)
    intrinsics = K_to_intrinsics(prediction.intrinsics)  # f 3 3
    frames_resized = prediction.processed_images  # f h w 3

    _, height_resized, width_resized, _ = frames_resized.shape
    print_verbose(f"DA3 processed resolution: {width_resized}x{height_resized}")
    print_verbose(
        f"Depth map distribution: min={depths.min():.5f}, median={np.median(depths):.5f}, "
        f"mean={depths.mean():.5f}, max={depths.max():.5f}"
    )

    # Convert w2c (f 3 4) to c2w (f 4 4)
    cam_w2c_44 = np.zeros((num_frames, 4, 4), dtype=np.float32)
    cam_w2c_44[:, :3, :4] = cam_w2c
    cam_w2c_44[:, 3, 3] = 1.0
    cam_c2w = np.linalg.inv(cam_w2c_44)  # f 4 4, camera-to-world

    # Align camera poses so first frame is identity transformation
    cam_c2w = np.linalg.inv(cam_c2w[0])[None] @ cam_c2w

    if resize_output:
        # Resize outputs to original video resolution
        depths = resize_2d(depths, height, width, mode="bilinear", crop=False, inverse=True)
        mask_sky = resize_2d(mask_sky, height, width, mode="bilinear", crop=False, inverse=False)
        print_verbose(f"Sky mask coverage: {mask_sky.astype(np.int64).sum() / np.prod(mask_sky.shape) * 100:.3f}%")

        intrinsics[:, 0::2] *= width / width_resized
        intrinsics[:, 1::2] *= height / height_resized
        print_verbose(
            f"Scaled intrinsics (first frame): fx={intrinsics[0, 0]:.3f}, fy={intrinsics[0, 1]:.3f}, "
            f"cx={intrinsics[0, 2]:.3f}, cy={intrinsics[0, 3]:.3f}"
        )

    del prediction
    cleanup()

    return depths, mask_sky, cam_c2w, intrinsics
