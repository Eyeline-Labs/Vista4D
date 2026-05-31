from math import sqrt
from typing import Union

import numpy as np
import numpy.typing as npt
from PIL import Image
import torch
import torchvision.transforms as vT

from pi3.models.pi3 import Pi3
from pi3.models.pi3x import Pi3X
from utils.media import focal_to_intrinsics, resize_2d
from utils.misc import cleanup
from utils.recon_and_seg.recover_focal_shift import recover_focal_shift


def init_pi3(model_id: str = "yyfz233/Pi3X", device: Union[torch.device, str] = "cuda"):
    print(f"Loading Pi3(X) model with model_id=`{model_id}`")
    model_class = Pi3X if model_id.endswith("Pi3X") else Pi3
    model = model_class.from_pretrained(model_id).to(device).eval()
    return model


def preprocess_frames_pi3(video: npt.NDArray[np.uint8], pixel_limit: int = 255000):  # Pixel limit is approximately 384p
    num_frames, height, width, _ = video.shape

    scale = sqrt(pixel_limit / (height * width)) if height * width > pixel_limit else 1.0
    new_height, new_width = height * scale, width * scale
    new_height_14, new_width_14 = round(new_height / 14), round(new_width / 14)
    while (new_height_14 * 14) * (new_width_14 * 14) > pixel_limit:
        if new_height_14 / new_width_14 > height / width:
            new_height_14 -= 1
        else:
            new_width_14 -= 1
    new_height, new_width = max(1, new_height_14) * 14, max(1, new_width_14) * 14

    frames_tensor = []
    to_tensor = vT.ToTensor()
    for i in range(num_frames):
        frame = Image.fromarray(video[i])
        frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
        frame = to_tensor(frame)
        frames_tensor.append(frame)
    frames = torch.stack(frames_tensor, dim=0)
    return frames, (new_height, new_width)


def run_pi3(
    video: npt.NDArray[np.uint8],
    model: Pi3 | Pi3X,
    pixel_limit: int = 255000,
    resize_output: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[torch.device, str] = "cuda",
    verbose: bool = True,
):
    cleanup()
    num_frames, height, width, _ = video.shape

    def print_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Run Pi3(X) inference
    print_verbose("Running Pi3(X) inference ...")
    frames, (height_new, width_new) = preprocess_frames_pi3(video, pixel_limit=pixel_limit)
    frames = frames.to(dtype=dtype, device=device)
    print_verbose(f"Pi3 processed resolution: {width_new}x{height_new}")

    with torch.amp.autocast(device, dtype=dtype):
        results = model(frames[None])

    local_points = results["local_points"][0].detach().cpu().to(torch.float32)
    focal, shift = recover_focal_shift(local_points, downsample_size=(height_new // 2, width_new // 2))
    local_points = local_points.numpy().astype(np.float32)
    focal = focal.detach().cpu().numpy().astype(np.float32)
    shift = shift.detach().cpu().numpy().astype(np.float32)

    depths = local_points[..., 2] + shift[..., None, None]  # f h w
    cam_c2w = results["camera_poses"][0].detach().cpu().numpy().astype(np.float32)  # f 4 4
    intrinsics = focal_to_intrinsics(focal, height=height_new, width=width_new)  # f 4

    print_verbose(
        f"Depth map distribution: min={depths.min():.5f}, median={np.median(depths):.5f}, "
        f"mean={depths.mean():.5f}, max={depths.max():.5f}"
    )

    # Align camera poses so first frame is identity transformation
    cam_c2w = np.linalg.inv(cam_c2w[0])[None] @ cam_c2w

    if resize_output:
        # Resize outputs to original video resolution
        depths = resize_2d(depths, height, width, mode="bilinear", crop=False, inverse=True)

        intrinsics[:, 0::2] *= width / width_new
        intrinsics[:, 1::2] *= height / height_new
        print_verbose(
            f"Scaled intrinsics (first frame): fx={intrinsics[0, 0]:.3f}, fy={intrinsics[0, 1]:.3f}, "
            f"cx={intrinsics[0, 2]:.3f}, cy={intrinsics[0, 3]:.3f}"
        )

    del results
    cleanup()

    return depths, cam_c2w, intrinsics
