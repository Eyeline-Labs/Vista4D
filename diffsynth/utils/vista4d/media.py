from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from PIL import Image


def apply_num_frames(l: Union[List, Tuple, npt.NDArray, torch.Tensor], num_frames: int, batched: bool = False):

    def get_center_slice(num_frames_input, num_frames):
        assert num_frames_input >= num_frames,\
            f"Input ({num_frames_input}) must have same or more frames as `num_frames` ({num_frames})."
        start_index = (num_frames_input - num_frames) // 2
        return slice(start_index, start_index + num_frames, 1)

    if batched:
        if isinstance(l, (list, tuple)):
            l = [l_[get_center_slice(len(l_), num_frames)] for l_ in l]
        elif isinstance(l, (np.ndarray, torch.Tensor)):
            l = l[:, get_center_slice(l.shape[1], num_frames)]
    else:
        l = l[get_center_slice(len(l), num_frames)]
    return l


def pil_resampling(name: str):
    return {
        "nearest": Image.Resampling.NEAREST,
        "lanczos": Image.Resampling.LANCZOS,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "box": Image.Resampling.BOX,
        "hamming": Image.Resampling.HAMMING,
    }[name]


def crop_and_resize_pil(
    video: List[Image.Image], height: int, width: int, resample: str = "lanczos", batched: bool = False,
):
    if batched:
        return [crop_and_resize_pil(video_, height, width, resample=resample, batched=False) for video_ in video]
    if isinstance(resample, str):
        resample = pil_resampling(resample)

    video_cropped_and_resized = []
    for frame in video:
        width_input, height_input = frame.size

        if (height_input, width_input) != (height, width):
            if height_input / width_input > height / width:  # Input is taller than target
                height_crop = int(width_input * height / width)
                height_start = (height_input - height_crop) // 2
                frame = frame.crop((0, height_start, width_input, height_start + height_crop))
            else:  # Input is wider than target
                width_crop = int(height_input * width / height)
                width_start = (width_input - width_crop) // 2
                frame = frame.crop((width_start, 0, width_start + width_crop, height_input))
            frame = frame.resize((width, height), resample)

        video_cropped_and_resized.append(frame)
    return video_cropped_and_resized


def crop_and_resize_tensor(t: torch.Tensor, height: int, width: int, mode: str = "bilinear", inverse: bool = False):
    height_input, width_input = t.shape[-2:]
    if (height_input, width_input) == (height, width):
        return t

    dtype = t.dtype
    leading_dims = t.shape[:-2]
    t = t.to(torch.float64).reshape(-1, *t.shape[-2:])  # Always cast to float64, flatten leading dimensions

    if height_input / width_input > height / width:  # Input is taller than target
        height_crop = int(width_input * height / width)
        height_start = (height_input - height_crop) // 2
        t = t[..., height_start:height_start + height_crop, :]
    else:  # Input is wider than target
        width_crop = int(height_input * width / height)
        width_start = (width_input - width_crop) // 2
        t = t[..., :, width_start:width_start + width_crop]
    if inverse:
        t = 1 / t

    t = F.interpolate(t[:, None], size=(height, width), mode=mode)[:, 0]  # Add extra channel dimension

    if inverse:
        t = 1 / t

    t = t.reshape(*leading_dims, *t.shape[-2:]).to(dtype)
    return t
