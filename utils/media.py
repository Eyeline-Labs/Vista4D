import json
from random import randint
from os import listdir, makedirs, path, rename, replace
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import imageio
import Imath
from matplotlib import colormaps
from natsort import natsorted
import numpy as np
import numpy.typing as npt
import OpenEXR
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from diffsynth.utils.vista4d.media import apply_num_frames, crop_and_resize_pil


def np_to_pil(video: npt.NDArray[np.uint8]):
    return [Image.fromarray(video[i]) for i in range(video.shape[0])]


def pil_to_np(video: List[Image.Image]):
    return np.stack([np.array(frame) for frame in video], axis=0)


def load_video(video_path: str, desc: str = "Loading video"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_array = np.empty((num_frames, height, width, 3), dtype=np.uint8)

    i = 0
    with tqdm(total=num_frames, desc=desc, leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV reads in BGR format, convert to RGB
            video_array[i] = frame
            i += 1
            pbar.update(1)
    cap.release()

    if i < num_frames:
        video_array = video_array[:num_frames]
    return video_array, fps


def save_video(output_path: str, video: npt.NDArray, fps: float, quality=None, imageio_params=None):
    imageio_params = imageio_params if imageio_params is not None else {}
    if quality is not None:
        imageio_params["quality"] = quality
    if path.splitext(output_path)[1] == ".gif":
        imageio_params["loop"] = 0
    else:
        # Disable imageio's frame resizing: the H.264 encoder handles non-multiples-of-16
        # natively via the crop rectangle in the bitstream, so no pre-resizing is needed.
        imageio_params.setdefault("macro_block_size", 1)

    makedirs(path.split(output_path)[0], exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, **imageio_params)

    for i in range(video.shape[0]):
        writer.append_data(video[i])
    writer.close()


def save_mp4_with_gif(
    output_path: str, video: npt.NDArray[np.uint8], fps: int = 24, quality: int = 9, gif_folder: Optional[str] = None,
):  # save_path should *not* have an extension!
    save_video(f"{output_path}.mp4", video, fps=fps, quality=quality)
    if gif_folder is not None:
        save_folder, save_file = path.split(output_path)
        save_folder_gif = path.join(save_folder, gif_folder)
        save_video(path.join(save_folder_gif, f"{save_file}.gif"), video, fps=fps)


def safe_save(file_path: str, save_fn: Callable):  # Save files while mitigating race conditions
    file_name, file_ext = path.splitext(file_path)
    tmp_path = f"{file_name}_tmp{randint(0, 2 ** 31)}{file_ext}"

    save_fn(tmp_path)  # save_fn should take one (1) argument, and that is the path to save to
    if path.exists(file_path):
        replace(tmp_path, file_path)
    else:
        try:
            rename(tmp_path, file_path)
        except Exception:  # Prevent path existence race condition
            replace(tmp_path, file_path)


def dtype_to_pixel_type(dtype: np.dtype):
    return Imath.PixelType({"float16": Imath.PixelType.HALF, "float32": Imath.PixelType.FLOAT}[np.dtype(dtype).name])


def save_depths(output_folder: str, depths: npt.NDArray[np.float16 | np.float32], dtype: np.dtype = np.float16):
    pixel_type = dtype_to_pixel_type(dtype)
    if depths.dtype != dtype:
        depths = depths.astype(dtype)

    num_frames, height, width = depths.shape
    makedirs(output_folder, exist_ok=True)
    for i in range(num_frames):
        header = OpenEXR.Header(width, height)
        header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
        header["channels"] = {"Y": Imath.Channel(pixel_type)}
        file = OpenEXR.OutputFile(path.join(output_folder, f"{i:05d}.exr"), header)
        file.writePixels({"Y": depths[i].tobytes()})
        file.close()


def load_depths(input_folder: str, dtype: np.dtype = np.float16, desc: str = "Loading depths"):
    pixel_type = dtype_to_pixel_type(dtype)
    files = sorted(file for file in listdir(input_folder) if file.endswith(".exr"))
    assert len(files) > 0, f"No EXR files found in {input_folder}"

    frames = []
    for filename in tqdm(files, desc=desc, leave=False):
        file = OpenEXR.InputFile(path.join(input_folder, filename))
        data_window = file.header()["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1
        frames.append(np.frombuffer(file.channel("Y", pixel_type), dtype=dtype).reshape(height, width))
        file.close()

    return np.stack(frames, axis=0)


def save_masks(output_folder: str, masks: npt.NDArray[np.bool_]):
    num_frames, height, width = masks.shape
    makedirs(output_folder, exist_ok=True)
    for i in range(num_frames):
        cv2.imwrite(path.join(output_folder, f"{i:05d}.png"), masks[i].astype(np.uint8) * 255)


def load_masks(input_folder: str, desc: str = "Loading masks"):
    files = sorted(file for file in listdir(input_folder) if file.endswith(".png"))
    assert len(files) > 0, f"No PNG files found in {input_folder}"
    frames = []
    for file in tqdm(files, desc=desc, leave=False):
        frame = cv2.imread(path.join(input_folder, file), cv2.IMREAD_GRAYSCALE) > 127
        frames.append(frame)
    return np.stack(frames)


def save_cameras(output_path: str, cam_c2w: npt.NDArray, intrinsics: npt.NDArray):
    makedirs(path.split(output_path)[0], exist_ok=True)
    np.savez(output_path, cam_c2w=cam_c2w, intrinsics=intrinsics)


def load_cameras(input_path: str, force_same_first_frame: bool = False):
    data = np.load(input_path)
    if force_same_first_frame:
        return data["cam_c2w_fsff"], data["intrinsics_fsff"]
    return data["cam_c2w"], data["intrinsics"]


def save_clips(output_path: str, clips: Dict[str, Tuple[int, int]]):
    makedirs(path.split(output_path)[0], exist_ok=True)
    with open(output_path, "w") as file:
        json.dump({name: [int(start), int(end)] for name, (start, end) in clips.items()}, file, indent=2)


def load_clips(input_path: str):
    if not path.isfile(input_path):
        return None
    with open(input_path, "r") as file:
        data = json.load(file)
    return {name: (int(start), int(end)) for name, (start, end) in data.items()}


def load_recon_and_seg(folder: str, depths_dtype: np.dtype = np.float16):
    # Load a recon_and_seg folder into a single dict. Fills in the default static_mask (~dynamic_mask) and
    # the default clips ({"src": (0, num_frames)}) when their source files are absent.
    video, fps = load_video(path.join(folder, "video.mp4"), desc="Loading video")
    depths = load_depths(path.join(folder, "depths"), dtype=depths_dtype, desc="Loading depths").astype(np.float32)
    cam_c2w, intrinsics = load_cameras(path.join(folder, "cameras.npz"))
    dynamic_mask = load_masks(path.join(folder, "dynamic_mask"), desc="Loading dynamic masks")
    static_mask = load_masks(path.join(folder, "static_mask"), desc="Loading static masks")\
        if path.isdir(path.join(folder, "static_mask")) else ~dynamic_mask
    sky_mask = load_masks(path.join(folder, "sky_mask"), desc="Loading sky masks")
    clips = load_clips(path.join(folder, "clips.json")) or {"src": (0, video.shape[0])}
    return {
        "video": video, "fps": fps,
        "depths": depths, "cam_c2w": cam_c2w, "intrinsics": intrinsics,
        "dynamic_mask": dynamic_mask, "static_mask": static_mask, "sky_mask": sky_mask,
        "clips": clips,
    }


def intrinsics_to_K(intrinsics: npt.NDArray):
    num_frames = intrinsics.shape[0]
    K = np.zeros((num_frames, 3, 3), dtype=intrinsics.dtype)
    K[:, 0, 0] = intrinsics[:, 0]
    K[:, 1, 1] = intrinsics[:, 1]
    K[:, 0, 2] = intrinsics[:, 2]
    K[:, 1, 2] = intrinsics[:, 3]
    K[:, 2, 2] = 1.0
    return K


def K_to_intrinsics(K: npt.NDArray):
    num_frames = K.shape[0]
    intrinsics = np.zeros((num_frames, 4), dtype=K.dtype)
    intrinsics[:, 0] = K[:, 0, 0]
    intrinsics[:, 1] = K[:, 1, 1]
    intrinsics[:, 2] = K[:, 0, 2]
    intrinsics[:, 3] = K[:, 1, 2]
    return intrinsics


def focal_to_intrinsics(focal, height, width):
    diagonal = pow(pow(height, 2) + pow(width, 2), 0.5)
    ff = focal * (diagonal / 2.0)
    intrinsics = np.zeros((*focal.shape, 4), dtype=np.float32)
    intrinsics[..., 0] = ff
    intrinsics[..., 1] = ff
    intrinsics[..., 2] = width / 2.0
    intrinsics[..., 3] = height / 2.0
    return intrinsics


def slice_center_frames(video: Union[List, Tuple, npt.NDArray, torch.Tensor], num_frames: int):
    return apply_num_frames(video, num_frames, batched=False)


def resize_2d(
    t: npt.NDArray, height: int, width: int, mode: str = "bilinear", crop: bool = False, inverse: bool = False,
):
    # Fast path: input already at target resolution → skip the (float64 cast + F.interpolate) entirely.
    # With matching sizes no crop is needed either (aspect ratios match trivially), so returning the
    # input unchanged is semantically identical to running the full pipeline but avoids a potentially
    # expensive pass on large video/depth tensors and the `1/depth` round-trip for inverse=True.
    if t.shape[-2:] == (height, width):
        return t.copy()

    t = t.copy()
    dtype = t.dtype

    t = torch.from_numpy(t)
    t = t.to(torch.float64)  # Always cast to float64
    leading_dims = t.shape[:-2]
    t = t.reshape(-1, *t.shape[-2:])  # Flatten leading dimensions

    if crop:
        height_input, width_input = t.shape[-2:]
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

    t = t.reshape(*leading_dims, *t.shape[-2:])
    if dtype == np.bool_:
        t = t >= 0.5
    elif dtype == np.uint8:
        t = t.clamp(0.0, 255.0)
    t = t.numpy().astype(dtype)
    return t


def resize_intrinsics(
    intrinsics: npt.NDArray, height, width, height_input: bool = None, width_input: bool = None, crop: bool = False,
):
    intrinsics = intrinsics.copy()
    width_input = intrinsics[:, 2].mean() * 2.0 if width_input is None else width_input
    height_input = intrinsics[:, 3].mean() * 2.0 if height_input is None else height_input

    if crop:
        if height_input / width_input > height / width:  # Input is taller than target
            height_crop = int(width_input * height / width)
            intrinsics[..., 3] -= (height_input - height_crop) / 2  # Shift cy
            height_input = height_crop
        else:  # Input is wider than target
            width_crop = int(height_input * width / height)
            intrinsics[..., 2] -= (width_input - width_crop) / 2 # Shift cx
            width_input = width_crop

    fx_scale = width / width_input
    fy_scale = height / height_input

    intrinsics[..., 0] *= fx_scale  # fx
    intrinsics[..., 1] *= fy_scale  # fy
    intrinsics[..., 2] *= fx_scale  # cx
    intrinsics[..., 3] *= fy_scale  # cy
    return intrinsics


def crop_and_resize_video(
    video: Union[List[Image.Image], npt.NDArray[np.uint8]], height: int, width: int, resample: str = "lanczos",
):
    is_numpy = isinstance(video, np.ndarray)
    if is_numpy:
        video = [Image.fromarray(video[i]) for i in range(video.shape[0])]
    video = crop_and_resize_pil(video, height, width, resample=resample, batched=False)
    if is_numpy:
        video = np.stack([np.asarray(frame) for frame in video], axis=0)
    return video


def depths_to_disparity_video(depths: npt.NDArray, mask: npt.NDArray, use_colormap: bool = True):
    depths = depths.astype(np.float64)
    mask = mask & (depths != 0.0)  # Avoid division by 0
    if mask.astype(np.int64).sum() == 0:
        disparity = np.zeros(depths.shape + (3,), dtype=np.uint8)
    else:
        depths[~mask] = 1.0
        disparity = 1 / depths
        disparity[~mask] = 0.0
        disparity_min, disparity_max = max(disparity[mask].min(), 0.001), min(disparity[mask].max(), 1e3)
        disparity = (disparity - disparity_min) / (disparity_max - disparity_min)  # Normalize to [0, 1]
        disparity = (np.clip(disparity, 0.0, 1.0) * 255).astype(np.uint8)
    if use_colormap:
        colormap = np.array(colormaps.get_cmap("inferno").colors)
        disparity = (colormap[disparity] * 255).astype(np.uint8)
        disparity[~mask] = 0
    return disparity


def mask_to_video(mask: npt.NDArray[np.bool_]):
    return mask[..., None].repeat(3, axis=-1).astype(np.uint8) * 255


def concat_2d(videos: List[List[npt.NDArray[np.uint8]]]):
    videos = [np.concatenate(row, axis=2) for row in videos]
    return np.concatenate(videos, axis=1)
