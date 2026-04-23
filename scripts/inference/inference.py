from argparse import ArgumentParser
from glob import glob
from os import path
from time import time
from typing import Any, Dict

import numpy as np
from PIL import Image
import torch
import yaml

from diffsynth.pipelines.wan_video_vista4d import Vista4DPipeline, ModelConfig, apply_num_frames
from diffsynth.utils.vista4d.media import crop_and_resize_pil

from utils.media import load_cameras, load_masks, load_video, np_to_pil, pil_to_np, save_mp4_with_gif


def get_pipeline(args, vista4d_config: Dict[str, Any]):
    model_id_with_origin_paths = args.model_id_with_origin_paths.split(",")
    model_configs = [
        ModelConfig(
            path=glob(path.join(args.local_model_folder, id.split(":")[0], id.split(":")[1])),
            skip_download=True,
        )
        for id in model_id_with_origin_paths
    ]
    tokenizer_config = ModelConfig(
        path=glob(path.join(
            args.local_model_folder,
            args.tokenizer_id_with_origin_path.split(":")[0],
            args.tokenizer_id_with_origin_path.split(":")[1],
        )),
        skip_download=True,
    )
    pipe = Vista4DPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        vista4d_config=vista4d_config,
        vista4d_checkpoint=args.vista4d_checkpoint,
        use_usp=args.use_usp,
        vram_limit=args.vram_limit,
    )
    return pipe


def get_inputs(args, vista4d_config: Dict[str, Any]):
    video_src, fps = load_video(path.join(args.input_folder, "video_src.mp4"))
    video_pc, _ = load_video(path.join(args.input_folder, "video_pc.mp4"))
    cam_c2w_tgt, intrinsics_tgt = load_cameras(path.join(args.input_folder, "cameras_tgt.npz"))

    def load_mask_or_default(mask_name, default=True):
        mask_path = path.join(args.input_folder, mask_name)
        if path.isdir(mask_path):
            return load_masks(mask_path)
        num_frames, height, width, _ = video_src.shape
        return (np.ones if default else np.zeros)((num_frames, height, width), dtype=np.bool_)

    assert args.prompt is not None, "`--prompt` argument must be provided."

    batch_size = len(args.seed)
    np_to_pil = lambda t: [Image.fromarray(t[i]) for i in range(t.shape[0])]

    inputs = {
        # Prompts
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        # Vista4D: Videos and masks
        "source_video": np_to_pil(video_src),
        "point_cloud_video": np_to_pil(video_pc),
        "source_alpha_mask": load_mask_or_default("alpha_mask_src"),
        "source_motion_mask": load_mask_or_default("dynamic_mask_src"),
        "point_cloud_alpha_mask": load_mask_or_default("alpha_mask_pc"),
        "point_cloud_motion_mask": load_mask_or_default("dynamic_mask_pc"),
        # Vista4D: Target cameras
        "target_cam_c2w": cam_c2w_tgt,
        "target_intrinsics": intrinsics_tgt,
    }
    if "I2V" in args.model_id_with_origin_paths:
        inputs["input_image"] = inputs["source_video"][0]
    for k, v in inputs.items():
        inputs[k] = v[None].repeat(batch_size, axis=0) if isinstance(v, np.ndarray) else [v for _ in range(batch_size)]
    inputs = {  # Add inputs that do not need to be batch-repeated
        **inputs,
        # Shape and frames
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        # Noise augmentation
        "source_noise_level": vista4d_config["dit"]["augmentation"]["source_noise_level"],
        "point_cloud_noise_level": vista4d_config["dit"]["augmentation"]["point_cloud_noise_level"],
        "image_noise_level": vista4d_config["dit"]["augmentation"]["image_noise_level"],
    }

    return inputs, fps


def get_input_videos_for_vis(inputs):
    num_frames, height, width = inputs["num_frames"], inputs["height"], inputs["width"]

    point_cloud_masks = np.stack((
        inputs["point_cloud_alpha_mask"][0],
        inputs["point_cloud_motion_mask"][0],
        inputs["point_cloud_alpha_mask"][0] | inputs["point_cloud_motion_mask"][0],
    ), axis=-1).astype(np.uint8) * 255
    point_cloud_masks = np_to_pil(point_cloud_masks)

    source_video = apply_num_frames(inputs["source_video"][0], num_frames, batched=False)
    point_cloud_video = apply_num_frames(inputs["point_cloud_video"][0], num_frames, batched=False)
    point_cloud_masks = apply_num_frames(point_cloud_masks, num_frames, batched=False)
    source_video = crop_and_resize_pil(source_video, height, width, resample="lanczos")
    point_cloud_video = crop_and_resize_pil(point_cloud_video, height, width, resample="lanczos")
    point_cloud_masks = crop_and_resize_pil(point_cloud_masks, height, width, resample="bilinear")

    return pil_to_np(source_video), pil_to_np(point_cloud_video), pil_to_np(point_cloud_masks)


@torch.no_grad()
def main(args):
    with open(args.vista4d_config_path, "r") as file:
        vista4d_config = yaml.safe_load(file)

    inputs, fps = get_inputs(args, vista4d_config)
    pipe = get_pipeline(args, vista4d_config)  # Also initializes USP

    if args.num_inference_time_trials is not None:
        print(f"Running inference time trials n={args.num_inference_time_trials} times, not saving outputs.")
        for i in range(args.num_inference_time_trials):
            t = time()
            pipe(**inputs, seed=args.seed, cfg_merge=args.cfg_merge, tiled=args.tile_vae)
            print(f"Inference wall time (trial={i}): {time() - t}s")
        return

    if args.use_usp:
        import torch.distributed as dist

    def save_video_fn(save_path, video, quality=9):
        if args.use_usp and dist.get_rank() != 0:  # Only save videos with rank 0
            return
        save_mp4_with_gif(save_path, video, fps=fps, quality=quality, gif_folder="gifs" if args.save_gif else None)

    source_video, point_cloud_video, point_cloud_masks = get_input_videos_for_vis(inputs)
    save_video_fn(path.join(args.output_folder, "source"), source_video)
    save_video_fn(path.join(args.output_folder, "point_cloud"), point_cloud_video)
    save_video_fn(path.join(args.output_folder, "point_cloud_masks"), point_cloud_masks)

    videos = pipe(**inputs, seed=args.seed, cfg_merge=args.cfg_merge, tiled=args.tile_vae)
    for video, seed in zip(videos, args.seed):
        video = np.stack([np.array(frame) for frame in video], axis=0)
        save_video_fn(path.join(args.output_folder, f"video_seed={seed}"), video)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id_with_origin_paths", required=True, type=str)
    parser.add_argument("--tokenizer_id_with_origin_path", required=True, type=str)
    parser.add_argument("--local_model_folder", required=True, type=str)
    parser.add_argument("--vista4d_checkpoint", required=True, type=str)
    parser.add_argument("--vista4d_config_path", type=str, required=True)

    parser.add_argument("--input_folder", required=True, type=str)
    parser.add_argument("--output_folder", required=True, type=str)
    parser.add_argument("--save_gif", action="store_true", default=False)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "gaudy, overexposed, static, blurred details, subtitles, style, artwork, painting, still, "
            "overall gray, worst quality, low quality, JPEG compression, ugly, mutilated, extra fingers, "
            "poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, "
            "still image, cluttered background, three legs, many people in background, walking backwards"
        )
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--num_frames", type=int, default=49)

    parser.add_argument("--use_usp", action="store_true", default=False)
    parser.add_argument("--cfg_merge", action="store_true", default=False)
    parser.add_argument("--tile_vae", action="store_true", default=False)
    parser.add_argument(
        "--vram_limit", type=float, default=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
    )  # Can set default to None if VRAM is not a problem
    parser.add_argument("--seed", type=int, nargs="+", default=[10027])  # batch_size > 1 multi-seed generation

    parser.add_argument("--num_inference_time_trials", type=int, default=None)  # Run N inference trials, no video saves
    args = parser.parse_args()
    main(args)
