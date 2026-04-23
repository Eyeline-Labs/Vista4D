from argparse import ArgumentParser
import csv
from functools import partial
from os import path
from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch
from tqdm.auto import tqdm

from utils.misc import cleanup
from utils.media import (
    depths_to_disparity_video, load_cameras, load_depths, load_masks, load_video,
    resize_2d, resize_intrinsics, save_cameras, save_depths, save_masks, save_mp4_with_gif, save_video,
)
from utils.point_cloud.filter import contract_mask, get_depths_outliers
from utils.point_cloud.render_video import render_video


SKY_DEPTH = 1e3


def visualize(
    videos: Tuple[npt.NDArray[np.uint8]],
    depths: Tuple[npt.NDArray],
    alpha_masks: Tuple[npt.NDArray[np.bool_]],
    dynamic_masks: Tuple[npt.NDArray[np.bool_]],
    static_masks: Tuple[npt.NDArray[np.bool_]],
):
    assert len(videos) == len(depths) == len(alpha_masks) == len(dynamic_masks) == len(static_masks),\
        "Length of all five input tuples for visualization must match."

    video_row = np.concatenate(videos, axis=2)
    depths_row = np.concatenate([
        depths_to_disparity_video(d, a, use_colormap=True) for d, a in zip(depths, alpha_masks)
    ], axis=2)
    masks_row = np.concatenate([
        np.stack((a, d, s), axis=-1).astype(np.uint8) * 255
        for a, d, s in zip(alpha_masks, dynamic_masks, static_masks)
    ], axis=2)
    return np.concatenate((video_row, depths_row, masks_row), axis=1)


@torch.no_grad()
def main(args):
    dtype = torch.float32
    device = "cuda"
    # When render_only_necessary is True, we only render (and save) the necessary point cloud renders and masks for
    # Vista4D inference. When False, we additionally render the point cloud
    # 1) without temporal persistence and
    # 2) double-reprojected back to the source camera
    # which can be useful for running on baselines/other works, ablations, and other tests
    render_all = not args.render_only_necessary

    with open(args.metadata_path, "r", newline="", encoding="utf-8") as file:
        render_data = list(csv.DictReader(file))

    progress = tqdm(render_data)
    for data_info in progress:
        save_name = data_info["name"]
        video_name = data_info["video"]
        cam_name = data_info["camera"]

        recon_and_seg_folder = path.join(args.recon_and_seg_folder, video_name)
        video_src, fps = load_video(path.join(recon_and_seg_folder, "video.mp4"))
        num_frames, height, width, _ = video_src.shape

        progress.set_description(
            f"Rendering point cloud, video=`{video_name}`, camera=`{cam_name}`, "
            f"num_frames={num_frames}, resolution={args.width}x{args.height}"
        )

        depths_src = load_depths(path.join(recon_and_seg_folder, "depths")).astype(np.float32)
        cam_c2w_src, intrinsics_src = load_cameras(path.join(recon_and_seg_folder, "cameras.npz"))
        cam_c2w_tgt, intrinsics_tgt = load_cameras(
            path.join(args.cam_folder, video_name, f"{cam_name}.npz"),
            force_same_first_frame=args.force_same_first_frame,
        )
        dynamic_mask_src = load_masks(path.join(recon_and_seg_folder, "dynamic_mask"))
        static_mask_src = load_masks(path.join(recon_and_seg_folder, "static_mask"))\
            if path.isdir(path.join(recon_and_seg_folder, "static_mask")) else None
        sky_mask_src = load_masks(path.join(recon_and_seg_folder, "sky_mask"))\
            if path.isdir(path.join(recon_and_seg_folder, "sky_mask")) else np.zeros_like(dynamic_mask_src)

        resize_2d_fn = partial(resize_2d, height=args.height, width=args.width, crop=True)
        resize_intrinsics_fn = partial(
            resize_intrinsics, height=args.height, width=args.width, height_input=height, width_input=width, crop=True,
        )
        video_src = np.moveaxis(resize_2d_fn(np.moveaxis(video_src, -1, -3), mode="bicubic", inverse=False), -3, -1)
        depths_src = resize_2d_fn(depths_src, mode="bilinear", inverse=True)
        intrinsics_src = resize_intrinsics_fn(intrinsics_src)
        intrinsics_tgt = resize_intrinsics_fn(intrinsics_tgt)
        dynamic_mask_src = resize_2d_fn(dynamic_mask_src, mode="bilinear", inverse=False)
        static_mask_src = resize_2d_fn(static_mask_src, mode="bilinear", inverse=False)\
            if static_mask_src is not None else ~dynamic_mask_src
        sky_mask_src = resize_2d_fn(sky_mask_src, mode="bilinear", inverse=False)

        static_mask_src = contract_mask(static_mask_src, radius=6, iterations=3)

        dynamic_mask_src =\
            dynamic_mask_src & ~get_depths_outliers(depths_src, dynamic_mask_src, mode=args.depth_outliers)
        static_mask_src = static_mask_src & ~get_depths_outliers(depths_src, static_mask_src, mode=args.depth_outliers)

        dynamic_mask_src = dynamic_mask_src & ~sky_mask_src  # Remove sky mask, uncontracted to be conservative
        sky_mask_src = contract_mask(sky_mask_src, radius=6, iterations=3)
        static_mask_src = static_mask_src | sky_mask_src  # Ensure the sky is also static
        depths_src[sky_mask_src] = SKY_DEPTH

        render_outputs = render_video(
            video=video_src,
            depths=depths_src,
            cam_c2w_src=cam_c2w_src,
            cam_c2w_tgt=cam_c2w_tgt,
            intrinsics_src=intrinsics_src,
            intrinsics_tgt=intrinsics_tgt,
            dynamic_mask=dynamic_mask_src,
            static_mask=static_mask_src,
            render_dynamic_mask=True,
            double_reprojection=render_all,
            dtype=dtype,
            device=device,
            verbose=False,
        )
        if render_all:
            (
                (video_pc, depths_pc, alpha_mask_pc, dynamic_mask_pc),
                (video_pc_ntp, depths_pc_ntp, alpha_mask_pc_ntp, dynamic_mask_pc_ntp),
                (video_pc2, depths_pc2, alpha_mask_pc2, dynamic_mask_pc2),
            ) = render_outputs
        else:
            video_pc, depths_pc, alpha_mask_pc, dynamic_mask_pc = render_outputs

        static_mask_pc = alpha_mask_pc & ~dynamic_mask_pc
        if render_all:
            static_mask_pc2 = alpha_mask_pc2 & ~dynamic_mask_pc2

        depths_src = np.minimum(depths_src, SKY_DEPTH)
        depths_pc = np.minimum(depths_pc, SKY_DEPTH)
        if render_all:
            depths_pc_ntp = np.minimum(depths_pc_ntp, SKY_DEPTH)
            depths_pc2 = np.minimum(depths_pc2, SKY_DEPTH)

        true_mask = np.ones_like(alpha_mask_pc)
        depths_dtype = np.float16

        output_folder = path.join(args.output_folder, save_name)
        # Video
        video_quality = 9
        save_video(path.join(output_folder, "video_src.mp4"), video_src, fps=fps, quality=video_quality)
        save_video(path.join(output_folder, "video_pc.mp4"), video_pc, fps=fps, quality=video_quality)
        if render_all:
            save_video(path.join(output_folder, "video_pc_ntp.mp4"), video_pc_ntp, fps=fps, quality=video_quality)
            save_video(path.join(output_folder, "video_pc2.mp4"), video_pc2, fps=fps, quality=video_quality)
        # Depths
        save_depths(path.join(output_folder, "depths_src"), depths_src, dtype=depths_dtype)
        save_depths(path.join(output_folder, "depths_pc"), depths_pc, dtype=depths_dtype)
        if render_all:
            save_depths(path.join(output_folder, "depths_pc_ntp"), depths_pc_ntp, dtype=depths_dtype)
            save_depths(path.join(output_folder, "depths_pc2"), depths_pc2, dtype=depths_dtype)
        # Masks: Original (source cameras)
        save_masks(path.join(output_folder, "alpha_mask_src"), true_mask)
        save_masks(path.join(output_folder, "dynamic_mask_src"), dynamic_mask_src)
        save_masks(path.join(output_folder, "static_mask_src"), static_mask_src)
        save_masks(path.join(output_folder, "sky_mask_src"), sky_mask_src)
        # Masks: Point cloud (target cameras)
        save_masks(path.join(output_folder, "alpha_mask_pc"), alpha_mask_pc)
        save_masks(path.join(output_folder, "dynamic_mask_pc"), dynamic_mask_pc)
        save_masks(path.join(output_folder, "static_mask_pc"), static_mask_pc)
        if render_all:
            # Masks: Point cloud (target cameras), no temporal persistence
            save_masks(path.join(output_folder, "alpha_mask_pc_ntp"), alpha_mask_pc_ntp)
            save_masks(path.join(output_folder, "dynamic_mask_pc_ntp"), dynamic_mask_pc_ntp)
            save_masks(path.join(output_folder, "static_mask_pc_ntp"), ~true_mask)
            # Masks: Point cloud w/ double reprojection (source cameras)
            save_masks(path.join(output_folder, "alpha_mask_pc2"), alpha_mask_pc2)
            save_masks(path.join(output_folder, "dynamic_mask_pc2"), dynamic_mask_pc2)
            save_masks(path.join(output_folder, "static_mask_pc2"), static_mask_pc2)
        # Cameras
        save_cameras(path.join(output_folder, "cameras_src.npz"), cam_c2w_src, intrinsics_src)
        save_cameras(path.join(output_folder, "cameras_tgt.npz"), cam_c2w_tgt, intrinsics_tgt)

        if args.save_vis:
            if render_all:
                vis_collage = visualize(
                    videos=(video_src, video_pc, video_pc_ntp, video_pc2),
                    depths=(depths_src, depths_pc, depths_pc_ntp, depths_pc2),
                    alpha_masks=(true_mask, alpha_mask_pc, alpha_mask_pc_ntp, alpha_mask_pc2),
                    dynamic_masks=(dynamic_mask_src, dynamic_mask_pc, dynamic_mask_pc_ntp, dynamic_mask_pc2),
                    static_masks=(static_mask_src, static_mask_pc, ~true_mask, static_mask_pc2),
                )
            else:
                vis_collage = visualize(
                    videos=(video_src, video_pc),
                    depths=(depths_src, depths_pc),
                    alpha_masks=(true_mask, alpha_mask_pc),
                    dynamic_masks=(dynamic_mask_src, dynamic_mask_pc),
                    static_masks=(static_mask_src, static_mask_pc),
                )
            save_mp4_with_gif(
                path.join(output_folder, "vis"), vis_collage, fps=fps, quality=6,
                gif_folder="" if args.save_vis_gif else None,
            )

        cleanup()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--metadata_path", type=str, default="./eval_data/metadata.csv")
    parser.add_argument("--recon_and_seg_folder", type=str, default="./eval_data/recon_and_seg")
    parser.add_argument("--cam_folder", type=str, default="./eval_data/cameras")
    parser.add_argument("--output_folder", type=str, default="./eval_data/render")

    parser.add_argument("--render_only_necessary", action="store_true", default=False)
    parser.add_argument("--force_same_first_frame", action="store_true", default=False)

    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)

    parser.add_argument("--depth_outliers", type=str, default="gaussian", choices=("gaussian", "pool"))
    parser.add_argument("--save_vis", action="store_true", default=False)
    parser.add_argument("--save_vis_gif", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
