from argparse import ArgumentParser
from os import path
from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch

from utils.media import (
    depths_to_disparity_video, load_cameras, load_clips, load_recon_and_seg, load_video,
    resize_intrinsics, save_cameras, save_depths, save_mp4_with_gif, save_masks, save_video,
)
from utils.point_cloud.edit import build_edit_fn, load_edits
from utils.point_cloud.preprocess import SKY_DEPTH, preprocess_scene
from utils.point_cloud.render_video import render_video


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
    depths_dtype = np.float16
    device = "cuda"
    render_all = not args.render_only_necessary

    scene = load_recon_and_seg(args.recon_and_seg_folder, depths_dtype=depths_dtype)
    fps = scene["fps"]
    clips = scene["clips"]
    height_in, width_in = scene["video"].shape[1:3]
    cam_c2w_tgt, intrinsics_tgt = load_cameras(args.cam_path)

    src_start, src_end = clips["src"]
    num_src_frames = src_end - src_start
    assert num_src_frames >= args.num_frames,\
        f"Source clip doesn't have enough frames ({num_src_frames}) compared to requested output ({args.num_frames})"
    # Render window: center-sliced within the src clip range
    win_start = src_start + (num_src_frames - args.num_frames) // 2
    win_indices = np.arange(win_start, win_start + args.num_frames)

    # Unprojection set: render window + (optionally) DSE frames strided by dse_frame_interval
    if "dse" in clips:
        dse_start, dse_end = clips["dse"]
        dse_indices = np.arange(dse_start, dse_end, args.dse_frame_interval)
        unproj_indices = np.concatenate([win_indices, dse_indices])
        print(
            f"Unprojection set: {len(win_indices)} render-window src frames + "
            f"{len(dse_indices)} DSE frames (stride {args.dse_frame_interval})"
        )
    else:
        unproj_indices = win_indices

    src = preprocess_scene(
        scene, indices=unproj_indices, height=args.height, width=args.width,
        depth_outliers=args.depth_outliers, ignore_sky_mask=args.ignore_sky_mask,
    )
    video_src = src["video"]
    depths_src = src["depths"]
    cam_c2w_src = src["cam_c2w"]
    intrinsics_src = src["intrinsics"]
    dynamic_mask_src = src["dynamic_mask"]
    static_mask_src = src["static_mask"]
    sky_mask_src = src["sky_mask"]

    # Target intrinsics are specified at the src input resolution, so resize them the same way as src
    intrinsics_tgt = resize_intrinsics(
        intrinsics_tgt, height=args.height, width=args.width,
        height_input=height_in, width_input=width_in, crop=True,
    )

    edit_fn = build_edit_fn(
        edits_path=args.edits_path,
        video_src=video_src,
        sky_mask_src=sky_mask_src,
        num_frames_tgt=args.num_frames,
        height=args.height,
        width=args.width,
        depth_outliers=args.depth_outliers,
        ignore_sky_mask=args.ignore_sky_mask,
        depths_dtype=depths_dtype,
    )
    has_edits = edit_fn is not None

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
        edit_fn=edit_fn,
        dtype=dtype,
        device=device,
        verbose=True,
    )
    if render_all:
        (
            (video_pc, depths_pc, alpha_mask_pc, dynamic_mask_pc),
            (video_pc_ntp, depths_pc_ntp, alpha_mask_pc_ntp, dynamic_mask_pc_ntp),
            (video_pc2, depths_pc2, alpha_mask_pc2, dynamic_mask_pc2),
        ) = render_outputs
    else:
        video_pc, depths_pc, alpha_mask_pc, dynamic_mask_pc = render_outputs

    # Edit mode: re-render the post-edit point cloud at the SOURCE camera so the `_src` row of the output
    # (video, depths, masks) reflects the edited scene instead of the raw input
    if has_edits:
        # The rerender uses an uncontracted and unfiltered preprocess of src (contract_masks=False) so the result stays
        # as close to the raw source video as possible
        src_raw = preprocess_scene(
            scene, indices=unproj_indices, height=args.height, width=args.width,
            depth_outliers=args.depth_outliers, ignore_sky_mask=args.ignore_sky_mask,
            contract_masks=False, filter_depth_outliers=False,
        )
        # `no_temporal_persistence=True`: each rendered frame shows only the points that originated in that frame
        # (one-hot visibility), so the rerendered source stays as close to the raw input as possible with only the edit
        video_srcedit, depths_srcedit, alpha_mask_srcedit, dynamic_mask_srcedit = render_video(
            video=src_raw["video"],
            depths=src_raw["depths"],
            cam_c2w_src=src_raw["cam_c2w"],
            cam_c2w_tgt=src_raw["cam_c2w"][:args.num_frames],
            intrinsics_src=src_raw["intrinsics"],
            intrinsics_tgt=src_raw["intrinsics"][:args.num_frames],
            dynamic_mask=src_raw["dynamic_mask"],
            static_mask=src_raw["static_mask"],
            render_dynamic_mask=True,
            double_reprojection=False,
            no_temporal_persistence=True,
            edit_fn=edit_fn,
            dtype=dtype,
            device=device,
            verbose=True,
        )

    static_mask_pc = alpha_mask_pc & ~dynamic_mask_pc
    if render_all:
        static_mask_pc2 = alpha_mask_pc2 & ~dynamic_mask_pc2

    depths_src = np.minimum(depths_src, SKY_DEPTH)
    depths_pc = np.minimum(depths_pc, SKY_DEPTH)
    if render_all:
        depths_pc_ntp = np.minimum(depths_pc_ntp, SKY_DEPTH)
        depths_pc2 = np.minimum(depths_pc2, SKY_DEPTH)

    true_mask = np.ones_like(alpha_mask_pc)
    # Alpha for the src-side outputs. All-True for the raw source video (every pixel covered). In edit mode this is
    # overwritten below with the rerender's alpha (holes where the edited scene no longer has geometry at a given pixel)
    alpha_mask_src = np.ones_like(alpha_mask_pc)

    # Drop DSE tail from source-frame outputs so saves/visualization are render-window only
    video_src = video_src[:args.num_frames]
    depths_src = depths_src[:args.num_frames]
    cam_c2w_src = cam_c2w_src[:args.num_frames]
    intrinsics_src = intrinsics_src[:args.num_frames]
    dynamic_mask_src = dynamic_mask_src[:args.num_frames]
    static_mask_src = static_mask_src[:args.num_frames]
    sky_mask_src = sky_mask_src[:args.num_frames]

    video_quality = 9
    # Edit mode: The raw input is preserved alongside under the `_srcraw` prefix in the render_{384p,720p} folder, and
    # the rerender is saved as `_src` as the rerender is used as the source video for the video diffusion model
    if has_edits:
        video_srcraw = video_src  # Snapshot the raw src values before we overwrite them
        depths_srcraw = depths_src
        alpha_mask_srcraw = alpha_mask_src
        dynamic_mask_srcraw = dynamic_mask_src
        static_mask_srcraw = static_mask_src
        sky_mask_srcraw = sky_mask_src

        save_video(path.join(args.output_folder, "video_srcraw.mp4"), video_srcraw, fps=fps, quality=video_quality)
        save_depths(path.join(args.output_folder, "depths_srcraw"), depths_srcraw, dtype=depths_dtype)
        save_masks(path.join(args.output_folder, "alpha_mask_srcraw"), alpha_mask_srcraw)
        save_masks(path.join(args.output_folder, "dynamic_mask_srcraw"), dynamic_mask_srcraw)
        save_masks(path.join(args.output_folder, "static_mask_srcraw"), static_mask_srcraw)
        save_masks(path.join(args.output_folder, "sky_mask_srcraw"), sky_mask_srcraw)

        video_src = video_srcedit[:args.num_frames]
        depths_src = np.minimum(depths_srcedit[:args.num_frames], SKY_DEPTH)
        alpha_mask_src = alpha_mask_srcedit[:args.num_frames]
        dynamic_mask_src = dynamic_mask_srcedit[:args.num_frames]
        static_mask_src = alpha_mask_src & ~dynamic_mask_src
        # Keep raw sky pixels that the rerender didn't draw a non-sky surface in front of (sky points were placed at
        # SKY_DEPTH during preprocessing, so anything drawn there with depth < SKY_DEPTH is a new foreground object
        # occluding what used to be sky)
        sky_mask_src = sky_mask_srcraw & ~(alpha_mask_src & (depths_src < SKY_DEPTH * 0.99))

    # Video
    save_video(path.join(args.output_folder, "video_src.mp4"), video_src, fps=fps, quality=video_quality)
    save_video(path.join(args.output_folder, "video_pc.mp4"), video_pc, fps=fps, quality=video_quality)
    if render_all:
        save_video(path.join(args.output_folder, "video_pc_ntp.mp4"), video_pc_ntp, fps=fps, quality=video_quality)
        save_video(path.join(args.output_folder, "video_pc2.mp4"), video_pc2, fps=fps, quality=video_quality)
    # Depths
    save_depths(path.join(args.output_folder, "depths_src"), depths_src, dtype=depths_dtype)
    save_depths(path.join(args.output_folder, "depths_pc"), depths_pc, dtype=depths_dtype)
    if render_all:
        save_depths(path.join(args.output_folder, "depths_pc_ntp"), depths_pc_ntp, dtype=depths_dtype)
        save_depths(path.join(args.output_folder, "depths_pc2"), depths_pc2, dtype=depths_dtype)
    # Masks: Original (source cameras)
    save_masks(path.join(args.output_folder, "alpha_mask_src"), alpha_mask_src)
    save_masks(path.join(args.output_folder, "dynamic_mask_src"), dynamic_mask_src)
    save_masks(path.join(args.output_folder, "static_mask_src"), static_mask_src)
    save_masks(path.join(args.output_folder, "sky_mask_src"), sky_mask_src)

    # Raw input video per unique insert source from the edits JSON (first-appearance order), sliced to each insert's src
    # clip. Single-insert runs get `video_ins.mp4`, and multi-insert runs get `video_ins0.mp4`, `video_ins1.mp4`, ...
    if has_edits:
        insert_sources: list = []
        for edit in load_edits(args.edits_path):
            target = edit.get("target", {})
            if target.get("kind") == "insert":
                source = target.get("source")
                if source and source not in insert_sources:
                    insert_sources.append(source)
        single_insert = len(insert_sources) == 1
        for i, source in enumerate(insert_sources):
            ins_video, ins_fps = load_video(path.join(source, "video.mp4"))
            ins_clips = load_clips(path.join(source, "clips.json"))
            if ins_clips and "src" in ins_clips:
                ins_start, ins_end = ins_clips["src"]
                ins_video = ins_video[ins_start:ins_end]
            ins_name = "video_ins.mp4" if single_insert else f"video_ins{i}.mp4"
            save_video(
                path.join(args.output_folder, ins_name),
                ins_video, fps=ins_fps, quality=video_quality,
            )

    # Masks: Point cloud (target cameras)
    save_masks(path.join(args.output_folder, "alpha_mask_pc"), alpha_mask_pc)
    save_masks(path.join(args.output_folder, "dynamic_mask_pc"), dynamic_mask_pc)
    save_masks(path.join(args.output_folder, "static_mask_pc"), static_mask_pc)
    if render_all:
        # Masks: Point cloud (target cameras), no temporal persistence
        save_masks(path.join(args.output_folder, "alpha_mask_pc_ntp"), alpha_mask_pc_ntp)
        save_masks(path.join(args.output_folder, "dynamic_mask_pc_ntp"), dynamic_mask_pc_ntp)
        save_masks(path.join(args.output_folder, "static_mask_pc_ntp"), ~true_mask)
        # Masks: Point cloud w/ double reprojection (source cameras)
        save_masks(path.join(args.output_folder, "alpha_mask_pc2"), alpha_mask_pc2)
        save_masks(path.join(args.output_folder, "dynamic_mask_pc2"), dynamic_mask_pc2)
        save_masks(path.join(args.output_folder, "static_mask_pc2"), static_mask_pc2)

    # Cameras
    save_cameras(path.join(args.output_folder, "cameras_src.npz"), cam_c2w_src, intrinsics_src)
    save_cameras(path.join(args.output_folder, "cameras_tgt.npz"), cam_c2w_tgt, intrinsics_tgt)

    if args.save_vis:
        # Edit mode: Prepend a `srcraw` column (the raw input scene) so the collage reads srcraw → srcedit → pc
        # (and optionally → pc_ntp → pc2). Non-edit mode keeps the two-column (or four-column with render_all) layout
        if has_edits:
            vis_videos = (video_srcraw, video_src, video_pc)
            vis_depths = (depths_srcraw, depths_src, depths_pc)
            vis_alpha = (alpha_mask_srcraw, alpha_mask_src, alpha_mask_pc)
            vis_dyn = (dynamic_mask_srcraw, dynamic_mask_src, dynamic_mask_pc)
            vis_static = (static_mask_srcraw, static_mask_src, static_mask_pc)
            if render_all:
                vis_videos += (video_pc_ntp, video_pc2)
                vis_depths += (depths_pc_ntp, depths_pc2)
                vis_alpha += (alpha_mask_pc_ntp, alpha_mask_pc2)
                vis_dyn += (dynamic_mask_pc_ntp, dynamic_mask_pc2)
                vis_static += (~true_mask, static_mask_pc2)
            vis_collage = visualize(
                videos=vis_videos, depths=vis_depths,
                alpha_masks=vis_alpha, dynamic_masks=vis_dyn, static_masks=vis_static,
            )
        elif render_all:
            vis_collage = visualize(
                videos=(video_src, video_pc, video_pc_ntp, video_pc2),
                depths=(depths_src, depths_pc, depths_pc_ntp, depths_pc2),
                alpha_masks=(alpha_mask_src, alpha_mask_pc, alpha_mask_pc_ntp, alpha_mask_pc2),
                dynamic_masks=(dynamic_mask_src, dynamic_mask_pc, dynamic_mask_pc_ntp, dynamic_mask_pc2),
                static_masks=(static_mask_src, static_mask_pc, ~true_mask, static_mask_pc2),
            )
        else:
            vis_collage = visualize(
                videos=(video_src, video_pc),
                depths=(depths_src, depths_pc),
                alpha_masks=(alpha_mask_src, alpha_mask_pc),
                dynamic_masks=(dynamic_mask_src, dynamic_mask_pc),
                static_masks=(static_mask_src, static_mask_pc),
            )
        save_mp4_with_gif(
            path.join(args.output_folder, "vis"), vis_collage, fps=fps, quality=5,
            gif_folder="" if args.save_vis_gif else None,
        )

    print(f"Point cloud render is saved to: {args.output_folder}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--recon_and_seg_folder", type=str, required=True)
    parser.add_argument("--cam_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--edits_path", type=str, default=None)  # Optional edits JSON (see utils/point_cloud/edit.py)

    parser.add_argument("--render_only_necessary", action="store_true", default=False)
    parser.add_argument("--ignore_sky_mask", action="store_true", default=False)
    parser.add_argument("--depth_outliers", type=str, default="gaussian", choices=("gaussian", "pool"))

    parser.add_argument("--save_vis", action="store_true", default=False)
    parser.add_argument("--save_vis_gif", action="store_true", default=False)

    parser.add_argument("--height", default=720, type=int)
    parser.add_argument("--width", default=1280, type=int)
    parser.add_argument("--num_frames", default=49, type=int)
    parser.add_argument("--dse_frame_interval", default=1, type=int)  # Stride through DSE frames at unprojection
    args = parser.parse_args()

    main(args)
