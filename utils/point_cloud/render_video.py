from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import torch

from utils.media import intrinsics_to_K
from utils.misc import cleanup
from utils.point_cloud.point_cloud import render, unproject


@torch.no_grad()
def render_video(
    video: npt.NDArray[np.uint8],  # f h w 3
    depths: npt.NDArray[np.float32],  # f h w
    cam_c2w_src: npt.NDArray[np.float32],  # f 4 4
    cam_c2w_tgt: npt.NDArray[np.float32],  # f 4 4
    intrinsics_src: npt.NDArray[np.float32],  # f 4, 4 is [fx fy cx cy]
    intrinsics_tgt: npt.NDArray[np.float32],  # f 4, 4 is [fx fy cx cy]
    dynamic_mask: Optional[npt.NDArray[np.bool_]] = None,  # f h w
    static_mask: Optional[npt.NDArray[np.bool_]] = None,  # f h w
    render_dynamic_mask: bool = True,
    double_reprojection: bool = False,
    no_temporal_persistence: bool = False,
    edit_fn: Optional[Callable] = None,  # (points_color, points_pos, visible, visible_ntp, indices) -> first four
    dtype: torch.dtype = torch.float32,  # Use bfloat16 for speed and memory
    device: str = "cuda",
    verbose: bool = False,
):
    cam_c2w_src = np.diag([-1, -1, 1, 1])[None] @ cam_c2w_src  # Convert to correct coordinate convention for rendering
    cam_c2w_tgt = np.diag([-1, -1, 1, 1])[None] @ cam_c2w_tgt

    num_frames, height, width, _ = video.shape
    num_frames_tgt = cam_c2w_tgt.shape[0]  # May be < num_frames when src has extra unprojection frames (e.g. DSE)

    video = torch.from_numpy(video).to(dtype=dtype, device=device) / 255  # [0, 255] -> [0, 1]
    depths = torch.from_numpy(depths).to(dtype=dtype, device=device)
    cam_c2w_src = torch.from_numpy(cam_c2w_src).to(dtype=dtype, device=device)
    cam_c2w_tgt = torch.from_numpy(cam_c2w_tgt).to(dtype=dtype, device=device)
    K_src = torch.from_numpy(intrinsics_to_K(intrinsics_src)).to(dtype=dtype, device=device)
    K_tgt = torch.from_numpy(intrinsics_to_K(intrinsics_tgt)).to(dtype=dtype, device=device)
    dynamic_mask = torch.from_numpy(dynamic_mask).to(dtype=torch.bool, device=device)
    static_mask = torch.from_numpy(static_mask).to(dtype=torch.bool, device=device)

    # Unproject src once; derive both visibilities (temporally-persistent `visible` and one-hot `visible_ntp`)
    points_color, points_pos, visible, indices = unproject(
        video=video,
        depths=depths,
        cam_c2w=cam_c2w_src,
        K=K_src,
        dynamic_mask=dynamic_mask,
        static_mask=static_mask,
    )
    # Drop any extra src-only columns (e.g. DSE frames beyond the render window) so `render` iterates tgt-aligned
    # Static points stay visible (row broadcast to True); dynamic points from dropped columns are culled
    visible = visible[:, :num_frames_tgt]
    # `visible_ntp` is one-hot by origin frame — scatter from indices avoids a redundant `unproject` call. This is
    # needed by both `double_reprojection` (its NTP-target render) and `no_temporal_persistence` (which swaps
    # `visible` → `visible_ntp` for the main render)
    if double_reprojection or no_temporal_persistence:
        visible_ntp = torch.zeros(indices.shape[0], num_frames, dtype=torch.bool, device=device)
        visible_ntp.scatter_(1, indices[:, 0:1], True)  # indices[:, 0] is the origin frame per point
        visible_ntp = visible_ntp[:, :num_frames_tgt]  # Drop src-only columns, matching `visible` above
    else:
        visible_ntp = None

    ### Optional: Edit the point cloud via `edit_fn(points_color, points_pos, visible, visible_ntp, indices)` ###
    if edit_fn is not None:
        points_color, points_pos, visible, visible_ntp = edit_fn(
            points_color, points_pos, visible, visible_ntp, indices,
        )
    del indices

    # `dynamic_mask="visible"` lets `render` compute dyn = `visible.sum == 1`. Under no_temporal_persistence that would
    # be True for every point (NTP is strictly one-hot) and classify everything as dynamic, so we compute the mask from
    # the temporally-persistent `visible` explicitly and pass it as a tensor.
    render_visible = visible_ntp if no_temporal_persistence else visible
    if no_temporal_persistence:
        render_dyn = (visible.sum(dim=1) == 1) if (render_dynamic_mask or double_reprojection) else None
    else:
        render_dyn = "visible" if (render_dynamic_mask or double_reprojection) else None
    video_pc, depths_pc, alpha_mask_pc, dynamic_mask_pc = render(
        points_color=points_color,
        points_pos=points_pos,
        visible=render_visible,
        cam_c2w=cam_c2w_tgt,
        K=K_tgt,
        height=height,
        width=width,
        dynamic_mask=render_dyn,
        verbose=verbose,
    )
    cleanup()
    depths_pc[~alpha_mask_pc] = 0.0  # Set all undefined depths to 0

    def to_numpy(video, depths, alpha_mask, dynamic_mask):
        video = (video.cpu().to(torch.float32).numpy().clip(0.0, 1.0) * 255).astype(np.uint8)
        depths = depths.cpu().to(torch.float32).numpy()
        alpha_mask = alpha_mask.cpu().numpy()
        dynamic_mask = dynamic_mask.cpu().numpy() if dynamic_mask is not None else None
        return video, depths, alpha_mask, dynamic_mask

    if not double_reprojection:
        return to_numpy(video_pc, depths_pc, alpha_mask_pc, dynamic_mask_pc)

    # Render source at target without temporal persistence (`visible_ntp` was derived upfront, reuse it here)
    assert dynamic_mask_pc is not None
    del video, depths, dynamic_mask, static_mask  # No longer needed on GPU
    video_pc_ntp, depths_pc_ntp, alpha_mask_pc_ntp, dynamic_mask_pc_ntp = render(
        points_color=points_color,
        points_pos=points_pos,
        visible=visible_ntp,
        cam_c2w=cam_c2w_tgt,
        K=K_tgt,
        height=height,
        width=width,
        dynamic_mask=visible.sum(dim=1) == 1,
        verbose=verbose,
    )
    del points_color, points_pos, visible, visible_ntp  # No longer needed on GPU
    cleanup()
    depths_pc_ntp[~alpha_mask_pc_ntp] = 0.0  # Set all undefined depths to 0

    # Double reprojection: Render rendered target (which has no temporal persistence) at source
    points_color_pc, points_pos_pc, visible_pc, indices_pc = unproject(
        video=video_pc_ntp,
        depths=depths_pc_ntp,
        cam_c2w=cam_c2w_tgt,
        K=K_tgt,
        dynamic_mask=dynamic_mask_pc_ntp,
        static_mask=alpha_mask_pc_ntp & ~dynamic_mask_pc_ntp,
    )
    del indices_pc, cam_c2w_tgt, K_tgt  # No longer needed on GPU
    # Double-reprojection target: render-window src cameras only (first num_frames_tgt entries of cam_c2w_src)
    video_pc2, depths_pc2, alpha_mask_pc2, dynamic_mask_pc2 = render(
        points_color=points_color_pc,
        points_pos=points_pos_pc,
        visible=visible_pc,
        cam_c2w=cam_c2w_src[:num_frames_tgt],
        K=K_src[:num_frames_tgt],
        height=height,
        width=width,
        dynamic_mask="visible" if render_dynamic_mask else None,
        verbose=verbose,
    )
    del points_color_pc, points_pos_pc, visible_pc, cam_c2w_src, K_src  # No longer needed on GPU
    cleanup()
    depths_pc2[~alpha_mask_pc2] = 0.0  # Set all undefined depths to 0

    dynamic_mask_pc = None if not render_dynamic_mask else dynamic_mask_pc
    return (
        to_numpy(video_pc, depths_pc, alpha_mask_pc, dynamic_mask_pc),
        to_numpy(video_pc_ntp, depths_pc_ntp, alpha_mask_pc_ntp, dynamic_mask_pc_ntp),
        to_numpy(video_pc2, depths_pc2, alpha_mask_pc2, dynamic_mask_pc2),
    )
