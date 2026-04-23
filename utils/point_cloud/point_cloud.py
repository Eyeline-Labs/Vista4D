from typing import Optional, Union

import torch
from tqdm.auto import tqdm


def inverse(t, dtype=torch.float32):
    return t.to(dtype=dtype).inverse().to(dtype=t.dtype)


# `autocast(enabled=False)`: SAM3's tracking predictor enters a global bf16 autocast on init
# and never exits it (sam3_tracking_predictor.py:50). Every CUDA matmul after the first SAM3
# load then runs in bf16 — including the projection math in this function, which quantizes
# pixel coordinates coarsely enough to produce screen-wide stripe artifacts. The guard lives
# here (and in `render` + `apply_op_to_subset`) rather than on outer wrappers so it can't
# accidentally cover a SAM3 call site.
@torch.autocast(device_type="cuda", enabled=False)
def unproject(
    video: torch.Tensor,  # f h w 3, uint8
    depths: torch.Tensor,  # f h w, (b)float
    cam_c2w: torch.Tensor,  # f 4 4, (b)float
    K: torch.Tensor,  # f 3 3, (b)float
    dynamic_mask: Optional[torch.Tensor] = None,  # f h w, bool
    static_mask: Optional[torch.Tensor] = None,  # f h w, bool
):
    num_frames, height, width, _ = video.shape
    dtype, device = depths.dtype, depths.device

    depths = depths.to(dtype=dtype, device=device)
    cam_c2w = cam_c2w.to(dtype=dtype, device=device)
    K = K.to(dtype=dtype, device=device)

    index_frame, index_height, index_width = torch.meshgrid(  # Create frame & pixel coordinate matrix
        torch.arange(num_frames, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )

    # Create pixel coordinates [u v 1] (homogeneous), u is width (x) and v is height (y)
    pixel_coords = torch.stack([index_width, index_height, torch.ones_like(index_height)], dim=-1)  # f h w 3
    pixel_coords = pixel_coords.to(dtype=dtype, device=device)
    pixel_coords[..., :2] += 0.5  # Shift to pixel centers

    points_cam = (pixel_coords.flatten(1, 2) @ inverse(K).mT) * depths.flatten(1, 2)[..., None]  # f (h w) 3
    #             ^ f h w 3 -> f (h w) 3       ^ f 3 3          ^ f h w -> f (h w) 1

    R = cam_c2w[:, :3, :3]  # f 3 3
    T = cam_c2w[:, None, :3, 3]  # f 1 3
    points_world = (points_cam @ R.mT) + T  # f (h w) 3

    get_true_mask = lambda: torch.ones(*depths.shape, dtype=torch.bool, device=device)
    dynamic_mask = dynamic_mask if dynamic_mask is not None else get_true_mask()  # f h w
    static_mask = static_mask if static_mask is not None else ~get_true_mask()  # f h w
    mask_flat = (dynamic_mask | static_mask).view(-1)  # All points we should keep after unprojection, (f h w)

    indices = torch.stack([index_frame, index_height, index_width], dim=-1).view(-1, 3)  # Indices [frame height width]

    frame_indices = index_frame.reshape(-1)[mask_flat]  # n, int64
    visible = torch.zeros(frame_indices.shape[0], num_frames, dtype=torch.bool, device=device)  # n f
    visible.scatter_(1, frame_indices.unsqueeze(1), True)  # Equivalent to one_hot but avoids int64 intermediate
    del frame_indices
    visible = (visible & dynamic_mask.view(-1)[mask_flat][:, None]) | static_mask.view(-1)[mask_flat][:, None]

    return (
        video.reshape(-1, 3)[mask_flat],  # Points rgb, n 3, uint8
        points_world.reshape(-1, 3)[mask_flat],  # Points xyz, n 3, (b)float16
        visible,  # n f
        indices[mask_flat],  # Indices, n 3, 3 is [f h w]
    )


def render_frame(
    points_color: torch.Tensor,  # n 3, (b)float16
    points_pos: torch.Tensor,  # n 3, in world coordinates
    cam_c2w: torch.Tensor,  # 4 4
    K: torch.Tensor,  # 3 3
    height: int,
    width: int,
    dynamic_mask: Optional[torch.Tensor] = None,  # n, bool
    batch_size: Optional[int] = int(1e7),  # Process points in chunks to prevent GPU OOM
):
    """
    Renders a single frame using bilinear splatting with z-buffering.
    """
    dtype, device = points_pos.dtype, points_pos.device
    num_points = points_pos.shape[0]
    batch_size = int(batch_size)  # If we put 1e7 then the result is actually a float, convert to int!

    z_buffer = torch.full((height, width), float("inf"), dtype=dtype, device=device)
    frame_rgb = torch.zeros((height, width, 3), dtype=dtype, device=device)
    valid_mask = torch.zeros((height, width), dtype=torch.bool, device=device)

    has_dynamic_mask = dynamic_mask is not None
    dynamic_mask_buffer = torch.zeros((height, width), dtype=dtype, device=device) if has_dynamic_mask else None

    cam_w2c = inverse(cam_c2w)  # Precompute camera matrices (project from world to camera)
    R_w2c = cam_w2c[:3, :3]
    T_w2c = cam_w2c[:3, 3]

    for i in range(0, num_points, batch_size):  # Process in batches to avoid OOM

        ### BATCHING & PROJECTION ###
        points_pos_batch = points_pos[i:i + batch_size]  # b 3
        points_color_batch = points_color[i:i + batch_size]  # b 3
        if has_dynamic_mask:
            dynamic_mask_batch = dynamic_mask[i:i + batch_size]  # b

        points_cam_batch = (points_pos_batch @ R_w2c.T) + T_w2c  # World -> camera, b 3,
        depths_batch = points_cam_batch[:, 2]
        valid_z_batch = depths_batch > 1e-5  # Filter points behind camera (z <= 0)
        if not valid_z_batch.any():
            continue
        points_cam_batch = points_cam_batch[valid_z_batch]
        points_color_batch = points_color_batch[valid_z_batch]
        depths_batch = depths_batch[valid_z_batch]
        if has_dynamic_mask:
            dynamic_mask_batch = dynamic_mask_batch[valid_z_batch]

        points_uvz = (K @ points_cam_batch.T).T  # Project from camera coords to image plane
        u = points_uvz[:, 0] / points_uvz[:, 2]  # Do P_uv = (K @ P_c.T).T, then divide by z
        v = points_uvz[:, 1] / points_uvz[:, 2]

        ### BILINEAR EXPANSION ###
        u_0 = torch.floor(u).to(torch.int32)  # Bilinear expansion (1 point -> 4 fragments)
        v_0 = torch.floor(v).to(torch.int32)
        du = u - u_0  # Fractional offsets for weights
        dv = v - v_0

        # Bilinear weights of four neighbors: (0,0), (1,0), (0,1), (1,1)
        w_00 = (1 - du) * (1 - dv)  # Top-left
        w_01 = (1 - du) * dv  # Bottom-left
        w_10 = du * (1 - dv)  # Top-right
        w_11 = du * dv  # Bottom-right

        u_frag = torch.cat([u_0, u_0, u_0 + 1, u_0 + 1])  # All four fragments, result size of 4 * batch
        v_frag = torch.cat([v_0, v_0 + 1, v_0, v_0 + 1])
        w_frag = torch.cat([w_00, w_01, w_10, w_11])
        d_frag = depths_batch.repeat(4)
        c_frag = points_color_batch.repeat(4, 1)
        if has_dynamic_mask:
            dyn_frag = dynamic_mask_batch.repeat(4)

        ### FILTERING & SORTING ###
        valid_uv = (u_frag >= 0) & (u_frag < width) & (v_frag >= 0) & (v_frag < height)
        if not valid_uv.any():  # Filter out-of-bounds pixels (outside of the height & width box)
            continue
        u_frag, v_frag = u_frag[valid_uv], v_frag[valid_uv]
        d_frag, c_frag = d_frag[valid_uv], c_frag[valid_uv]
        w_frag = w_frag[valid_uv]
        if has_dynamic_mask:
            dyn_frag = dyn_frag[valid_uv]

        sort_idx = torch.argsort(d_frag, descending=True)  # Sort by depth (Painter's algorithm), process far -> near
        u_frag, v_frag = u_frag[sort_idx], v_frag[sort_idx]
        d_frag, c_frag = d_frag[sort_idx], c_frag[sort_idx]
        w_frag = w_frag[sort_idx]
        if has_dynamic_mask:
            dyn_frag = dyn_frag[sort_idx]

        ### Z-BUFFER UPDATE ###
        indices_flat = v_frag * width + u_frag
        is_closer = d_frag < z_buffer.view(-1)[indices_flat]  # Get indices to update (if closer than global z-buffer)
        indices_flat = indices_flat[is_closer]  # Only update the ones that are closer than the current buffer

        z_buffer.view(-1)[indices_flat] = d_frag[is_closer]  # Update z-buffer

        frame_rgb_current = frame_rgb.view(-1, 3)[indices_flat]  # Update color with alpha blending
        alpha = w_frag[is_closer].unsqueeze(-1)
        frame_rgb.view(-1, 3)[indices_flat] = c_frag[is_closer] * alpha + frame_rgb_current * (1 - alpha)
        valid_mask.view(-1)[indices_flat] = True
        if has_dynamic_mask:
            dynamic_mask_current = dynamic_mask_buffer.view(-1)[indices_flat]
            alpha_flat = alpha[:, 0]
            dynamic_mask_buffer.view(-1)[indices_flat] =\
                dyn_frag[is_closer].to(dtype=dtype) * alpha_flat + dynamic_mask_current * (1 - alpha_flat)

    if has_dynamic_mask:
        dynamic_mask_buffer = dynamic_mask_buffer > 0  # Be conservative about static points, hence >0 and not ==1
    return frame_rgb, z_buffer, valid_mask, dynamic_mask_buffer


def render_frame(
    points_color: torch.Tensor,  # n 3, (b)float16
    points_pos: torch.Tensor,  # n 3, in world coordinates
    cam_c2w: torch.Tensor,  # 4 4
    K: torch.Tensor,  # 3 3
    height: int,
    width: int,
    dynamic_mask: Optional[torch.Tensor] = None,  # n, bool
    z_tolerance: float = 0.02,
    occlusion_sharpness: float = 10.0,
    batch_size: int = int(1e7),  # Process points in chunks to prevent GPU OOM
):
    dtype, device = points_pos.dtype, points_pos.device
    num_points = points_pos.shape[0]
    batch_size = int(batch_size)  # If we put 1e7 then the number is actually a float, convert to int!

    accum_color = torch.zeros((height * width, 3), dtype=torch.float32, device=device)
    accum_depths = torch.zeros((height * width,), dtype=torch.float32, device=device)
    accum_weight = torch.zeros((height * width,), dtype=torch.float32, device=device)
    z_buffer = torch.full((height * width,), float("inf"), dtype=dtype, device=device)

    # Temporary buffer for intra-batch occlusion culling
    min_z_batch = torch.empty((height * width,), dtype=torch.float32, device=device)

    has_dynamic_mask = dynamic_mask is not None
    accum_dyn_mask = torch.zeros((height * width,), dtype=torch.float32, device=device) if has_dynamic_mask else None

    cam_w2c = inverse(cam_c2w)  # Precompute camera matrices (project from world to camera)
    R_w2c = cam_w2c[:3, :3]
    T_w2c = cam_w2c[:3, 3]

    for i in range(0, num_points, batch_size):

        ### BATCHING & PROJECTION ###
        points_pos_batch = points_pos[i:i + batch_size]
        points_color_batch = points_color[i:i + batch_size]
        if has_dynamic_mask:
            dynamic_mask_batch = dynamic_mask[i:i + batch_size]

        points_cam_batch = (points_pos_batch @ R_w2c.T) + T_w2c  # World -> camera
        depths_batch = points_cam_batch[:, 2]
        valid_z = depths_batch > 1e-5  # Filter points behind camera (z <= 0)
        if not valid_z.any():
            continue

        points_cam_batch = points_cam_batch[valid_z]
        points_color_batch = points_color_batch[valid_z]
        depths_batch = depths_batch[valid_z]
        if has_dynamic_mask:
            dynamic_mask_batch = dynamic_mask_batch[valid_z]

        points_uvz = (K @ points_cam_batch.T).T  # Project from camera coords to image plane
        u = points_uvz[:, 0] / points_uvz[:, 2]  # Do P_uv = (K @ P_c.T).T, then divide by z
        v = points_uvz[:, 1] / points_uvz[:, 2]

        ### BILINEAR EXPANSION ###
        u_0 = torch.floor(u).to(torch.int32)  # Bilinear expansion (1 point -> 4 fragments)
        v_0 = torch.floor(v).to(torch.int32)
        du = u - u_0  # Fractional offsets for weights
        dv = v - v_0

        # Bilinear weights of four neighbors: (0,0), (1,0), (0,1), (1,1)
        w_00 = (1 - du) * (1 - dv)  # Top-left
        w_01 = (1 - du) * dv  # Bottom-left
        w_10 = du * (1 - dv)  # Top-right
        w_11 = du * dv  # Bottom-right

        u_frag = torch.cat([u_0, u_0, u_0 + 1, u_0 + 1])  # All four fragments, result size of 4 * batch
        v_frag = torch.cat([v_0, v_0 + 1, v_0, v_0 + 1])
        w_geom = torch.cat([w_00, w_01, w_10, w_11])  # Geometric weight (amount for each fragment)
        d_frag = depths_batch.repeat(4)
        c_frag = points_color_batch.repeat(4, 1)
        if has_dynamic_mask:
            dyn_frag = dynamic_mask_batch.repeat(4).float()

        # Depth weighting
        d_safe = d_frag.to(torch.float32).clamp(min=1e-1)  # Clamp min weight to prevent far points from disappearing
        w_depth = torch.pow(d_safe, -occlusion_sharpness).clamp(min=1e-10)
        w_final = (w_geom.to(torch.float32) * w_depth)  # Weight geometric weight by depth weight

        # FILTERING
        # Also drop zero-weight bilinear corners (du=0 or dv=0 makes two of the 4 fragments degenerate). A zero-weight
        # fragment contributes nothing to accum but could still trigger `should_reset` below, clearing prior non-zero
        # contributions at that pixel and locking z_buffer to a depth no later fragment can blend into. This produces
        # column/row stripes on planar surfaces when src and insert points interleave with different depth profiles.
        valid_uv = (u_frag >= 0) & (u_frag < width) & (v_frag >= 0) & (v_frag < height) & (w_final > 0)
        if not valid_uv.any():  # Filter out-of-bounds pixels (outside of the height & width box)
            continue

        idx_flat = v_frag[valid_uv] * width + u_frag[valid_uv]
        d_frag = d_frag[valid_uv]
        c_frag = c_frag[valid_uv]
        w_final = w_final[valid_uv]
        if has_dynamic_mask:
            dyn_frag = dyn_frag[valid_uv]

        # Sort depth by descending (far -> near) for painter's algorithm: Last write to an index is the closest point
        sort_idx = torch.argsort(d_frag, descending=True)
        idx_flat = idx_flat[sort_idx]
        d_frag = d_frag[sort_idx]
        c_frag = c_frag[sort_idx]
        w_final = w_final[sort_idx]
        if has_dynamic_mask:
            dyn_frag = dyn_frag[sort_idx]

        # INTRA-BATCH CULLING
        min_z_batch.fill_(float("inf"))  # Reset local batch z-buffer

        # Get minimum depth within batch (far -> near sorting means last write is closest point)
        min_z_batch[idx_flat] = d_frag.to(torch.float32)  # If two points has the same idx, last write will take over

        min_d_batch = min_z_batch[idx_flat]  # Filter points occluded by other points in the same batch
        rel_diff_batch = (torch.log1p(d_frag.to(torch.float32)) - torch.log1p(min_d_batch)).abs()
        visible_in_batch = rel_diff_batch <= z_tolerance

        if not visible_in_batch.any():
            continue

        idx_flat = idx_flat[visible_in_batch]  # Only keep points that are not occluded/closest within threshold
        d_frag = d_frag[visible_in_batch]
        c_frag = c_frag[visible_in_batch]
        w_final = w_final[visible_in_batch]
        if has_dynamic_mask:
            dyn_frag = dyn_frag[visible_in_batch]

        ### GLOBAL Z-BUFFER UPDATE ###
        z_curr = z_buffer[idx_flat]

        log_curr = torch.log1p(z_curr.to(torch.float32))  # We calculate relative diff using float32 for stability
        log_new = torch.log1p(d_frag.to(torch.float32))
        rel_diff = (log_new - log_curr).abs()
        is_closer = d_frag < z_curr
        should_reset = is_closer & (rel_diff > z_tolerance)  # Reset pixels which are noticeably closer than buffer

        if should_reset.any():
            reset_indices = idx_flat[should_reset].to(torch.int64)
            accum_color.index_fill_(0, reset_indices, 0.0)
            accum_depths.index_fill_(0, reset_indices, 0.0)
            accum_weight.index_fill_(0, reset_indices, 0.0)
            if has_dynamic_mask:
                accum_dyn_mask.index_fill_(0, reset_indices, 0.0)  # Also reset dynamic mask if provided
            z_buffer[reset_indices] = d_frag[should_reset]

        should_blend = should_reset | (rel_diff <= z_tolerance)  # Blend reset pixels & pixels within tolerance
        if not should_blend.any():
            continue
        idx_blend = idx_flat[should_blend]
        w_blend = w_final[should_blend]

        accum_color.index_add_(0, idx_blend, c_frag[should_blend].to(torch.float32) * w_blend[..., None])
        accum_depths.index_add_(0, idx_blend, d_frag[should_blend].to(torch.float32) * w_blend)
        accum_weight.index_add_(0, idx_blend, w_blend)
        if has_dynamic_mask:
            accum_dyn_mask.index_add_(0, idx_blend, dyn_frag[should_blend] * w_blend)

    valid_mask = (accum_weight > 1e-16).squeeze()  # Alpha mask of the point cloud render frame

    frame_rgb = torch.zeros((height * width, 3), dtype=dtype, device=device)  # Normalize colors
    result_rgb = accum_color[valid_mask] / accum_weight[valid_mask][..., None]
    frame_rgb[valid_mask] = result_rgb.to(dtype=dtype)

    frame_depths = torch.zeros((height * width,), dtype=dtype, device=device)  # Normalize depths
    frame_depths[valid_mask] = (accum_depths[valid_mask] / accum_weight[valid_mask]).to(dtype=dtype)

    dynamic_mask_pc = None
    if has_dynamic_mask:
        dynamic_mask_pc = torch.zeros((height * width,), dtype=torch.bool, device=device)
        dynamic_mask_pc[valid_mask] = (accum_dyn_mask[valid_mask] / accum_weight[valid_mask]) > 0.0
        dynamic_mask_pc = dynamic_mask_pc.reshape(height, width)

    return (
        frame_rgb.reshape(height, width, 3),
        frame_depths.reshape(height, width),
        valid_mask.reshape(height, width),
        dynamic_mask_pc,
    )


# `autocast(enabled=False)`: see `unproject` for context. Keeps `render_frame`'s projection
# and bilinear-splatting math in fp32 even after SAM3 has leaked a bf16 autocast globally.
@torch.autocast(device_type="cuda", enabled=False)
def render(
    points_color: torch.Tensor,  # n 3, 3 is [r g b], (b)float
    points_pos: torch.Tensor,  # n 3, 3 is [x y z], (b)float
    visible: torch.Tensor,  # n f, bool
    cam_c2w: torch.Tensor,  # f 4 4, (b)float
    K: torch.Tensor,  # f 3 3, (b)float
    height: int,
    width: int,
    dynamic_mask: Optional[Union[str, torch.Tensor]] = None,  # n, dynamic mask to render, if "visible", use visible
    verbose: bool = False,
):
    num_points, num_frames = visible.shape
    dtype, device = points_pos.dtype, points_pos.device

    assert points_color.shape == (num_points, 3),\
        f"`points_color` must have shape (n, 3) where n={num_points} (from `visible`), got {points_color.shape}."
    assert points_pos.shape == (num_points, 3),\
        f"`points_pos` must have shape (n, 3) where n={num_points} (from `visible`), got {points_pos.shape}."

    video_pc = torch.zeros(num_frames, height, width, 3, device=device)
    alpha_mask_pc = torch.zeros(num_frames, height, width, device=device, dtype=torch.bool)
    depths_pc = torch.zeros(num_frames, height, width, device=device)

    render_dynamic_mask = dynamic_mask is not None
    dynamic_mask_pc = torch.zeros(num_frames, height, width, device=device, dtype=torch.bool)\
        if render_dynamic_mask else None
    if dynamic_mask == "visible":
        dynamic_mask = visible.sum(dim=1) == 1  # Dynamic points are points which appear for exactly one (1) frame

    progress = range(num_frames)
    progress = tqdm(progress, desc="Rendering point cloud") if verbose else progress
    for i in progress:
        visible_i = visible[:, i]
        points_color_i = points_color[visible_i]
        points_pos_i = points_pos[visible_i]
        dynamic_mask_i = dynamic_mask[visible_i] if render_dynamic_mask else None
        cam_c2w_i = cam_c2w[i]
        K_i = K[i]

        video_pc[i], depths_pc[i], alpha_mask_pc[i], dynamic_mask_pc_i = render_frame(
            points_color=points_color_i,
            points_pos=points_pos_i,
            cam_c2w=cam_c2w_i,
            K=K_i,
            height=height,
            width=width,
            dynamic_mask=dynamic_mask_i,
            batch_size=1e7,
        )
        if render_dynamic_mask:
            dynamic_mask_pc[i] = dynamic_mask_pc_i

    return video_pc, depths_pc, alpha_mask_pc, dynamic_mask_pc
