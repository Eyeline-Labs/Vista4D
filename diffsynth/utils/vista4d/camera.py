from functools import partial

import torch


def get_plucker_embedding(intrinsics, cam_c2w, height, width, height_dit=None, width_dit=None, flip_flag=None):
    """
    Computes the Plucker embedding given camera intrinsics and extrinsics

    Params:
        intrinsics (torch.Tensor): Camera intrinsics, shape b f 4, where 4 is [ fx fy cx cy ]
        cam_c2w (torch.Tensor): Camera extrinsics, shape b f 4 4
        ...

    Returns:
        plucker (torch.Tensor): Plucker embedding, shape b f h w 6

    From AC3D: https://github.com/snap-research/ac3d/blob/3c1e29e688f4a6d0f0ad41f1bf75d2eab709dac2/training/controlnet_datasets_camera.py#L111
    """

    custom_meshgrid = partial(torch.meshgrid, indexing="ij")

    batch_size, num_frames = intrinsics.shape[:2]

    use_dit_hw = True
    if height_dit is None or width_dit is None:
        use_dit_hw = False
        height_dit = height
        width_dit = width
    else:
        patch_height = height / height_dit
        patch_width = width / width_dit

    j, i = custom_meshgrid(
        torch.linspace(0, height_dit - 1, height_dit, device=cam_c2w.device, dtype=cam_c2w.dtype),
        torch.linspace(0, width_dit - 1, width_dit, device=cam_c2w.device, dtype=cam_c2w.dtype),
    )
    # b f (h w)
    i = i.reshape([1, 1, height_dit * width_dit]).expand([batch_size, num_frames, height_dit * width_dit]) + 0.5
    j = j.reshape([1, 1, height_dit * width_dit]).expand([batch_size, num_frames, height_dit * width_dit]) + 0.5

    if use_dit_hw:
        i = i * patch_width + (patch_width / 2)
        j = j * patch_height + (patch_height / 2)

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, height_dit - 1, height_dit, device=cam_c2w.device, dtype=cam_c2w.dtype),
            torch.linspace(width_dit - 1, 0, width_dit, device=cam_c2w.device, dtype=cam_c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, height_dit * width_dit]).expand(batch_size, 1, height_dit * width_dit) + 0.5
        j_flip = j_flip.reshape([1, 1, height_dit * width_dit]).expand(batch_size, 1, height_dit * width_dit) + 0.5
        if use_dit_hw:
            i_flip = i_flip * patch_width + (patch_width / 2)
            j_flip = j_flip * patch_height + (patch_height / 2)

        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)  # b f 1

    zs = torch.ones_like(i)  # b f (h w)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # b f (h w) 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # b f (h w) 3

    rays_d = directions @ cam_c2w[..., :3, :3].transpose(-1, -2)  # b f (h w) 3
    rays_o = cam_c2w[..., :3, 3]  # b f 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # b f (h w) 3
    # cam_c2w @ directions
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # b f (h w) 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(batch_size, cam_c2w.shape[1], height_dit, width_dit, 6)  # b f h w 6
    return plucker
