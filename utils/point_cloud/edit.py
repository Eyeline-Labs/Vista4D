import json
import math
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from utils.media import intrinsics_to_K, load_recon_and_seg
from utils.point_cloud.filter import contract_mask
from utils.point_cloud.point_cloud import unproject
from utils.point_cloud.preprocess import preprocess_scene


OP_NAMES = ("translate", "rotate", "scale", "remove")
TARGET_KINDS = ("existing", "insert", "duplicate")
# Legacy aliases accepted at validation time and normalized to the canonical name above. Kept
# so JSONs written before the `"sam3"` → `"existing"` rename still load without a migration.
TARGET_KIND_ALIASES = {"sam3": "existing"}
SCOPES = ("global", "frame")


def load_edits(input_path):
    # Parse an edits JSON and lightly validate it. Schema:
    # {"edits": [
    #     {"target": {"kind": "existing"|"insert"|"duplicate", "prompt": "...", ("source": "<path>")},
    #      "ops": [{"op": "translate"|"rotate"|"scale"|"remove", "params": ...}, ...],
    #      "scope": "global"|"frame",
    #      ("mask_expansion": [radius, iterations]), ("centroid_threshold": 0.6)},
    #     ...]}
    with open(input_path, "r") as f:
        data = json.load(f)
    edits = data.get("edits", [])
    for i, edit in enumerate(edits):
        validate_edit(edit, index=i)
    return edits


def validate_edit(edit, index):
    assert "target" in edit, f"Edit #{index} missing required field `target`."
    assert "ops" in edit, f"Edit #{index} missing required field `ops`."
    assert "scope" in edit, f"Edit #{index} missing required field `scope`."
    assert edit["scope"] in SCOPES,\
        f"Edit #{index} has invalid `scope` `{edit['scope']}`; must be one of {SCOPES}."

    target = edit["target"]
    assert "kind" in target, f"Edit #{index} target missing required field `kind`."
    # Normalize legacy aliases (e.g., old "sam3") to the canonical name in-place so every
    # downstream comparison can check against `TARGET_KINDS` without re-aliasing.
    if target["kind"] in TARGET_KIND_ALIASES:
        target["kind"] = TARGET_KIND_ALIASES[target["kind"]]
    assert target["kind"] in TARGET_KINDS,\
        f"Edit #{index} target has invalid `kind` `{target['kind']}`; must be one of {TARGET_KINDS}."
    assert "prompt" in target, f"Edit #{index} target missing required field `prompt`."
    if target["kind"] == "insert":
        assert "source" in target,\
            f"Edit #{index} target has `kind=insert` but is missing required field `source`."

    for j, op in enumerate(edit["ops"]):
        assert "op" in op, f"Edit #{index}, op #{j} missing required field `op`."
        assert op["op"] in OP_NAMES,\
            f"Edit #{index}, op #{j} has invalid `op` `{op['op']}`; must be one of {OP_NAMES}."


def safe_quantile(tensor, q, max_samples=int(1e6)):
    # torch.quantile chokes above ~16M elements, so subsample when necessary
    if tensor.numel() > max_samples:
        idx = torch.randperm(tensor.numel(), device=tensor.device)[:max_samples]
        tensor = tensor.flatten()[idx]
    return torch.quantile(tensor, q)


def get_centroid(points, threshold):
    # Robust centroid: trim points beyond the `threshold` quantile of distance-to-median, then mean
    if points.shape[0] == 0:
        return points.new_zeros(3, dtype=torch.float32)
    points_f = points.to(torch.float32)
    median = torch.median(points_f, dim=0).values
    dists = (points_f - median).norm(p=2, dim=1)
    inliers = dists <= safe_quantile(dists, threshold)
    return points_f[inliers].mean(dim=0)


def rotation_matrix(rx_deg, ry_deg, rz_deg, device, dtype=torch.float32):
    # Euler ZYX rotation (matches world_rerender convention): R = Rz @ Ry @ Rx
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=dtype, device=device)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=dtype, device=device)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=dtype, device=device)
    return Rz @ Ry @ Rx


# `autocast(enabled=False)`: `rotate` does a matmul and `get_centroid` does a reduction —
# both get quantized to bf16 under SAM3's leaked global autocast (sam3_tracking_predictor.py:50),
# which is visible as rotation angles snapping or centroids drifting. Guarding the math here
# keeps it fp32; SAM3's own bf16 context is untouched.
@torch.autocast(device_type="cuda", enabled=False)
def apply_op_to_subset(
    points_pos: torch.Tensor,  # n 3
    subset_idx: torch.Tensor,  # k, long (indices into points_pos)
    op: Dict,
    centroid_threshold: float,
):
    # Apply a single translate/rotate/scale op to the subset, out-of-place on `points_pos`.
    # Always computes in float32 for stable matmul/centroid math, then casts back to input dtype.
    # `remove` is handled by the caller (it drops rows rather than transforming positions).
    if subset_idx.numel() == 0 or op["op"] == "remove":
        return points_pos

    orig_dtype = points_pos.dtype
    # `copy=True`: `.to(torch.float32)` is a no-op when points_pos is already float32 (returns
    # the same tensor), so without an explicit copy the `points_f[subset_idx] = ...` below
    # would mutate the input in place. That silently poisoned cam_ui's unproject_insert_cache
    # (insert+translate compounded the translate on every Apply) and violates this function's
    # own out-of-place contract. Force a copy so the caller's tensor is never touched.
    points_f = points_pos.to(torch.float32, copy=True)
    sel = points_f[subset_idx]

    if op["op"] == "translate":
        t = torch.tensor(op["params"], dtype=torch.float32, device=points_f.device)
        assert t.shape == (3,), f"`translate` params must be length 3, got {op['params']}."
        points_f[subset_idx] = sel + t
    elif op["op"] == "rotate":
        assert len(op["params"]) == 3, f"`rotate` params must be length 3, got {op['params']}."
        centroid = get_centroid(sel, threshold=centroid_threshold)
        R = rotation_matrix(*op["params"], device=points_f.device)
        points_f[subset_idx] = (sel - centroid) @ R.T + centroid
    elif op["op"] == "scale":
        params = op["params"]
        factor = (params,) * 3 if isinstance(params, (int, float)) else tuple(params)
        assert len(factor) == 3, f"`scale` params must be a scalar or length 3, got {params}."
        centroid = get_centroid(sel, threshold=centroid_threshold)
        factor_t = torch.tensor(factor, dtype=torch.float32, device=points_f.device)
        points_f[subset_idx] = (sel - centroid) * factor_t + centroid
    else:
        raise NotImplementedError(f"Unrecognized op `{op['op']}`.")

    return points_f.to(orig_dtype)


def expand_mask(mask, radius, iterations):
    # Dilate a (f h w) mask; mirrors `--edit_mask_expansion` in world_rerender
    return ~contract_mask(~mask, radius=radius, iterations=iterations)


def lookup_mask_at_points(
    mask: torch.Tensor,     # f h w bool
    indices: torch.Tensor,  # n 3 int64, 3 is [frame height width]
    valid: torch.Tensor,    # n bool, which points are eligible (e.g. same source, not sky)
    num_frames_mask: int,
):
    # Look up `mask` at each point's origin pixel; out-of-range frames (e.g. DSE tail) are never selected
    selected = torch.zeros(indices.shape[0], dtype=torch.bool, device=mask.device)
    in_range = indices[:, 0] < num_frames_mask
    valid = valid & in_range
    if valid.any():
        v = valid.nonzero(as_tuple=False).squeeze(-1)
        f_idx, h_idx, w_idx = indices[v, 0], indices[v, 1], indices[v, 2]
        selected[v] = mask[f_idx, h_idx, w_idx]
    return selected


# Per-point state carried through all edits: each key is a tensor with leading dim = num points.
# Kept as a plain dict (rather than a dataclass) so it composes naturally with `render_video`'s existing
# `points_color`/`points_pos`/`visible`/`visible_ntp`/`indices` tensors.
BUFFER_KEYS = ("points_color", "points_pos", "visible", "visible_ntp", "indices", "source_id", "is_sky")


def filter_buffers(buffers, keep):
    # Drop rows where `keep` is False across all buffer keys
    for key in BUFFER_KEYS:
        if buffers[key] is not None:
            buffers[key] = buffers[key][keep]
    return buffers


def append_buffers(buffers, other):
    # Concatenate `other` into `buffers` along the point dimension
    for key in BUFFER_KEYS:
        if buffers[key] is None:
            assert other[key] is None, f"Buffers inconsistent: `{key}` is None in target but set in other."
            continue
        buffers[key] = torch.cat([buffers[key], other[key]], dim=0)
    return buffers


def clone_buffers_subset(buffers, subset_idx):
    # Deep-clone the subset as a new buffers dict (for `duplicate` target-mode)
    return {key: (buffers[key][subset_idx].clone() if buffers[key] is not None else None) for key in BUFFER_KEYS}


def unproject_insert(
    insert: Dict,
    source_id: int,
    num_frames_tgt: int,
    sam_mask_np: npt.NDArray[np.bool_],
    double_reprojection: bool,
    device: str,
):
    # Unproject an insert scene filtered to the SAM-selected pixels, returning buffers ready to concat
    video = torch.from_numpy(insert["video"]).to(dtype=torch.float32, device=device) / 255
    depths = torch.from_numpy(insert["depths"]).to(dtype=torch.float32, device=device)
    cam_c2w = np.diag([-1, -1, 1, 1])[None] @ insert["cam_c2w"]  # Match render_video's coordinate flip
    cam_c2w = torch.from_numpy(cam_c2w).to(dtype=torch.float32, device=device)
    K = torch.from_numpy(insert["K"]).to(dtype=torch.float32, device=device)
    dynamic_mask = torch.from_numpy(insert["dynamic_mask"] & sam_mask_np).to(dtype=torch.bool, device=device)
    static_mask = torch.from_numpy(insert["static_mask"] & sam_mask_np).to(dtype=torch.bool, device=device)

    points_color, points_pos, visible, indices = unproject(
        video=video, depths=depths, cam_c2w=cam_c2w, K=K,
        dynamic_mask=dynamic_mask, static_mask=static_mask,
    )
    visible = visible[:, :num_frames_tgt]
    if double_reprojection:
        num_frames_total = depths.shape[0]
        visible_ntp = torch.zeros(indices.shape[0], num_frames_total, dtype=torch.bool, device=device)
        visible_ntp.scatter_(1, indices[:, 0:1], True)
        visible_ntp = visible_ntp[:, :num_frames_tgt]
    else:
        visible_ntp = None

    # Inserts never touch sky points; look up the insert scene's own sky_mask for is_sky per point
    sky_np = insert["sky_mask"].astype(np.bool_)
    sky_t = torch.from_numpy(sky_np).to(device=device)
    is_sky = lookup_mask_at_points(
        sky_t, indices, torch.ones(indices.shape[0], dtype=torch.bool, device=device), sky_np.shape[0],
    )
    source_id_t = torch.full((indices.shape[0],), source_id, dtype=torch.long, device=device)
    return {
        "points_color": points_color, "points_pos": points_pos,
        "visible": visible, "visible_ntp": visible_ntp,
        "indices": indices, "source_id": source_id_t, "is_sky": is_sky,
    }


def select_subset_idx(buffers, target_source_id, mask, num_frames_mask):
    # Return long-tensor indices into `buffers` of points: matching source, under mask, not sky
    valid = (buffers["source_id"] == target_source_id) & ~buffers["is_sky"]
    selected = lookup_mask_at_points(mask, buffers["indices"], valid, num_frames_mask)
    return selected.nonzero(as_tuple=False).squeeze(-1)


def apply_edit_to_buffers(buffers, subset_idx, ops, scope, centroid_threshold):
    # Run all ops for one edit over `subset_idx`, respecting scope (global or per-origin-frame)
    if subset_idx.numel() == 0:
        return buffers

    remove_after = any(op["op"] == "remove" for op in ops)
    transform_ops = [op for op in ops if op["op"] != "remove"]

    if scope == "global":
        for op in transform_ops:
            buffers["points_pos"] = apply_op_to_subset(buffers["points_pos"], subset_idx, op, centroid_threshold)
    else:  # scope == "frame"
        frame_per_point = buffers["indices"][subset_idx, 0]
        for op in transform_ops:
            for f in torch.unique(frame_per_point).tolist():
                frame_subset = subset_idx[frame_per_point == f]
                buffers["points_pos"] = apply_op_to_subset(
                    buffers["points_pos"], frame_subset, op, centroid_threshold,
                )

    if remove_after:
        keep = torch.ones(buffers["points_pos"].shape[0], dtype=torch.bool, device=buffers["points_pos"].device)
        keep[subset_idx] = False
        filter_buffers(buffers, keep)
    return buffers


def apply_edits_to_buffers(
    buffers: Dict,                        # Must have all BUFFER_KEYS populated (`source_id=0`, `is_sky` set)
    edits: List[Dict],
    src_video: npt.NDArray[np.uint8],     # f h w 3, uint8
    src_num_frames: int,                  # Frames in the src scene (used to bounds-check index lookups)
    num_frames_tgt: int,
    sam3_fn: Callable,        # (video, keywords) -> (f h w) bool mask
    load_insert_fn: Optional[Callable] = None,  # source_path -> insert dict (see `unproject_insert`)
    default_centroid_threshold: float = 0.6,
    verbose: bool = False,
    # Persistent caller-owned cache for `unproject_insert` results. Keyed by
    # (source_path, normalized_prompt, mask_expansion_tuple, num_frames_tgt, double_reprojection).
    # Stored value is the unproject_insert output dict MINUS `source_id` (which depends on edit ordering, not geometry);
    # the caller-provided source_id tensor is rebuilt on each hit. Default None disables caching.
    unproject_insert_cache: Optional[Dict] = None,
):
    # Apply a list of edits to the unprojected point cloud buffers, returning the mutated `buffers` dict.
    #   * All `scope=global` edits run first, then all `scope=frame` edits; JSON order is preserved within each group.
    #   * Each edit's SAM3 mask is applied to its target scene (current for existing/duplicate, the insert for insert).
    #   * Sky points (per-source `sky_mask`) are excluded from every selection.
    #   * `mask_expansion=[r, i]` dilates the 2D mask before point selection.
    #   * `target.kind="duplicate"` clones the selected subset, applies ops to the clone, and appends to the pool.
    #   * `target.kind="insert"` unprojects the insert scene's SAM-filtered pixels, applies ops, and appends.
    device = buffers["points_pos"].device

    # SAM3 masks are expensive; cache by (source_id, prompt, mask_expansion_key)
    mask_cache: Dict[Tuple[int, str, Optional[Tuple[int, int]]], torch.Tensor] = {}
    # Reload the same insert name only once
    insert_source_ids: Dict[str, int] = {}
    # Per-source num_frames is needed to bounds-check indices during selection
    source_num_frames: Dict[int, int] = {0: src_num_frames}
    double_reprojection = buffers["visible_ntp"] is not None

    def get_mask(target_source_id, video_np, prompt, mask_expansion):
        key = (target_source_id, prompt, tuple(mask_expansion) if mask_expansion else None)
        if key in mask_cache:
            return mask_cache[key]
        keywords = [k.strip() for k in prompt.split(",") if k.strip()]
        mask_np = sam3_fn(video_np, keywords).astype(np.bool_)
        if mask_expansion is not None and tuple(mask_expansion) != (0, 0):
            mask_np = expand_mask(mask_np, radius=int(mask_expansion[0]), iterations=int(mask_expansion[1]))
        mask_t = torch.from_numpy(mask_np).to(device=device)
        mask_cache[key] = mask_t
        return mask_t

    # Run globals first, then per-frame (each group preserves JSON ordering)
    ordered_edits = (
        [(i, e) for i, e in enumerate(edits) if e["scope"] == "global"]
        + [(i, e) for i, e in enumerate(edits) if e["scope"] == "frame"]
    )

    for i, edit in ordered_edits:
        target = edit["target"]
        kind = target["kind"]
        centroid_threshold = float(edit.get("centroid_threshold", default_centroid_threshold))
        mask_expansion = edit.get("mask_expansion", None)
        if verbose:
            print(f"[edit {i}] kind={kind}, scope={edit['scope']}, prompt=`{target['prompt']}`")

        if kind in ("existing", "duplicate"):
            sam_mask = get_mask(
                target_source_id=0, video_np=src_video, prompt=target["prompt"], mask_expansion=mask_expansion,
            )
            subset_idx = select_subset_idx(buffers, 0, sam_mask, source_num_frames[0])

            if kind == "existing":
                apply_edit_to_buffers(buffers, subset_idx, edit["ops"], edit["scope"], centroid_threshold)
            else:  # duplicate
                clone = clone_buffers_subset(buffers, subset_idx)
                # Re-run the edit over the clone's full range (all its points are the "selected" subset)
                clone_idx = torch.arange(clone["points_pos"].shape[0], device=device)
                apply_edit_to_buffers(clone, clone_idx, edit["ops"], edit["scope"], centroid_threshold)
                append_buffers(buffers, clone)

        elif kind == "insert":
            assert load_insert_fn is not None,\
                "Edits include `insert` targets but no `load_insert_fn` was provided."
            source_name = target["source"]
            if source_name not in insert_source_ids:
                insert_source_ids[source_name] = len(insert_source_ids) + 1  # 0 is reserved for src
            new_source_id = insert_source_ids[source_name]

            insert = load_insert_fn(source_name)
            source_num_frames[new_source_id] = insert["video"].shape[0]
            sam_mask = get_mask(
                target_source_id=new_source_id, video_np=insert["video"],
                prompt=target["prompt"], mask_expansion=mask_expansion,
            )

            # Cache key: (source, normalized prompt, mask expansion, target frame count, ntp flag).
            # `unproject_insert` is by far the heaviest per-edit op for `kind=insert` (full GPU
            # unproject of the insert scene, not just the small selected subset), so memoizing
            # its output across edit-list updates lets the user tweak ops without re-paying it.
            new_buffers = None
            if unproject_insert_cache is not None:
                norm_prompt = tuple(sorted(
                    k.strip() for k in target["prompt"].split(",") if k.strip()
                ))
                cache_key = (
                    source_name, norm_prompt,
                    tuple(mask_expansion) if mask_expansion else None,
                    num_frames_tgt, double_reprojection,
                )
                cached = unproject_insert_cache.get(cache_key)
                if cached is not None:
                    # Shallow-copy the dict so apply_edit_to_buffers' reassignments
                    # (`buffers["points_pos"] = ...`) don't poison the cache. Tensors aren't
                    # cloned because apply_op_to_subset / filter_buffers always return new
                    # tensors rather than mutating in place.
                    new_buffers = dict(cached)
                    # source_id depends on edit order, not on the unprojected geometry — rebuild.
                    n_pts = new_buffers["points_pos"].shape[0]
                    new_buffers["source_id"] = torch.full(
                        (n_pts,), new_source_id, dtype=torch.long, device=device,
                    )

            if new_buffers is None:
                sam_mask_np = sam_mask.detach().cpu().numpy()
                new_buffers = unproject_insert(
                    insert=insert, source_id=new_source_id, num_frames_tgt=num_frames_tgt,
                    sam_mask_np=sam_mask_np, double_reprojection=double_reprojection, device=device,
                )
                if unproject_insert_cache is not None:
                    # Cache everything except source_id (which we just built per-call above).
                    unproject_insert_cache[cache_key] = {
                        k: v for k, v in new_buffers.items() if k != "source_id"
                    }

            new_idx = torch.arange(new_buffers["points_pos"].shape[0], device=device)
            apply_edit_to_buffers(new_buffers, new_idx, edit["ops"], edit["scope"], centroid_threshold)
            append_buffers(buffers, new_buffers)

    return buffers


def apply_edits(
    points_color: torch.Tensor,
    points_pos: torch.Tensor,
    visible: torch.Tensor,
    visible_ntp: Optional[torch.Tensor],
    indices: torch.Tensor,
    edits: List[Dict],
    src_video: npt.NDArray[np.uint8],     # f h w 3, uint8
    src_sky_mask: npt.NDArray[np.bool_],  # f h w
    num_frames_tgt: int,
    sam3_fn: Callable,
    load_insert_fn: Optional[Callable] = None,
    default_centroid_threshold: float = 0.6,
    verbose: bool = False,
):
    # Tuple-based edit entry point used by `render_video`; returns the four buffers it cares about.
    # cam_ui (which pre-builds a buffers dict on scene load) calls `apply_edits_to_buffers` directly.
    device = points_pos.device
    src_sky_t = torch.from_numpy(src_sky_mask.astype(np.bool_)).to(device=device)
    is_sky = lookup_mask_at_points(
        src_sky_t, indices, torch.ones(indices.shape[0], dtype=torch.bool, device=device), src_sky_mask.shape[0],
    )
    source_id = torch.zeros(indices.shape[0], dtype=torch.long, device=device)
    buffers = {
        "points_color": points_color, "points_pos": points_pos,
        "visible": visible, "visible_ntp": visible_ntp,
        "indices": indices, "source_id": source_id, "is_sky": is_sky,
    }
    buffers = apply_edits_to_buffers(
        buffers, edits=edits, src_video=src_video, src_num_frames=src_sky_mask.shape[0],
        num_frames_tgt=num_frames_tgt, sam3_fn=sam3_fn, load_insert_fn=load_insert_fn,
        default_centroid_threshold=default_centroid_threshold, verbose=verbose,
    )
    return buffers["points_color"], buffers["points_pos"], buffers["visible"], buffers["visible_ntp"]


def build_edit_fn(
    edits_path: Optional[str],
    video_src: npt.NDArray[np.uint8],      # Preprocessed src video, f h w 3
    sky_mask_src: npt.NDArray[np.bool_],   # Preprocessed src sky mask, f h w
    num_frames_tgt: int,
    height: int,
    width: int,
    depth_outliers: str = "gaussian",
    ignore_sky_mask: bool = False,
    depths_dtype: np.dtype = np.float16,
):
    # Build the `edit_fn` passed to `render_video`, or return None if no edits JSON was provided.
    # SAM3 is initialized lazily on first segmentation call, and insert scenes go through the same
    # `load_recon_and_seg` + `preprocess_scene` pipeline as the src; both are cached in-closure so
    # repeated SAM prompts or multi-edit inserts pay their cost once.
    if edits_path is None:
        return None
    edits = load_edits(edits_path)

    sam3_predictor = None  # Lazy: we only pay the model-load cost if an edit actually needs a mask

    def sam3_fn(video, keywords):
        from utils.recon_and_seg.seg_sam3_official import init_sam3_video, run_sam3_video
        nonlocal sam3_predictor
        if sam3_predictor is None:
            sam3_predictor = init_sam3_video()
        mask, _ = run_sam3_video(video, sam3_predictor, keywords)
        return mask

    insert_cache = {}

    def load_insert_fn(source_path):
        # Inserts always use the full src clip of their recon_and_seg folder (no DSE tail, no render-window
        # slice) — they contribute points wherever their SAM mask says, nothing more.
        if source_path in insert_cache:
            return insert_cache[source_path]
        scene = load_recon_and_seg(source_path, depths_dtype=depths_dtype)
        src_start, src_end = scene["clips"]["src"]
        indices = np.arange(src_start, src_end)
        processed = preprocess_scene(
            scene, indices=indices, height=height, width=width,
            depth_outliers=depth_outliers, ignore_sky_mask=ignore_sky_mask,
        )
        insert = {
            "video": processed["video"], "depths": processed["depths"],
            "cam_c2w": processed["cam_c2w"], "K": intrinsics_to_K(processed["intrinsics"]),
            "dynamic_mask": processed["dynamic_mask"], "static_mask": processed["static_mask"],
            "sky_mask": processed["sky_mask"],
        }
        insert_cache[source_path] = insert
        return insert

    return partial(
        apply_edits,
        edits=edits,
        src_video=video_src,
        src_sky_mask=sky_mask_src,
        num_frames_tgt=num_frames_tgt,
        sam3_fn=sam3_fn,
        load_insert_fn=load_insert_fn,
        verbose=True,
    )
