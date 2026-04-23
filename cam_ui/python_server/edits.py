"""
cam_ui edits module: build unified point-cloud buffers on scene load, apply live edits from the
right-side panel, and convert back to per-frame point clouds for viser display.

Buffers vs. per-frame clouds:
- Buffers (unified) are the natural form for `apply_edits_to_buffers` — one flat point list with
  per-point (frame, h, w) origin indices and a `visible` matrix. Shared with render_video.
- Per-frame clouds are what viser renders: one (n, 3) array per timeline frame.
- On scene load we build buffers once and keep them frozen. Each edits update clones the skeleton,
  applies edits into the clone, then slices by `visible[:, f]` to produce a fresh per-frame list.

Coordinate handling: `apply_edits_to_buffers` internally uses the `np.diag([-1,-1,1,1])` cam_c2w
flip that `render_video` applies. cam_ui displays in raw world coordinates, so we un-flip positions
(`* [-1,-1,1]`) when materializing per-frame clouds.
"""

import gc
import os
import threading
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import torch
from tqdm.auto import tqdm

from utils.media import intrinsics_to_K, load_recon_and_seg
from utils.point_cloud.edit import (
    BUFFER_KEYS, apply_edits_to_buffers, lookup_mask_at_points, validate_edit,
)
from utils.point_cloud.point_cloud import unproject
from utils.point_cloud.preprocess import preprocess_scene


# Serialize buffer updates. SAM3 inference and large torch ops don't compose with concurrent
# requests, and we only care about the latest edits landing in viser.
edits_lock = threading.Lock()

state = {
    "edits": [],                  # Latest validated edits list
    "buffers_src": None,          # Frozen src-only buffers (source_id=0 everywhere)
    "src_video": None,            # f h w 3 uint8, held stable so `id(video)` keys stay valid
    "num_frames_src": 0,
    "folder_path": None,          # Cached on scene load; read by `ensure_buffers_built` on first edit
    "height": 0,
    "width": 0,
    "depth_outliers": "gaussian",
    "ignore_sky_mask": False,
    "device": "cuda",
    # Optional: main.py has already loaded video/depths/cam_c2w/masks from disk to build the UI
    # point clouds; re-reading them in ensure_buffers_built (via load_recon_and_seg) is pure
    # duplication. When main.py passes a pre-loaded dict via load_src_from_folder, we skip the
    # disk round-trip. Left None for standalone callers (none today; render scripts don't use
    # this module).
    "cached_scene": None,

    # Lazy — only pay the ~30s model load if an edit actually needs a mask
    "sam3_predictor": None,

    # Persistent caches keyed across edits updates (not just within one apply_edits call)
    "sam_mask_cache": {},   # (id(video), sorted-keywords tuple) -> np.ndarray (f h w bool)
    "insert_cache": {},     # source_path -> insert dict (see utils.point_cloud.edit.unproject_insert)
    # (source_path, normalized_prompt_tuple, mask_expansion_tuple|None, num_frames_tgt,
    # double_reprojection) -> dict of pristine unprojected insert buffers (everything except
    # source_id, which depends on edit ordering and is rebuilt per-call). Lets the user tweak
    # translate/rotate/scale on an insert edit without re-paying the full GPU unproject.
    "unproject_insert_cache": {},

    # Human-readable progress for the UI to poll during a long-running apply. Updated at each
    # milestone (SAM3 load, per-edit segmentation, insert loading, etc.) so the frontend can
    # replace "Applying..." with something specific like "Running SAM3 for prompt: person, dog".
    "progress": "idle",
}


EDITS_LOG_PATH = "/tmp/vista4d_edits.log"


def set_progress(message: str):
    # Single-writer status (apply_edits holds `edits_lock` so there's no contention) that the
    # UI polls. Also written to:
    # - stderr fd via raw `os.write(2, ...)` — bypasses Python's stdio layer in case something
    #   in uvicorn/anyio is intercepting `print` and `sys.stderr.write` on worker threads.
    # - a dedicated log file at EDITS_LOG_PATH — guaranteed visibility even if fd 2 itself is
    #   redirected by the environment; `tail -f /tmp/vista4d_edits.log` in another shell is a
    #   reliable live trace.
    state["progress"] = message
    line = f"[edits] {message}\n"
    try:
        os.write(2, line.encode())
    except OSError:
        pass
    try:
        with open(EDITS_LOG_PATH, "a") as f:
            f.write(line)
    except OSError:
        pass


def get_progress() -> str:
    return state["progress"]


def reset_scene_state():
    # Clear per-scene state on folder reload / factory-reset. Keep the SAM3 predictor loaded
    # (its ~30s model-load cost shouldn't be repaid every scene swap; only the per-scene
    # caches on top of it get cleared).
    state["edits"] = []
    state["buffers_src"] = None
    state["src_video"] = None
    state["num_frames_src"] = 0
    state["folder_path"] = None
    state["height"] = 0
    state["width"] = 0
    state["cached_scene"] = None
    state["sam_mask_cache"] = {}
    state["insert_cache"] = {}
    state["unproject_insert_cache"] = {}
    # Dropping the Python refs above only marks the GPU buffers (5GB+ of points_pos / visible /
    # indices / etc.) and CPU scene arrays as unreachable; without an explicit collect the
    # caching allocator keeps the VRAM reserved and it accumulates across scene reloads until
    # the next edit OOMs. A sync `gc.collect()` forces the refs to actually drop now, and
    # `empty_cache()` returns the freed blocks to the driver pool.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_edits_list(edits):
    if not isinstance(edits, list):
        raise ValueError(f"`edits` must be a list, got {type(edits).__name__}.")
    for i, edit in enumerate(edits):
        validate_edit(edit, index=i)


def sam3_fn(video, keywords):
    # Persistent mask cache keyed by id(video). Stable because we hold src_video and insert
    # dicts in module state for the life of the scene.
    norm_keywords = tuple(sorted(k.strip() for k in keywords if k.strip()))
    # Empty-prompt guard: UI drafts can PUT an edit that hasn't been filled in yet. Skip SAM3
    # entirely (including the ~30s model load) so just opening the edits panel doesn't spin up
    # the predictor on the user's GPU.
    if not norm_keywords:
        set_progress("Skipping SAM3 (empty prompt)")
        return np.zeros(video.shape[:3], dtype=np.bool_)
    from utils.recon_and_seg.seg_sam3_official import init_sam3_video, run_sam3_video
    key = (id(video), norm_keywords)
    if key in state["sam_mask_cache"]:
        set_progress(f"Using cached SAM3 mask for prompt: {', '.join(norm_keywords)}")
        return state["sam_mask_cache"][key]
    if state["sam3_predictor"] is None:
        set_progress("Loading SAM3 predictor (first use, ~30s)")
        state["sam3_predictor"] = init_sam3_video()
    set_progress(f"Running SAM3 for prompt: {', '.join(norm_keywords)}")
    mask, _ = run_sam3_video(video, state["sam3_predictor"], list(norm_keywords))
    state["sam_mask_cache"][key] = mask
    return mask


def load_insert_fn(source_path):
    if source_path in state["insert_cache"]:
        set_progress(f"Using cached insert scene: {source_path}")
        return state["insert_cache"][source_path]
    set_progress(f"Loading insert scene: {source_path}")
    scene = load_recon_and_seg(source_path, depths_dtype=np.float16)
    src_start, src_end = scene["clips"]["src"]
    indices = np.arange(src_start, src_end)
    set_progress(f"Preprocessing insert scene: {source_path}")
    processed = preprocess_scene(
        scene, indices=indices, height=state["height"], width=state["width"],
        depth_outliers=state["depth_outliers"], ignore_sky_mask=state["ignore_sky_mask"],
        progress_fn=set_progress,
    )
    insert = {
        "video": processed["video"], "depths": processed["depths"],
        "cam_c2w": processed["cam_c2w"], "K": intrinsics_to_K(processed["intrinsics"]),
        "dynamic_mask": processed["dynamic_mask"], "static_mask": processed["static_mask"],
        "sky_mask": processed["sky_mask"],
    }
    state["insert_cache"][source_path] = insert
    return insert


def build_src_buffers(
    video: npt.NDArray[np.uint8],        # f h w 3
    depths: npt.NDArray,                 # f h w
    cam_c2w: npt.NDArray[np.float32],    # f 4 4
    intrinsics: npt.NDArray[np.float32], # f 4
    sky_mask: npt.NDArray[np.bool_],     # f h w
    device: str = "cuda",
):
    # Unproject the src scene with dynamic=all-True, static=all-False so `visible` becomes one-hot
    # by origin frame — matches cam_ui's existing per-frame display semantics.
    num_frames, height, width, _ = video.shape
    cam_c2w_flipped = np.diag([-1, -1, 1, 1])[None] @ cam_c2w  # Match render_video's coord flip
    K = intrinsics_to_K(intrinsics)

    video_t = torch.from_numpy(video).to(dtype=torch.float32, device=device) / 255
    depths_t = torch.from_numpy(depths).to(dtype=torch.float32, device=device)
    cam_c2w_t = torch.from_numpy(cam_c2w_flipped).to(dtype=torch.float32, device=device)
    K_t = torch.from_numpy(K).to(dtype=torch.float32, device=device)
    dyn_t = torch.ones((num_frames, height, width), dtype=torch.bool, device=device)
    stat_t = torch.zeros((num_frames, height, width), dtype=torch.bool, device=device)

    points_color, points_pos, visible, indices = unproject(
        video=video_t, depths=depths_t, cam_c2w=cam_c2w_t, K=K_t,
        dynamic_mask=dyn_t, static_mask=stat_t,
    )
    sky_t = torch.from_numpy(sky_mask.astype(np.bool_)).to(device=device)
    is_sky = lookup_mask_at_points(
        sky_t, indices, torch.ones(indices.shape[0], dtype=torch.bool, device=device), num_frames,
    )
    source_id = torch.zeros(indices.shape[0], dtype=torch.long, device=device)

    # `visible_ntp` is only used by render_video's double-reprojection path; cam_ui doesn't need it
    return {
        "points_color": points_color, "points_pos": points_pos,
        "visible": visible, "visible_ntp": None,
        "indices": indices, "source_id": source_id, "is_sky": is_sky,
    }


def load_src_from_folder(
    folder_path: str,
    height: int,
    width: int,
    depth_outliers: str = "gaussian",
    ignore_sky_mask: bool = False,
    device: str = "cuda",
    cached_scene: Optional[dict] = None,
):
    # Called from `load_and_process_folder` after the existing load pipeline finishes. This only
    # stores the load parameters — the actual preprocessing + ~45M-point GPU buffer build is
    # deferred until the first edit is applied, so camera-design-only sessions (and CPU-only
    # nodes) don't pay VRAM or compute cost they'll never use.
    #
    # `cached_scene` (optional): a pre-loaded scene dict from main.py. Same shape as
    # `load_recon_and_seg`'s return value (video, depths, cam_c2w, intrinsics, dynamic_mask,
    # static_mask, sky_mask, clips). When provided, `ensure_buffers_built` skips the redundant
    # disk re-read. Must have all required keys; partial dicts fall back to a disk reload.
    with edits_lock:
        reset_scene_state()
        state["folder_path"] = folder_path
        state["height"] = height
        state["width"] = width
        state["depth_outliers"] = depth_outliers
        state["ignore_sky_mask"] = ignore_sky_mask
        state["device"] = device
        state["cached_scene"] = cached_scene
        print("Edit buffers will be built on first edit (skipped at scene load to save VRAM)")


_REQUIRED_SCENE_KEYS = (
    "video", "depths", "cam_c2w", "intrinsics", "dynamic_mask", "static_mask", "sky_mask", "clips",
)


def ensure_buffers_built():
    # Lazy build of the src buffer skeleton. Idempotent — a no-op once buffers are populated.
    # Must be called under `edits_lock` (apply_edits already holds it).
    if state["buffers_src"] is not None:
        return
    if state.get("folder_path") is None:
        raise RuntimeError("No scene loaded — load a reconstruction folder before editing.")
    cached = state.get("cached_scene")
    if cached is not None and all(cached.get(k) is not None for k in _REQUIRED_SCENE_KEYS):
        set_progress("Using pre-loaded scene from UI (skipping disk re-read)")
        scene = cached
    else:
        set_progress("Loading reconstruction for edits")
        scene = load_recon_and_seg(state["folder_path"], depths_dtype=np.float16)
    src_start, src_end = scene["clips"]["src"]
    indices = np.arange(src_start, src_end)
    set_progress("Preprocessing scene for edits")
    processed = preprocess_scene(
        scene, indices=indices, height=state["height"], width=state["width"],
        depth_outliers=state["depth_outliers"], ignore_sky_mask=state["ignore_sky_mask"],
        progress_fn=set_progress,
    )
    state["src_video"] = processed["video"]
    state["num_frames_src"] = processed["video"].shape[0]
    set_progress("Building edit-ready source buffers")
    state["buffers_src"] = build_src_buffers(
        video=processed["video"],
        depths=processed["depths"].astype(np.float32),
        cam_c2w=processed["cam_c2w"].astype(np.float32),
        intrinsics=processed["intrinsics"].astype(np.float32),
        sky_mask=processed["sky_mask"],
        device=state["device"],
    )
    n_pts = state["buffers_src"]["points_pos"].shape[0]
    set_progress(f"Source buffers ready: {n_pts:,} points")


def clone_buffers(buffers):
    # Shallow dict copy: not a tensor-by-tensor `.clone()`. Every mutation path in
    # `apply_edits_to_buffers` (apply_op_to_subset → copy=True, filter_buffers via boolean
    # indexing, append_buffers via torch.cat, clone_buffers_subset via explicit .clone)
    # produces a fresh tensor and *replaces* the dict entry rather than mutating the tensor's
    # storage in place. A shallow dict copy is therefore sufficient to shield `buffers_src`'s
    # dict entries from reassignment; the underlying tensors are never touched. Saves ~5 GiB
    # GPU per edit apply on a 50M-point src buffer set.
    return {key: buffers[key] for key in BUFFER_KEYS}


def compute_edited_buffers(edits: List[Dict]):
    if not edits:
        return state["buffers_src"]
    buffers = clone_buffers(state["buffers_src"])
    return apply_edits_to_buffers(
        buffers=buffers, edits=edits,
        src_video=state["src_video"], src_num_frames=state["num_frames_src"],
        num_frames_tgt=state["num_frames_src"],
        sam3_fn=sam3_fn, load_insert_fn=load_insert_fn,
        verbose=True,
        unproject_insert_cache=state["unproject_insert_cache"],
    )


def buffers_to_frame_point_clouds(buffers, num_frames: int):
    # Per-frame (positions, colors) materialization for viser display. Handles every visibility
    # pattern (one-hot src buffers, all-False rows from removes, multi-frame-visible static
    # inserts) on the same path: expand visibility to (point_idx, frame_idx) pairs via nonzero,
    # sort by frame, bulk-gather positions and colors, bulk-transfer, slice per frame.
    #
    # Memory-peak tricks (stay bulk for speed; the per-frame-gather variant was ~50× slower
    # due to allocator + sync serialization). All three trim the concurrent live set during
    # stage 3/4 without adding syncs that would slow the bulk path:
    #   1. `frame_idx_sorted` is freed after `bincount` (saves ~2 GiB for M=250M).
    #   2. Position gather is done in fp16 — pre-cast `points_pos` before indexing so the
    #      gather is born in fp16 (~1.5 GiB) rather than fp32 (~3 GiB). Upcast to fp32 on the
    #      CPU side for viser, which expects fp32 arrays; fp16 precision (~0.01 at 10-unit
    #      scene scale) is well below the rendered point cloud's per-pixel resolution, so the
    #      dtype round-trip is invisible on the display.
    #   3. Color gather is sequenced *before* position gather — the ~3 GiB fp32 color
    #      intermediate collapses to ~0.75 GiB uint8 and is freed before the next big
    #      allocation, so the two gather transients never coexist.
    #
    # This is strictly the UI-display path; `render_edit.py` consumes the float buffers from
    # `apply_edits_to_buffers` directly and never goes through here.
    points_pos = buffers["points_pos"]
    points_color = buffers["points_color"]
    visible = buffers["visible"]
    device = points_pos.device
    flip = torch.tensor([-1.0, -1.0, 1.0], dtype=points_pos.dtype, device=device)

    stages = tqdm(total=4, desc="Building per-frame point clouds", leave=False)

    # Stage 1: expand visibility to (point, frame) pairs and sort by frame.
    set_progress("Per-frame PCs: 1/4 nonzero + argsort")
    point_idx, frame_idx = visible.nonzero(as_tuple=True)
    order = frame_idx.argsort()
    frame_idx_sorted = frame_idx[order]
    point_idx_sorted = point_idx[order]
    del point_idx, frame_idx, order
    if device.type == "cuda":
        torch.cuda.synchronize()
    stages.update(1)

    # Stage 2: frame boundaries; free `frame_idx_sorted` immediately (trick 1).
    set_progress("Per-frame PCs: 2/4 frame boundaries")
    counts = torch.bincount(frame_idx_sorted, minlength=num_frames)
    boundaries_gpu = torch.cat(
        [torch.zeros(1, dtype=counts.dtype, device=device), counts.cumsum(0)]
    )
    del frame_idx_sorted, counts
    if device.type == "cuda":
        torch.cuda.synchronize()
    stages.update(1)

    # Stage 3: gather colors → uint8, sequenced before the position gather (trick 3).
    set_progress("Per-frame PCs: 3/4 gather colors → uint8")
    colors_fp32 = points_color[point_idx_sorted]  # fp32, ~3 GiB at 250M entries
    colors_fp32.mul_(255).clamp_(0, 255)           # in-place, no extra alloc
    col_u8_gpu = colors_fp32.to(torch.uint8)       # uint8, ~0.75 GiB
    del colors_fp32                                # releases the ~3 GiB fp32 intermediate
    if device.type == "cuda":
        torch.cuda.synchronize()
    stages.update(1)

    # Stage 4: gather positions in fp16 (trick 2) + bulk transfer everything.
    set_progress("Per-frame PCs: 4/4 gather positions (fp16) + transfer")
    points_pos_half = points_pos.to(torch.float16)                # ~0.3 GiB for a 55M-pt buffer
    pos_sorted_gpu = points_pos_half[point_idx_sorted]            # fp16, ~1.5 GiB at 250M
    pos_sorted_gpu.mul_(flip.to(torch.float16))                   # in-place flip
    del points_pos_half, point_idx_sorted
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Bulk GPU→CPU transfer. fp16 cuts the positions transfer in half vs fp32; upcast to
    # fp32 on the CPU side since viser's `add_point_cloud` expects fp32 arrays.
    pos_sorted = pos_sorted_gpu.cpu().numpy().astype(np.float32)
    col_sorted = col_u8_gpu.cpu().numpy()
    boundaries = boundaries_gpu.cpu().numpy()
    del pos_sorted_gpu, col_u8_gpu, boundaries_gpu

    point_clouds = [pos_sorted[boundaries[f]:boundaries[f + 1]] for f in range(num_frames)]
    colors_list = [col_sorted[boundaries[f]:boundaries[f + 1]] for f in range(num_frames)]
    stages.update(1)
    stages.close()
    return point_clouds, colors_list


def apply_edits(edits: List[Dict]):
    # Top-level entry for `PUT /api/edits`. Validates, clones src buffers, runs edits, materializes
    # per-frame point clouds. Serialized via `edits_lock` so concurrent PUTs don't interleave.
    validate_edits_list(edits)
    with edits_lock:
        ensure_buffers_built()  # lazy: first edit pays the buffer-build cost, not every scene load
        state["edits"] = edits
        try:
            if edits:
                set_progress(f"Applying {len(edits)} edit(s)")
                buffers = compute_edited_buffers(edits)
                set_progress("Building per-frame point clouds")
                point_clouds, colors_list = buffers_to_frame_point_clouds(
                    buffers, state["num_frames_src"],
                )
                # Edited buffers can be multi-GB; drop them. The frozen src buffers stay.
                del buffers
            else:
                set_progress("Clearing edits — restoring source point clouds")
                buffers = state["buffers_src"]
                point_clouds, colors_list = buffers_to_frame_point_clouds(
                    buffers, state["num_frames_src"],
                )
            torch.cuda.empty_cache()
            set_progress("idle")
            return point_clouds, colors_list
        except Exception:
            set_progress("idle")
            raise


def get_edits() -> List[Dict]:
    return list(state["edits"])


def has_edits() -> bool:
    return len(state["edits"]) > 0
