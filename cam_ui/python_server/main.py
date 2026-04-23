"""
Vista4D: 4D point cloud visualization and camera editor

CRITICAL INDEXING RULE:
- User-facing frame numbers: 1 to 500 (displayed in UI, sliders, logs)
- Internal array indices: 0 to 499 (used for list/array access)
- Always convert: frame_index = frame_ui - 1 before array access
- Always display: frame_ui = frame_index + 1 for user output

TRAJECTORY COMPUTATION SYSTEM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The system computes the final 500-frame camera trajectory using two distinct modes:

1. ORIGINAL MODE (No User Keyframes)
   ────────────────────────────────────
   - When: User has NOT added any keyframes
   - Behavior: Return the original cameras from the loaded folder AS-IS
   - No interpolation, no Catmull-Rom, just the raw original trajectory
   - Visual: Blue spline through all 500 original positions

2. CATMULL-ROM MODE (User Has Keyframes)
   ───────────────────────────────────────
   - When: User HAS added at least one keyframe
   - Behavior: Catmull-Rom spline interpolation through control points
   - Control Points: [frame_0, user_keyframe_1, ..., user_keyframe_n, frame_48]
   - Visual: Green spline through control points

   Algorithm:
   a) Build control points from keyframes
   b) For each segment between control points (e.g., frame 12 → 24):
      - Sample positions uniformly along Catmull-Rom curve
      - Slerp orientations smoothly between keyframes
      - Interpolate vertical FOV (optical parameter), then recalculate fx/fy
      - Linear interpolate aspect ratio and principal point (cx, cy)
   c) Use configurable tension parameter (0.0-1.0)

KEYFRAME LIFECYCLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

State Transitions:
- Load .npz → ORIGINAL MODE (blue trajectory)
- Add first keyframe → Switch to CATMULL-ROM MODE (green trajectory)
- Add more keyframes → Recompute Catmull-Rom with new control points
- Delete keyframe → Recompute Catmull-Rom
- Delete ALL keyframes → Switch back to ORIGINAL MODE (blue trajectory)

The system automatically detects which mode to use by checking if new_keyframes is empty.

VISUAL SPLINE vs GROUND TRUTH:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Visual Spline: High-resolution (500 segments) Catmull-Rom through control points
  Purpose: Beautiful smooth curve visualization in Viser

- Ground Truth: Computed 500 camera positions via Catmull-Rom sampling
  Purpose: Actual camera positions used for rendering and export
  These positions lie ON the Catmull-Rom curve

Both use the same tension parameter and control points, ensuring consistency.
"""

from argparse import ArgumentParser
from pathlib import Path
import os
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import viser
import viser.transforms as tf
import uvicorn

from utils.media import load_video, load_depths, load_cameras, load_clips, load_masks
from utils.point_cloud.filter import contract_mask, get_depths_outliers

import edits as edits_mod  # Sibling module — sys.path[0] is this script's directory


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

server = None  # Initialized in __main__ after args are parsed

# ============================================================================
# Global State
# ============================================================================

MAX_FRAMES = 500  # Fallback max frames (actual limit comes from loaded data)

# Image resolution (updated dynamically when data loads)
IMAGE_HEIGHT = 720  # Default height, will be updated from loaded depths shape
IMAGE_WIDTH = 1280  # Default width, will be updated from loaded depths shape

# Trajectory visualization settings
trajectory_tension = 0.5  # Curve tightness: 0.0 (loose/curved) to 1.0 (tight/straight)
point_size = 0.003  # Point cloud particle size

# Static background constants (same pipeline as render_single.py)
SKY_DEPTH = 1e3
STATIC_BG_FRAME_STEP = 4       # Use every 4th frame for persistent overlay
STATIC_BG_SPATIAL_FRAC = 0.25  # Keep 25% of static pixels per subsampled frame

# Server-side playback state
playback_thread = None
playback_stop_event = threading.Event()
playback_stop_event.set()  # Initially stopped

# Viser GUI element handles (None until created)
frame_slider = None
follow_camera_checkbox = None
status_text = None
frame_info_text = None
captured_count_text = None

scene_data = {
    "loaded": False,
    "point_clouds": None,
    "colors": None,
    # Unedited per-frame clouds (restored when edits list is empty so raw view is pixel-identical)
    "point_clouds_original": None,
    "colors_original": None,
    "cameras": None,
    "dse_cameras": None,
    "fps": 30.0,
    "num_frames": 0,
    "image_height": 0,
    "image_width": 0,
    "current_frame": 0,
    "playing": False,
    "file_path": None,
    # Static background: src base (computed once) + DSE inputs (reaggregated on stride change)
    "src_static_points": None,
    "src_static_colors": None,
    "dse_static_cache": None,
    "show_static_bg": False,
    "has_static_bg": False,
    # Source / DSE camera overlays
    "show_source_cameras": False,
    "show_dse_cameras": False,
    # DSE
    "has_dse": False,
    "num_dse_frames": 0,
    "dse_frame_interval": 1,
    # Edits
    "edits_ready": False,   # True once `edits_mod.load_src_from_folder` finishes building src buffers
}

new_camera_trajectory = []
client_follow_mode = {}  # Track which clients have follow mode enabled
new_keyframes = {}  # Track user-defined keyframes: {frame: {position, wxyz, fov, aspect}}
captured_cameras = [None] * MAX_FRAMES  # Array of 500 captured cameras (None if not captured)
viewport_fov_multiplier = 1.0  # Global viewport FOV multiplier for live preview

# Cache for output trajectory (only recompute when keyframes change)
cached_output_trajectory = None
trajectory_needs_update = True

# Handles for scene elements that need show/hide toggling
_static_bg_handle = None
_source_cameras_frustum_handle = None
_source_cameras_path_handle = None
_dse_cameras_frustum_handles = []  # One per DSE frame
_dse_cameras_path_handle = None


# ============================================================================
# Data Loading and Processing
# ============================================================================

def _aggregate_static_points(depths_for_bg, images, intrinsics, cam_c2w, static_mask, indices):
    """Unproject masked static pixels from selected frames and return concatenated points/colors.

    `indices` selects frames to aggregate; per-frame spatial subsampling uses STATIC_BG_SPATIAL_FRAC.
    Returns (points, colors) as float32/uint8 arrays, or (None, None) if no valid static pixels.
    """
    all_points, all_colors = [], []
    for i in indices:
        pixel_indices = np.where(static_mask[i].ravel())[0]
        if len(pixel_indices) == 0:
            continue
        n_keep = max(1, int(len(pixel_indices) * STATIC_BG_SPATIAL_FRAC))
        chosen = np.random.choice(pixel_indices, size=n_keep, replace=False)
        rows, cols = np.unravel_index(chosen, (IMAGE_HEIGHT, IMAGE_WIDTH))
        sparse_depth = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        sparse_depth[rows, cols] = depths_for_bg[i][rows, cols]
        mask_2d = np.zeros(IMAGE_HEIGHT * IMAGE_WIDTH, dtype=bool)
        mask_2d[chosen] = True
        pts, clrs = depth_to_pointcloud(sparse_depth, images[i], intrinsics[i], cam_c2w[i])
        all_points.append(pts[mask_2d])
        all_colors.append(clrs[mask_2d])

    if not all_points:
        return None, None
    return np.concatenate(all_points).astype(np.float32), np.concatenate(all_colors).astype(np.uint8)


def load_and_process_folder(folder_path: str):
    """Load a 4D reconstruction folder (output of recon_and_seg_single.py) and process data.

    If the folder has a clips.json (DSE), timeline data comes from the src window only;
    DSE frames are cached separately for on-demand static-background augmentation.
    """
    folder = Path(folder_path)
    print(f"Loading reconstruction folder: {folder_path}")

    # Load video/depths/cameras (joint src+DSE if DSE was used)
    images, fps = load_video(str(folder / "video.mp4"))  # (F, H, W, 3)
    depths = load_depths(str(folder / "depths"), dtype=np.float32)  # (F, H, W)
    cam_c2w, intrinsics = load_cameras(str(folder / "cameras.npz"))

    num_frames_total = min(images.shape[0], depths.shape[0], cam_c2w.shape[0])
    images = images[:num_frames_total]
    depths = depths[:num_frames_total]
    cam_c2w = cam_c2w[:num_frames_total]
    intrinsics = intrinsics[:num_frames_total]

    # Detect DSE via clips.json (clamp to available frames for safety)
    clips = load_clips(str(folder / "clips.json")) or {"src": (0, num_frames_total)}
    src_start, src_end = clips["src"]
    src_end = min(src_end, num_frames_total)
    num_src_frames = max(0, src_end - src_start)
    has_dse = "dse" in clips
    if has_dse:
        dse_start, dse_end = clips["dse"]
        dse_end = min(dse_end, num_frames_total)
        num_dse_frames = max(0, dse_end - dse_start)
        has_dse = num_dse_frames > 0
    if not has_dse:
        dse_start, dse_end, num_dse_frames = 0, 0, 0

    global IMAGE_HEIGHT, IMAGE_WIDTH
    IMAGE_HEIGHT = depths.shape[1]
    IMAGE_WIDTH = depths.shape[2]
    dse_note = f" (+{num_dse_frames} DSE)" if has_dse else ""
    print(f"Loaded {num_src_frames} src frames{dse_note} at {IMAGE_HEIGHT}h x {IMAGE_WIDTH}w, fps={fps:.2f}")

    # Build per-frame point clouds / cameras for src frames (timeline-editable)
    point_clouds, colors_list, cameras = [], [], []
    for i in range(src_start, src_end):
        print(f"Processing frame {i-src_start+1}/{num_src_frames}...", end='\r')
        points, colors = depth_to_pointcloud(depths[i], images[i], intrinsics[i], cam_c2w[i])
        point_clouds.append(points)
        colors_list.append(colors)
        cameras.append(process_camera(cam_c2w[i], intrinsics[i], IMAGE_HEIGHT, IMAGE_WIDTH))
    print(f"\nPoint clouds built!")

    # Build DSE camera dicts (for optional overlay)
    dse_cameras = [
        process_camera(cam_c2w[i], intrinsics[i], IMAGE_HEIGHT, IMAGE_WIDTH)
        for i in range(dse_start, dse_end)
    ]

    # Load raw masks once. Used by both the static-background pipeline below AND the edits
    # pipeline (via `cached_scene_for_edits`), so we hold raw references here and avoid a
    # second full-disk reload when the user applies their first edit.
    dyn_mask_folder = folder / "dynamic_mask"
    static_mask_folder = folder / "static_mask"
    sky_mask_folder = folder / "sky_mask"
    dynamic_mask_raw = load_masks(str(dyn_mask_folder))[:num_frames_total] if dyn_mask_folder.is_dir() else None
    static_mask_raw = load_masks(str(static_mask_folder))[:num_frames_total] if static_mask_folder.is_dir() else None
    sky_mask_raw = load_masks(str(sky_mask_folder))[:num_frames_total] if sky_mask_folder.is_dir() else None

    # Static background: compute src base; cache DSE frames for on-demand augmentation
    src_static_points = None
    src_static_colors = None
    dse_static_cache = None
    if dynamic_mask_raw is not None:
        print("Processing masks for static background...")
        static_mask_bg = static_mask_raw if static_mask_raw is not None else ~dynamic_mask_raw
        static_mask_bg = contract_mask(static_mask_bg, radius=6, iterations=3)
        static_mask_bg = static_mask_bg & ~get_depths_outliers(depths, static_mask_bg, mode="gaussian")

        depths_for_bg = depths.copy()
        if sky_mask_raw is not None:
            contracted_sky = contract_mask(sky_mask_raw, radius=6, iterations=3)
            static_mask_bg = static_mask_bg | contracted_sky
            depths_for_bg[contracted_sky] = SKY_DEPTH

        # Src static base (stride-4, computed once)
        src_indices = list(range(src_start, src_end, STATIC_BG_FRAME_STEP))
        src_static_points, src_static_colors = _aggregate_static_points(
            depths_for_bg, images, intrinsics, cam_c2w, static_mask_bg, src_indices,
        )
        if src_static_points is not None:
            print(f"Src static background: {len(src_static_points):,} points")
        else:
            print("Src static background: no valid static pixels found")

        # Cache DSE inputs; actual DSE static points rebuilt on demand per frame interval
        if has_dse:
            dse_static_cache = {
                "depths_for_bg": depths_for_bg,
                "images": images,
                "intrinsics": intrinsics,
                "cam_c2w": cam_c2w,
                "static_mask": static_mask_bg,
                "dse_start": dse_start,
                "dse_end": dse_end,
            }
    else:
        print("No dynamic_mask/ found; static pixel temporal persistence unavailable")

    # Pre-loaded scene for the edits pipeline. If all required masks were present we pass the raw
    # arrays through; otherwise edits.py will fall back to re-reading from disk (and likely fail,
    # since it assumes dynamic_mask/sky_mask exist). Keeping these references alive costs ~260MB
    # for a 49-frame 720p scene — negligible vs the multi-GB GPU buffers edits will build.
    cached_scene_for_edits = None
    if dynamic_mask_raw is not None and sky_mask_raw is not None:
        cached_scene_for_edits = {
            "video": images, "fps": fps,
            "depths": depths, "cam_c2w": cam_c2w, "intrinsics": intrinsics,
            "dynamic_mask": dynamic_mask_raw,
            "static_mask": static_mask_raw if static_mask_raw is not None else ~dynamic_mask_raw,
            "sky_mask": sky_mask_raw,
            "clips": clips,
        }

    has_static_bg = src_static_points is not None

    return {
        "point_clouds": point_clouds,
        "colors": colors_list,
        "cameras": cameras,
        "dse_cameras": dse_cameras,
        "fps": fps,
        "num_frames": num_src_frames,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "src_static_points": src_static_points,
        "src_static_colors": src_static_colors,
        "dse_static_cache": dse_static_cache,
        "has_dse": has_dse,
        "num_dse_frames": num_dse_frames,
        "has_static_bg": has_static_bg,
        "show_static_bg": has_static_bg,  # On by default when available
        "show_source_cameras": False,
        "show_dse_cameras": False,
        "dse_frame_interval": 1,
        "cached_scene_for_edits": cached_scene_for_edits,
    }


def depth_to_pointcloud(depth_map, rgb_image, intrinsic, c2w):
    """Convert depth map to 3D point cloud in world coordinates."""
    H, W = depth_map.shape
    fx, fy, cx, cy = intrinsic

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_map.astype(np.float32)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)  # (H, W, 4)
    points_world = np.einsum('ij,hwj->hwi', c2w, points_cam)
    points_3d = points_world[..., :3]  # (H, W, 3)

    points_flat = points_3d.reshape(-1, 3)
    colors_flat = rgb_image.reshape(-1, 3)

    return points_flat.astype(np.float32), colors_flat.astype(np.uint8)


def process_camera(c2w, intrinsic, height, width):
    """
    Process camera parameters for Viser visualization.

    CRITICAL: This computes FOV from the actual intrinsics to ensure visualization
    matches the camera calibration. However, Viser's camera model has limitations:
    - Uses single FOV (we average fx/fy if they differ)
    - Assumes centered principal point (cx=width/2, cy=height/2)

    For cameras with non-square pixels or off-center principal points, the
    visualization will be approximate. The ORIGINAL intrinsics are always
    preserved and used during export.

    Args:
        c2w: 4x4 camera-to-world transformation matrix
        intrinsic: [fx, fy, cx, cy] camera intrinsics
        height: image height in pixels
        width: image width in pixels

    Returns:
        Dictionary with position, wxyz, fov, aspect, and original intrinsics
    """
    fx, fy, cx, cy = intrinsic

    # Check for non-standard intrinsics
    if abs(fx - fy) > 0.1:
        print(f"[WARN] Non-square pixels detected (fx={fx:.2f}, fy={fy:.2f}); using average focal length")

    if abs(cx - width/2) > 1.0 or abs(cy - height/2) > 1.0:
        print(f"[WARN] Off-center principal point (cx={cx:.2f}, cy={cy:.2f})")
        print(f"   Viser visualization assumes centered principal point")

    # Calculate BOTH vertical and horizontal FOV
    # We need both because:
    # - Camera frustums (visual representation) use horizontal FOV
    # - Viewport camera (when snapping to match intrinsics) uses vertical FOV
    f_avg = (fx + fy) / 2.0

    # Vertical FOV (for viewport camera)
    fov_vertical_rad = 2 * np.arctan(height / (2 * f_avg))
    fov_vertical_deg = np.degrees(fov_vertical_rad)

    # Horizontal FOV (for camera frustum visualization)
    fov_horizontal_rad = 2 * np.arctan(width / (2 * f_avg))
    fov_horizontal_deg = np.degrees(fov_horizontal_rad)

    # Use horizontal FOV for the camera frustum visualization
    fov_deg = fov_horizontal_deg

    aspect = width / height

    # Extract rotation and position
    rotation_matrix = c2w[:3, :3]
    position = c2w[:3, 3]

    # Convert rotation matrix to quaternion (wxyz format)
    wxyz = tf.SO3.from_matrix(rotation_matrix).wxyz

    return {
        "position": position,
        "wxyz": wxyz,
        "fov": fov_deg,  # Horizontal FOV for frustum visualization
        "fov_vertical": fov_vertical_deg,  # Vertical FOV for viewport camera
        "aspect": aspect,
        "fx": float(fx),  # Store original intrinsics
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }


# ============================================================================
# Camera Interpolation and Output Trajectory
# ============================================================================

def catmull_rom_segment(p0, p1, p2, p3, t, tension):
    """
    Evaluate Catmull-Rom spline at parameter t in [0, 1].
    Returns position on curve between p1 and p2.

    At t=0: returns p1 (start control point)
    At t=1: returns p2 (end control point)

    Args:
        p0, p1, p2, p3: Control points (4 consecutive points)
        t: Parameter in [0, 1]
        tension: Tension parameter (0.5 = standard Catmull-Rom)
    """
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom basis matrix coefficients
    # This is the standard centripetal Catmull-Rom formulation
    c0 = -tension * t + 2.0 * tension * t2 - tension * t3
    c1 = 1.0 + (tension - 3.0) * t2 + (2.0 - tension) * t3
    c2 = tension * t + (3.0 - 2.0 * tension) * t2 + (tension - 2.0) * t3
    c3 = -tension * t2 + tension * t3

    return c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3


def compute_output_trajectory():
    """
    Compute the final 500-frame output trajectory using Catmull-Rom spline interpolation.

    Algorithm:
    1. Build control points from keyframes (frame_0, user_keyframes..., frame_499)
    2. For each segment between control points:
       - Sample positions uniformly along Catmull-Rom spline
       - Slerp orientations smoothly
       - Interpolate vertical FOV (optical parameter), recalculate fx/fy
       - Linear interpolate aspect ratio and principal point (cx, cy)
    3. Cache result for performance

    Returns: List of 500 camera dicts with {position, wxyz, fov, aspect, fx, fy, cx, cy}
    """
    global cached_output_trajectory, trajectory_needs_update

    # Return cached trajectory if valid
    if not trajectory_needs_update and cached_output_trajectory is not None:
        return cached_output_trajectory

    if not scene_data["loaded"]:
        return None

    # Check if user has added ANY keyframes
    has_user_keyframes = len(new_keyframes) > 0

    # ═══════════════════════════════════════════════════════════
    # ORIGINAL MODE: No user keyframes
    # ═══════════════════════════════════════════════════════════
    if not has_user_keyframes:
        cached_output_trajectory = list(scene_data["cameras"])
        trajectory_needs_update = False
        return cached_output_trajectory

    # ═══════════════════════════════════════════════════════════
    # CATMULL-ROM MODE: User has keyframes
    # ═══════════════════════════════════════════════════════════
    print(f"Computing trajectory: Catmull-Rom through {len(new_keyframes)} keyframe(s)...")

    # Build list of control points (keyframes)
    control_indices = []
    control_cameras = []

    # Always start with frame 0
    control_indices.append(0)
    control_cameras.append(captured_cameras[0] if captured_cameras[0] is not None else scene_data["cameras"][0])

    # Add user keyframes (excluding endpoints)
    num_frames = scene_data["num_frames"]
    for i in range(1, num_frames - 1):
        if captured_cameras[i] is not None:
            control_indices.append(i)
            control_cameras.append(captured_cameras[i])

    # Always end with last frame
    last_frame_idx = num_frames - 1
    control_indices.append(last_frame_idx)
    control_cameras.append(captured_cameras[last_frame_idx] if captured_cameras[last_frame_idx] is not None else scene_data["cameras"][last_frame_idx])

    # Build output trajectory using Catmull-Rom interpolation
    output_traj = [None] * num_frames

    # Place control points directly
    for i, frame_idx in enumerate(control_indices):
        output_traj[frame_idx] = control_cameras[i]

    # Interpolate between each pair of control points
    for seg_idx in range(len(control_indices) - 1):
        start_frame = control_indices[seg_idx]
        end_frame = control_indices[seg_idx + 1]

        idx_p1 = seg_idx
        idx_p2 = seg_idx + 1
        idx_p0 = max(0, seg_idx - 1)
        idx_p3 = min(len(control_indices) - 1, seg_idx + 2)

        p1_pos = np.array(control_cameras[idx_p1]["position"])
        p2_pos = np.array(control_cameras[idx_p2]["position"])

        # Use phantom (reflected) endpoints instead of clamping p0=p1 or p3=p2.
        # Clamping halves the endpoint velocity, causing ease-in/out on the first
        # and last segments. Reflection maintains constant velocity at endpoints.
        if idx_p0 == idx_p1:
            p0_pos = 2.0 * p1_pos - p2_pos  # Reflect p2 through p1
        else:
            p0_pos = np.array(control_cameras[idx_p0]["position"])

        if idx_p3 == idx_p2:
            p3_pos = 2.0 * p2_pos - p1_pos  # Reflect p1 through p2
        else:
            p3_pos = np.array(control_cameras[idx_p3]["position"])

        p1_wxyz = control_cameras[idx_p1]["wxyz"]
        p2_wxyz = control_cameras[idx_p2]["wxyz"]

        num_interior_frames = end_frame - start_frame - 1

        if num_interior_frames > 0:
            for i in range(1, num_interior_frames + 1):
                frame_idx = start_frame + i
                t = i / (num_interior_frames + 1)

                position = catmull_rom_segment(p0_pos, p1_pos, p2_pos, p3_pos, t, trajectory_tension)

                rot1 = R.from_quat([p1_wxyz[1], p1_wxyz[2], p1_wxyz[3], p1_wxyz[0]])
                rot2 = R.from_quat([p2_wxyz[1], p2_wxyz[2], p2_wxyz[3], p2_wxyz[0]])
                key_rots = R.from_quat([rot1.as_quat(), rot2.as_quat()])
                slerp_interp = Slerp([0, 1], key_rots)
                rot_interp = slerp_interp(t)
                quat_xyzw = rot_interp.as_quat()
                wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                # Interpolate intrinsics: log-space interpolation of 1/f
                f1_avg = (control_cameras[idx_p1]["fx"] + control_cameras[idx_p1]["fy"]) / 2.0
                f2_avg = (control_cameras[idx_p2]["fx"] + control_cameras[idx_p2]["fy"]) / 2.0
                x_interp = np.exp(np.log(1.0 / f1_avg) * (1 - t) + np.log(1.0 / f2_avg) * t)
                f_avg = 1.0 / x_interp
                fx = fy = f_avg

                fov_vertical_rad = 2 * np.arctan(IMAGE_HEIGHT / (2.0 * f_avg))
                fov_vertical = np.degrees(fov_vertical_rad)
                fov = np.degrees(2 * np.arctan((IMAGE_WIDTH / 2.0) / f_avg))

                aspect = control_cameras[idx_p1]["aspect"] * (1 - t) + control_cameras[idx_p2]["aspect"] * t
                cx = control_cameras[idx_p1]["cx"] * (1 - t) + control_cameras[idx_p2]["cx"] * t
                cy = control_cameras[idx_p1]["cy"] * (1 - t) + control_cameras[idx_p2]["cy"] * t

                output_traj[frame_idx] = {
                    "position": position,
                    "wxyz": wxyz,
                    "fov": fov,
                    "fov_vertical": fov_vertical,
                    "aspect": aspect,
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(cx),
                    "cy": float(cy),
                }

    # Cache the result
    cached_output_trajectory = output_traj
    trajectory_needs_update = False

    return output_traj


def compute_output_trajectory_fsff():
    """
    Compute FSFF (Force Same First Frame) trajectory using Catmull-Rom spline interpolation.

    This is similar to compute_output_trajectory(), but ALWAYS uses the original frame 0
    camera as the first control point, even if the user captured a different pose at frame 0.
    This ensures the exported trajectory starts exactly at the original first frame.

    Algorithm:
    1. Build control points: [ORIGINAL_frame_0, user_keyframes..., frame_499]
    2. Run full Catmull-Rom interpolation with these control points
    3. The spline naturally passes through the original frame 0

    Returns: List of 500 camera dicts with {position, wxyz, fov, aspect, fx, fy, cx, cy}
    """
    if not scene_data["loaded"]:
        return None

    # Check if user has added ANY keyframes (excluding frame 0)
    has_user_keyframes = len(new_keyframes) > 0

    if not has_user_keyframes:
        return list(scene_data["cameras"])

    # Build list of control points (keyframes)
    control_indices = []
    control_cameras = []

    # FSFF: ALWAYS use ORIGINAL frame 0, ignoring any user capture at frame 0
    control_indices.append(0)
    control_cameras.append(scene_data["cameras"][0])  # Force original

    # Add user keyframes (excluding endpoints)
    num_frames = scene_data["num_frames"]
    for i in range(1, num_frames - 1):
        if captured_cameras[i] is not None:
            control_indices.append(i)
            control_cameras.append(captured_cameras[i])

    # Always end with last frame (use captured if available, otherwise original)
    last_frame_idx = num_frames - 1
    control_indices.append(last_frame_idx)
    if captured_cameras[last_frame_idx] is not None:
        control_cameras.append(captured_cameras[last_frame_idx])
    else:
        control_cameras.append(scene_data["cameras"][last_frame_idx])

    # Build output trajectory using Catmull-Rom interpolation
    output_traj = [None] * num_frames

    # Place control points directly
    for i, frame_idx in enumerate(control_indices):
        output_traj[frame_idx] = control_cameras[i]

    # Interpolate between each pair of control points
    for seg_idx in range(len(control_indices) - 1):
        start_frame = control_indices[seg_idx]
        end_frame = control_indices[seg_idx + 1]

        # Get 4 control points for Catmull-Rom (p0, p1, p2, p3)
        idx_p1 = seg_idx
        idx_p2 = seg_idx + 1
        idx_p0 = max(0, seg_idx - 1)
        idx_p3 = min(len(control_indices) - 1, seg_idx + 2)

        p1_pos = np.array(control_cameras[idx_p1]["position"])
        p2_pos = np.array(control_cameras[idx_p2]["position"])

        # Phantom endpoints to avoid ease-in/out (same as compute_output_trajectory)
        if idx_p0 == idx_p1:
            p0_pos = 2.0 * p1_pos - p2_pos
        else:
            p0_pos = np.array(control_cameras[idx_p0]["position"])

        if idx_p3 == idx_p2:
            p3_pos = 2.0 * p2_pos - p1_pos
        else:
            p3_pos = np.array(control_cameras[idx_p3]["position"])

        p1_wxyz = control_cameras[idx_p1]["wxyz"]
        p2_wxyz = control_cameras[idx_p2]["wxyz"]

        num_interior_frames = end_frame - start_frame - 1

        if num_interior_frames > 0:
            for i in range(1, num_interior_frames + 1):
                frame_idx = start_frame + i
                t = i / (num_interior_frames + 1)

                position = catmull_rom_segment(p0_pos, p1_pos, p2_pos, p3_pos, t, trajectory_tension)

                rot1 = R.from_quat([p1_wxyz[1], p1_wxyz[2], p1_wxyz[3], p1_wxyz[0]])
                rot2 = R.from_quat([p2_wxyz[1], p2_wxyz[2], p2_wxyz[3], p2_wxyz[0]])
                key_rots = R.from_quat([rot1.as_quat(), rot2.as_quat()])
                slerp_interp = Slerp([0, 1], key_rots)
                rot_interp = slerp_interp(t)
                quat_xyzw = rot_interp.as_quat()
                wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                f1_avg = (control_cameras[idx_p1]["fx"] + control_cameras[idx_p1]["fy"]) / 2.0
                f2_avg = (control_cameras[idx_p2]["fx"] + control_cameras[idx_p2]["fy"]) / 2.0
                x_interp = np.exp(np.log(1.0 / f1_avg) * (1 - t) + np.log(1.0 / f2_avg) * t)
                f_avg = 1.0 / x_interp
                fx = fy = f_avg

                fov_vertical = np.degrees(2 * np.arctan(IMAGE_HEIGHT / (2.0 * f_avg)))
                fov = np.degrees(2 * np.arctan((IMAGE_WIDTH / 2.0) / f_avg))

                aspect = control_cameras[idx_p1]["aspect"] * (1 - t) + control_cameras[idx_p2]["aspect"] * t
                cx = control_cameras[idx_p1]["cx"] * (1 - t) + control_cameras[idx_p2]["cx"] * t
                cy = control_cameras[idx_p1]["cy"] * (1 - t) + control_cameras[idx_p2]["cy"] * t

                output_traj[frame_idx] = {
                    "position": position,
                    "wxyz": wxyz,
                    "fov": fov,
                    "fov_vertical": fov_vertical,
                    "aspect": aspect,
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(cx),
                    "cy": float(cy),
                }

    return output_traj


def invalidate_trajectory_cache():
    """Mark trajectory cache as invalid - will recompute on next access."""
    global trajectory_needs_update
    trajectory_needs_update = True


def visualize_output_trajectory():
    """
    Visualize the output trajectory with color-coded segments:
    - Blue: Original trajectory (when NO user keyframes exist)
    - Purple: Segments between original cameras (when user keyframes exist)
    - Green: Segments connecting to/from user-defined cameras
    """
    try:
        server.scene.remove("/target_paths/spline")
    except:
        pass

    output_traj = compute_output_trajectory()
    if output_traj is None or len(output_traj) == 0:
        return

    # Identify which frames are user-defined (captured cameras)
    user_defined_frames = set()
    for frame_ui in new_keyframes.keys():
        user_defined_frames.add(frame_ui - 1)  # Convert to 0-based index


    # Build control points for VISUAL spline: frame_0 + keyframes + frame_48
    # The Catmull-Rom spline will pass through these control points
    control_points = []
    control_point_frames = []

    # Always start with frame 0 (original)
    control_points.append(output_traj[0]["position"])
    control_point_frames.append(0)

    # Add all user-defined keyframes in order (excluding endpoints)
    last_frame_idx = len(output_traj) - 1
    for frame_idx in sorted(user_defined_frames):
        if frame_idx != 0 and frame_idx != last_frame_idx:
            control_points.append(output_traj[frame_idx]["position"])
            control_point_frames.append(frame_idx)

    # Always end with last frame (original)
    if len(control_points) == 1 or control_point_frames[-1] != last_frame_idx:
        control_points.append(output_traj[last_frame_idx]["position"])
        control_point_frames.append(last_frame_idx)

    control_points = np.array(control_points)

    color = (100, 100, 255) if len(user_defined_frames) == 0 else (100, 255, 100)

    # Create high-resolution Catmull-Rom spline through control points
    # High segment count ensures smooth visual representation
    server.scene.add_spline_catmull_rom(
        name="/target_paths/spline",
        positions=control_points,
        color=color,
        line_width=3.0,
        tension=trajectory_tension,
        segments=500,  # High resolution for smooth visualization
    )



# ============================================================================
# Scene Visualization
# ============================================================================

def update_scene(frame_idx: int, update_trajectory_line: bool = False):
    """
    Update the Viser scene for the given frame.

    Args:
        frame_idx: Frame index (0-48)
        update_trajectory_line: If True, also update the purple trajectory line (slow)
                               Only set to True when keyframes change, not on every frame change
    """
    if not scene_data["loaded"]:
        return

    frame_idx = max(0, min(frame_idx, scene_data["num_frames"] - 1))
    scene_data["current_frame"] = frame_idx

    # Update point cloud for current frame
    server.scene.add_point_cloud(
        name="/pointcloud",
        points=scene_data["point_clouds"][frame_idx],
        colors=scene_data["colors"][frame_idx],
        point_size=point_size,
    )

    # Get output trajectory (uses cache, very fast)
    output_traj = compute_output_trajectory()

    if output_traj is not None and len(output_traj) > frame_idx:
        # Show ONLY the output trajectory camera for current frame (purple)
        output_cam = output_traj[frame_idx]

        server.scene.add_camera_frustum(
            name="/target_cameras",
            fov=output_cam["fov"],  # Horizontal FOV for frustum visualization
            aspect=output_cam["aspect"],
            scale=0.15,
            wxyz=output_cam["wxyz"],
            position=output_cam["position"],
            color=(0, 127, 255),
            visible=False,
        )

        # Update source camera frustum if enabled
        if scene_data.get("show_source_cameras", False):
            _show_source_cameras_frustum(frame_idx)

        # Only update trajectory line when explicitly requested (when keyframes change)
        if update_trajectory_line:
            visualize_output_trajectory()


def _show_source_cameras_frustum(frame_idx: int):
    """Show source camera frustum at the given frame (synced per-frame)."""
    global _source_cameras_frustum_handle
    cam = scene_data["cameras"][frame_idx]
    _source_cameras_frustum_handle = server.scene.add_camera_frustum(
        name="/source_cameras/frustum",
        fov=cam["fov"],
        aspect=cam["aspect"],
        scale=0.15,
        wxyz=cam["wxyz"],
        position=cam["position"],
        color=(255, 64, 64),
    )
    _source_cameras_frustum_handle.visible = True


def _show_source_cameras_path():
    """Add a static spline through all source camera positions."""
    global _source_cameras_path_handle
    positions = np.array([c["position"] for c in scene_data["cameras"]])
    _source_cameras_path_handle = server.scene.add_spline_catmull_rom(
        name="/source_cameras/path",
        positions=positions,
        color=(255, 64, 64),
        line_width=2.0,
        tension=0.5,
        segments=200,
    )
    _source_cameras_path_handle.visible = True


def _hide_source_cameras():
    """Hide source camera frustum and path."""
    global _source_cameras_frustum_handle, _source_cameras_path_handle
    if _source_cameras_frustum_handle is not None:
        _source_cameras_frustum_handle.visible = False
    if _source_cameras_path_handle is not None:
        _source_cameras_path_handle.visible = False


def _build_static_bg_points():
    """Concatenate src static base with DSE static points at the current frame interval.

    Returns (points, colors) or (None, None) if no src base is available.
    """
    src_pts = scene_data.get("src_static_points")
    src_clrs = scene_data.get("src_static_colors")
    if src_pts is None:
        return None, None

    cache = scene_data.get("dse_static_cache")
    if cache is None:
        return src_pts, src_clrs

    interval = max(1, int(scene_data.get("dse_frame_interval", 1)))
    dse_indices = list(range(cache["dse_start"], cache["dse_end"], interval))
    dse_pts, dse_clrs = _aggregate_static_points(
        cache["depths_for_bg"], cache["images"], cache["intrinsics"],
        cache["cam_c2w"], cache["static_mask"], dse_indices,
    )
    if dse_pts is None:
        return src_pts, src_clrs
    return (
        np.concatenate([src_pts, dse_pts]).astype(np.float32),
        np.concatenate([src_clrs, dse_clrs]).astype(np.uint8),
    )


def show_static_bg():
    """Show the persistent static background overlay (src + optional DSE)."""
    global _static_bg_handle
    points, colors = _build_static_bg_points()
    if points is None:
        return
    _static_bg_handle = server.scene.add_point_cloud(
        name="/static_bg",
        points=points,
        colors=colors,
        point_size=point_size,
    )
    _static_bg_handle.visible = True


def hide_static_bg():
    """Hide the persistent static background point cloud overlay."""
    global _static_bg_handle
    if _static_bg_handle is not None:
        _static_bg_handle.visible = False


def rebuild_static_bg_if_visible():
    """Re-aggregate and redraw the static background overlay if currently shown."""
    if scene_data.get("show_static_bg", False):
        show_static_bg()


def _show_dse_cameras():
    """Show all DSE camera frustums + a spline through their positions."""
    global _dse_cameras_frustum_handles, _dse_cameras_path_handle
    _hide_dse_cameras()  # Clear any stale handles first
    dse_cams = scene_data.get("dse_cameras") or []
    if not dse_cams:
        return
    _dse_cameras_frustum_handles = []
    for i, cam in enumerate(dse_cams):
        handle = server.scene.add_camera_frustum(
            name=f"/dse_cameras/frustum_{i}",
            fov=cam["fov"],
            aspect=cam["aspect"],
            scale=0.1,
            wxyz=cam["wxyz"],
            position=cam["position"],
            color=(64, 200, 255),
        )
        handle.visible = True
        _dse_cameras_frustum_handles.append(handle)
    positions = np.array([c["position"] for c in dse_cams])
    if len(positions) >= 2:
        _dse_cameras_path_handle = server.scene.add_spline_catmull_rom(
            name="/dse_cameras/path",
            positions=positions,
            color=(64, 200, 255),
            line_width=2.0,
            tension=0.5,
            segments=200,
        )
        _dse_cameras_path_handle.visible = True


def _hide_dse_cameras():
    """Hide all DSE camera frustums and the path spline."""
    global _dse_cameras_frustum_handles, _dse_cameras_path_handle
    for handle in _dse_cameras_frustum_handles:
        handle.visible = False
    _dse_cameras_frustum_handles = []
    if _dse_cameras_path_handle is not None:
        _dse_cameras_path_handle.visible = False
        _dse_cameras_path_handle = None


@app.get("/api/health")
async def health():
    return {"status": "ok", "viser_port": 8080}


class LoadFolderRequest(BaseModel):
    folder_path: str


@app.post("/api/load-folder")
async def load_folder(request: LoadFolderRequest):
    """Load a 4D reconstruction folder and visualize in Viser."""
    try:
        folder_path = Path(request.folder_path).expanduser()

        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="Folder not found")

        if not folder_path.is_dir():
            raise HTTPException(status_code=400, detail="Path must be a directory")

        required = ["video.mp4", "depths", "cameras.npz"]
        missing = [r for r in required if not (folder_path / r).exists()]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required files/folders: {missing}")

        # Load and process in background thread to avoid blocking
        def load_data():
            global scene_data, captured_cameras, new_keyframes
            processed = load_and_process_folder(str(folder_path))
            # Keep an immutable copy of the raw per-frame clouds so toggling edits off restores them
            processed["point_clouds_original"] = list(processed["point_clouds"])
            processed["colors_original"] = list(processed["colors"])
            scene_data.update(processed)
            scene_data["loaded"] = True
            scene_data["file_path"] = str(folder_path)
            scene_data["edits_ready"] = False

            # Reset captured cameras to match actual frame count
            captured_cameras = [None] * scene_data["num_frames"]
            new_keyframes = {}
            invalidate_trajectory_cache()

            # Register the scene for edits; the heavy GPU buffer build is deferred to the first
            # edit so camera-design-only sessions (and CPU nodes that can't run SAM3 anyway)
            # don't pay the VRAM/compute cost. Enable the panel only if CUDA is present.
            try:
                edits_mod.load_src_from_folder(
                    str(folder_path),
                    height=scene_data["image_height"], width=scene_data["image_width"],
                    cached_scene=scene_data.get("cached_scene_for_edits"),
                )
                import torch
                scene_data["edits_ready"] = torch.cuda.is_available()
                if not scene_data["edits_ready"]:
                    print("[edits] CUDA not available — edits panel disabled (camera design still works)")
            except Exception as err:
                print(f"[WARN] Edit registration failed ({err}) — edits panel will be disabled")

            # Update GUI controls
            if frame_slider is not None:
                frame_slider.max = scene_data["num_frames"]
                frame_slider.value = 1  # Start at frame 1 (internal index 0)

            # Display first frame with trajectory line
            update_scene(0, update_trajectory_line=True)

            # Show static background by default if available
            if scene_data.get("has_static_bg", False):
                show_static_bg()

            # Snap viewport to working camera (output trajectory) at frame 0
            output_traj = compute_output_trajectory()
            if output_traj and len(output_traj) > 0:
                cam = output_traj[0]
                for client in server.get_clients().values():
                    client.camera.wxyz = cam["wxyz"]
                    client.camera.position = cam["position"]
                    client.camera.fov = np.radians(cam["fov_vertical"])

            print("Reconstruction folder loaded and ready!")

        threading.Thread(target=load_data, daemon=True).start()

        return {
            "status": "loading",
            "message": "Loading reconstruction folder ...",
            "folder_name": folder_path.name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/load-status")
async def get_load_status():
    """Get current loading status and file info."""
    if not scene_data["loaded"]:
        return {
            "status": "not_loaded",
            "loaded": False
        }

    # Get first frame camera data (frame 1, index 0)
    first_cam = scene_data["cameras"][0]

    # Build the extrinsics matrix (4x4 camera-to-world transform)
    rotation_matrix = tf.SO3(first_cam["wxyz"]).as_matrix()
    position = first_cam["position"]
    extrinsics_c2w = np.eye(4)
    extrinsics_c2w[:3, :3] = rotation_matrix
    extrinsics_c2w[:3, 3] = position

    # Get intrinsics [fx, fy, cx, cy]
    intrinsics_frame1 = [
        float(first_cam["fx"]),
        float(first_cam["fy"]),
        float(first_cam["cx"]),
        float(first_cam["cy"])
    ]

    return {
        "status": "loaded",
        "loaded": True,
        "file_path": scene_data.get("file_path", "Unknown"),
        "num_frames": scene_data["num_frames"],
        "fps": scene_data["fps"],
        "parsing_info": {
            "depths_shape": f"({scene_data['num_frames']}, {IMAGE_HEIGHT}, {IMAGE_WIDTH})",
            "video_shape": f"({scene_data['num_frames']}, {IMAGE_HEIGHT}, {IMAGE_WIDTH}, 3)",
            "intrinsics_shape": f"({scene_data['num_frames']}, 4)",
            "cameras_shape": f"({scene_data['num_frames']}, 4, 4)",
            "point_clouds_generated": scene_data["num_frames"],
        },
        "first_frame_camera": {
            "intrinsics": {
                "fx": intrinsics_frame1[0],
                "fy": intrinsics_frame1[1],
                "cx": intrinsics_frame1[2],
                "cy": intrinsics_frame1[3],
                "matrix_3x3": [
                    [intrinsics_frame1[0], 0.0, intrinsics_frame1[2]],
                    [0.0, intrinsics_frame1[1], intrinsics_frame1[3]],
                    [0.0, 0.0, 1.0]
                ]
            },
            "extrinsics_c2w": {
                "matrix_4x4": extrinsics_c2w.tolist(),
                "rotation_3x3": rotation_matrix.tolist(),
                "translation": position.tolist(),
                "quaternion_wxyz": first_cam["wxyz"].tolist()
            },
            "fov_degrees": float(first_cam["fov"]),
            "fov_vertical_degrees": float(first_cam["fov_vertical"]),
            "aspect_ratio": float(first_cam["aspect"])
        }
    }


@app.post("/api/scene/update")
async def update_scene_api(data: dict):
    return {"status": "updated", "data": data}


class KeyframeRequest(BaseModel):
    frame: int
    intrinsics_override: Optional[dict] = None  # Optional {fx, fy, cx, cy}


class KeyframeData(BaseModel):
    frame: int
    position: list
    wxyz: list
    fov: float
    aspect: float


@app.post("/api/keyframe/capture")
async def capture_keyframe(request: KeyframeRequest):
    """Capture current camera pose as a keyframe at specified frame."""
    if not scene_data["loaded"]:
        raise HTTPException(status_code=400, detail="No scene loaded")

    frame = request.frame
    if frame < 1 or frame > scene_data["num_frames"]:
        raise HTTPException(status_code=400, detail=f"Frame must be between 1 and {scene_data['num_frames']}")

    # Get all connected clients
    clients = server.get_clients()
    if not clients:
        raise HTTPException(status_code=400, detail="No connected clients")

    frame_index = frame - 1  # Convert to internal index

    # Get the first client's camera state
    client = list(clients.values())[0]
    camera_state = client.camera

    # Get intrinsics (use override if provided, else current frame)
    original_frame_cam = scene_data["cameras"][frame_index]

    if request.intrinsics_override:
        fx = float(request.intrinsics_override['fx'])
        fy = float(request.intrinsics_override['fy'])
        # ALWAYS use centered principal point for user-defined keyframes
        cx = IMAGE_WIDTH / 2.0  # width / 2
        cy = IMAGE_HEIGHT / 2.0  # height / 2

        # Recalculate FOV from custom fx/fy (matching process_camera calculation)
        f_avg = (fx + fy) / 2.0
        fov_vertical_rad = 2 * np.arctan(IMAGE_HEIGHT / (2 * f_avg))
        fov_vertical_deg = np.degrees(fov_vertical_rad)
        fov_horizontal_rad = 2 * np.arctan(IMAGE_WIDTH / (2 * f_avg))
        fov_deg = np.degrees(fov_horizontal_rad)

    else:
        fx = float(original_frame_cam["fx"])
        fy = float(original_frame_cam["fy"])
        cx = float(original_frame_cam["cx"])
        cy = float(original_frame_cam["cy"])
        fov_deg = float(original_frame_cam["fov"])
        fov_vertical_deg = float(original_frame_cam["fov_vertical"])

    # Create captured pose with BOTH frame_ui and frame_index (matching Viser GUI)
    captured_pose = {
        "frame_ui": frame,  # Store user-facing frame number
        "frame_index": frame_index,  # Store internal index
        "position": camera_state.position.tolist(),
        "wxyz": camera_state.wxyz.tolist(),
        "fov": float(fov_deg),  # Horizontal FOV
        "fov_vertical": float(fov_vertical_deg),  # Vertical FOV
        "aspect": float(original_frame_cam["aspect"]),  # Use current frame's aspect
        "fx": float(fx),  # Preserve exact intrinsics (custom or original)
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }

    # Check if this frame already has a captured camera
    was_existing = captured_cameras[frame_index] is not None

    # Store in captured_cameras array (overwrites if exists)
    captured_cameras[frame_index] = {
        "position": np.array(captured_pose["position"]),
        "wxyz": np.array(captured_pose["wxyz"]),
        "fov": captured_pose["fov"],
        "fov_vertical": captured_pose["fov_vertical"],
        "aspect": captured_pose["aspect"],
        "fx": captured_pose["fx"],
        "fy": captured_pose["fy"],
        "cx": captured_pose["cx"],
        "cy": captured_pose["cy"],
    }

    # Also store in keyframes dict for timeline sync (overwrites if exists)
    new_keyframes[frame] = {
        "position": captured_pose["position"],
        "wxyz": captured_pose["wxyz"],
        "fov": captured_pose["fov"],
        "fov_vertical": captured_pose["fov_vertical"],
        "aspect": captured_pose["aspect"],
        "fx": captured_pose["fx"],
        "fy": captured_pose["fy"],
        "cx": captured_pose["cx"],
        "cy": captured_pose["cy"],
    }

    # Add to trajectory list (matching Viser GUI behavior)
    new_camera_trajectory.append(captured_pose)

    print(f"Captured keyframe at frame {frame} (total: {len(new_keyframes)})")

    # Invalidate cache and update trajectory line
    invalidate_trajectory_cache()
    update_scene(frame_index, update_trajectory_line=True)

    return {
        "status": "captured",
        "frame": frame,
        "pose": {
            "position": captured_pose["position"],
            "wxyz": captured_pose["wxyz"],
            "fov": captured_pose["fov"],
            "aspect": captured_pose["aspect"]
        }
    }


@app.post("/api/keyframe/add")
async def add_keyframe(keyframe: KeyframeData):
    """Add a keyframe with explicit pose data."""
    frame = keyframe.frame
    max_frames = scene_data["num_frames"] if scene_data["loaded"] else MAX_FRAMES
    if frame < 1 or frame > max_frames:
        raise HTTPException(status_code=400, detail=f"Frame must be between 1 and {max_frames}")

    frame_index = frame - 1

    # Store in captured_cameras array (overwrites if exists)
    captured_cameras[frame_index] = {
        "position": np.array(keyframe.position),
        "wxyz": np.array(keyframe.wxyz),
        "fov": keyframe.fov,
        "aspect": keyframe.aspect
    }

    # Also store in keyframes dict (overwrites if exists)
    new_keyframes[frame] = {
        "position": keyframe.position,
        "wxyz": keyframe.wxyz,
        "fov": keyframe.fov,
        "aspect": keyframe.aspect
    }

    # Invalidate cache and update trajectory line
    invalidate_trajectory_cache()
    update_scene(scene_data["current_frame"], update_trajectory_line=True)

    return {"status": "added", "frame": frame}


@app.delete("/api/keyframe/{frame}")
async def delete_keyframe(frame: int):
    """Remove a keyframe at specified frame."""
    if frame in new_keyframes:
        frame_index = frame - 1

        captured_cameras[frame_index] = None
        del new_keyframes[frame]
        invalidate_trajectory_cache()
        update_scene(scene_data["current_frame"], update_trajectory_line=True)
        print(f"Deleted keyframe at frame {frame} (remaining: {len(new_keyframes)})")

        return {"status": "deleted", "frame": frame}
    else:
        raise HTTPException(status_code=404, detail="Keyframe not found")


@app.get("/api/keyframes")
async def get_keyframes():
    """Get all current keyframes."""
    return {
        "keyframes": new_keyframes,
        "count": len(new_keyframes)
    }


@app.post("/api/keyframes/clear")
async def clear_keyframes():
    """Clear all keyframes."""
    global new_keyframes, captured_cameras, client_follow_mode
    new_keyframes = {}
    captured_cameras = [None] * (scene_data["num_frames"] if scene_data["loaded"] else MAX_FRAMES)

    # Reset follow mode for all clients to prevent stuck camera
    # When keyframes are cleared, the trajectory changes, so disable auto-follow
    client_follow_mode = {}

    # Invalidate cache and update trajectory line (will revert to original)
    if scene_data["loaded"]:
        invalidate_trajectory_cache()
        update_scene(scene_data["current_frame"], update_trajectory_line=True)

    return {"status": "cleared"}


@app.get("/api/frame/current")
async def get_current_frame():
    """Get the current frame index."""
    return {
        "frame": scene_data["current_frame"] + 1,  # Convert to user-facing (1-500)
        "loaded": scene_data["loaded"]
    }


@app.post("/api/frame/set")
async def set_current_frame(data: dict):
    """Set the current frame."""
    frame_ui = data.get("frame", 1)
    if frame_ui < 1 or frame_ui > scene_data["num_frames"]:
        raise HTTPException(status_code=400, detail=f"Frame must be between 1 and {scene_data['num_frames']}")

    frame_index = frame_ui - 1
    update_scene(frame_index)

    # Update the frame slider if it exists
    if frame_slider is not None:
        frame_slider.value = frame_ui

    # Update camera for all clients with follow mode enabled (using output trajectory)
    if scene_data["loaded"]:
        output_traj = compute_output_trajectory()
        if output_traj and len(output_traj) > frame_index:
            cam = output_traj[frame_index]
            for client in server.get_clients().values():
                client_id = client.client_id
                if client_follow_mode.get(client_id, False):
                    client.camera.wxyz = cam["wxyz"]
                    client.camera.position = cam["position"]
                    client.camera.fov = np.radians(cam["fov_vertical"])

    return {"status": "updated", "frame": frame_ui}


def _playback_worker(fps: float):
    """Background thread that advances frames for server-side playback.

    Uses pace-adjusted sleep: if a frame update takes longer than the target
    interval, the next frame fires immediately (no sleep) so no frames are
    ever skipped. Playback just slows down gracefully under load.
    """
    interval = 1.0 / fps
    while not playback_stop_event.is_set():
        if not scene_data["loaded"]:
            playback_stop_event.wait(timeout=0.1)
            continue
        t_start = time.time()

        current = scene_data["current_frame"]
        next_frame = (current + 1) % scene_data["num_frames"]
        update_scene(next_frame)
        if frame_slider is not None:
            frame_slider.value = next_frame + 1

        # Apply follow camera for clients in follow mode
        output_traj = compute_output_trajectory()
        if output_traj and len(output_traj) > next_frame:
            cam = output_traj[next_frame]
            for client in server.get_clients().values():
                if client_follow_mode.get(client.client_id, False):
                    client.camera.wxyz = cam["wxyz"]
                    client.camera.position = cam["position"]
                    client.camera.fov = np.radians(cam["fov_vertical"])

        # Sleep only for the remaining time in the interval; if the update
        # took longer than the interval, proceed immediately (no skip)
        elapsed = time.time() - t_start
        remaining = interval - elapsed
        if remaining > 0:
            playback_stop_event.wait(timeout=remaining)


@app.post("/api/playback/start")
async def start_playback(data: dict):
    """Start server-side playback."""
    global playback_thread
    fps = float(data.get("fps", 30))
    fps = max(1.0, min(60.0, fps))

    # Stop existing playback thread
    playback_stop_event.set()
    if playback_thread is not None and playback_thread.is_alive():
        playback_thread.join(timeout=1.0)

    # Start new playback thread
    playback_stop_event.clear()
    playback_thread = threading.Thread(target=_playback_worker, args=(fps,), daemon=True)
    playback_thread.start()

    return {"status": "playing", "fps": fps}


@app.post("/api/playback/stop")
async def stop_playback():
    """Stop server-side playback."""
    playback_stop_event.set()
    return {"status": "stopped"}


@app.post("/api/static-bg/toggle")
async def toggle_static_bg(data: dict):
    """Toggle the persistent static background point cloud overlay."""
    show = data.get("show", False)
    scene_data["show_static_bg"] = show
    show_static_bg() if show else hide_static_bg()
    return {"status": "updated", "show_static_bg": show}


@app.post("/api/source-cameras/toggle")
async def toggle_source_cameras(data: dict):
    """Toggle per-frame source camera frustum + path."""
    show = data.get("show", False)
    scene_data["show_source_cameras"] = show
    if show:
        _show_source_cameras_path()
        _show_source_cameras_frustum(scene_data["current_frame"])
    else:
        _hide_source_cameras()
    return {"status": "updated", "show_source_cameras": show}


@app.post("/api/dse-cameras/toggle")
async def toggle_dse_cameras(data: dict):
    """Toggle DSE camera frustums + path overlay."""
    show = data.get("show", False)
    scene_data["show_dse_cameras"] = show
    if show:
        _show_dse_cameras()
    else:
        _hide_dse_cameras()
    return {"status": "updated", "show_dse_cameras": show}


@app.get("/api/dse/frame-interval")
async def get_dse_frame_interval():
    """Get current DSE static-background frame interval."""
    return {"frame_interval": scene_data.get("dse_frame_interval", 1)}


@app.post("/api/dse/frame-interval")
async def set_dse_frame_interval(data: dict):
    """Set DSE static-background frame interval and rebuild the overlay if visible."""
    interval = max(1, int(data.get("frame_interval", 1)))
    scene_data["dse_frame_interval"] = interval
    rebuild_static_bg_if_visible()
    return {"status": "updated", "frame_interval": interval}


@app.post("/api/cameras/follow")
async def toggle_follow_camera(data: dict):
    """Toggle follow camera mode for all clients."""
    follow = data.get("follow", False)

    if follow_camera_checkbox is not None:
        follow_camera_checkbox.value = follow

    # Update follow mode for all clients
    for client in server.get_clients().values():
        client_id = client.client_id
        client_follow_mode[client_id] = follow

        if follow and scene_data["loaded"]:
            # Snap to current frame's output trajectory camera
            # Use current frame from scene_data (maintained by React app)
            frame_index = scene_data["current_frame"]
            frame_ui = frame_index + 1
            output_traj = compute_output_trajectory()
            if output_traj and len(output_traj) > frame_index:
                cam = output_traj[frame_index]
                client.camera.wxyz = cam["wxyz"]
                client.camera.position = cam["position"]
                # Use VERTICAL FOV for viewport camera (Viser expects vertical FOV)
                client.camera.fov = np.radians(cam["fov_vertical"])

    return {"status": "updated", "follow": follow}


@app.post("/api/cameras/snap")
async def snap_to_camera():
    """Snap all clients to current frame's output trajectory camera."""
    if not scene_data["loaded"]:
        raise HTTPException(status_code=400, detail="No scene loaded")

    # Use current frame from scene_data (maintained by React app)
    frame_index = scene_data["current_frame"]
    frame_ui = frame_index + 1
    output_traj = compute_output_trajectory()

    if output_traj and len(output_traj) > frame_index:
        cam = output_traj[frame_index]

        for client in server.get_clients().values():
            client.camera.wxyz = cam["wxyz"]
            client.camera.position = cam["position"]
            client.camera.fov = np.radians(cam["fov_vertical"])

    return {"status": "snapped", "frame": frame_ui}


class ExportRequest(BaseModel):
    filename: str = "output_camera_trajectory.npz"


@app.post("/api/trajectory/export")
async def export_trajectory(request: ExportRequest = None):
    """Export the interpolated output trajectory."""
    if not scene_data["loaded"]:
        raise HTTPException(status_code=400, detail="No scene loaded")

    # Get filename from request or use default
    filename = request.filename if request else "output_cameras.npz"
    if not filename.endswith('.npz'):
        filename += '.npz'

    # Save to cam_ui/exported_cameras/ (relative to repo root where server runs)
    export_dir = Path("cam_ui/exported_cameras")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Compute the full output trajectory
    output_traj = compute_output_trajectory()
    if output_traj is None:
        raise HTTPException(status_code=500, detail="Failed to compute trajectory")

    # Convert to cam_c2w format (500, 4, 4) - matching input format
    cam_c2w_output = []
    intrinsics_output = []

    for i, cam in enumerate(output_traj):
        # Build 4x4 transformation matrix (EXTRINSICS)
        rotation_matrix = tf.SO3(cam["wxyz"]).as_matrix()
        position = cam["position"]

        c2w = np.eye(4)
        c2w[:3, :3] = rotation_matrix
        c2w[:3, 3] = position

        cam_c2w_output.append(c2w)

        # CRITICAL: Use exact intrinsics from the original frame
        # These are the same intrinsics used for visualization
        # Ensures perfect fidelity between what you see and what you export
        fx = cam.get("fx") or scene_data["cameras"][i]["fx"]
        fy = cam.get("fy") or scene_data["cameras"][i]["fy"]
        cx = cam.get("cx") or scene_data["cameras"][i]["cx"]
        cy = cam.get("cy") or scene_data["cameras"][i]["cy"]
        intrinsics_output.append([fx, fy, cx, cy])


    cam_c2w_output = np.array(cam_c2w_output)  # Shape: (num_frames, 4, 4)
    intrinsics_output = np.array(intrinsics_output)  # Shape: (num_frames, 4)

    # Create cam_c2w_fsff (Force Same First Frame)
    output_traj_fsff = compute_output_trajectory_fsff()

    if output_traj_fsff is not None:
        cam_c2w_fsff_list = []
        intrinsics_fsff_list = []

        for i, cam in enumerate(output_traj_fsff):
            # Build 4x4 transformation matrix
            rotation_matrix = tf.SO3(cam["wxyz"]).as_matrix()
            position = cam["position"]

            c2w = np.eye(4)
            c2w[:3, :3] = rotation_matrix
            c2w[:3, 3] = position

            cam_c2w_fsff_list.append(c2w)

            # Use intrinsics from FSFF trajectory
            fx = cam.get("fx") or scene_data["cameras"][i]["fx"]
            fy = cam.get("fy") or scene_data["cameras"][i]["fy"]
            cx = cam.get("cx") or scene_data["cameras"][i]["cx"]
            cy = cam.get("cy") or scene_data["cameras"][i]["cy"]
            intrinsics_fsff_list.append([fx, fy, cx, cy])

        cam_c2w_fsff = np.array(cam_c2w_fsff_list)
        intrinsics_fsff = np.array(intrinsics_fsff_list)

    else:
        print(f"[WARN] FSFF computation failed, using naive frame 0 replacement")
        cam_c2w_fsff = cam_c2w_output.copy()
        original_frame_0_rot = tf.SO3(scene_data["cameras"][0]["wxyz"]).as_matrix()
        original_frame_0_pos = scene_data["cameras"][0]["position"]
        cam_c2w_fsff[0, :3, :3] = original_frame_0_rot
        cam_c2w_fsff[0, :3, 3] = original_frame_0_pos
        intrinsics_fsff = intrinsics_output

    output_path = export_dir / filename
    np.savez(
        output_path,
        cam_c2w=cam_c2w_output,
        cam_c2w_fsff=cam_c2w_fsff,
        intrinsics=intrinsics_output,
        intrinsics_fsff=intrinsics_fsff,
    )
    print(f"Exported cameras: {len(output_traj)} frames → {output_path.absolute()} ({output_path.stat().st_size / 1024:.1f} KB)")

    return {
        "status": "exported",
        "path": str(output_path.absolute()),
        "num_frames": len(output_traj),
        "num_keyframes": len(new_keyframes),
        "shapes": {
            "cam_c2w": list(cam_c2w_output.shape),
            "intrinsics": list(intrinsics_output.shape),
        },
        "file_size_kb": round(output_path.stat().st_size / 1024, 2)
    }


# ============================================================================
# Edits
# ============================================================================

class EditsRequest(BaseModel):
    edits: list


class EditsExportRequest(BaseModel):
    filename: str = "edits.json"


@app.get("/api/edits")
async def get_edits():
    return {
        "edits": edits_mod.get_edits(),
        "ready": scene_data.get("edits_ready", False),
    }


@app.get("/api/edits/progress")
async def get_edits_progress():
    # Lightweight poll target for the UI — reports the current stage of an in-flight apply
    # (e.g., "Running SAM3 for prompt: ...", "Loading insert scene: ...") or "idle" when done.
    return {"progress": edits_mod.get_progress()}


@app.put("/api/edits")
async def put_edits(request: EditsRequest):
    # Log the moment the PUT lands — both to fd 2 (bypasses Python stdio) and to the edits log
    # file (`/tmp/vista4d_edits.log`), which is a guaranteed trace in case something in the
    # hosting environment is redirecting fd 2 itself.
    _put_line = f"[edits] PUT received with {len(request.edits)} edit(s)\n"
    try: os.write(2, _put_line.encode())
    except OSError: pass
    try:
        with open(edits_mod.EDITS_LOG_PATH, "a") as _f:
            _f.write(_put_line)
    except OSError: pass
    if not scene_data["loaded"]:
        raise HTTPException(status_code=400, detail="No scene loaded")
    if not scene_data.get("edits_ready", False):
        raise HTTPException(status_code=409, detail="Edit buffers are not ready yet — try again shortly")

    try:
        edits_mod.validate_edits_list(request.edits)
    except (AssertionError, ValueError) as err:
        raise HTTPException(status_code=400, detail=f"Invalid edits: {err}")

    # SAM3 inference + insert preprocessing are blocking torch ops; run off the event loop so
    # concurrent API calls (e.g., /api/controls/state polls, frame scrubs) stay responsive while
    # a heavy apply is in flight.
    try:
        # Always serve `apply_edits`'s return so the per-frame clouds go through the same
        # `buffers_to_frame_point_clouds(buffers_src + edits)` materializer for both the
        # edits-applied and edits-cleared cases. The old empty-case path swapped in
        # `point_clouds_original` (a `depth_to_pointcloud` numpy pass from scene load), which
        # has subtly different point counts / ordering than the GPU-buffer path. For a remove
        # edit that's the visible-symptom bug: apply-remove → viser shows N_filtered via
        # buffers_to_frame; undo → viser gets N_original via depth_to_pointcloud, but the
        # shape mismatch can confuse viser's point_cloud replacement, leaving the removed
        # subset still absent on screen. Unifying the path makes apply/undo symmetric.
        point_clouds, colors_list = await run_in_threadpool(edits_mod.apply_edits, request.edits)
        scene_data["point_clouds"] = point_clouds
        scene_data["colors"] = colors_list
    except Exception as err:
        # Include exception class so the frontend shows something actionable (e.g. RuntimeError vs
        # CUDA OOM vs SAM3 load failure) rather than just a bare message.
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply edits ({type(err).__name__}): {err}",
        )

    update_scene(scene_data["current_frame"])
    return {"status": "applied", "num_edits": len(request.edits)}


@app.post("/api/edits/export")
async def export_edits(request: EditsExportRequest):
    import json as _json
    filename = request.filename if request.filename.endswith(".json") else request.filename + ".json"
    export_dir = Path("cam_ui/exported_edits")
    export_dir.mkdir(parents=True, exist_ok=True)
    output_path = export_dir / filename
    with open(output_path, "w") as f:
        _json.dump({"edits": edits_mod.get_edits()}, f, indent=2)
    print(f"Exported edits: {len(edits_mod.get_edits())} edit(s) → {output_path.absolute()}")
    return {
        "status": "exported",
        "path": str(output_path.absolute()),
        "num_edits": len(edits_mod.get_edits()),
    }


@app.get("/api/trajectory/inspect")
async def inspect_trajectory():
    """Inspect the exported trajectory file."""
    output_path = Path("output_camera_trajectory.npz")

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="No exported trajectory file found")

    try:
        data = np.load(output_path)

        result = {
            "file_exists": True,
            "file_path": str(output_path.absolute()),
            "file_size_kb": round(output_path.stat().st_size / 1024, 2),
            "arrays": {}
        }

        for key in data.files:
            arr = data[key]
            array_info = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "size_kb": round(arr.nbytes / 1024, 2),
                "non_zero_elements": int(np.count_nonzero(arr)),
                "total_elements": int(arr.size)
            }

            # Add sample data
            if arr.ndim == 0:  # Scalar
                array_info["value"] = float(arr) if np.isscalar(arr) else arr.tolist()
            elif arr.ndim == 1 and len(arr) <= 5:
                array_info["data"] = arr.tolist()
            elif arr.ndim == 1:
                array_info["first_3"] = arr[:3].tolist()
                array_info["last_3"] = arr[-3:].tolist()
            elif arr.ndim == 2:
                array_info["first_row"] = arr[0].tolist()
                array_info["last_row"] = arr[-1].tolist()
            elif arr.ndim == 3:
                array_info["first_matrix"] = arr[0].tolist()
                array_info["last_matrix"] = arr[-1].tolist()

            result["arrays"][key] = array_info

        data.close()
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect file: {str(e)}")


@app.post("/api/factory-reset")
async def factory_reset():
    """Perform a complete factory reset."""
    global scene_data, new_camera_trajectory, new_keyframes, captured_cameras, client_follow_mode, \
        _static_bg_handle, _source_cameras_frustum_handle, _source_cameras_path_handle, \
        _dse_cameras_frustum_handles, _dse_cameras_path_handle

    # Stop any running playback
    playback_stop_event.set()

    _static_bg_handle = None
    _source_cameras_frustum_handle = None
    _source_cameras_path_handle = None
    _dse_cameras_frustum_handles = []
    _dse_cameras_path_handle = None

    # Clear the entire Viser scene
    server.scene.reset()

    # Reset all data structures
    scene_data = {
        "loaded": False,
        "point_clouds": None,
        "colors": None,
        "point_clouds_original": None,
        "colors_original": None,
        "cameras": None,
        "dse_cameras": None,
        "fps": 30.0,
        "num_frames": 0,
        "image_height": 0,
        "image_width": 0,
        "current_frame": 0,
        "playing": False,
        "file_path": None,
        "src_static_points": None,
        "src_static_colors": None,
        "dse_static_cache": None,
        "show_static_bg": False,
        "has_static_bg": False,
        "show_source_cameras": False,
        "show_dse_cameras": False,
        "has_dse": False,
        "num_dse_frames": 0,
        "dse_frame_interval": 1,
        "edits_ready": False,
    }
    edits_mod.reset_scene_state()
    new_camera_trajectory = []
    new_keyframes = {}
    captured_cameras = [None] * MAX_FRAMES
    client_follow_mode = {}

    # Reset GUI controls
    if frame_slider is not None:
        frame_slider.disabled = True
        frame_slider.value = 1
        frame_slider.max = 500
    if follow_camera_checkbox is not None:
        follow_camera_checkbox.value = False
    if status_text is not None:
        status_text.value = "Factory reset complete - no data loaded"
    if frame_info_text is not None:
        frame_info_text.value = "N/A"
    if captured_count_text is not None:
        captured_count_text.value = "0"

    return {"status": "reset"}


@app.get("/api/controls/state")
async def get_controls_state():
    """Get current state of all controls."""
    # Check if any client has follow mode enabled (since GUI checkbox is commented out)
    follow_camera_active = any(client_follow_mode.values()) if client_follow_mode else False

    return {
        "loaded": scene_data["loaded"],
        "current_frame": scene_data["current_frame"] + 1 if scene_data["loaded"] else 1,
        "num_frames": scene_data["num_frames"],
        "follow_camera": follow_camera_checkbox.value if follow_camera_checkbox else follow_camera_active,
        "captured_count": len(new_keyframes),
        "status": status_text.value if status_text else "No data loaded",
        "trajectory_tension": trajectory_tension,
        "point_size": point_size,
        "show_static_bg": scene_data.get("show_static_bg", False),
        "has_static_bg": scene_data.get("has_static_bg", False),
        "show_source_cameras": scene_data.get("show_source_cameras", False),
        "has_dse": scene_data.get("has_dse", False),
        "num_dse_frames": scene_data.get("num_dse_frames", 0),
        "dse_frame_interval": scene_data.get("dse_frame_interval", 1),
        "show_dse_cameras": scene_data.get("show_dse_cameras", False),
        "edits_ready": scene_data.get("edits_ready", False),
        "num_edits": len(edits_mod.get_edits()),
    }


@app.get("/api/trajectory/tension")
async def get_trajectory_tension():
    """Get current trajectory tension value."""
    return {"tension": trajectory_tension}


@app.post("/api/trajectory/tension")
async def set_trajectory_tension(data: dict):
    """Set trajectory tension and recompute visualization."""
    global trajectory_tension

    tension = data.get("tension", 0.5)
    tension = max(0.0, min(1.0, tension))  # Clamp to [0.0, 1.0]

    trajectory_tension = tension

    if scene_data["loaded"]:
        invalidate_trajectory_cache()
        update_scene(scene_data["current_frame"], update_trajectory_line=True)

    return {"status": "updated", "tension": trajectory_tension}


@app.get("/api/pointcloud/size")
async def get_point_size():
    """Get current point cloud size value."""
    return {"point_size": point_size}


@app.get("/api/viewport/fov")
async def get_viewport_fov():
    """Get current viewport FOV multiplier."""
    return {
        "fov_multiplier": float(viewport_fov_multiplier),
        "base_fov": float(scene_data["cameras"][0]["fov"]) if scene_data["loaded"] else None
    }


@app.post("/api/viewport/fov")
async def set_viewport_fov(data: dict):
    """Set viewport FOV multiplier and update all clients' viewport in real-time."""
    global viewport_fov_multiplier

    fov_multiplier = data.get("fov_multiplier", 1.0)
    fov_multiplier = max(0.1, min(5.0, fov_multiplier))  # Clamp to [0.1, 5.0]

    viewport_fov_multiplier = fov_multiplier

    # Update FOV for all connected clients in real-time
    # Always relative to frame 0 so the multiplier has consistent meaning across frames
    if scene_data["loaded"]:
        frame0_cam = scene_data["cameras"][0]
        frame0_f_avg = (frame0_cam["fx"] + frame0_cam["fy"]) / 2.0
        new_f_avg = frame0_f_avg * fov_multiplier
        new_fov_vertical_rad = 2 * np.arctan((IMAGE_HEIGHT / 2.0) / new_f_avg)

        for client in server.get_clients().values():
            client.camera.fov = new_fov_vertical_rad

    return {
        "status": "updated",
        "fov_multiplier": viewport_fov_multiplier
    }


@app.post("/api/pointcloud/size")
async def set_point_size(data: dict):
    """Set point cloud size and update visualization."""
    global point_size

    size = data.get("point_size", 0.003)
    size = max(0.0, min(0.01, size))

    point_size = size

    if scene_data["loaded"]:
        update_scene(scene_data["current_frame"])
        if scene_data.get("show_static_bg", False):
            show_static_bg()

    return {"status": "updated", "point_size": point_size}


def run_fastapi(port: int):
    # access_log=False suppresses the per-request "INFO: 127.0.0.1:xxxx - ... 200 OK" noise. Our
    # debounced edits PUT + the React status poll generate a lot of these, and they drown out the
    # progress prints from the edits pipeline (SAM3, insert loading). Errors still log.
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False)


# ============================================================================
# Main Application
# ============================================================================

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--viser-port", type=int, default=9997)
    parser.add_argument("--fastapi-port", type=int, default=9998)
    args = parser.parse_args()

    # Initialize Viser server with the specified port
    server = viser.ViserServer(port=args.viser_port)
    server.gui.configure_theme(
        control_layout="floating",
        show_logo=False,
        show_share_button=False,
    )

    # Snap any newly-connected client to the working camera at frame 0. The `load_data` thread
    # also snaps all currently-connected clients at load time, but that fires a one-shot: if the
    # browser connects *after* load completes (common — user loads the scene, then opens the tab),
    # that snap misses them and they land in viser's default free-orbit view. This handler
    # closes the gap.
    @server.on_client_connect
    def _snap_new_client_to_working_camera(client: viser.ClientHandle):
        if not scene_data.get("loaded"):
            return
        try:
            output_traj = compute_output_trajectory()
        except Exception:
            return
        if not output_traj:
            return
        cam = output_traj[0]
        client.camera.wxyz = cam["wxyz"]
        client.camera.position = cam["position"]
        client.camera.fov = np.radians(cam["fov_vertical"])

    # Start FastAPI in background
    fastapi_thread = threading.Thread(target=run_fastapi, args=(args.fastapi_port,), daemon=True)
    fastapi_thread.start()

    print("=" * 60)
    print("Vista4D: Viser Python Server")
    print("=" * 60)
    print(f"Viser UI:   http://localhost:{args.viser_port}")
    print(f"FastAPI:    http://localhost:{args.fastapi_port}")
    print("=" * 60)
    print("\nWaiting for reconstruction folder to be loaded via the React UI...\n")

    # ========================================================================
    # CONFIGURATION: Set your reconstruction folder path here
    # ========================================================================
    FOLDER_PATH = "/path/to/recon_and_seg/folder"  # Change this path
    AUTO_LOAD = False  # Set to True to auto-load on startup

    if AUTO_LOAD and Path(FOLDER_PATH).exists():
        print(f"\nAuto-loading: {FOLDER_PATH}")
        processed = load_and_process_folder(FOLDER_PATH)
        scene_data.update(processed)
        scene_data["loaded"] = True

    # Keep server running
    print("\nServer ready! Press Ctrl+C to stop.\n")

    while True:
        time.sleep(0.1)
