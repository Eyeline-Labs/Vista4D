from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont


def color_from_id(obj_id: int):  # Deterministic bright-ish RGB color from an integer ID
    x = (obj_id * 2654435761) & 0xFFFFFFFF
    r = 64 + (x & 0xFF) // 2
    g = 64 + ((x >> 8) & 0xFF) // 2
    b = 64 + ((x >> 16) & 0xFF) // 2
    return int(r), int(g), int(b)


def overlay_instances_on_frame(frame: npt.NDArray[np.uint8], instances: List[Dict[str, Any]], alpha: float = 0.5):
    frame = frame.copy()
    for instance in instances:  # Apply mask overlays with color
        mask = instance["mask"]
        color = np.array(color_from_id(instance["id"]), dtype=np.uint8)[None, None]  # 3 -> 1 1 3 (for broadcasting)
        frame[mask] = np.clip(frame[mask] * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)

    frame = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(frame)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=16)
    except OSError:
        font = ImageFont.load_default()
    for inst in instances:  # Draw boxes + `keyword: score` labels
        x_0, y_0, x_1, y_1 = inst["box_xyxy"].tolist()
        color = color_from_id(inst["id"])
        draw.rectangle([x_0, y_0, x_1, y_1], outline=color, width=3)

        label = f"{inst['keyword']}: {inst['score']:.2f}"
        bbox = draw.textbbox((x_0 + 3, y_0 + 3), label, font=font)
        pad = 2
        # Draw semi-transparent white background behind label using a composited overlay
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=(255, 255, 255, 160),
        )
        frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(frame)

        draw.text((x_0 + 3, y_0 + 3), label, fill=color, font=font)

    frame = np.asarray(frame)
    return frame


def overlay_instances_on_video(
    video: npt.NDArray[np.uint8], seg_frames: List[List[Dict[str, Any]]], alpha: float = 0.5,
):
    return np.stack(
        [overlay_instances_on_frame(video[i], seg_frames[i], alpha=alpha) for i in range(video.shape[0])], axis=0,
    )
