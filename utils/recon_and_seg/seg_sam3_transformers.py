"""This script uses the Transformers implementation of SAM3 and requires transformers>=5.0.0.

The problem is, the newest xfuser version (at the time of writing) is 0.4.5, which is *not* compatible with
transformers>=5.0.0 (some import errors due to transformers API changes), which is required for multi-GPU inference
(i.e., unified sequence parallel) to work for Vista4D inference.

To avoid having to install two environments, we have written a new seg_sam3.py which uses the SAM3 official repo's
implementation, but you should be able to switch to this transformers implementation once xfuser catches up to
transformers>=5.0.0 with some relatively minimal code changes. We will keep this here for reference.
"""

from typing import List, Union

import numpy as np
import numpy.typing as npt
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from utils.misc import cleanup


def init_sam3_video(
    model_id: str = "facebook/sam3", dtype: torch.dtype = torch.bfloat16, device: Union[torch.device, str] = "cuda",
):
    print(f"Loading Segment Anything 3 model with model_id=`{model_id}`")
    model = Sam3VideoModel.from_pretrained(model_id).to(dtype=dtype, device=device)
    model.eval()
    processor = Sam3VideoProcessor.from_pretrained(model_id)
    return model, processor


def invert_prompt_to_obj_ids(prompt_to_obj_ids):
    obj_ids_to_prompt = {}
    for k, v in prompt_to_obj_ids.items():
        for v_ in v:
            obj_ids_to_prompt[v_] = k
    return obj_ids_to_prompt


def run_sam3_video(
    video: npt.NDArray[np.uint8],
    model: Sam3VideoModel,
    processor: Sam3VideoProcessor,
    keywords: List[str],
    dtype: torch.dtype = torch.bfloat16,
    device: Union[torch.device, str] = "cuda",
):
    cleanup()

    num_frames, height, width, _ = video.shape
    keywords = [keyword.strip() for keyword in keywords]

    print(f"Running SAM3 inference with keywords={keywords}")
    session = processor.init_video_session(
        video=video, inference_device=device, processing_device=device, video_storage_device=device, dtype=dtype,
    )
    session = processor.add_text_prompt(inference_session=session, text=keywords)

    dynamic_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    seg_frames = []
    with torch.inference_mode():
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=session, max_frame_num_to_track=num_frames - 1,
        ):
            processed_outputs = processor.postprocess_outputs(session, model_outputs)  # TODO: Filter by scores?

            object_ids = processed_outputs.get("object_ids", None)
            if object_ids is None or object_ids.shape[0] == 0:
                seg_frames.append([])
                continue
            object_ids = object_ids.detach().cpu().numpy().astype(int)
            scores = processed_outputs["scores"].detach().cpu().numpy()
            boxes = processed_outputs["boxes"].detach().cpu().numpy()  # n 4, 4 is [x y x y]
            masks = processed_outputs["masks"].detach().cpu().numpy()  # n h w

            obj_ids_to_prompt = invert_prompt_to_obj_ids(processed_outputs["prompt_to_obj_ids"])

            instances = []
            for i in range(object_ids.shape[0]):
                instances.append({
                    "id": int(object_ids[i]),
                    "keyword": obj_ids_to_prompt[int(object_ids[i])],
                    "score": float(scores[i]),
                    "box_xyxy": boxes[i].astype(np.float32),
                    "mask": masks[i],
                })

            dynamic_mask[model_outputs.frame_idx] = masks.any(axis=0)  # n h w -> h w
            seg_frames.append(instances)
    print(f"Dynamic mask coverage: {dynamic_mask.astype(np.int64).sum() / np.prod(dynamic_mask.shape) * 100:.3f}%")

    del processed_outputs
    cleanup()

    return dynamic_mask, seg_frames
