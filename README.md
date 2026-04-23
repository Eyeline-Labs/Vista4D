# Vista4D: Video Reshooting with 4D Point Clouds (CVPR 2026 Highlight)

[![Project Page](https://img.shields.io/badge/Project-Page-yellow?logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ5ZWxsb3ciIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIvPjxsaW5lIHgxPSIyIiB5MT0iMTIiIHgyPSIyMiIgeTI9IjEyIi8+PHBhdGggZD0iTTEyIDJhMTUuMyAxNS4zIDAgMCAxIDQgMTAgMTUuMyAxNS4zIDAgMCAxLTQgMTAgMTUuMyAxNS4zIDAgMCAxLTQtMTAgMTUuMyAxNS4zIDAgMCAxIDQtMTB6Ii8+PC9zdmc+)](https://eyeline-labs.github.io/Vista4D)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2604.21915)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Vista4D-blue)](https://huggingface.co/Eyeline-Labs/Vista4D)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Eval%20Data-blue)](https://huggingface.co/datasets/Eyeline-Labs/Vista4D-Eval-Data)

[Kuan Heng Lin](https://kuanhenglin.github.io)<sup>1,3&lowast;</sup>, [Zhizheng Liu](https://bosmallear.github.io)<sup>1,4&lowast;</sup>, [Pablo Salamanca](https://pablosalaman.ca)<sup>1,2</sup>, [Yash Kant](https://yashkant.github.io)<sup>1,2</sup>, [Ryan Burgert](https://ryanndagreat.github.io)<sup>1,2,5&lowast;</sup>, [Yuancheng Xu](https://yuancheng-xu.github.io)<sup>1,2</sup>, [Koichi Namekata](https://kmcode1.github.io)<sup>1,2,6&lowast;</sup>, [Yiwei Zhao](https://zhaoyw007.github.io)<sup>2</sup>, [Bolei Zhou](https://boleizhou.github.io)<sup>4</sup>, [Micah Goldblum](https://goldblum.github.io)<sup>3</sup>, [Paul Debevec](https://www.pauldebevec.com)<sup>1,2</sup>, [Ning Yu](https://ningyu1991.github.io)<sup>1,2</sup> <br/>
<sup>1</sup>Eyeline Labs, <sup>2</sup>Netflix, <sup>3</sup>Columbia University, <sup>4</sup>UCLA, <sup>5</sup>Stony Brook University, <sup>6</sup>University of Oxford<br>

<sup>&lowast;</sup>*Work done during an internship at Eyeline Labs*

# &#128483; Updates

- **2026/04/23:** Vista4D inference code, model weights, and evaluation dataset have been released!
- **2026/04/09:** Vista4D has been selected as a Highlight paper &#10024;
- **2026/02/21:** Vista4D has been accepted to CVPR 2026 &#127881;

# &#128064; Overview

![Vista4D teaser figure](media/docs/teaser.jpg)

**Vista4D** is a *video reshooting* framework which synthesizes the dynamic scene represented by an input source video from novel camera trajectories and viewpoints. We bridge the distribution shift between training and inference for point-cloud-grounded video reshooting, as Vista4D is robust to point cloud artifacts from imprecise 4D reconstruction of real-world videos by training on noisy, reconstructed multiview videos. Our 4D point cloud with temporally-persistent static points also explicitly preserves scene content and improved camera control. Vista4D generalizes to real-world applications such as dynamic scene expansion (casual video capture of scene as background reference), 4D scene recomposition (point cloud editing), and long video inference with memory.

This repository contains the following:

- [Inference code for Vista4D for video reshooting](#vista4d-code)
  - [Instructions to download and use our model weights](#wan-21-and-vista4d-checkpoints)
- [Instructions to run Vista4D on our evaluation dataset](#evaluation-dataset-and-inference)
- [Inference code (and UI) for 4D scene recomposition (point cloud editing)](#application-4d-scene-recomposition-point-cloud-editing)
- [Inference code for dynamic scene expansion](#application-dynamic-scene-expansion-dse)
- [Our camera UI, for all of the above (video reshooting *and* all applications!)](#camera-ui-optional)

# &#127909; Vista4D code

## Environment setup

First, create a new Conda environment
```bash
conda create --name vista4d python=3.12
conda activate vista4d
```
We will use CUDA 12.8 for installation and compilation. If your system's default CUDA version is not 12.8 (or if you are unsure), run
```bash
conda install -c nvidia cuda-toolkit=12.8
conda install -c conda-forge gxx_linux-64  # Ensure a compatible C++ compiler
export CUDA_HOME=$CONDA_PREFIX  # Required for the environment setup
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
to install a self-contained CUDA toolkit into the environment.

Next, install PyTorch (we tested with version 2.10.0, though others may work)
```bash
pip3 install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```
and install all the other dependencies
```bash
pip3 install -r requirements.txt
```

<details>
<summary><b>⚠️ Setting up on SLURM login nodes or multiple GPU types</b></summary>

Before installing Flash Attention with the commands below, if you are installing on a GPU-less login node, or plan to run this environment across different GPU architectures (e.g., compiling on an A100 but running on an H100), you must explicitly specify the target architectures. *Failing to do this will cause the compiler to skip building the necessary kernels, resulting in a silent fallback to standard Math attention and causing out of memory (OOM) errors when doing Vista4D inference.* If this applies to you, run
```bash
# Architecture guide (select *only* the ones you need to keep compile times reasonable):
# 8.0 = A100 / A30
# 8.6 = RTX 3090 / A40 / RTX A6000
# 8.9 = RTX 4090 / RTX 6000 Ada / L40
# 9.0 = H100 / H200
# 9.0+PTX = JIT fallback for Blackwell GPUs (RTX PRO 6000 Blackwell / B100 / B200)

# Example: Compiling for A100, Ada generation, and Blackwell
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0+PTX"
```
</details>

Then, install Flash Attention and XFuser (for unified sequence parallel) with

```bash
export FLASH_ATTENTION_FORCE_BUILD="TRUE"  # Optional: If you want to force a compilation from source
pip3 install flash-attn==2.8.3 --no-build-isolation  # This might take a while if you are compiling from source
pip3 install "xfuser[flash-attn]==0.4.5"
```

## Video preprocessing

### 4D reconstruction and dynamic mask segmentation

For 4D reconstruction, you can pick from [Depth Anything 3 (DA3)](https://github.com/ByteDance-Seed/Depth-Anything-3) or [Pi3X](https://github.com/yyfz/Pi3). For dynamic mask segmentation, we use [Segment Anything 3 (SAM3)](https://github.com/facebookresearch/sam3), which requires you to request checkpoint access on their Hugging Face [repo](https://huggingface.co/facebook/sam3). Then, do
```bash
hf auth login
```
and paste your Hugging Face access token as instructed.

To run 4D reconstruction and dynamic mask segmentation, do
```bash
EXAMPLE=couple-newspaper RECON_METHOD=pi3 bash scripts/preprocess/example_recon_and_seg_single.sh
```
We have provided eight (8) example videos, `couple-newspaper`, `couple-walk`, `elderly-tennis`, `mountain-hike`, `park-selfie`, `parkour`, `snowboard`, and `soapbox` (all from our evaluation dataset and provided in `./media/single/`) which you configure through `EXAMPLE`. You can also pick from Pi3X (`pi3`) or DA3 (`da3`) as your 4D reconstruction method through `RECON_METHOD`, though our provided target cameras are designed with Pi3X's reconstruction. The reconstruction and segmentation results and visualization will be found in `./results/single/$EXAMPLE/recon_and_seg/` after running the script.

Our default reconstruction method is Pi3X, as we find it to contains less temporal flickering and geometric artifacts than DA3 while using less VRAM. However, DA3 supports a higher base resolution than Pi3(X), which can be useful for highly detailed source videos.

### Camera UI (optional)

We include an interactive camera design UI for designing your own target cameras to use as input to `render_single.py`, though we provide an example one made with this exact UI for each example video in `./media/single/`. It visualizes the loaded 4D reconstruction as a navigable 4D point cloud, capture keyframe camera poses and zooms at desired positions, and export the interpolated target cameras as a `.npz` file ready for point cloud rendering. (Our camera UI also supports point cloud editing/4D scene recomposition, which we detail later.)

Our camera design UI is built on top of [Viser](https://viser.studio), which runs as a Python/FastAPI backend paired with a React frontend. Node.js is required for the React frontend, and you can install it with
```bash
conda install conda-forge::nodejs=25  # Slightly older versions probably work, this is just the newest one at the time of writing
```
The Python dependencies (`viser`, `fastapi`, `uvicorn`) are already included in `requirements.txt`.

To launch the UI, run from the repo root
```bash
bash cam_ui/startup.sh
```
This starts the Viser server (port 9997), FastAPI backend (port 9998), and React frontend (port 9999). Open `http://localhost:9999` in your browser once all three are running.

<details>
<summary><b>⚠️ Remote GPU node without inbound port access (e.g., firewalled SLURM)</b></summary>

**Before you do this**, note that we do not recommend this setup, as
1. it routes your UI through a public third-party relay, so check with your sysadmin that this is acceptable for your cluster's network policy, and
2. the UI will feel noticeably slower than a direct SSH port-forward since every click and frame round-trips through the tunnel. If you can get a normal SSH into the GPU node, then that's much preferred. If you only need camera design (no editing), the UI runs fine on a CPU node that you *can* SSH into, since only point cloud editing (for 4D scene recomposition) requires GPU access as it loads SAM3 on first use.

If your GPU node has internet access but you can't directly reach its ports (e.g., a SLURM cluster that firewalls inbound TCP to most ports, or a GPU node you can't SSH into), you can forward 9999 and 9997 through an outbound tunneling service like [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) quick tunnels (free, no account needed). On the GPU node, download cloudflared once with
```bash
curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared
```
and after running `bash cam_ui/startup.sh`, open two separate shells (e.g., tmux panes) on the GPU node and start one tunnel per port
```bash
./cloudflared tunnel --url http://localhost:9999  # React frontend
./cloudflared tunnel --url http://localhost:9997  # Viser backend (for the iframe)
```
Each prints a random `https://<random-words>.trycloudflare.com` URL. Since the Viser iframe is loaded cross-origin by the browser (not proxied through the React app), the UI needs to know the Viser tunnel URL at runtime. Pass it as a `?viser=` query param when opening the React tunnel URL on your laptop
```
https://<react-random-words>.trycloudflare.com/?viser=https://<viser-random-words>.trycloudflare.com
```
The UI reads `?viser=` on page load and points the iframe at that URL; without the query param it falls back to `http://localhost:9997` (the normal local-machine / SSH-forwarded case). The FastAPI backend does not need its own tunnel since it is proxied through the React frontend via `/api/`.
</details>

**Camera UI usage:** Enter the path to a `recon_and_seg` output folder (e.g., `results/single/couple-newspaper/recon_and_seg/`) in the *Folder path* field and click **Load**. Once loaded, navigate the 3D point cloud in the Viser viewport: WASD + Q/E for translation and mouse drag for rotation. (Due to Viser limitations, our UI doesn't support camera roll, only pitch/tilt and yaw/pan for rotations.) You can play and pause the video/point cloud, or you can manually scrub the timeline frames. When you selected a frame you want as a keyframe, click **Capture current view** and also set your zoom there. When satisfied with the target cameras (which you can preview with **Auto-follow camera** during playback), click **Export cameras** to write the interpolated camera path to `cam_ui/exported_cameras/output_cameras.npz` (which is customizable in *Output filename*).

Pass the exported `.npz` as `--cam_path` to `render_single.py` instead of the provided cameras in `./media/single/`.

### Point cloud rendering

To unproject our 4D reconstruction to a point cloud, then render that point cloud in target cameras, run
```bash
EXAMPLE=couple-newspaper RESOLUTION=720p bash scripts/preprocess/example_render_single.sh
```
For resolution, we support `384p` and `720p`, corresponding to our two model checkpoints/variants. We also provide sample target cameras for each of the provided eight (8) source videos in the script under `./media/single/` which the script automatically loads and renders in. The results of point cloud rendering can be found in `./results/single/$EXAMPLE/render_$RESOLUTION/`.

By default, `--render_only_necessary` is enabled, which only renders the point cloud outputs needed for Vista4D inference. To also output point cloud renders without temporal persistence and the double-reprojected (useful for baselines, ablations, etc.), add `RENDER_ONLY_NECESSARY=false` to the start of the script run.

## Vista4D inference

### Wan 2.1 and Vista4D checkpoints

We provide two Vista4D checkpoints ([`Eyeline-Labs/Vista4D`](https://huggingface.co/Eyeline-Labs/Vista4D)) finetuned on [`Wan-AI/Wan2.1-T2V-14B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B):

| Checkpoint | Base model | Training resolution | Training steps | Notes |
|---|---|---|---|---|
| `384p49_step=30000` | [`Wan2.1-T2V-14B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | 672 &times; 384, 49 frames | 30000 | N/A |
| `720p49_step=3000` | [`Wan2.1-T2V-14B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | 1280 &times; 720, 49 frames | 3000 | Finetuned from `384p49_step=30000` |

To do Vista4D inference, first download the Wan 2.1 and Vista4D checkpoints to `./checkpoints/`. The Vista4D checkpoints are hosted on [Eyeline-Labs/Vista4D](https://huggingface.co/Eyeline-Labs/Vista4D). Download both the `384p` and `720p` checkpoints into `./checkpoints/vista4d/` with
```bash
hf download Eyeline-Labs/Vista4D --local-dir ./checkpoints/vista4d
```
If you only need one resolution, pass `--include` to grab just that variant with
```bash
hf download Eyeline-Labs/Vista4D --local-dir ./checkpoints/vista4d --include "384p49_step=30000/*" OR "720p49_step=3000/*"
```
You'll also need the `Wan2.1-T2V-14B` base model. Download it from [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) into `./checkpoints/wan/Wan2.1-T2V-14B/` with
```bash
hf download Wan-AI/Wan2.1-T2V-14B --local-dir ./checkpoints/wan/Wan2.1-T2V-14B
```

### Inference

After running the example point cloud rendering script above, we can run inference on it with
```bash
EXAMPLE=couple-newspaper RESOLUTION=720p bash scripts/inference/example_inference_single.sh
```
where the outputs will be stored in `./results/single/$EXAMPLE/vista4d_$RESOLUTION/`. Just like with the preprocessing steps, the inference script supports all eight (8) of the provided example videos and their target cameras via `$EXAMPLE`.

#### Unified sequence parallel (USP)

To speed up inference and reduce per-GPU VRAM usage, you can do multi-GPU inference with USP by prepending `USE_USP=true` in front of the inference bash script and optionally `NUM_GPUS=?` to specify number of GPU used (default is `NUM_GPUS=8`), so
```bash
USE_USP=true NUM_GPUS=8 EXAMPLE=couple-newspaper RESOLUTION=720p bash scripts/inference/example_inference_single.sh
```

## Evaluation dataset and inference

We provide 110 video-camera pairs to evaluate Vista4D ([`Eyeline-Labs/Vista4D-Eval-Data`](https://huggingface.co/datasets/Eyeline-Labs/Vista4D-Eval-Data)). We select 13 videos from [DAVIS](https://davischallenge.org/) and 38 videos from [Pexels](https://www.pexels.com/). We use [Pi3](https://yyfz.github.io/pi3/) for 4D reconstruction and [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) to do dynamic pixel segmentation. Then, for each video, we hand-design two to three target cameras for each video using our camera UI.

### Downloading the evaluation dataset

From the root directory of the project, run
```bash
huggingface-cli download Eyeline-Labs/Vista4D-Eval-Data --repo-type dataset --local-dir eval_data
```
to download the Vista4D evaluation dataset into `./eval_data/` and then run
```bash
tar -xvf eval_data/eval_data.tar -C eval_data/
```
to extract the contents. It should have the following structure:
```
eval_data/
    metadata.csv
    recon_and_seg/  # 4D reconstruction and dynamic mask segmentation
        avocado-slice/  # There should be 51 total videos
            cameras.npz  # Source intrinsics and extrinsics
            video.mp4
            depths/
                00000.exr
                ...
            dynamic_mask/
                00000.png
                ...
            sky_mask/  # Sky segmentation (to set them to a large depth)
                00000.png
                ...
        [video_name]/
            ...
        ...
    cameras/
        avocado-slice/  # Two to three target cameras per video
            close-crane-above.npz
            left-front-zoom.npz
        [video_name]/
            [camera_name].npz
            ...
        ...
```

`metadata.csv` contains the following information:

- `name`: Name of video-camera pair, in the format `[video]_[camera]`
- `video`: Name of source video, the 4D reconstruction and segmentation can be found in `eval_data/recon_and_seg/[video]/`
- `camera`: Name of camera, corresponds to a `video`, can be found in `eval_data/cameras/[video]/[camera].npz`
- `seed`: Randomly-generated fixed seed for evaluation
- `prompt`: Prompt for the video-camera pair, usually just the prompt of the source video
- `dynamic`: Dynamic keywords used to obtain the segmentation map
- `do_sky_seg`: Whether the video contains sky (and thus we need to segment it separately)
- `source`: Source of the video, `davis` or `pexels`
- `video_id`: For videos from `pexels` only, original ID of the video on Pexels, full link is `https://www.pexels.com/video/[video_id]`

### Point cloud rendering for evaluation data

First, render the point clouds for the evaluation dataset with
```bash
RESOLUTION=720p bash scripts/preprocess/example_render_eval.sh
```
The rendered outputs will be stored in `./eval_data/render_$RESOLUTION/`. As with single-video rendering, `RENDER_ONLY_NECESSARY` defaults to `true`. You can also set `RESOLUTION=720p` for the higher-resolution model variant. The point cloud rendering result should be stored in `./eval_data/` as
```
eval_data/
    ...
    render_$RESOLUTION/
        avocado-slice_close-crane-above/
            alpha_mask_pc/  # pc means point cloud render
            alpha_mask_src/  # src means source video
            depths_pc/
            depths_src/
            dynamic_mask_pc/  # pc means point cloud render
        [video_name]_[camera_name]/
            ...
        ...
    ...
```

### Evaluation inference

After rendering the point clouds, run Vista4D inference on the evaluation dataset with
```bash
RESOLUTION=720p bash scripts/inference/example_inference_eval.sh
```
The outputs will be stored in `./results/eval/vista4d_$RESOLUTION/`. The prompts and seeds for each video-camera pair are specified in `./eval_data/metadata.csv`. (The seeds were randomly generated at the creation of `metadata.csv`.) The script also supports USP for multi-GPU inference (see [above](#unified-sequence-parallel-usp)).

#### Sharding

Since inference over the full evaluation dataset (with 110 video-camera pairs) can be slow on a single GPU, you can split the workload into contiguous shards and run them in parallel across separate scripts. For example, to split into 8 shards and run the third shard, run the inference script with
```bash
NUM_SHARDS=8 SHARD_ID=2 RESOLUTION=720p bash scripts/inference/example_inference_eval.sh
```
Each shard processes a contiguous chunk of the evaluation data, and already-completed video-camera pairs are automatically skipped. Simply run the above script with the corresponding `CUDA_VISIBLE_DEVICES` on each GPU.

## Application: 4D scene recomposition (Point cloud editing)

Vista4D's training on 4D-reconstructed multiview data makes it robust to point cloud artifacts, which lets us directly edit and recompose the 4D point cloud itself: Manipulating, duplicating, deleting, or even inserting subjects from other scenes, while maintaining their dynamics. 4D scene recomposition applies these edits once to the unprojected point cloud, allowing the video diffusion model to synthesize physically plausible results for the edited scene. To prevent conditioning conflicts between the unedited source video and the render of the edited point cloud, the model is conditioned on an edited source video, i.e., the edited point cloud rerendered from the source cameras without static pixel temporal persistence.

We provide eight (8) 4D scene recomposition examples, organized as four source clips (before the underscore) with two edits each: `couple-hug_duplicate-car`/`couple-hug_couple-newspaper`, `funeral-procession_remove-priest`/`funeral-procession_rhino`, `hike_enlarge-backpack`/`hike_cow`, and `swing_shrink-person`/`swing_couple-walk`.

All of the below editing operations are supported by the camera UI. Edits you make in the panel stay as a local draft until you click **Apply edits**, at which point the server re-applies the full edit list and the viewport updates with the new point cloud. Once you're happy with the scene and target cameras, click **Export edits** (in addition to **Export cameras**) to write the edit session to a JSON file that pairs with the exported `.npz`.

Each edit targets a subset of points via a SAM3 text prompt, and runs one or more of `translate` / `rotate` / `scale` / `remove`. Target modes include:
- `existing`: Edit points from the current scene whose pixels match the prompt
- `duplicate`: Clone those points, transform the clone, and add it alongside the originals
- `insert`: Unproject another scene's recon_and_seg folder, filter with the SAM3 mask, and insert the transformed points into the existing 4D point cloud

`rotate` and `scale` requires a transformation with respect to a centroid, which we compute from the segmented point cloud. Edits can run `global`ly (one centroid over the whole subset) or per `frame` (one centroid per origin frame, so e.g. a per-frame rotation tracks a moving subject). Sky points are excluded from every selection automatically.

### Edits JSON schema

An edits JSON pairs with a camera `.npz` and is written by the cam UI when you export an edit session. The schema is:
```json
{"edits": [
    {
        "target": {
            "kind": "existing" | "duplicate" | "insert",
            "prompt": "comma, separated, keywords",
            "source": "<path/to/recon_and_seg>"
        },
        "ops": [
            {"op": "translate", "params": [x, y, z]},
            {"op": "rotate",    "params": [rx_deg, ry_deg, rz_deg]},
            {"op": "scale",     "params": s  /* or [sx, sy, sz] for per-axis scale */},
            {"op": "remove"}  // no params
        ],
        "scope": "global" | "frame",
        "mask_expansion": [radius, iterations],
        "centroid_threshold": 0.6
    }
]}
```
`target.source` is required for `insert` and points to another recon_and_seg folder; `mask_expansion` and `centroid_threshold` are optional. All `global`-scope edits run before any `frame`-scope edits, and edits within each group run in JSON order. Legacy JSONs written with `"kind": "sam3"` are still accepted for backward compatibility (silently normalized to `"existing"` at load time).

### 4D reconstruction for edits

To reconstruct and segment a source clip together with its paired insert scene(s), do
```bash
EXAMPLE=hike RECON_METHOD=pi3 bash scripts/preprocess/example_recon_and_seg_edit.sh
```
Each `EXAMPLE` (one of `couple-hug`, `funeral-procession`, `hike`, `swing`) reconstructs both the main scene and its insert scene(s) into separate `./results/edit/$SCENE_NAME/recon_and_seg/` folders, with per-scene SAM3 keywords preconfigured. `RECON_METHOD` picks between Pi3X (`pi3`) and DA3 (`da3`) the same as for single-video reconstruction.

### Point cloud rendering with edits

```bash
EXAMPLE=hike_cow RESOLUTION=720p bash scripts/preprocess/example_render_edit.sh
```
This reads the scene's recon_and_seg (from `./results/edit/$EXAMPLE_NAME/recon_and_seg/` where `$EXAMPLE_NAME` is the part before the underscore, e.g. `hike` for `hike_cow`), applies the edits JSON under `./media/edit/$EXAMPLE.json` to the unprojected point cloud, and renders in the target cameras to `./results/edit/$EXAMPLE/render_$RESOLUTION/`. `RESOLUTION` and `RENDER_ONLY_NECESSARY` behave the same way as with single-video rendering.

The edit-render output folder at `./results/edit/$EXAMPLE/render_$RESOLUTION/` contains the usual target-camera render (`video_pc.mp4` and its `depths_pc/`, `alpha_mask_pc/`, `dynamic_mask_pc/`, `static_mask_pc/` dirs) plus a source-camera rerender of the edited point cloud (`video_src.mp4`, `depths_src/`, `alpha_mask_src/`, `dynamic_mask_src/`, `static_mask_src/`, `sky_mask_src/`). The source-camera re-render is what Vista4D inference conditions on in place of the raw input. The raw input source video is preserved alongside as `video_srcraw.mp4` with matching `*_srcraw/` mask and depth dirs, and each unique insert source is written as `video_ins.mp4` (single-insert runs) or `video_ins0.mp4`, `video_ins1.mp4`, ... (multi-insert runs).

### Vista4D inference with edits

```bash
EXAMPLE=hike_cow RESOLUTION=720p bash scripts/inference/example_inference_edit.sh
```
Outputs are saved in `./results/edit/$EXAMPLE/vista4d_$RESOLUTION/`. The script has per-example prompts and seeds preconfigured for each of the eight 4D scene recomposition examples, and USP multi-GPU inference (with `USE_USP=true` and setting `NUM_GPUS`) is supported the same way as single-video inference.

## Application: Dynamic Scene Expansion (DSE)

Video reshooting often requires Vista4D to hallucinate pixels not observed in the source video, even when we already have additional visual information of the scene (e.g., a casual capture of the environment or an alternate camera angle). DSE incorporates this extra information by jointly reconstructing the source video and the additional scene frames into a single temporally-persistent 4D point cloud, which reduces video model hallucinations and gives stronger control beyond the source video. The extra scene capture only contributes context to the point cloud, and the source video passed to the video diffusion model is still just the original source video.

We provide eight (8) DSE examples, organized as five scenes (before the hyphen) with one to two source videos each: `conference-punch`/`conference-study`, `hall-cartwheel`, `lounge-cup`/`lounge-drink`, `plaza-point`, and `room-lift`/`room-walk`, all filmed by the authors.

### 4D reconstruction with DSE

To jointly reconstruct and segment the source clip and its paired scene capture, do
```bash
EXAMPLE=lounge-cup RECON_METHOD=pi3 bash scripts/preprocess/example_recon_and_seg_dse.sh
```
Both videos are reconstructed in a single coordinate frame and segmented across both, with results saved to `./results/dse/$EXAMPLE/recon_and_seg/`. A `clips.json` inside this folder records the source and DSE frame ranges, which downstream scripts automatically detect.

### Camera UI with DSE (optional)

The cam UI loads DSE reconstructions the same way (enter `./results/dse/$EXAMPLE/recon_and_seg/` in *Folder path* and click **Load**). The editable timeline only covers the source frames. Two DSE-specific controls appear once a DSE reconstruction is loaded:
- **Show DSE cameras** overlays the scene-capture camera frustums and their path.
- **DSE frame interval** strides through DSE frames when rebuilding the static background overlay (higher = sparser). The equivalent config for point cloud rendering is `DSE_FRAME_INTERVAL`.

### Point cloud rendering with DSE

```bash
EXAMPLE=lounge-cup RESOLUTION=720p bash scripts/preprocess/example_render_dse.sh
```
The source clip and DSE frames are unprojected together into a single temporally-persistent 4D point cloud, which is then rendered in the target cameras to `./results/dse/$EXAMPLE/render_$RESOLUTION/`. We also provide sample target cameras for each of the provided ten examples in the script under `./media/dse/` which the script automatically loads and renders in. Change `DSE_FRAME_INTERVAL` (default `4`) to subsample DSE frames during unprojection (higher means fewer DSE frames), and `RESOLUTION` and `RENDER_ONLY_NECESSARY` can be configured the same way with single-video rendering.

### Vista4D inference with DSE

```bash
EXAMPLE=lounge-cup RESOLUTION=720p bash scripts/inference/example_inference_dse.sh
```
Outputs are saved in `./results/dse/$EXAMPLE/vista4d_$RESOLUTION/`. USP multi-GPU inference (with `USE_USP=true` and setting `NUM_GPUS`) is supported the same way as single-video inference. Note that `example_inference_dse.sh` and `example_inference_single.sh` are essentially the same script, just that the DSE script has DSE examples preconfigured.

## Coming soon (a.k.a. TODOs)

- [ ] Release I2V-finetuned checkpoint(s)
- [ ] Release code and sample for long video inference with memory (application), requires I2V checkpoint
- [ ] TBD: Release data preprocessing and training code
- [ ] TBD: Release training data

# &#128591; Acknowledgements

We would like to thank Aleksander Hołyński, Wenqi Xian, Dan Zheng, Mohsen Mousavi, Li Ma, and Lingxiao Li for their technical discussions; Ryan Tabrizi, Tianyi Lorena Yan, and Shreyas Havaldar for appearing in our demo videos; Lukas Lepicovsky, David Rhodes, Nhat Phong Tran, Dacklin Young, and Johnson Thomasson for their production support; Jeffrey Shapiro, Ritwik Kumar, and Hossein Taghavi for their executive support; Jennifer Lao and Lianette Alnaber for their operational support.

Moreover, our work is inspired by many awesome prior works such as [TrajectoryCrafter](https://trajectorycrafter.github.io/), [ReCamMaster](https://jianhongbai.github.io/ReCamMaster/), [GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/), and [EX-4D](https://tau-yihouxiang.github.io/projects/EX-4D/EX-4D.html), and we built Vista4D on top of amazing projects such as [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Wan 2.1](https://github.com/Wan-Video/Wan2.1), [MultiCamVideo](https://jianhongbai.github.io/ReCamMaster/), [OpenVid](https://github.com/NJU-PCALab/OpenVid-1M), [Pi3(X)](https://github.com/yyfz/Pi3), [STream3R](https://github.com/NIRVANALAN/STream3R), [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3), [Segment Anything 3](https://github.com/facebookresearch/sam3), [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2), [Llama 3](https://github.com/meta-llama/llama3), [Recognize Anything](https://github.com/xinyu1205/recognize-anything), and [Viser](https://viser.studio).

# &#128269; Reference

If you use our paper in your research, please cite the following work.

```bibtex
@inproceedings{lin2026vista4d,
    author = {Lin, {Kuan Heng} and Liu, Zhizheng and Salamanca, Pablo and Kant, Yash and Burgert, Ryan and Xu, Yuancheng and Namekata, Koichi and Zhao, Yiwei and Zhou, Bolei and Goldblum, Micah and Debevec, Paul and Yu, Ning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    title = {{Vista4D}: Video Reshooting with 4D Point Clouds},
    year = {2026}
}
```

For any questions, thoughts, discussions, and any other things you want to reach out for, please contact [Jordan Lin](https://kuanhenglin.github.io) (`jordan at cs dot columbia dot edu`).
