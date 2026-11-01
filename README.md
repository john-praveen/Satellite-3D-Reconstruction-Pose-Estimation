# Satellite 3D Reconstruction + Pose Estimation

Given photo(s) of an object, this pipeline does two things:
1. **3D reconstruction** — builds a 3D mesh of the object using Hunyuan3D-2 (a generative
   single/multi-image-to-3D model), from either **one photo** or **three photos**
   (front/back/left, which generally gives a more accurate mesh).
2. **Pose estimation** — given that mesh and a photo, estimates the object's **rotation**
   in the photo, using the **ACTUAL official [FoundPose](https://github.com/facebookresearch/foundpose)**
   (ECCV 2024) repo — rendering, DINOv2 matching via its "cyclic buddies" correspondence
   filter, PnP-RANSAC — wired up automatically against your reconstructed object. There is
   no separate from-scratch reimplementation of the algorithm here: this repo's job is
   just to build a mesh, package it as a minimal fake BOP dataset the official repo can
   consume (`satpose/foundpose_official/build_bop_dataset.py`), shell out to its actual
   `gen_templates.py` / `gen_repre.py` / `infer.py`, and render a mask+axes visual check
   on the resulting poses (`satpose/pipeline.py`, `satpose/visualize.py`).
   - `run_single_view_pipeline.py` — one photo in, pose estimated on that photo (or
     others) against the resulting mesh.
   - `run_multiview_pipeline.py` — three (or four, with an optional right view) photos
     in, for a more complete reconstruction; pose estimated the same way against the
     resulting mesh.
   - `run_pipeline.py` — a single interactive entry point wrapping both of the above:
     asks "single or multi view?" in the terminal (or skips the prompt if pre-set),
     then everything else (photo paths, camera intrinsics, where query photos come
     from, whether to render visualizations) comes from `satpose/config.py` instead of CLI
     flags. **Recommended if you're handing this project to someone else** — see
     section 2 below.

   All three need a second conda environment (the official FoundPose repo's own
   dependency stack). see "Official pipeline setup" below.

**Important limitation up front:** the reconstructed mesh is not metrically scaled to the
real object's true size. That means the estimated **rotation is the trustworthy output**;
the estimated **translation/distance is not** (there's no way to recover true scale from
a photo without a calibrated stereo setup or known object dimensions). Every script in
this repo reflects that — translation is reported but flagged as unreliable.

## Project layout

```
.
├── run_pipeline.py               # interactive entry point (settings from satpose/config.py)
├── run_single_view_pipeline.py   # CLI entry point: one photo -> mesh -> pose
├── run_multiview_pipeline.py     # CLI entry point: 3-4 photos -> mesh -> pose
├── simulate_closed_loop.py       # runs the closed loop with fake stages (no GPU needed)
├── verify_orientation.py         # checks the pose stage on a synthetic satellite with known poses
├── requirements.txt
├── data/
│   └── starsat/                  # example photos
├── tests/                        # unit tests for the closed-loop modules (no GPU needed)
└── satpose/                      # library code
    ├── config.py                 # all user-tunable settings (paths, camera intrinsics, run options)
    ├── pipeline.py               # query resolution, FoundPose subprocess calls, CSV/overlay output
    ├── visualize.py              # mask + axes overlay rendering
    ├── viewpoints.py             # Contribution 1: viewing directions, keyframes, Hunyuan view slots
    ├── provenance.py             # Contribution 2: observed vs. hallucinated mesh, weighted pose refinement
    ├── runtime.py                # Contribution 3: fast pose loop + background rebuild, atomic model swap
    ├── metrics.py                # rotation / pointing / translation error
    ├── synthetic.py              # box satellite, STL writer, numpy renderer, wireframe overlay
    ├── verification.py           # generate / run / evaluate the synthetic orientation check
    ├── reconstruction/
    │   ├── single_view.py        # Hunyuan3D-2, one image -> mesh
    │   └── multiview.py          # Hunyuan3D-2, front/back/left(/right) -> mesh
    └── foundpose_official/
        ├── build_bop_dataset.py  # packages mesh + photos as a minimal BOP dataset
        └── configs/              # gen_templates / gen_repre / infer settings for FoundPose
```

Run all scripts from the repository root (`output/` and the `data/...` defaults are
relative to it).

## Closed-loop modules (research contributions)

On top of the baseline pipeline, three self-contained modules implement the research
contributions. They need only `numpy` and `scipy`, take the reconstruction and pose back-ends
as plain callables, and are covered by tests that run without a GPU.

1. **Pose-guided closed loop** (`satpose/viewpoints.py`) — each estimated pose gives the
   direction the camera sees the target from. `KeyframeSelector` keeps frames that show
   a new side, and `assign_slots` maps them to Hunyuan3D-2mv's front / left / back / right inputs.
2. **Observation-aware weighting** (`satpose/provenance.py`) — `ObservationMap` records which
   mesh vertices a camera has really seen, gives guessed geometry a low trust weight, and
   `refine_pose_weighted` uses those weights when polishing a pose from 2D-3D matches.
3. **Dual-rate runtime** (`satpose/runtime.py`) — `ClosedLoopPipeline` estimates pose on every
   frame against the current model while a background thread rebuilds the model from the
   keyframes; the new model is swapped in atomically and a failed rebuild never stops tracking.

```
python -m unittest discover -s tests -v     # run the tests
python simulate_closed_loop.py              # watch the loop run on a simulated fly-around
```

**Status.** The logic and the scheduling are tested on synthetic data. Not done yet:
running the loop with the real Hunyuan3D-2mv and FoundPose (needs a GPU machine), feeding
the trust weights into the official FoundPose `infer.py` (it does not export its
2D-3D correspondences), and the ground-truth evaluation on SPE3R. Two assumptions to verify
on real meshes: that the mesh's up axis is +Y (`viewpoints.DEFAULT_UP_AXIS`), and that a
rebuilt mesh keeps the same object frame as the first one (the first keyframe is always the
"front" view, which is what makes this likely, but it is not guaranteed).

## Verifying the pose stage on a synthetic satellite

`verify_orientation.py` checks the pose pipeline against ground truth. It builds a box-shaped
satellite (an asymmetric one, so only one orientation matches each view), saves it as
`satellite.stl`, renders 12 views at known poses, and draws wireframes of the true pose.
Hunyuan3D is not involved: the STL itself is the mesh, so only FoundPose is tested.

```
python verify_orientation.py generate    # STL + renders + ground-truth wireframes (no GPU)
python verify_orientation.py selftest    # report using FAKE estimates, to see what pass/fail looks like
python verify_orientation.py run         # generate, run the real FoundPose on it, compare, draw overlays
python verify_orientation.py evaluate    # re-compare an existing FoundPose result
```

Output goes to `output/synthetic_satellite/`:
- `gt_contact_sheet.png` — the 12 views with the true wireframe (green).
- `wireframe_eval/contact_sheet.png` — after `run`: true wireframe (green) and the wireframe at
  the **estimated rotation** (red), with the rotation error per view. Red covering green means
  the orientation is right. Only rotation is compared; the red wireframe uses the true translation.

`run` needs the FoundPose setup from "Official pipeline setup" below. The renders are
plain grey-shaded, like FoundPose's own templates, so this is a best case: if it fails here, the
setup is wrong; if it passes, real photos are the next test.

## 1. Setup

### Prerequisites
- Python 3.10, a CUDA-capable GPU strongly recommended (both the reconstruction and the
  pose estimation lean on it heavily; CPU will work but be very slow).
- conda (or another virtualenv tool).

### Install steps

**a) Clone Hunyuan3D-2** (this repo does the actual 3D reconstruction; we don't bundle it):
```
git clone https://github.com/Tencent/Hunyuan3D-2
```

**b) Create an environment and install PyTorch with CUDA:**
```
conda create -n satpose python=3.10 -y
conda activate satpose
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
(Use whichever `cuXXX` index matches your GPU driver's CUDA version — check with `nvidia-smi`.)

**c) Install Hunyuan3D-2's own dependencies:**
```
pip install -r Hunyuan3D-2/requirements.txt
```

**d) Install this repo's additional dependencies:**
```
pip install -r requirements.txt
```

**e) Point this repo at your Hunyuan3D-2 clone.** Either set an environment variable:
```
set HY3DGEN_PATH=D:\path\to\Hunyuan3D-2
```
or edit the default at the top of `satpose/config.py`.

First run of any reconstruction script downloads the Hunyuan3D-2 model weights from
Hugging Face automatically (~10-25GB depending on which model, one-time).

### Official pipeline setup (needed for both `run_single_view_pipeline.py` and `run_multiview_pipeline.py`)

The official FoundPose repo's dependencies (a pinned older PyTorch, `faiss`, `kornia`,
specific `pyrender`/`OpenGL` versions) don't mix cleanly with Hunyuan3D-2's, so it needs
its **own, separate conda environment**:

```
git clone --recurse-submodules https://github.com/facebookresearch/foundpose
conda create -n foundpose_repo python=3.9 -y
conda activate foundpose_repo
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
pip install faiss-cpu omegaconf torchmetrics fvcore iopath torchinfo scipy matplotlib pillow scikit-learn scikit-image imageio kornia pypng triangle pyopengl pyglet pyrender opencv-python webdataset pytz cython numpy==1.26.4
```
(`faiss-cpu` substitutes for the official `faiss-gpu`, which has no Windows wheel — fine
at this scale, it's just k-means/kNN over a few thousand vectors. `xformers` is skipped
too — it's an optional DINOv2 speedup, not a hard requirement, and the pinned version
doesn't line up with a newer CUDA toolchain.)

Then set two things so the pipelines can find it — either environment variables or the
defaults in `satpose/config.py`:
```
set FOUNDPOSE_OFFICIAL_REPO_PATH=D:\path\to\foundpose
set FOUNDPOSE_OFFICIAL_CONDA_ENV=foundpose_repo
```

`run_single_view_pipeline.py` / `run_multiview_pipeline.py` themselves still run in your
main `satpose` environment (from step b/c/d above) — they shell out to `foundpose_repo`
only for the 3 official-algorithm steps (template rendering, representation building,
inference), then come back to the main environment for reconstruction and the mask+axes
visualization. You don't run them from inside `foundpose_repo` yourself.

## 2. The interactive pipeline (`run_pipeline.py`) — recommended for handoff

If you don't want to remember CLI flags (`--image`, `--front`, `--query`, etc.), use
`run_pipeline.py` instead of `run_single_view_pipeline.py`/`run_multiview_pipeline.py`
directly. It's a thin wrapper around the same underlying code — everything is configured
by editing variables in `satpose/config.py` instead:
```
python run_pipeline.py
```
It asks one question in the terminal (unless pre-answered, see below):
```
Single view or multi view? [s/m]:
```
then reconstructs (printing a progress spinner while Hunyuan3D-2 works, which also
prints its own step-by-step progress underneath), estimates pose, and reports where
everything was saved.

**Everything else lives in `satpose/config.py`, under "run_pipeline.py":**
- `PIPELINE_MODE` — set to `"single"` or `"multi"` to skip the terminal prompt entirely
  (e.g. for a scripted/scheduled run); leave `None` to be asked each time.
- `SINGLE_VIEW_IMAGE` — the one reconstruction photo, for single-view mode.
- `FRONT_IMAGE` / `BACK_IMAGE` / `LEFT_IMAGE` / `RIGHT_IMAGE` — reconstruction photos for
  multi-view mode. `RIGHT_IMAGE` is optional (`None` to reconstruct from just 3 views).
- `QUERY_SOURCE` — what to estimate pose on. Leave `None` to default to the
  reconstruction photo(s) themselves, or set it to:
  - a single photo path, or a list of photo paths
  - a **folder** path — every image file inside it is used (non-recursive)
  - a **video file** path (`.mp4`/`.avi`/`.mov`/`.mkv`/…) — frames are extracted first
    (see `VIDEO_FRAME_STRIDE`: `1` = every frame, `5` = every 5th frame, etc.), one query
    photo per extracted frame
- `SAVE_MASK` / `SAVE_AXES` — turn off either (or both) to skip rendering those
  visualization images — useful for large batches (a folder or video with many frames)
  where you only want the numbers quickly. Estimated poses are **always** saved
  regardless, as both JSON and CSV.
- `CAMERA_K` / `IMAGE_WIDTH` / `IMAGE_HEIGHT` — same camera intrinsics setting used
  everywhere else (see "Adjusting parameters" below) — every query photo (including
  every extracted video frame) must match this resolution, checked automatically before
  any work starts.

**Output**, under `output/<name>/` same as the CLI scripts:
- `estimated_poses.csv` — one row per query photo: filename, rotation (both a flattened
  3x3 matrix and a quaternion), translation, inlier count, match score.
- `bop_datasets/outputs/inference/custom_v1/1/estimated-poses.json` — the raw official
  FoundPose output the CSV is derived from.
- `*_mask.png` / `*_mask_axes.png` (or just `*_axes.png` if `SAVE_MASK` is off) — per
  "Reading the output" below, unless disabled via `SAVE_MASK`/`SAVE_AXES`.

## 3. Quick start — try the included example

```
python run_single_view_pipeline.py
```
Reconstructs a mesh from `data/starsat/starsat.jpeg` and estimates pose on **all 5**
included starsat photos (that's the default query set for this script — pass `--query`
to override), saving results under `output/starsat/`. Open any `*_mask_axes.png` there —
see "Reading the output" below for how to judge it.

## 4. Running on your own data

### Single-view (one photo)
```
python run_single_view_pipeline.py --image path/to/recon_photo.jpg --query path/to/pose_photo.jpg
```
`--query` accepts multiple photos (`--query a.jpg b.jpg c.jpg`) — all get pose-estimated
against the one reconstructed mesh in a single run, which is much cheaper than
reconstructing/re-templating per photo. `--skip-recon` reuses an existing mesh;
`--skip-templates` reuses an existing template bank + representation (both cached under
`output/<name>/`).

### Multi-view (three photos: front/back/left)
Generally more accurate than single-view, since the reconstruction gets real geometric
constraints from 3 real vantage points instead of guessing occluded geometry from one.
```
python run_multiview_pipeline.py --front front.jpg --back back.jpg --left left.jpg
```
By default, pose is estimated on the same `--front`/`--back`/`--left` triplet used for
reconstruction; pass `--query` to estimate pose on different or additional photos
instead:
```
python run_multiview_pipeline.py --front front.jpg --back back.jpg --left left.jpg \
    --query front.jpg extra_photo.jpg
```
Results save under `output/<front image name>_multiview/`. No multiview example data is
included — the 5 starsat photos are independent single views, not a matched
front/back/left triplet.

### Re-running without redoing expensive steps
Each pipeline caches its outputs under `output/<name>/`. Skip steps you don't need to redo:
```
python run_single_view_pipeline.py --skip-recon --skip-templates --query new_photo.jpg
```
`--skip-recon` reuses the existing mesh; `--skip-templates` reuses the existing BOP
dataset, rendered template bank, and object representation — useful if you just want to
test a *different photo* of the *same already-reconstructed* object.

## 5. Adjusting parameters

- **`satpose/config.py`** — `CAMERA_K` / `IMAGE_WIDTH` / `IMAGE_HEIGHT`, the camera intrinsics for
  your photos. Defaults to a generic guess (`fx=fy=image width`, principal point
  centered) if you don't have real calibration. **If you have real calibration (phone
  EXIF, known lens spec), use it** — see "Known limitations" below for why this matters
  more than it might seem. Also `FOUNDPOSE_OFFICIAL_REPO_PATH` / `FOUNDPOSE_OFFICIAL_CONDA_ENV`
  (see "Official pipeline setup" above).
- **`satpose/foundpose_official/configs/{gen_templates,gen_repre,infer}/custom.json`** — every
  other tunable (template-bank size/coverage, DINOv2 backbone/layer, matching, PnP-RANSAC
  settings) is the official repo's own config, passed straight through via `--opts-path`.
  Notable ones:
  - `gen_templates/custom.json`: `min_num_viewpoints` / `num_inplane_rotations` control
    template-bank size (SO(3) coverage) — more = better coverage but linearly slower to
    render. `ssaa_factor` is turned down from the official default of `4.0` to `1.0` here
    — an ~12x speedup with no visible accuracy loss in testing.
  - `infer/custom.json`: `extractor_name` selects the DINOv2 backbone/layer (paper default
    is `vitl14_reg` layer 18; the included config uses the smaller/faster `vits14_reg`
    layer 9). `match_top_k_buddies` is the "cyclic buddies" correspondence filter size —
    keep it meaningfully smaller than your typical foreground patch count for it to
    actually filter anything.

## 6. Reading the output

Each run saves `<name>_mask_axes.png`: the reconstructed mesh, re-rendered at the
estimated pose and overlaid on your photo (cyan fill + contour = the mesh's projected
silhouette), plus a red/green/blue X/Y/Z axis triad at the object's center.

**This is how you judge whether a result is trustworthy — always check it, don't just
trust the numbers in the JSON output:**
- If the cyan silhouette **tightly hugs** the real object's outline in the photo → the
  rotation estimate is good.
- If the cyan silhouette is a **different shape, offset, or rotated relative to** the
  real object → the pose estimate is wrong for that photo. Don't trust it.

## 7. Known limitations (from testing on real photos)

- **Always check the mask overlay yourself, critically — never trust the numbers alone.**
  In testing (5 photos of one object), the official FoundPose algorithm (this repo's only
  pose engine, lower template resolution for speed) gave visibly tight, correct results on
  **all 5** test photos. That's still a single small test, not a validated benchmark —
  "always check the mask, don't assume success" still applies. Symmetric, flat objects are
  especially prone to a *wrong* rotation still producing a plausible-looking silhouette by
  coincidence (a flat panel looks roughly like a flat panel from many angles) — check
  whether edges and internal detail actually line up, not just the rough outline shape.
- **Texture-less, symmetric objects are much harder.** An object with real surface detail
  (panel seams, visible components, asymmetric shape) matches far better than a flat,
  featureless, symmetric one — DINOv2 needs *something* locally distinctive to match on.
- **Camera intrinsics matter more than you'd expect.** Unlike the mesh's absolute scale
  (which PnP automatically compensates for via translation, at zero cost to the visual
  result), a wrong focal length is *not* fully compensated — it can bias the rotation
  estimate itself, especially at oblique viewing angles. If results look worse on
  angled photos than face-on ones, this is a likely contributing factor.
- **Only rotation is trustworthy**, as stated up top — translation/depth reflects the
  mesh's arbitrary reconstructed scale, not the real object's physical distance.
- **No ground truth was used to numerically score this** — all accuracy assessments were
  done by eye via the mask overlay, not a formal benchmark. Treat it as a working
  research prototype, not a validated production system.

## 8. Troubleshooting

- **On Linux (headless server, no display):** set `PYOPENGL_PLATFORM=egl` in your shell
  *before* running any script that renders (anything that imports `pyrender`/`satpose/visualize.py`).
- **First run is slow / seems stuck:** it's downloading model weights (several GB) —
  check your network, it only happens once.
- **Out of GPU memory:** reduce `min_num_viewpoints` / `num_inplane_rotations` in
  `satpose/foundpose_official/configs/gen_templates/custom.json`, or switch `extractor_name` in
  `infer/custom.json` to a smaller DINOv2 backbone.
- **Pipeline fails with a path/"Unknown BOP dataset" error:** check
  `FOUNDPOSE_OFFICIAL_REPO_PATH` actually points at the folder *containing* `scripts/`
  (i.e. where you ran `git clone`), and that `conda activate foundpose_repo && python -c
  "import torch, faiss, pyrender"` succeeds with no errors in that environment.
- **Pipeline fails with `FileNotFoundError` on `dataset_meta.json`:** usually a
  relative-path issue if you passed a relative `--out-dir` — use an absolute path, or
  leave `--out-dir` unset to use the (already-absolute) default.
