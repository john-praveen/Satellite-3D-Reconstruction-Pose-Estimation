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
the estimated **translation/distance is only as good as the mesh scale** (there's no way to
recover true scale from a photo without a calibrated stereo setup or known object dimensions).
With a mesh of known scale, as for the synthetic test objects, it is accurate to under 1% of
the distance (see "Rotation and translation accuracy"); with a reconstructed mesh the scale is
arbitrary, so treat translation as unreliable and rely on rotation.

## Project layout

```
.
├── run_pipeline.py               # interactive entry point (settings from satpose/config.py)
├── run_single_view_pipeline.py   # CLI entry point: one photo -> mesh -> pose
├── run_multiview_pipeline.py     # CLI entry point: 3-4 photos -> mesh -> pose
├── scripts/                      # evaluation and demo tools (run from the repository root)
│   ├── verify_orientation.py     # quick check of the pose stage on a synthetic satellite (12 views)
│   ├── evaluate_synthetic.py     # 80 random views: generate / infer / overlays / accuracy
│   ├── make_tracking_video.py    # turntable video with the estimated wireframe + per-frame timing / fps
│   ├── run_on_photos.py          # pose on photos of a known mesh (no ground truth), outline overlays
│   └── simulate_closed_loop.py   # the closed loop with fake stages (no GPU needed)
├── requirements.txt
├── data/
│   └── starsat/                  # example photos
├── output/                       # results (git-ignored): synthetic_eval, objects/, turntable_synthetic, turntable_hunyuan
├── tests/                        # unit tests (no GPU needed)
└── satpose/                      # library code
    ├── config.py                 # all user-tunable settings (paths, camera intrinsics, run options)
    ├── pipeline.py               # query resolution, FoundPose subprocess calls, CSV/overlay output
    ├── visualize.py              # mask + axes overlay rendering
    ├── viewpoints.py             # Contribution 1: viewing directions, keyframes, Hunyuan view slots
    ├── provenance.py             # Contribution 2: observed vs. hallucinated mesh, weighted pose refinement
    ├── runtime.py                # Contribution 3: fast pose loop + background rebuild, atomic model swap
    ├── refinement.py             # silhouette matching: refine every FoundPose candidate, pick the best
    ├── postprocess.py            # applies the refinement to FoundPose's results (modes none / full)
    ├── metrics.py                # rotation / pointing / translation error, SPEED score
    ├── evaluation.py             # scores the pipeline on synthetic views with known poses
    ├── synthetic.py              # box satellite, STL writer, numpy renderer, wireframe overlay
    ├── objects.py                # other test objects (textured satellite, rocket body, debris) and mesh-file loader
    ├── verification.py           # generate / run / evaluate the synthetic orientation check
    ├── reconstruction/
    │   ├── single_view.py        # Hunyuan3D-2, one image -> mesh
    │   └── multiview.py          # Hunyuan3D-2, front/back/left(/right) -> mesh
    └── foundpose_official/
        ├── build_bop_dataset.py  # packages mesh + photos as a minimal BOP dataset
        ├── patch_bop_toolkit.py  # registers the custom dataset and candidate export in the official repo
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
python scripts/simulate_closed_loop.py      # watch the loop run on a simulated fly-around
```

**Status.** The logic and the scheduling are tested on synthetic data. Not done yet:
running the loop with the real Hunyuan3D-2mv and FoundPose (needs a GPU machine), feeding
the trust weights into the official FoundPose `infer.py` (it does not export its
2D-3D correspondences), and the ground-truth evaluation on SPE3R. Two assumptions to verify
on real meshes: that the mesh's up axis is +Y (`viewpoints.DEFAULT_UP_AXIS`), and that a
rebuilt mesh keeps the same object frame as the first one (the first keyframe is always the
"front" view, which is what makes this likely, but it is not guaranteed).

## Verifying the pose stage on a synthetic satellite

`scripts/verify_orientation.py` checks the pose pipeline against ground truth. It builds a box-shaped
satellite (an asymmetric one, so only one orientation matches each view), saves it as
`satellite.stl`, renders 12 views at known poses, and draws wireframes of the true pose.
Hunyuan3D is not involved: the STL itself is the mesh, so only FoundPose is tested.

```
python scripts/verify_orientation.py generate    # STL + renders + ground-truth wireframes (no GPU)
python scripts/verify_orientation.py selftest    # report using FAKE estimates, to see what pass/fail looks like
python scripts/verify_orientation.py run         # generate, run the real FoundPose on it, compare, draw overlays
python scripts/verify_orientation.py evaluate    # re-compare an existing FoundPose result
```

Output goes to `output/synthetic_satellite/`:
- `gt_contact_sheet.png` — the 12 views with the true wireframe (green).
- `wireframe_eval/contact_sheet.png` — after `run`: true wireframe (green) and the wireframe at
  the **estimated rotation** (red), with the rotation error per view. Red covering green means
  the orientation is right. Only rotation is compared; the red wireframe is drawn at the true
  position, so any mismatch is rotation error alone.

`run` needs the FoundPose setup from "Official pipeline setup" below, and must be started with
that environment's Python (it imports `pyrender` and `trimesh`, which live there):

```
set CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe
D:\conda_envs\foundpose_repo\python.exe scripts/verify_orientation.py run
```

The first run patches the official repo's `bop_toolkit` automatically (a `custom` dataset and
an output folder next to the dataset; see `satpose/foundpose_official/patch_bop_toolkit.py`).
Expect about 6 minutes on an RTX 3050 (798 templates, small DINOv2 model).

The renders are plain grey-shaded, like FoundPose's own templates, so this is a best case: if it
fails here, the setup is wrong; if it passes, real photos are the next test.

### Silhouette refinement (improves FoundPose's poses)

FoundPose matches the photo to rendered templates. As shipped, this repo kept only its single best
coarse pose, which is accurate to roughly the template spacing and sometimes a flipped copy of the
right answer. `satpose/refinement.py` and `satpose/postprocess.py` fix both:

1. The patched `infer.py` saves **all** coarse candidates (20 per image, `INFER_CONFIG`).
2. Each candidate is **refined**: rotation and translation are nudged to maximise the overlap
   between the mesh's outline and the photo's object mask (bounded to 20 deg / 15% so it cannot
   drift to another pose).
3. The candidate with the best **score** is kept (`POSE_REFINEMENT = "full"`), where
   score = outline overlap + 0.2 x (FoundPose's matched points / the most any candidate has).
   The outline alone can mistake a pose for its flipped twin, or be misled by an inexact mesh;
   FoundPose's own appearance matching is independent of both.

Variants tried and **not kept**: re-ranking the candidates without refining them (worse than doing
nothing, since unrefined candidates are too rough to compare), refining only FoundPose's own best pose
(fixes small errors, not flips), a pull towards FoundPose's own pose (no help), and a confidence gate that
falls back to FoundPose when the fit is poor (the threshold was too fragile). The code for these was removed.

**Measured results.** All on synthetic views with an exact mesh and exact masks, on an RTX 3050 laptop
(4 GB). "Gross failure" means a rotation error above 45 deg.

*Kept in `output/synthetic_eval/` and `output/objects/`*: 80 random views of the box satellite (seed 99) and
40 random views each of three more objects (seed 7). The textured satellite uses the same geometry with texture and
lighting in the photos while FoundPose still matches against the plain grey mesh, as it would with a generated
mesh. Reproduce with `scripts/evaluate_synthetic.py generate [--object NAME] | infer | overlays | accuracy`
(FoundPose environment's Python).

| object | FoundPose as shipped: rotation median / mean, within 10 deg | our pipeline: rotation median / mean, within 10 deg | our translation median | our SPEED score (median) |
|---|---|---|---|---|
| box satellite (80 views) | 6.2 / 35.0 deg, 69% | **0.52 / 1.4 deg, 99%** | 0.27% | 0.012 |
| textured satellite (40) | 10.8 / 46.3 deg, 48% | **0.58 / 10.1 deg, 95%** | 0.27% | 0.013 |
| irregular debris (40) | 19.7 / 52.8 deg, 30% | **0.50 / 1.4 deg, 98%** | 0.39% | 0.014 |
| rocket body, nearly symmetric (40) | 160.0 / 106.6 deg, 2% | **7.8 / 35.1 deg, 55%** | 0.65% | 0.139 |
| all four pooled (200 views) | 14.0 / 55.1 deg, 44% | **0.72 / 9.9 deg, 89%** | | |

Translation error is the distance between the estimated and true mesh-centroid position, as a % of the
camera distance (about 4 mesh units). The SPEED score is the rotation error in radians plus that relative
translation error (the ESA SPEED/SPEC metric, lower is better). Per-view numbers are in `accuracy.csv`, and the
wireframe overlays (green = truth, red = estimate) are in `wireframe_eval/` (our pipeline) and
`wireframe_baseline/` (as shipped), each with a contact sheet and a sheet of the 8 worst views. The sub-1%
translation holds only for a mesh of known scale; these are synthetic images, not real photos.

*Recorded earlier; the experiment code was not kept, so these cannot be re-run from this repo.* Ablation (160 random views from two independent sets, before the bias fix below):

| | median | mean | within 5 deg | within 10 deg | gross failures |
|---|---|---|---|---|---|
| FoundPose as shipped | 7.4 deg | 30.3 deg | 36% | 64% | 14.4% |
| + refine best candidate only | 1.4 deg | 25.4 deg | 81% | 83% | 14.4% |
| 20 candidates, no refinement | 6.7 deg | 22.9 deg | 39% | 70% | 10.0% |
| 20 candidates + full refinement | 1.1 deg | 4.7 deg | 94% | 98% | 1.9% |

Texture alone makes FoundPose worse (grey templates against coloured photos), and the refinement recovers most
of it. A nearly symmetric body is the weak case: the outline barely changes under a flip, FoundPose often does not
retrieve the right pose among its candidates, and 20% of the rocket-body views end up flipped about 180 deg. The
three object sets were rebuilt from scratch after the clean-up; our pipeline reproduced its earlier medians to within
0.2 deg, while FoundPose as shipped changed a lot between runs on the flip-prone objects (rocket body median 81 to
160 deg, debris 34 to 20 deg), so treat its exact numbers there as noisy.

Robustness (80 views, refinement given a bad mask or an inexact mesh while FoundPose still matches the true mesh;
this is why the matched-point bonus of 0.2 is on): noisy masks (outline +-4 px, holes, a stray blob) cost almost
nothing (96% within 10 deg); a mesh missing 5 small parts gave 90% within 10 deg (5% gross); a mesh with the solar
panels 25% too long left a typical error of about 12 deg (7.5% gross).

*Bias fix.* Translation first scored a consistent +1.1% too far away (97% of views). The outline matcher drew
shapes about half a pixel too big (PIL fills every pixel a polygon touches), and the optimiser compensated by
moving the object away. Drawing the outline 4x supersampled, with the right pixel-centre convention, cut the
bias to about +0.2% and, as a side effect, the median rotation error from 1.4-1.6 deg to 0.5-0.6 deg.

**Limits.** The refinement trusts the mesh outline. It tolerates a poor mask well, but a mesh that is clearly the
wrong shape leaves a typical error of about 12 deg. Real Hunyuan3D meshes will have errors of this kind, so these
numbers are an upper bound until the pipeline is run on real photos with real reconstructed meshes (below).

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
- **For a reconstructed mesh only rotation is trustworthy**, as stated up top — translation/depth
  reflects the mesh's arbitrary reconstructed scale, not the real object's physical distance.
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

## Speed (frames per second)

`scripts/make_tracking_video.py generate | infer | video` runs the pipeline on a 48-frame turntable of the test
satellite and times every frame, one at a time, on an idle RTX 3050 laptop (4 GB). It writes
`output/turntable_synthetic/tracking.mp4` / `.gif` (left: FoundPose as shipped, right: our pipeline (with refinement);
green = truth, red = estimate), `fps_report.txt` and per-frame errors.

| variant | time per frame | fps | rotation median | within 10 deg | position error median |
|---|---|---|---|---|---|
| FoundPose as shipped | 0.53 s | 1.9 | 6.8 deg | 60% | 3.5% |
| our pipeline (with refinement, 20 candidates) | 13.1 s | 0.08 | 0.40 deg | 100% | 0.26% |

None of these is real time. FoundPose itself spends most of its 0.53 s on the coarse pose (0.27 s), image
preparation (0.11 s) and feature extraction (0.09 s). Our pipeline's extra 12 s is the silhouette
refinement, which runs on the CPU in a single thread, so a faster GPU alone would not help it. The video plays at
8 fps for viewing; that is not the processing rate.

### The same test on a Hunyuan3D-generated mesh

The mesh made by the public Hunyuan3D-2 Space from one starsat photo (`output/turntable_hunyuan/starsat2_hunyuan.glb`)
was simplified to 4,284 triangles and rendered as a 48-frame turntable, so the true pose of every frame is known
(`make_tracking_video.py generate --object <mesh.glb>`, then `infer`, then `video --thin`; results in
`output/turntable_hunyuan/`). The renders come from the mesh itself, so this measures the pipeline on this shape,
not how closely the generated mesh matches the real satellite.

| variant | time per frame | fps | rotation median | rotation mean | within 10 deg | position error median |
|---|---|---|---|---|---|---|
| FoundPose as shipped | 0.25 s | 4.1 | 9.0 deg | 25.6 deg | 58% | 3.6% |
| our pipeline (with refinement, 20 candidates) | 73 s | 0.014 | 0.51 deg | 8.2 deg | 96% | 0.34% |

Our pipeline (with refinement) fails on 2 of 48 frames (turntable positions 67.5 and 90 deg), both 180 deg flips seen almost
edge-on, where the thin panel looks the same from either side; all other frames are within 4.7 deg. Refinement is
about 5.6 times slower here than on the box satellite (73 s against 13 s) because the generated mesh has far more
triangles to draw.

### The same pipeline on the real starsat photos

`scripts/run_on_photos.py --mesh <mesh.glb>` runs the pipeline on the five photos in `data/starsat/` (embedded in
1024x1024 canvases, focal length guessed as one photo width) and draws the mesh outline over each photo. There is no
ground truth, so this was judged by how well the mesh outline overlaps the photo's outline, not by accuracy
(recorded once; those outputs were not kept):

| variant | time per photo | mean outline overlap (IoU) |
|---|---|---|
| FoundPose as shipped | 0.4 s | 0.76 |
| our pipeline (with refinement) | 94 s | 0.92 |

The overlap is much better with our pipeline (with refinement), but the poses are not trustworthy: `starsat` and `starsat3`
show the same front view, yet their estimated rotations differ by 176 deg, and the photo-to-photo turns (117, 77,
135 and 176 deg) are far larger than the views suggest. The generated mesh is nearly left-right symmetric in outline,
so an outline match cannot tell front from back; both flips fit equally well (overlap 0.97 on both photos). Resolving
it needs internal detail (the dark compartments, the dish) or continuity between frames, not outline fit. This is the
main open problem.
