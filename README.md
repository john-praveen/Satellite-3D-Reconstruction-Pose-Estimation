# Satellite 3D Reconstruction + Pose Estimation

Given photo(s) of an object, this pipeline does two things:
1. **3D reconstruction** — builds a 3D mesh of the object using Hunyuan3D-2 (a generative
   single/multi-image-to-3D model), from either **one photo** or **three photos**
   (front/back/left, which generally gives a more accurate mesh).
2. **Pose estimation** — given that mesh and a photo, estimates the object's **rotation**
   in the photo. **Two engines are available, run through the same mask-overlay visual
   check:**
   - `run_official_pipeline.py` — runs the ACTUAL [FoundPose](https://github.com/facebookresearch/foundpose)
     (ECCV 2024) repo (rendering, DINOv2 matching via their "cyclic buddies" correspondence
     filter, PnP-RANSAC), wired up automatically against your object. **Recommended** —
     gave the tightest, most consistent results in testing (see `Known limitations` below).
     Needs a second conda environment (the official repo's own dependency stack) — see
     "Official pipeline setup".
   - `run_single_view_pipeline.py` / `run_multiview_pipeline.py` — a lighter, from-scratch
     reimplementation of the same algorithm (`foundpose_lite/`) that only needs the one
     environment from step 1. Faster to set up, but was less consistently accurate in
     testing than the official pipeline above.

**Important limitation up front:** the reconstructed mesh is not metrically scaled to the
real object's true size. That means the estimated **rotation is the trustworthy output**;
the estimated **translation/distance is not** (there's no way to recover true scale from
a single photo without a calibrated stereo setup or known object dimensions). Every
script in this repo reflects that — translation is reported but flagged as unreliable.

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
or edit the default at the top of `config.py`.

First run of any reconstruction script downloads the Hunyuan3D-2 model weights from
Hugging Face automatically (~10-25GB depending on which model, one-time).

### Official pipeline setup (needed for `run_official_pipeline.py`)

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

Then set two things so `run_official_pipeline.py` can find it — either environment
variables or the defaults in `config.py`:
```
set FOUNDPOSE_OFFICIAL_REPO_PATH=D:\path\to\foundpose
set FOUNDPOSE_OFFICIAL_CONDA_ENV=foundpose_repo
```

`run_official_pipeline.py` itself still runs in your main `satpose` environment (from
step b/c/d above) — it shells out to `foundpose_repo` only for the 3 official-algorithm
steps (template rendering, representation building, inference), then comes back to the
main environment for reconstruction and the mask+axes visualization. You don't run it
from inside `foundpose_repo` yourself.

## 2. Quick start — try the included example

```
python run_official_pipeline.py
```
Reconstructs a mesh from `data/starsat/starsat.jpeg` and estimates pose on **all 5**
included starsat photos (that's the default query set for this script — pass `--query`
to override), saving results under `output/starsat_official/`. Open any
`*_mask_axes.png` there — see "Reading the output" below for how to judge it.

Prefer the lighter/faster single-environment path instead? Use:
```
python run_single_view_pipeline.py
```
which does the same thing against `foundpose_lite/`, saving to `output/starsat/`.

## 3. Running on your own data

### Official pipeline (recommended)
```
python run_official_pipeline.py --image path/to/recon_photo.jpg --query path/to/pose_photo.jpg
```
`--query` accepts multiple photos (`--query a.jpg b.jpg c.jpg`) — all get pose-estimated
against the one reconstructed mesh in a single run, which is much cheaper than
reconstructing/re-templating per photo. `--skip-recon` reuses an existing mesh;
`--skip-templates` reuses an existing template bank + representation (both cached under
`output/<name>_official/`).

### Fast/single-environment pipeline
#### Single-view (one photo)
```
python run_single_view_pipeline.py --image path/to/your_photo.jpg
```
Optionally estimate pose on a *different* photo than the one used to build the mesh:
```
python run_single_view_pipeline.py --image recon_photo.jpg --query pose_photo.jpg
```

### Multi-view (three photos: front/back/left)
Generally more accurate than single-view, since the reconstruction gets real geometric
constraints from 3 real vantage points instead of guessing occluded geometry from one.
```
python run_multiview_pipeline.py --front front.jpg --back back.jpg --left left.jpg
```
No multiview example data is included — the 5 starsat photos are independent single
views, not a matched front/back/left triplet.

### Re-running without redoing expensive steps
Each pipeline caches its outputs under `output/<name>/`. Skip steps you don't need to redo:
```
python run_single_view_pipeline.py --skip-recon --skip-render --skip-repr --query new_photo.jpg
```
`--skip-recon` reuses the existing mesh, `--skip-render`/`--skip-repr` reuse the existing
rendered template bank and DINOv2 representation — useful if you just want to test a
*different photo* of the *same already-reconstructed* object.

## 4. Adjusting parameters

Everything tunable lives in **`config.py`**, with comments explaining each:
- `CAMERA_K` / `IMAGE_WIDTH` / `IMAGE_HEIGHT` — camera intrinsics for your photos. Defaults
  to a generic guess (`fx=fy=image width`, principal point centered) if you don't have real
  calibration. **If you have real calibration (phone EXIF, known lens spec), use it** —
  see "Known limitations" below for why this matters more than it might seem.
- `NUM_TEMPLATES` — speed/accuracy tradeoff for the template bank. 300 for a fast check,
  800 (default) for better accuracy. Roughly linear in render time (~2-3s per template).
- `MATCH_TOP_K_BUDDIES` — how many of the most self-consistent correspondences to keep
  before running PnP-RANSAC (a "cyclic buddies" filter ported from FoundPose's own
  matching code). **This mattered more than any other single parameter during testing.**
  It only works as a filter if it's smaller than your actual foreground patch count —
  watch the console output for the "N foreground patches" line and keep this comfortably
  below that number.
- `DINO_MODEL` / `DINO_LAYER` — which DINOv2 backbone to use for feature matching.
  Default (`vitl14_reg`, layer 18) is the paper's best-reported config; `vits14_reg` /
  layer 9 is smaller and faster if you're iterating quickly.
- Everything else (PCA dimension, cluster count, PnP settings) rarely needs changing.

## 5. Reading the output

Each run saves `<name>_mask_axes.png`: the reconstructed mesh, re-rendered at the
estimated pose and overlaid on your photo (cyan fill + contour = the mesh's projected
silhouette), plus a red/green/blue X/Y/Z axis triad at the object's center.

**This is how you judge whether a result is trustworthy — always check it, don't just
trust the numbers in the JSON output:**
- If the cyan silhouette **tightly hugs** the real object's outline in the photo → the
  rotation estimate is good.
- If the cyan silhouette is a **different shape, offset, or rotated relative to** the
  real object → the pose estimate is wrong for that photo. Don't trust it.

## 6. Known limitations (from testing on real photos)

- **Always check the mask overlay yourself, critically — never trust the numbers alone.**
  The `foundpose_lite` (fast) pipeline was inconsistent in testing (5 photos of one
  object): roughly 3 out of 5 gave visibly correct poses, and a `MATCH_TOP_K_BUDDIES`
  filter tweak that looked at first like it fixed the other 2 turned out, on a proper
  A/B test, to pick the identical template/pose as plain matching there — it wasn't
  actually a fix, just a coincidence in how it first looked. The `run_official_pipeline.py`
  path (actual FoundPose algorithm, lower template resolution for speed) then gave
  visibly tight, correct results on **all 5** of the same test photos, including the 2
  that were failing before. That's still a single small test, not a validated benchmark,
  and it's *why* the official pipeline is the recommended default above — but "always
  check the mask, don't assume success" still applies regardless of which engine you use.
  Symmetric, flat objects are especially prone to a *wrong* rotation still producing a
  plausible-looking silhouette by coincidence (a flat panel looks roughly like a flat
  panel from many angles) — check whether edges and internal detail actually line up,
  not just the rough outline shape.
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

## 7. Troubleshooting

- **On Linux (headless server, no display):** set `PYOPENGL_PLATFORM=egl` in your shell
  *before* running any script that renders (anything that imports `pyrender`/`foundpose_lite`).
- **First run is slow / seems stuck:** it's downloading model weights (several GB) —
  check your network, it only happens once.
- **Out of GPU memory:** reduce `NUM_TEMPLATES` in `config.py`, or switch `DINO_MODEL` to
  `dinov2_vits14_reg` (smaller backbone).
- **`run_official_pipeline.py` fails with a path/"Unknown BOP dataset" error:** check
  `FOUNDPOSE_OFFICIAL_REPO_PATH` actually points at the folder *containing* `scripts/`
  (i.e. where you ran `git clone`), and that `conda activate foundpose_repo && python -c
  "import torch, faiss, pyrender"` succeeds with no errors in that environment.
- **`run_official_pipeline.py` fails with `FileNotFoundError` on `dataset_meta.json`:**
  usually a relative-path issue if you passed a relative `--out-dir` — use an absolute
  path, or leave `--out-dir` unset to use the (already-absolute) default.
