"""All the knobs you're likely to want to change live here."""
import os

import numpy as np

# ============================================================================
# Path to a cloned Hunyuan3D-2 repo (https://github.com/Tencent/Hunyuan3D-2).
# This pipeline does NOT bundle Hunyuan3D-2 -- clone it separately (see
# README.md) and point this at it. It's used only for 3D reconstruction
# (turning photos into a mesh); everything after that (pose estimation) is
# self-contained in this repo.
# ============================================================================
HY3DGEN_PATH = os.environ.get("HY3DGEN_PATH", r"D:\Hunyuan3D-2")

# ============================================================================
# Camera intrinsics for your query photos.
# If you don't know the real focal length, fx=fy=image_width and the
# principal point at the image center is a reasonable default (~53 degree
# horizontal FOV) -- rotation estimates are fairly robust to this being
# approximate, but a wrong focal length CAN bias rotation too (not just
# translation), especially at oblique viewing angles. If you have real
# calibration (phone EXIF, a calibration file, a known lens spec), use it.
# ============================================================================
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900
CAMERA_K = np.array([
    [float(IMAGE_WIDTH), 0.0, IMAGE_WIDTH / 2.0],
    [0.0, float(IMAGE_WIDTH), IMAGE_HEIGHT / 2.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# ============================================================================
# Template bank (the "coarse pose" search space). More templates = better
# SO(3) coverage but linearly slower to render (~2-3s/template) and slower
# to build the representation from. 300 is fast for a quick check; 800 was
# used throughout testing and is a reasonable default.
# ============================================================================
NUM_TEMPLATES = 800
TEMPLATE_SIZE = 420          # must be a multiple of DINO_PATCH_SIZE (14)
DELTA = 0.6                  # object's longer bbox side, as a fraction of TEMPLATE_SIZE

# ============================================================================
# DINOv2 feature extractor. vitl14_reg + layer 18 is the paper's best-reported
# config (bigger/slower). vits14_reg + layer 9 is smaller/faster if you need
# to iterate quickly, at some cost to matching quality.
# ============================================================================
DINO_MODEL = "dinov2_vitl14_reg"
DINO_LAYER = 18
DINO_PATCH_SIZE = 14

# ============================================================================
# Bag-of-words retrieval + matching parameters. Rarely need changing.
# ============================================================================
PCA_DIM = 256
NUM_VISUAL_WORDS = 1024
SOFT_ASSIGN_K = 3
SOFT_ASSIGN_SIGMA = 10.0
TOP_K_TEMPLATES = 5           # how many candidate templates to try PnP on
MATCH_TOP_K_BUDDIES = 60      # keep only the N most "cyclically consistent" 2D-3D
                               # correspondences before PnP-RANSAC (ported from FoundPose's
                               # own matching, corresp_util.py's cyclic_buddies_matching).
                               # Must be meaningfully SMALLER than your typical foreground
                               # patch count to actually filter anything. NOTE: an A/B test
                               # found this does NOT reliably fix hard cases on its own (it
                               # can pick the identical template as plain matching) -- see
                               # foundpose_official/ below for the path that actually helped
                               # in testing.

# ============================================================================
# Official FoundPose repo (facebookresearch/foundpose), used as an alternative,
# more accurate (but heavier-setup) pose engine -- see run_official_pipeline.py.
# Needs the official repo cloned separately (README.md) plus its OWN conda env
# (different Python/deps than the rest of this pipeline), since its dependency
# stack doesn't mix cleanly with Hunyuan3D-2's. ssaa_factor in
# foundpose_official/configs/gen_templates/custom.json is turned down from the
# official default of 4.0 to 1.0 -- this was an ~12x speedup (30min -> 2.5min
# for 800 templates) with no visible accuracy loss in testing.
# ============================================================================
FOUNDPOSE_OFFICIAL_REPO_PATH = os.environ.get("FOUNDPOSE_OFFICIAL_REPO_PATH", r"D:\foundpose")
FOUNDPOSE_OFFICIAL_CONDA_ENV = os.environ.get("FOUNDPOSE_OFFICIAL_CONDA_ENV", "foundpose_repo")
BOP_DATASETS_ROOT = "output/bop_datasets"

# ============================================================================
# PnP-RANSAC parameters. Rarely need changing.
# ============================================================================
PNP_ITERS = 400
PNP_INLIER_THRESH_PX = 10.0

OUTPUT_DIR = "output"
