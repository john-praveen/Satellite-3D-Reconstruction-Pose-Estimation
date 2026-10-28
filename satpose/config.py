"""All the knobs you're likely to want to change live here."""
import os

import numpy as np

# ============================================================================
# Path to a cloned Hunyuan3D-2 repo (https://github.com/Tencent/Hunyuan3D-2).
# This pipeline does NOT bundle Hunyuan3D-2 -- clone it separately (see
# README.md) and point this at it. It's used only for 3D reconstruction
# (turning photos into a mesh); pose estimation itself is delegated entirely
# to the official FoundPose repo below.
# The fallback below (used only if the HY3DGEN_PATH env var isn't set) is
# THIS machine's actual clone location -- if you're setting this project up
# somewhere else, either change this literal to your own clone's path, or
# leave it and set the HY3DGEN_PATH environment variable instead (that always
# takes priority over this fallback).
# ============================================================================
HY3DGEN_PATH = os.environ.get("HY3DGEN_PATH", r"D:\3D_recon\Hunyuan3D-2")

# ============================================================================
# Camera intrinsics for your query photos.
# If you don't know the real focal length, fx=fy=image_width and the
# principal point at the image center is a reasonable default (~53 degree
# horizontal FOV) -- rotation estimates are fairly robust to this being
# approximate, but a wrong focal length CAN bias rotation too (not just
# translation), especially at oblique viewing angles. If you have real
# calibration (phone EXIF, a calibration file, a known lens spec), use it.
# If you have the camera intrinsics like focal length (fx, fy) and principal point (cx, cy) the matrix will be:
# CAMERA_K = np.array([[fx, 0, cx],
#                      [0, fy, cy],
#                      [0, 0, 1]], dtype=np.float64) 
#importantly, the units of fx, fy, cx, cy should be in pixels. If you have the sensor size and focal length in mm, you can convert them to pixels using the following formula:
# fx = (focal_length_mm / sensor_width_mm) * image_width_pixels
# fy = (focal_length_mm / sensor_height_mm) * image_height_pixels
#using camera intrinsics is the best way to get accurate pose estimation, but if you don't have them, the default values above will work reasonably well for many cases.
#
# IMPORTANT: these MUST match the actual pixel resolution of whatever photos
# you're currently running through the pipeline -- satpose/pipeline.py checks
# this at the start of every run and raises an error if a query photo's real
# size doesn't match IMAGE_WIDTH/IMAGE_HEIGHT below, since a mismatch would
# otherwise silently feed wrong intrinsics into the official FoundPose
# pipeline (not just break the mask+axes visualization). Update these two
# values (and CAMERA_K, unless you have real calibration for this exact
# resolution) whenever you switch to a different photo set.
# Currently set for the Aqua orbit_xy dataset (1024x1024): REAL simulator
# ground-truth intrinsics (monocular pinhole, DepthPerspective), not the
# fx=fy=width guess -- fx=fy=1097.99, cx=cy=512. The bundled starsat demo
# photos are 1600x900 with no known calibration (fx=fy=width guess applies
# there) -- change these back before re-running
# run_single_view_pipeline.py's default example.
# ============================================================================
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
CAMERA_K = np.array([
    [1097.99, 0.0, 512.0],
    [0.0, 1097.99, 512.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# ============================================================================
# The official FoundPose repo (facebookresearch/foundpose, ECCV 2024) does all
# pose estimation (template rendering, DINOv2 matching, PnP-RANSAC) -- see
# satpose/pipeline.py / run_single_view_pipeline.py / run_multiview_pipeline.py.
# Needs the official repo cloned separately (README.md "Official pipeline
# setup") plus its OWN conda env (different Python/deps than the rest of this
# pipeline), since its dependency stack doesn't mix cleanly with Hunyuan3D-2's.
# Template-bank size, DINOv2 backbone, matching, and PnP parameters are all
# tuned in satpose/foundpose_official/configs/*/custom.json instead of here. Notably,
# ssaa_factor in configs/gen_templates/custom.json is turned down from the
# official default of 4.0 to 1.0 -- this was an ~12x speedup (30min -> 2.5min
# for 800 templates) with no visible accuracy loss in testing.
# As with HY3DGEN_PATH above, the fallback below is THIS machine's actual
# clone location (note the nested \repo -- the outer foundpose/ folder also
# has unrelated scratch files, the real `git clone` output is one level
# deeper) -- change it or set the FOUNDPOSE_OFFICIAL_REPO_PATH env var for a
# different setup.
# ============================================================================
FOUNDPOSE_OFFICIAL_REPO_PATH = os.environ.get("FOUNDPOSE_OFFICIAL_REPO_PATH", r"D:\3D_recon\foundpose\repo")
FOUNDPOSE_OFFICIAL_CONDA_ENV = os.environ.get("FOUNDPOSE_OFFICIAL_CONDA_ENV", "foundpose_repo")

OUTPUT_DIR = "output"

# ============================================================================
# run_pipeline.py -- the single interactive entry point (asks "single or
# multi view?" in the terminal, then runs everything end to end using the
# settings below). Everything here can also just be edited directly instead
# of answering the prompt -- see each variable's comment.
# ============================================================================

# "single", "multi", or None. If set, run_pipeline.py uses this and skips the
# terminal prompt entirely -- useful for a fixed/scripted setup.
PIPELINE_MODE = None

# -- Single-view: the one photo reconstruction is built from. --
SINGLE_VIEW_IMAGE = "data/starsat/starsat.jpeg"

# -- Multi-view: reconstruction photos. FRONT/BACK/LEFT are required by the
# underlying model; RIGHT_IMAGE is optional (set to None to reconstruct from
# just 3 views). --
FRONT_IMAGE = "data/starsat/starsat.jpeg"
BACK_IMAGE = None
LEFT_IMAGE = None
RIGHT_IMAGE = None

# -- Where to pull QUERY photos (the ones pose is estimated on) from. Any of:
#   None                    -> defaults to the reconstruction photo(s) above
#   "path/to/photo.jpg"     -> a single photo
#   ["a.jpg", "b.jpg"]      -> an explicit list of photos
#   "path/to/a_folder"      -> every image file in that folder (non-recursive)
#   "path/to/a_video.mp4"   -> frames extracted from the video (see
#                              VIDEO_FRAME_STRIDE below), one query photo per
#                              extracted frame
# All query photos must match IMAGE_WIDTH/IMAGE_HEIGHT above -- videos/photos
# at a different resolution need those updated first. --
QUERY_SOURCE = None
VIDEO_FRAME_STRIDE = 1  # 1 = every frame, 5 = every 5th frame, etc.

# -- Visualization toggles. Estimated poses are ALWAYS saved (JSON + CSV)
# regardless of these; they only control whether the *_mask.png/*_mask_axes.png
# sanity-check images get rendered. Turning both off is faster for large
# batches (folders/videos with many frames) when you just want the numbers. --
SAVE_MASK = True
SAVE_AXES = True
