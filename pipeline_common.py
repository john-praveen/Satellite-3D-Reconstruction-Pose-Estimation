"""Shared pose-estimation + visualization logic used by both
run_single_view_pipeline.py and run_multiview_pipeline.py -- once you have a mesh, the
rest of the pipeline (render templates, match, estimate pose, visualize) doesn't care
whether that mesh came from one photo or three.
"""
import os
import sys

import torch
from PIL import Image

import config

sys.path.insert(0, config.HY3DGEN_PATH)
from hy3dgen.rembg import BackgroundRemover

from foundpose_lite.build_repr import build_object_representation
from foundpose_lite.estimate_pose import estimate_pose, save_pose_result
from foundpose_lite.render_templates import render_templates
from foundpose_lite.wireframe import draw_axes, render_mask_overlay

_remover = None


def remove_background(image_path: str, out_path: str):
    global _remover
    if _remover is None:
        _remover = BackgroundRemover()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_nobg = _remover(image)
    image_nobg.save(out_path)
    return out_path


def run_pose_and_visualize(mesh_path: str, query_raw_path: str, out_dir: str,
                            num_templates: int = None, skip_render: bool = False,
                            skip_repr: bool = False):
    """End-to-end: render templates from `mesh_path` (if not skipped), build the object
    representation, estimate the pose of the object in `query_raw_path`, and save a
    mask+axes visualization. Returns the pose result dict."""
    num_templates = num_templates or config.NUM_TEMPLATES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    templates_dir = os.path.join(out_dir, "templates")
    repr_path = os.path.join(out_dir, "object_repre.pkl")
    query_stem = os.path.splitext(os.path.basename(query_raw_path))[0]
    nobg_path = os.path.join(out_dir, f"{query_stem}_nobg.png")
    pose_path = os.path.join(out_dir, f"{query_stem}_pose.json")
    mask_path = os.path.join(out_dir, f"{query_stem}_mask.png")
    combined_path = os.path.join(out_dir, f"{query_stem}_mask_axes.png")

    if not skip_render:
        print(f"== rendering {num_templates} templates ==")
        render_templates(mesh_path, templates_dir, num_templates, config.TEMPLATE_SIZE,
                          config.DELTA)
    else:
        print("== skipping template rendering (reusing existing) ==")

    if not skip_repr:
        print("== building object representation (DINOv2 + PCA + bag-of-words) ==")
        build_object_representation(
            templates_dir, config.DINO_MODEL, config.DINO_LAYER, config.DINO_PATCH_SIZE,
            config.PCA_DIM, config.NUM_VISUAL_WORDS, config.SOFT_ASSIGN_K,
            config.SOFT_ASSIGN_SIGMA, device, repr_path)
    else:
        print("== skipping representation build (reusing existing) ==")

    print("== removing background from query image ==")
    remove_background(query_raw_path, nobg_path)

    print("== estimating pose ==")
    result = estimate_pose(
        repr_path, nobg_path, config.CAMERA_K, config.DINO_MODEL, config.DINO_LAYER,
        config.DINO_PATCH_SIZE, config.TEMPLATE_SIZE, config.DELTA, config.TOP_K_TEMPLATES,
        config.PNP_ITERS, config.PNP_INLIER_THRESH_PX, device, config.MATCH_TOP_K_BUDDIES)
    print(f"best template: {result['template_id']}  inliers: {result['num_inliers']}")
    print("R =", result["R"])
    print("t =", result["t"], " (NOT metrically trustworthy -- rotation is the reliable part)")
    save_pose_result(result, pose_path)

    print("== rendering mask + axes visualization ==")
    render_mask_overlay(query_raw_path, mesh_path, result["mesh_centroid"], result["R"],
                         result["t"], config.CAMERA_K, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                         mask_path)
    draw_axes(mask_path, mesh_path, result["mesh_centroid"], result["R"], result["t"],
              config.CAMERA_K, combined_path)

    print(f"saved: {pose_path}")
    print(f"saved: {combined_path}")
    print("Check the mask_axes image before trusting the result: if the cyan silhouette "
          "doesn't tightly line up with the real object, the pose is wrong -- this "
          "pipeline is not reliable on every input, see README.md limitations.")
    return result
