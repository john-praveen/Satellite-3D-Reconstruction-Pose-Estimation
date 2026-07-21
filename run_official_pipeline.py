"""Full single-view pipeline using the ACTUAL official FoundPose algorithm (not our fast
reimplementation in foundpose_lite/) -- reconstructs a mesh, builds a minimal fake BOP
dataset around it (see foundpose_official/build_bop_dataset.py), runs the official repo's
gen_templates.py -> gen_repre.py -> infer.py (in ITS OWN conda env -- see README.md), then
renders our mask+axes visualization on the resulting poses.

This needs TWO conda environments (this pipeline's own env, and a separate one for the
official repo -- their dependency stacks don't mix). See README.md "Official pipeline
setup" for how to create both.

Usage (uses the included starsat example by default, all 5 photos):
    python run_official_pipeline.py

Usage (your own image(s)):
    python run_official_pipeline.py --image path/to/photo.jpg --query path/to/photo.jpg
    python run_official_pipeline.py --image recon_photo.jpg --query photo1.jpg photo2.jpg
"""
import argparse
import json
import os
import subprocess
import sys

import config
from foundpose_official.build_bop_dataset import build_custom_bop_dataset
from foundpose_lite.wireframe import draw_axes, render_mask_overlay
from pipeline_common import remove_background
from single_view_recon import reconstruct_single_view

DEFAULT_IMAGES = ["data/starsat/starsat.jpeg", "data/starsat/starsat1.jpeg",
                   "data/starsat/starsat2.jpeg", "data/starsat/starsat3.jpeg",
                   "data/starsat/starsat4.jpeg"]


def run_official_script(script_name: str, config_name: str, bop_root: str):
    env = os.environ.copy()
    env["REPO_PATH"] = config.FOUNDPOSE_OFFICIAL_REPO_PATH
    env["BOP_PATH"] = bop_root
    env["PYTHONPATH"] = os.pathsep.join([
        config.FOUNDPOSE_OFFICIAL_REPO_PATH,
        os.path.join(config.FOUNDPOSE_OFFICIAL_REPO_PATH, "external", "bop_toolkit"),
        os.path.join(config.FOUNDPOSE_OFFICIAL_REPO_PATH, "external", "dinov2"),
    ])
    opts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foundpose_official",
                              "configs", script_name.replace(".py", ""), f"{config_name}.json")
    cmd = ["conda", "run", "-n", config.FOUNDPOSE_OFFICIAL_CONDA_ENV, "python",
           os.path.join("scripts", script_name), "--opts-path", opts_path]
    print(f"== running official {script_name} ==")
    result = subprocess.run(cmd, cwd=config.FOUNDPOSE_OFFICIAL_REPO_PATH, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed (exit code {result.returncode}) -- "
                            f"see README.md if this is an environment/setup issue")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default="data/starsat/starsat.jpeg",
                         help="photo to reconstruct a 3D mesh from")
    parser.add_argument("--query", nargs="+", default=None,
                         help="photo(s) to estimate pose on (defaults to all 5 starsat photos)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-recon", action="store_true")
    parser.add_argument("--skip-templates", action="store_true",
                         help="reuse an existing template bank + representation")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = os.path.abspath(args.out_dir or os.path.join(config.OUTPUT_DIR, f"{stem}_official"))
    os.makedirs(out_dir, exist_ok=True)
    mesh_path = os.path.join(out_dir, f"{stem}_mesh.glb")
    query_images = [os.path.abspath(q) for q in (args.query or DEFAULT_IMAGES)]
    bop_root = os.path.join(out_dir, "bop_datasets")

    if not args.skip_recon:
        print(f"== single-view reconstruction from {args.image} ==")
        reconstruct_single_view(args.image, mesh_path)
    else:
        print(f"== skipping reconstruction, reusing {mesh_path} ==")

    nobg_paths = []
    for q in query_images:
        q_stem = os.path.splitext(os.path.basename(q))[0]
        nobg_path = os.path.join(out_dir, f"{q_stem}_nobg.png")
        remove_background(q, nobg_path)
        nobg_paths.append(nobg_path)

    if not args.skip_templates:
        build_custom_bop_dataset(mesh_path, nobg_paths, query_images, config.CAMERA_K,
                                  config.IMAGE_WIDTH, config.IMAGE_HEIGHT, bop_root)
        run_official_script("gen_templates.py", "custom", bop_root)
        run_official_script("gen_repre.py", "custom", bop_root)
    else:
        print("== skipping template generation, reusing existing BOP dataset/templates ==")
    run_official_script("infer.py", "custom", bop_root)

    poses_path = os.path.join(bop_root, "outputs", "inference", "custom_v1", "1",
                               "estimated-poses.json")
    with open(poses_path) as f:
        poses = json.load(f)

    for pose in poses:
        im_id = int(pose["img_id"])
        raw_path = query_images[im_id]
        q_stem = os.path.splitext(os.path.basename(raw_path))[0]
        mesh_ply = os.path.join(bop_root, "custom", "models", "obj_000001.ply")

        mask_path = os.path.join(out_dir, f"{q_stem}_mask.png")
        combined_path = os.path.join(out_dir, f"{q_stem}_mask_axes.png")
        render_mask_overlay(raw_path, mesh_ply, [0.0, 0.0, 0.0], pose["R"], pose["t"],
                             config.CAMERA_K, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, mask_path)
        draw_axes(mask_path, mesh_ply, [0.0, 0.0, 0.0], pose["R"], pose["t"],
                  config.CAMERA_K, combined_path)
        print(f"{raw_path}: inliers={pose.get('inliers_est')} score={pose.get('score')} "
              f"-> {combined_path}")


if __name__ == "__main__":
    main()
