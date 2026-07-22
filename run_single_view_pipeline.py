"""Full single-view pipeline using the ACTUAL official FoundPose (facebookresearch/foundpose,
ECCV 2024) algorithm: one photo -> 3D reconstruction (Hunyuan3D-2) -> pose estimation
(the official repo's rendering, DINOv2 matching via its "cyclic buddies" correspondence
filter, PnP-RANSAC, run in its own conda env -- see foundpose_official/build_bop_dataset.py
and pipeline_common.py) -> mask+axes visualization.

Needs TWO conda environments (this pipeline's own env, and a separate one for the
official repo -- their dependency stacks don't mix). See README.md "Official pipeline
setup" for how to create both.

Usage (uses the included starsat example by default, all 5 photos):
    python run_single_view_pipeline.py

Usage (your own image(s)):
    python run_single_view_pipeline.py --image path/to/photo.jpg --query path/to/photo.jpg
    python run_single_view_pipeline.py --image recon_photo.jpg --query photo1.jpg photo2.jpg
"""
import argparse
import os

import config
from pipeline_common import run_official_foundpose
from single_view_recon import reconstruct_single_view

DEFAULT_IMAGES = ["data/starsat/starsat.jpeg", "data/starsat/starsat1.jpeg",
                   "data/starsat/starsat2.jpeg", "data/starsat/starsat3.jpeg",
                   "data/starsat/starsat4.jpeg"]


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
    out_dir = os.path.abspath(args.out_dir or os.path.join(config.OUTPUT_DIR, stem))
    os.makedirs(out_dir, exist_ok=True)
    mesh_path = os.path.join(out_dir, f"{stem}_mesh.glb")
    query_images = args.query or DEFAULT_IMAGES

    if not args.skip_recon:
        print(f"== single-view reconstruction from {args.image} ==")
        reconstruct_single_view(args.image, mesh_path)
    else:
        print(f"== skipping reconstruction, reusing {mesh_path} ==")

    run_official_foundpose(mesh_path, query_images, out_dir, skip_templates=args.skip_templates)


if __name__ == "__main__":
    main()
