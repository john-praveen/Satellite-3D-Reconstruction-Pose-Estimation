"""Full multi-view pipeline using the ACTUAL official FoundPose (facebookresearch/foundpose,
ECCV 2024) algorithm: three or four photos (front/back/left, optionally +right) -> 3D
reconstruction (Hunyuan3D-2 multi-view) -> pose estimation (the official repo's rendering,
DINOv2 matching, PnP-RANSAC, run in its own conda env -- see pipeline_common.py) ->
mask+axes visualization. Multi-view reconstruction is generally more accurate than
single-view, since the mesh gets real geometric constraints from multiple real vantage
points instead of guessing occluded geometry from one photo.

Needs TWO conda environments (this pipeline's own env, and a separate one for the
official repo -- their dependency stacks don't mix). See README.md "Official pipeline
setup" for how to create both.

No multiview example data is included (the starsat photos are 5 independent single
views, not a matched front/back/left/right set of the same object) -- provide your own
photos.

Usage:
    python run_multiview_pipeline.py --front front.jpg --back back.jpg --left left.jpg

Usage (with the optional 4th viewpoint):
    python run_multiview_pipeline.py --front front.jpg --back back.jpg --left left.jpg \\
        --right right.jpg

Usage (estimate pose on different/additional photos than the recon set):
    python run_multiview_pipeline.py --front front.jpg --back back.jpg --left left.jpg \\
        --query front.jpg extra_photo.jpg
"""
import argparse
import os

import config
from multiview_recon import reconstruct_multiview
from pipeline_common import run_official_foundpose


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--front", required=True)
    parser.add_argument("--back", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", default=None, help="optional 4th viewpoint")
    parser.add_argument("--query", nargs="+", default=None,
                         help="photo(s) to estimate pose on (defaults to --front, --back, "
                              "--left, --right)")
    parser.add_argument("--out-dir", default=None,
                         help="output directory (defaults to output/<front image name>_multiview)")
    parser.add_argument("--skip-recon", action="store_true",
                         help="reuse an existing mesh instead of reconstructing")
    parser.add_argument("--skip-templates", action="store_true",
                         help="reuse an existing template bank + representation")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.front))[0]
    out_dir = os.path.abspath(args.out_dir or os.path.join(config.OUTPUT_DIR, f"{stem}_multiview"))
    os.makedirs(out_dir, exist_ok=True)
    mesh_path = os.path.join(out_dir, f"{stem}_mesh_mv.glb")
    recon_views = [args.front, args.back, args.left] + ([args.right] if args.right else [])
    query_images = args.query or recon_views

    if not args.skip_recon:
        print(f"== multi-view reconstruction from {', '.join(recon_views)} ==")
        reconstruct_multiview(args.front, args.back, args.left, mesh_path, right_path=args.right)
    else:
        print(f"== skipping reconstruction, reusing {mesh_path} ==")

    run_official_foundpose(mesh_path, query_images, out_dir, skip_templates=args.skip_templates)


if __name__ == "__main__":
    main()
