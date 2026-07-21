"""Full single-view pipeline: one photo -> 3D reconstruction -> pose estimation ->
mask+axes visualization.

Usage (uses the included starsat example by default):
    python run_single_view_pipeline.py

Usage (your own image):
    python run_single_view_pipeline.py --image path/to/photo.jpg --out-dir output/myobject

Usage (estimate pose on a DIFFERENT photo than the one used to build the mesh):
    python run_single_view_pipeline.py --image photo_for_recon.jpg --query photo_to_pose.jpg
"""
import argparse
import os

import config
from pipeline_common import run_pose_and_visualize
from single_view_recon import reconstruct_single_view


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default="data/starsat/starsat.jpeg",
                         help="photo to reconstruct a 3D mesh from")
    parser.add_argument("--query", default=None,
                         help="photo to estimate pose on (defaults to --image)")
    parser.add_argument("--out-dir", default=None,
                         help="output directory (defaults to output/<image name>)")
    parser.add_argument("--num-templates", type=int, default=config.NUM_TEMPLATES)
    parser.add_argument("--skip-recon", action="store_true",
                         help="reuse an existing mesh instead of reconstructing")
    parser.add_argument("--skip-render", action="store_true",
                         help="reuse existing rendered templates")
    parser.add_argument("--skip-repr", action="store_true",
                         help="reuse an existing object representation")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = args.out_dir or os.path.join(config.OUTPUT_DIR, stem)
    mesh_path = os.path.join(out_dir, f"{stem}_mesh.glb")
    query_path = args.query or args.image

    if not args.skip_recon:
        print(f"== single-view reconstruction from {args.image} ==")
        reconstruct_single_view(args.image, mesh_path)
    else:
        print(f"== skipping reconstruction, reusing {mesh_path} ==")

    run_pose_and_visualize(mesh_path, query_path, out_dir, num_templates=args.num_templates,
                            skip_render=args.skip_render, skip_repr=args.skip_repr)


if __name__ == "__main__":
    main()
