"""Full multi-view pipeline: three photos (front/back/left) -> 3D reconstruction ->
pose estimation -> mask+axes visualization.

No multiview example data is included (the starsat photos are 5 independent single
views, not a front/back/left triplet of the same pose) -- provide your own 3 photos.

Usage:
    python run_multiview_pipeline.py --front front.jpg --back back.jpg --left left.jpg \
        --query front.jpg --out-dir output/myobject
"""
import argparse
import os

import config
from multiview_recon import reconstruct_multiview
from pipeline_common import run_pose_and_visualize


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--front", required=True)
    parser.add_argument("--back", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--query", default=None,
                         help="photo to estimate pose on (defaults to --front)")
    parser.add_argument("--out-dir", default=None,
                         help="output directory (defaults to output/<front image name>)")
    parser.add_argument("--num-templates", type=int, default=config.NUM_TEMPLATES)
    parser.add_argument("--skip-recon", action="store_true",
                         help="reuse an existing mesh instead of reconstructing")
    parser.add_argument("--skip-render", action="store_true",
                         help="reuse existing rendered templates")
    parser.add_argument("--skip-repr", action="store_true",
                         help="reuse an existing object representation")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.front))[0]
    out_dir = args.out_dir or os.path.join(config.OUTPUT_DIR, stem)
    mesh_path = os.path.join(out_dir, f"{stem}_mesh_mv.glb")
    query_path = args.query or args.front

    if not args.skip_recon:
        print(f"== multi-view reconstruction from {args.front}, {args.back}, {args.left} ==")
        reconstruct_multiview(args.front, args.back, args.left, mesh_path)
    else:
        print(f"== skipping reconstruction, reusing {mesh_path} ==")

    run_pose_and_visualize(mesh_path, query_path, out_dir, num_templates=args.num_templates,
                            skip_render=args.skip_render, skip_repr=args.skip_repr)


if __name__ == "__main__":
    main()
