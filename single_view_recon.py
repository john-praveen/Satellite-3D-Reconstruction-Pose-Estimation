"""3D reconstruction from ONE photo, using Hunyuan3D-2's single-image pipeline.

Usage:
    python single_view_recon.py --image data/starsat/starsat.jpeg --out output/starsat_mesh.glb
"""
import argparse
import os
import sys

from PIL import Image

import config

sys.path.insert(0, config.HY3DGEN_PATH)
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
    return _pipeline


def reconstruct_single_view(image_path: str, out_mesh_path: str):
    """Removes the background, then reconstructs a 3D mesh from the single photo."""
    image = Image.open(image_path).convert("RGB")
    image = BackgroundRemover()(image)

    mesh = get_pipeline()(image=image)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out_mesh_path)), exist_ok=True)
    mesh.export(out_mesh_path)
    print(f"saved {out_mesh_path}")
    return out_mesh_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="path to a single RGB photo")
    parser.add_argument("--out", required=True, help="output .glb mesh path")
    args = parser.parse_args()
    reconstruct_single_view(args.image, args.out)
