"""3D reconstruction from THREE photos (front/back/left), using Hunyuan3D-2's multiview
pipeline. Generally gives a more accurate/complete mesh than single-view, since the model
gets real geometric constraints from 3 real vantage points instead of guessing occluded
geometry from one photo.

Usage:
    python multiview_recon.py --front front.jpg --back back.jpg --left left.jpg \
        --out output/mesh.glb
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
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2mv', subfolder='hunyuan3d-dit-v2-mv', use_safetensors=True)
    return _pipeline


def reconstruct_multiview(front_path: str, back_path: str, left_path: str, out_mesh_path: str,
                           num_inference_steps: int = 30, octree_resolution: int = 380):
    """Removes background from all 3 photos, then reconstructs a 3D mesh from them jointly."""
    remover = BackgroundRemover()
    images = {}
    for view, path in [("front", front_path), ("back", back_path), ("left", left_path)]:
        img = Image.open(path).convert("RGB")
        images[view] = remover(img)

    pipeline = get_pipeline()
    mesh = pipeline(image=images, num_inference_steps=num_inference_steps,
                     octree_resolution=octree_resolution)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out_mesh_path)), exist_ok=True)
    mesh.export(out_mesh_path)
    print(f"saved {out_mesh_path}")
    return out_mesh_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True)
    parser.add_argument("--back", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--out", required=True, help="output .glb mesh path")
    args = parser.parse_args()
    reconstruct_multiview(args.front, args.back, args.left, args.out)
