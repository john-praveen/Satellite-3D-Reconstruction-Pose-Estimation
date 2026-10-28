"""3D reconstruction from THREE OR FOUR photos (front/back/left, optionally +right), using
Hunyuan3D-2's multiview pipeline. Generally gives a more accurate/complete mesh than
single-view, since the model gets real geometric constraints from multiple real vantage
points instead of guessing occluded geometry from one photo. The underlying model
(hunyuan3d-dit-v2-mv) supports up to 4 named views -- front/left/back/right (see
hy3dgen/shapegen/preprocessors.py's MVImageProcessorV2.view2idx in the Hunyuan3D-2 repo) --
right is optional; the official examples only demonstrate 3.

Usage:
    python -m satpose.reconstruction.multiview --front front.jpg --back back.jpg --left left.jpg \
        --out output/mesh.glb
    python -m satpose.reconstruction.multiview --front front.jpg --back back.jpg --left left.jpg \
        --right right.jpg --out output/mesh.glb
"""
import argparse
import os
import sys

from PIL import Image

from satpose import config

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
                           right_path: str = None, num_inference_steps: int = 30,
                           octree_resolution: int = 380):
    """Removes background from each photo, then reconstructs a 3D mesh from them jointly.
    `right_path` is optional -- pass it for a 4th viewpoint (front/back/left/right) when
    you have one, for tighter geometric constraints than the 3-view minimum."""
    remover = BackgroundRemover()
    views = [("front", front_path), ("back", back_path), ("left", left_path)]
    if right_path:
        views.append(("right", right_path))
    images = {}
    for view, path in views:
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
    parser.add_argument("--right", default=None, help="optional 4th viewpoint")
    parser.add_argument("--out", required=True, help="output .glb mesh path")
    args = parser.parse_args()
    reconstruct_multiview(args.front, args.back, args.left, args.out, right_path=args.right)
