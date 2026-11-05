"""Runs the pose pipeline on photos of an object with a known mesh and draws the result.

There is no ground truth here, so nothing is scored for accuracy. Instead each photo gets an
overlay of the mesh at the estimated pose (red) against the photo's own outline (green), and
the report gives the outline overlap (IoU) per photo. Run it with the FoundPose environment's
Python:

    set CONDA_EXE=%USERPROFILE%\\anaconda3\\Scripts\\conda.exe
    D:\\conda_envs\\foundpose_repo\\python.exe scripts/run_on_photos.py --mesh output/starsat/starsat2_hunyuan.glb

The photos must show the object on a pure black background (the mask is taken from that), as in
data/starsat. They are placed unscaled-in-shape into the 1024x1024 canvas the pipeline expects,
with the principal point at its centre. The focal length is unknown for these photos, so it is
a guess of ``focal_factor`` x photo width (1.0 is the guess satpose/config.py uses for them);
try other values to see how much the answer depends on it.
"""
import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes, binary_opening, label

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from satpose import config  # noqa: E402
from satpose.objects import crease_edges as feature_edges  # noqa: E402
from satpose.pipeline import run_official_foundpose  # noqa: E402
from satpose.postprocess import load_mesh_triangles  # noqa: E402
from satpose.refinement import RefinementConfig, SilhouetteMatcher  # noqa: E402

CANVAS = 1024
BLACK_LEVEL = 9          # background is exactly 0; the darkest part of the object is ~20


def object_mask(rgb: np.ndarray) -> np.ndarray:
    """Mask of an object on a black background: the largest solid blob above the noise."""
    mask = binary_fill_holes(binary_closing(rgb.max(axis=2) > BLACK_LEVEL, iterations=3))
    mask = binary_opening(mask, iterations=2)
    labels, _ = label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def embed_photo(path: str, out_dir: str) -> Tuple[str, str, np.ndarray]:
    """Writes the photo and its RGBA cut-out onto a 1024x1024 canvas (same scale both ways,
    photo centre at the canvas centre). Returns both paths and the mask on the canvas."""
    photo = np.array(Image.open(path).convert("RGB"))
    mask = object_mask(photo)
    height, width = mask.shape
    scale = CANVAS / width
    new_height = int(round(height * scale))
    top = (CANVAS - new_height) // 2
    stem = os.path.splitext(os.path.basename(path))[0]

    def place(image: Image.Image, mode: str) -> Image.Image:
        canvas = Image.new(mode, (CANVAS, CANVAS))
        canvas.paste(image.resize((CANVAS, new_height), Image.LANCZOS), (0, top))
        return canvas

    rgb = photo * mask[..., None]
    query = place(Image.fromarray(rgb), "RGB")
    cutout = place(Image.fromarray(np.dstack([rgb, mask.astype(np.uint8) * 255]), "RGBA"), "RGBA")
    query_path = os.path.join(out_dir, f"{stem}.png")
    cutout_path = os.path.join(out_dir, f"{stem}_cutout.png")
    query.save(query_path)
    cutout.save(cutout_path)
    return query_path, cutout_path, np.array(cutout)[..., 3] > 127


def project(points: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    cam = points @ R.T + t
    return np.stack([K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2],
                     K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]], axis=1)


def overlay_image(photo_path: str, mask: np.ndarray, matcher: SilhouetteMatcher, edges: np.ndarray,
                  R: np.ndarray, t: np.ndarray, K: np.ndarray, square: bool = False
                  ) -> Tuple[Image.Image, float]:
    """The photo with its outline (green) and the mesh at (R, t): outline (red) and creases
    (orange), cropped around the object (to a square if ``square``). Also returns the hard IoU
    of the two silhouettes at full canvas resolution."""
    rendered = matcher.silhouette(R, t) > 0.5
    iou = float((rendered & mask).sum() / max((rendered | mask).sum(), 1))
    image = Image.open(photo_path).convert("RGB")
    pixels = np.array(image)
    pixels[mask & ~binary_erosion(mask, iterations=2)] = (60, 255, 120)
    pixels[rendered & ~binary_erosion(rendered, iterations=2)] = (255, 70, 70)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for a, b in edges.reshape(-1, 2, 3):
        (x0, y0), (x1, y1) = project(np.array([a, b]), R, t, K)
        draw.line([(x0, y0), (x1, y1)], fill=(255, 150, 40), width=1)
    ys, xs = np.nonzero(mask | rendered)
    pad = 40
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    if square:                      # same framing for every photo; PIL pads outside the canvas
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, x1, y0, y1 = cx - side // 2, cx + side // 2, cy - side // 2, cy + side // 2
    else:
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, CANVAS), min(y1, CANVAS)
    return image.crop((int(x0), int(y0), int(x1), int(y1))), iou


def draw_overlay(photo_path: str, mask: np.ndarray, matcher: SilhouetteMatcher, edges: np.ndarray,
                 R: np.ndarray, t: np.ndarray, K: np.ndarray, label_text: str, out_path: str) -> float:
    """Saves ``overlay_image`` at 2x with a label; returns the outline IoU."""
    image, iou = overlay_image(photo_path, mask, matcher, edges, R, t, K)
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    ImageDraw.Draw(image).text((10, 8), f"{label_text}  outline IoU {iou:.2f}", fill=(255, 255, 255))
    image.save(out_path)
    return iou


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", required=True, help="mesh of the object (glb/obj/ply/stl)")
    parser.add_argument("--photos", nargs="+", default=sorted(glob.glob("data/starsat/*.jpeg")))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--focal-factor", type=float, default=1.0,
                        help="focal length as a multiple of the photo width (a guess)")
    args = parser.parse_args()

    first = Image.open(args.photos[0])
    width = first.size[0]
    focal = args.focal_factor * CANVAS          # photo width maps to the canvas width
    K = np.array([[focal, 0.0, CANVAS / 2], [0.0, focal, CANVAS / 2], [0.0, 0.0, 1.0]])
    config.CAMERA_K = K                         # the pipeline reads intrinsics from here
    out_dir = os.path.abspath(args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.mesh)), f"photos_focal{args.focal_factor:g}"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"photo width {width}px -> canvas {CANVAS}px, fx = fy = {focal:.0f} (factor {args.focal_factor:g})")

    queries, cutouts, masks = [], [], []
    for path in args.photos:
        query, cutout, mask = embed_photo(path, out_dir)
        queries.append(query)
        cutouts.append(cutout)
        masks.append(mask)

    poses = run_official_foundpose(os.path.abspath(args.mesh), queries, out_dir, save_mask=False,
                                   save_axes=False, nobg_paths=cutouts)

    bop_root = os.path.join(out_dir, "bop_datasets")
    triangles = load_mesh_triangles(os.path.join(bop_root, "custom", "models", "obj_000001.ply"))
    edges = feature_edges(triangles)
    overlay_dir = os.path.join(out_dir, "wireframe")
    os.makedirs(overlay_dir, exist_ok=True)
    report: List[dict] = []
    for pose in poses:
        index = int(pose["img_id"])
        R = np.array(pose["R"], dtype=np.float64).reshape(3, 3)
        t = np.array(pose["t"], dtype=np.float64).reshape(3)
        matcher = SilhouetteMatcher(triangles, masks[index], K, RefinementConfig(resolution=CANVAS, supersample=2))
        name = os.path.splitext(os.path.basename(args.photos[index]))[0]
        iou = draw_overlay(queries[index], masks[index], matcher, edges, R, t, K, name,
                           os.path.join(overlay_dir, f"{name}.png"))
        report.append({"photo": name, "iou": iou, "R": R.tolist(), "t": t.tolist(),
                       "inliers": pose.get("inliers_est")})
        print(f"{name}: outline IoU {iou:.3f}, FoundPose inliers {pose.get('inliers_est')}")
    with open(os.path.join(out_dir, "photo_report.json"), "w") as f:
        json.dump({"focal_factor": args.focal_factor, "photos": report}, f, indent=1)
    print(f"overlays in {overlay_dir}")


if __name__ == "__main__":
    main()
