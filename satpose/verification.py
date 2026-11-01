"""End-to-end orientation check on the synthetic box satellite.

``generate``  writes the satellite as an STL, renders it from known poses, and draws the
              ground-truth wireframe on each view (proves the renderer and conventions).
``run_foundpose``  feeds the STL and the renders to the real pipeline (needs the FoundPose
              setup) and returns the path of its ``estimated-poses.json``.
``evaluate``  compares estimated poses with ground truth and draws, per view, the true
              wireframe (green) and the wireframe at the *estimated rotation* (red).
``selftest``  does the same with fake estimates, so the report can be inspected without
              FoundPose.

Only rotation is compared: the pipeline's translation is in arbitrary mesh scale. The red
wireframe therefore uses the estimated rotation with the ground-truth translation.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from satpose import config
from satpose.metrics import rotation_error_deg
from satpose.synthetic import (
    build_satellite, draw_wireframe, render, sample_poses, write_binary_stl,
)

GT_COLOR = (60, 255, 120)
GT_HIDDEN_COLOR = (40, 120, 70)
ESTIMATE_COLOR = (255, 70, 70)
FOUNDPOSE_POSES = os.path.join("foundpose", "bop_datasets", "outputs", "inference",
                               "custom_v1", "1", "estimated-poses.json")


def _font(size: int):
    for loader in (lambda: ImageFont.truetype("arial.ttf", size),
                   lambda: ImageFont.load_default(size=size)):
        try:
            return loader()
        except (OSError, TypeError):
            continue
    return ImageFont.load_default()


def _contact_sheet(image_paths: List[str], out_path: str, columns: int = 4,
                   thumb: int = 420) -> None:
    thumbs = [Image.open(p).convert("RGB").resize((thumb, thumb)) for p in image_paths]
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * thumb), (10, 10, 14))
    for i, image in enumerate(thumbs):
        sheet.paste(image, ((i % columns) * thumb, (i // columns) * thumb))
    sheet.save(out_path)


def _paths(out_dir: str) -> Dict[str, str]:
    return {"stl": os.path.join(out_dir, "satellite.stl"),
            "renders": os.path.join(out_dir, "renders"),
            "gt": os.path.join(out_dir, "ground_truth.json"),
            "wire_gt": os.path.join(out_dir, "wireframe_gt")}


def generate(out_dir: str, num_views: int = 12, distance: float = 4.0) -> Dict[str, str]:
    """Writes the STL, the renders, ground truth, and ground-truth wireframe images.

    Images use the camera and resolution from ``satpose/config.py``, so they pass the
    pipeline's own size check. ``distance`` is in mesh units; 4.0 keeps the satellite well
    inside the pipeline's depth range and the frame.
    """
    paths = _paths(out_dir)
    for key in ("renders", "wire_gt"):
        os.makedirs(paths[key], exist_ok=True)
    satellite = build_satellite()
    write_binary_stl(paths["stl"], satellite.triangles, satellite.normals)

    size = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    views, wire_images = [], []
    for index, (R, t) in enumerate(sample_poses(num_views, distance)):
        rgb, mask, depth = render(satellite, R, t, config.CAMERA_K, size)
        name = f"view_{index:02d}"
        Image.fromarray(rgb).save(os.path.join(paths["renders"], f"{name}.png"))
        cutout = np.dstack([rgb, mask.astype(np.uint8) * 255])
        Image.fromarray(cutout, "RGBA").save(os.path.join(paths["renders"], f"{name}_cutout.png"))

        wire = Image.fromarray(rgb)
        draw_wireframe(wire, satellite, R, t, config.CAMERA_K, depth, GT_COLOR,
                       hidden_color=GT_HIDDEN_COLOR)
        wire_path = os.path.join(paths["wire_gt"], f"{name}.png")
        wire.save(wire_path)
        wire_images.append(wire_path)
        views.append({"image": f"renders/{name}.png", "cutout": f"renders/{name}_cutout.png",
                      "R": R.tolist(), "t": t.tolist()})

    with open(paths["gt"], "w") as f:
        json.dump({"image_size": list(size), "camera_K": config.CAMERA_K.tolist(),
                   "distance": distance, "views": views}, f, indent=1)
    _contact_sheet(wire_images, os.path.join(out_dir, "gt_contact_sheet.png"))
    return paths


def load_ground_truth(out_dir: str) -> dict:
    with open(_paths(out_dir)["gt"]) as f:
        return json.load(f)


def run_foundpose(out_dir: str) -> str:
    """Runs the real FoundPose pipeline on the synthetic data. Returns its poses file path."""
    from satpose.pipeline import run_official_foundpose  # needs the FoundPose setup

    truth = load_ground_truth(out_dir)
    queries = [os.path.join(out_dir, v["image"]) for v in truth["views"]]
    cutouts = [os.path.join(out_dir, v["cutout"]) for v in truth["views"]]
    run_official_foundpose(_paths(out_dir)["stl"], queries, os.path.join(out_dir, "foundpose"),
                           save_mask=False, save_axes=False, nobg_paths=cutouts)
    return os.path.join(out_dir, FOUNDPOSE_POSES)


def evaluate(out_dir: str, poses_path: Optional[str] = None, eval_name: str = "wireframe_eval",
             threshold_deg: float = 10.0) -> dict:
    """Compares estimated rotations with ground truth and writes overlays and a report.

    Per view, ``<eval_name>/view_XX.png`` shows the true wireframe in green and the
    wireframe at the estimated rotation in red; a perfect estimate hides the green under
    the red. Also writes ``report.csv`` and ``contact_sheet.png``. Returns a summary dict.
    """
    poses_path = poses_path or os.path.join(out_dir, FOUNDPOSE_POSES)
    with open(poses_path) as f:
        estimates = {int(p["img_id"]): np.array(p["R"], dtype=np.float64).reshape(3, 3)
                     for p in json.load(f)}
    truth = load_ground_truth(out_dir)
    camera_K = np.array(truth["camera_K"])
    size = tuple(truth["image_size"])
    satellite = build_satellite()
    eval_dir = os.path.join(out_dir, eval_name)
    os.makedirs(eval_dir, exist_ok=True)

    rows, overlay_paths = [], []
    for index, view in enumerate(truth["views"]):
        R_true, t_true = np.array(view["R"]), np.array(view["t"])
        R_est = estimates.get(index)
        error = None if R_est is None else rotation_error_deg(R_est, R_true)

        _, _, depth = render(satellite, R_true, t_true, camera_K, size)
        image = Image.open(os.path.join(out_dir, view["image"])).convert("RGB")
        draw_wireframe(image, satellite, R_true, t_true, camera_K, depth, GT_COLOR,
                       width=5, hidden_color=GT_HIDDEN_COLOR)
        if R_est is not None:
            draw_wireframe(image, satellite, R_est, t_true, camera_K, None, ESTIMATE_COLOR,
                           width=3)
        label = f"view {index:02d}: " + ("no estimate" if error is None
                                         else f"rotation error {error:.1f} deg")
        ImageDraw.Draw(image).text((20, 16), label, fill=(255, 255, 255), font=_font(40))
        overlay = os.path.join(eval_dir, f"view_{index:02d}.png")
        image.save(overlay)
        overlay_paths.append(overlay)
        rows.append((index, error))

    _contact_sheet(overlay_paths, os.path.join(eval_dir, "contact_sheet.png"))
    with open(os.path.join(eval_dir, "report.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["view", "rotation_error_deg"])
        writer.writerows((i, "" if e is None else f"{e:.3f}") for i, e in rows)

    errors = [e for _, e in rows if e is not None]
    return {
        "views": len(rows),
        "estimated": len(errors),
        "errors_deg": dict(rows),
        "median_deg": float(np.median(errors)) if errors else None,
        "within_threshold": sum(e <= threshold_deg for e in errors),
        "threshold_deg": threshold_deg,
        "eval_dir": eval_dir,
    }


def selftest(out_dir: str, noise_deg: float = 3.0, flipped_view: int = 5) -> dict:
    """Evaluates fake estimates (truth + small noise, one view flipped 180 deg about the
    satellite's up axis) to show what a good and a bad result look like."""
    truth = load_ground_truth(out_dir)
    rng = np.random.default_rng(0)
    flip = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    fake = []
    for index, view in enumerate(truth["views"]):
        R = np.array(view["R"])
        R = R @ flip if index == flipped_view else \
            Rotation.from_rotvec(rng.normal(0, np.radians(noise_deg) / 3, 3)).as_matrix() @ R
        fake.append({"img_id": index, "R": R.tolist(), "t": view["t"]})
    fake_path = os.path.join(out_dir, "selftest_poses.json")
    with open(fake_path, "w") as f:
        json.dump(fake, f)
    return evaluate(out_dir, fake_path, eval_name="wireframe_selftest")


def format_summary(summary: dict) -> str:
    lines = [f"{'view':>5} {'rotation error':>16}"]
    for index, error in summary["errors_deg"].items():
        lines.append(f"{index:5d} {'no estimate' if error is None else f'{error:12.2f} deg':>16}")
    lines.append(f"\nmedian rotation error: "
                 f"{'n/a' if summary['median_deg'] is None else format(summary['median_deg'], '.2f') + ' deg'}")
    lines.append(f"views within {summary['threshold_deg']:.0f} deg: "
                 f"{summary['within_threshold']} of {summary['views']}")
    lines.append(f"overlays: {summary['eval_dir']}")
    return "\n".join(lines)
