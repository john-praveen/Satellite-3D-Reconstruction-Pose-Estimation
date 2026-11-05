"""Scores the pose pipeline on synthetic views whose true poses are known.

``generate`` renders views of a test object; ``run_inference`` runs the real FoundPose on them
(once as shipped, once with the candidates the refinement needs); ``write_overlays`` applies the
refinement and draws wireframe overlays for both; ``write_accuracy`` scores rotation and
translation. ``run_inference`` and ``write_overlays`` need the FoundPose environment's Python
(they import trimesh/pyrender via the pipeline).
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from satpose import config, verification
from satpose.metrics import rotation_error_deg, speed_score, translation_components
from satpose.objects import get_object
from satpose.postprocess import improve_poses, load_mesh_triangles, save_poses

NUM_VIEWS = 80
SEED = 7
BASELINE_CONFIG = "custom"          # FoundPose as shipped: 5 templates, best pose only
INFER_CONFIGS = (BASELINE_CONFIG, config.INFER_CONFIG)
MM_SCALE = 500.0   # build_bop_dataset's mesh scale: FoundPose's translation is mesh units x this


def generate(out_dir: str, seed: int = SEED, num_views: int = NUM_VIEWS,
             object_name: str = "satellite") -> None:
    """Renders random views of a test object; a different seed gives an independent set."""
    verification.generate(out_dir, num_views=num_views, seed=seed, object_name=object_name)


def foundpose_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "foundpose")


def run_inference(out_dir: str, infer_configs: Sequence[str] = INFER_CONFIGS,
                  reuse_templates: bool = False) -> None:
    """Runs FoundPose: templates + representation once, then inference for each config.

    With ``reuse_templates`` the templates and representation from an earlier run are kept
    (they only depend on the mesh), so only inference is repeated.
    """
    from satpose.pipeline import run_official_foundpose

    truth = verification.load_ground_truth(out_dir)
    queries = [os.path.join(out_dir, v["image"]) for v in truth["views"]]
    cutouts = [os.path.join(out_dir, v["cutout"]) for v in truth["views"]]
    stl = os.path.join(out_dir, "satellite.stl")
    for position, infer_config in enumerate(infer_configs):
        run_official_foundpose(stl, queries, foundpose_dir(out_dir),
                               skip_templates=reuse_templates or position > 0,
                               save_mask=False, save_axes=False, nobg_paths=cutouts,
                               infer_config=infer_config, refinement="none")


def inference_dir(out_dir: str, infer_config: str) -> str:
    """Where one inference config's results (poses, candidates) were saved."""
    from satpose.pipeline import _inference_dir as locate

    return locate(os.path.join(foundpose_dir(out_dir), "bop_datasets"), infer_config)


def write_overlays(out_dir: str, infer_config: Optional[str] = None,
                   baseline_config: str = BASELINE_CONFIG) -> Dict[str, dict]:
    """Draws wireframe overlays for every view of a set, for our pipeline's result
    (``wireframe_eval/``) and for FoundPose as shipped (``wireframe_baseline/``), so the two
    can be compared, and saves the refined poses next to FoundPose's own.

    Each folder gets per-view images, ``contact_sheet.png`` (first 16 views) and, for larger
    sets, ``worst_views_contact_sheet.png`` (the 8 largest errors).
    """
    infer_config = infer_config or config.INFER_CONFIG
    truth = verification.load_ground_truth(out_dir)
    results = inference_dir(out_dir, infer_config)
    with open(os.path.join(results, "estimated-poses.json")) as f:
        poses = json.load(f)
    triangles = load_mesh_triangles(
        os.path.join(foundpose_dir(out_dir), "bop_datasets", "custom", "models", "obj_000001.ply"))
    cutouts = [os.path.join(out_dir, v["cutout"]) for v in truth["views"]]
    improved = improve_poses(poses, os.path.join(results, "candidates"), cutouts, triangles,
                             np.array(truth["camera_K"]), config.POSE_REFINEMENT)
    refined_path = save_poses(improved, os.path.join(results, "estimated-poses-refined.json"))
    baseline_path = os.path.join(inference_dir(out_dir, baseline_config), "estimated-poses.json")
    return {"pipeline": verification.evaluate(out_dir, refined_path, "wireframe_eval"),
            "as_shipped": verification.evaluate(out_dir, baseline_path, "wireframe_baseline")}


def pose_errors(out_dir: str, poses: Sequence[dict]) -> List[Optional[dict]]:
    """Rotation and translation error of every view (None where there is no estimate).

    FoundPose's pose is the mesh *centroid's* position in the camera, in mesh units times
    ``MM_SCALE``, so it is compared with the true centroid position ``R c + t``. Translation
    errors are in mesh units (the units of the ground truth); the ``*_relative`` entries are
    fractions of the camera distance and ``object_relative`` a fraction of the object's
    diameter.
    """
    truth = verification.load_ground_truth(out_dir)
    centroid = verification.mesh_centroid(os.path.join(out_dir, "satellite.stl"))
    diameter = 2.0 * get_object(truth.get("object", "satellite"))[0].radius
    estimates = {int(p["img_id"]): p for p in poses}
    results: List[Optional[dict]] = []
    for index, view in enumerate(truth["views"]):
        pose = estimates.get(index)
        if pose is None:
            results.append(None)
            continue
        R_true, t_true = np.array(view["R"]), np.array(view["t"])
        centroid_true = R_true @ centroid + t_true
        centroid_est = np.array(pose["t"], dtype=np.float64).reshape(3) / MM_SCALE
        rotation = rotation_error_deg(np.array(pose["R"], dtype=np.float64).reshape(3, 3), R_true)
        translation = translation_components(centroid_est, centroid_true)
        translation["object_relative"] = translation["total"] / diameter
        distance = float(np.linalg.norm(centroid_true))
        translation["lateral_relative"] = translation["lateral"] / distance
        translation["depth_relative"] = translation["depth"] / distance
        results.append({"rotation_deg": rotation, **translation,
                        "speed_score": speed_score(rotation, translation["relative"])})
    return results


def _median(values: Sequence[float]) -> float:
    return float(np.percentile(values, 50.0)) if len(values) else float("nan")


def accuracy_table(label_to_errors: Dict[str, List[Optional[dict]]]) -> List[str]:
    """One text table: rotation and translation accuracy of each labelled result set."""
    columns = ("rot med", "rot mean", "<=5deg", "<=10deg", "t med", "t mean", "lat med",
               "depth med", "t/size", "SPEED med", "SPEED mean")
    lines = [f"{'':<34}" + "".join(f"{c:>11}" for c in columns),
             f"{'':<34}" + "".join(f"{u:>11}" for u in
                                   ("deg", "deg", "views", "views", "% dist", "% dist", "% dist",
                                    "% dist", "% diam", "score", "score")),
             "-" * (34 + 11 * len(columns))]
    for label, errors in label_to_errors.items():
        found = [e for e in errors if e is not None]
        if not found:
            lines.append(f"{label:<34} no estimates")
            continue
        rot = [e["rotation_deg"] for e in found]
        rel = [100 * e["relative"] for e in found]
        cells = (f"{_median(rot):11.2f}{np.mean(rot):11.2f}"
                 f"{100 * np.mean(np.array(rot) <= 5):10.0f}%{100 * np.mean(np.array(rot) <= 10):10.0f}%"
                 f"{_median(rel):11.2f}{np.mean(rel):11.2f}"
                 f"{_median([100 * e['lateral_relative'] for e in found]):11.2f}"
                 f"{_median([100 * e['depth_relative'] for e in found]):11.2f}"
                 f"{_median([100 * e['object_relative'] for e in found]):11.2f}"
                 f"{_median([e['speed_score'] for e in found]):11.3f}"
                 f"{np.mean([e['speed_score'] for e in found]):11.3f}")
        lines.append(f"{label:<34}{cells}")
    return lines


def write_accuracy(out_dir: str, infer_config: Optional[str] = None,
                   baseline_config: str = BASELINE_CONFIG) -> Dict[str, List[Optional[dict]]]:
    """Scores our pipeline (with refinement) and FoundPose as shipped on one evaluated set, and writes
    ``accuracy.csv`` (every view) next to the set's other results. Needs ``write_overlays``
    to have saved the refined poses first."""
    infer_config = infer_config or config.INFER_CONFIG
    refined = os.path.join(inference_dir(out_dir, infer_config), "estimated-poses-refined.json")
    shipped = os.path.join(inference_dir(out_dir, baseline_config), "estimated-poses.json")
    results = {}
    for label, path in (("pipeline", refined), ("as_shipped", shipped)):
        with open(path) as f:
            results[label] = pose_errors(out_dir, json.load(f))
    with open(os.path.join(out_dir, "accuracy.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        fields = ["rotation_deg", "total", "lateral", "depth", "relative", "lateral_relative",
                  "depth_relative", "object_relative", "speed_score"]
        writer.writerow(["variant", "view"] + fields)
        for label, errors in results.items():
            for index, e in enumerate(errors):
                writer.writerow([label, index] + ([""] * len(fields) if e is None
                                                  else [f"{e[k]:.5f}" for k in fields]))
    return results
