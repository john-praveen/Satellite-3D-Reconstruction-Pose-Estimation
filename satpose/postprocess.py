"""Improves FoundPose's poses by refining every candidate against the photo's silhouette
(see refinement.py).

Reads what the patched official ``infer.py`` leaves behind -- ``estimated-poses.json`` (its
best coarse pose per image) and ``candidates/<im_id>.json`` (every coarse candidate) -- plus
the background-removed cut-out of each photo, and writes ``estimated-poses-refined.json``.

Modes (``improve_poses(mode=...)``): ``"none"`` keeps FoundPose's pose; ``"full"`` refines every
candidate and keeps the best (images without candidates refine FoundPose's own pose).
"""
from __future__ import annotations

import copy
import json
import os
from typing import List, Sequence

import numpy as np
from PIL import Image

from satpose.refinement import (
    Candidate, RefinementConfig, decimate_triangles, select_best_pose,
)

MODES = ("none", "full")
MAX_OUTLINE_TRIANGLES = 4000  # larger (e.g. generated) meshes are simplified first


def load_mask(cutout_path: str, alpha_threshold: int = 10) -> np.ndarray:
    """Boolean object mask from an RGBA cut-out's alpha channel."""
    return np.asarray(Image.open(cutout_path).convert("RGBA"))[:, :, 3] > alpha_threshold


def load_mesh_triangles(ply_path: str) -> np.ndarray:
    """Triangles (T, 3, 3) of a mesh file, simplified if it is large."""
    import trimesh  # available in the FoundPose environment

    mesh = trimesh.load(ply_path, force="mesh")
    if len(mesh.faces) > MAX_OUTLINE_TRIANGLES:
        return decimate_triangles(mesh.vertices, mesh.faces)
    return np.asarray(mesh.vertices)[np.asarray(mesh.faces)]


def load_candidates(candidates_dir: str, image_id: int) -> List[Candidate]:
    """Coarse pose candidates saved for one image (empty if there are none)."""
    path = os.path.join(candidates_dir, f"{int(image_id):06d}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [Candidate(np.array(c["R"], dtype=np.float64).reshape(3, 3),
                          np.array(c["t"], dtype=np.float64).reshape(3), float(c["inliers"]))
                for c in json.load(f)]


def _pose_entry(original: dict, R: np.ndarray, t: np.ndarray, iou: float) -> dict:
    entry = copy.deepcopy(original)
    entry["R"] = np.asarray(R).tolist()
    entry["t"] = [[float(v)] for v in np.asarray(t).reshape(3)]
    entry["silhouette_iou"] = iou
    return entry


def improve_poses(poses: Sequence[dict], candidates_dir: str, cutout_paths: Sequence[str],
                  mesh_triangles: np.ndarray, camera_K: np.ndarray, mode: str = "full",
                  config: RefinementConfig = RefinementConfig()) -> List[dict]:
    """Returns improved copies of FoundPose's pose entries (same schema, plus ``silhouette_iou``).

    ``poses`` are FoundPose's ``estimated-poses.json`` entries; ``cutout_paths[i]`` is the
    cut-out of image ``i``; ``mesh_triangles`` are in the units of the poses' translation.
    Images with no mask or no candidates keep their original pose.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "none":
        return [copy.deepcopy(p) for p in poses]

    improved = []
    for pose in poses:
        image_id = int(pose["img_id"])
        mask = load_mask(cutout_paths[image_id])
        original = Candidate(np.array(pose["R"], dtype=np.float64).reshape(3, 3),
                             np.array(pose["t"], dtype=np.float64).reshape(3),
                             float(pose.get("inliers_est") or 0.0))
        candidates = load_candidates(candidates_dir, image_id)

        best, _ = select_best_pose(candidates or [original], mesh_triangles, mask, camera_K, config)
        improved.append(_pose_entry(pose, best.R, best.t, best.iou))
    return improved


def save_poses(poses: Sequence[dict], path: str) -> str:
    with open(path, "w") as f:
        json.dump(list(poses), f)
    return path
