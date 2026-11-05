"""Silhouette-based selection and refinement of coarse pose candidates.

FoundPose matches the photo to pre-rendered templates, so its pose is only as fine as the
template spacing, and when the object looks alike from two sides it can pick the wrong one.
This module compares the mesh's *outline* at a candidate pose with the photo's object mask:

* ``refine_pose`` nudges a pose (rotation and translation) to maximise the overlap.
* ``select_best_pose`` refines every candidate and keeps the one whose outline fits best,
  which rejects candidates that are a flipped copy of the right answer when the outline
  differs.

An outline cannot tell apart two poses with the same silhouette (e.g. looking at a symmetric
body from the front or from the back), so it complements FoundPose's appearance matching
instead of replacing it. Everything is numpy / scipy / PIL; units follow the mesh and ``t``
(for the pipeline, millimetres) and ``camera_K`` must be in pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.optimize import Bounds, minimize
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RefinementConfig:
    """Tunable parameters of the silhouette comparison."""
    resolution: int = 192             # longest image side used for the comparison, in pixels
    blur_sigma_px: float = 2.0        # softens masks so the overlap changes smoothly with pose
    supersample: int = 4              # outline drawn this many times larger, then averaged down
    max_rotation_deg: float = 20.0    # how far refinement may move a candidate's rotation
    max_translation_frac: float = 0.15  # ... and its translation, as a fraction of the distance
    max_iterations: int = 6           # optimiser sweeps
    inlier_weight: float = 0.2        # bonus for candidates FoundPose matched well (see below)


@dataclass(frozen=True)
class Candidate:
    """A coarse pose candidate (object to camera) with FoundPose's inlier count."""
    R: np.ndarray
    t: np.ndarray
    inliers: float = 0.0


@dataclass(frozen=True)
class ScoredPose:
    """A pose and how well its outline overlaps the photo's mask (1 = perfect)."""
    R: np.ndarray
    t: np.ndarray
    iou: float
    inliers: float = 0.0
    score: float = 0.0   # what candidates are ranked by: iou plus the inlier bonus


class SilhouetteMatcher:
    """Renders a mesh outline and scores it against a fixed target mask."""

    def __init__(self, triangles: np.ndarray, mask: np.ndarray, camera_K: np.ndarray,
                 config: RefinementConfig = RefinementConfig()):
        mask = np.asarray(mask, dtype=bool)
        height, width = mask.shape
        scale = config.resolution / max(height, width)
        self.size = (max(int(round(width * scale)), 8), max(int(round(height * scale)), 8))
        # Intrinsics of the small image. Pixel i of an image shrunk by `scale` is centred on
        # source coordinate (i + 0.5) / scale - 0.5, so the principal point moves by
        # (scale - 1) / 2 beyond the plain scaling; without it the object sits a fraction
        # of a pixel off.
        K = np.asarray(camera_K, dtype=np.float64)
        self.camera_K = np.array([[scale * K[0, 0], 0.0, scale * K[0, 2] + (scale - 1.0) / 2.0],
                                  [0.0, scale * K[1, 1], scale * K[1, 2] + (scale - 1.0) / 2.0],
                                  [0.0, 0.0, 1.0]])
        self.triangles = np.asarray(triangles, dtype=np.float64)
        self.config = config
        small = Image.fromarray(mask.astype(np.uint8) * 255).resize(self.size, Image.BILINEAR)
        self._target = gaussian_filter(np.asarray(small, dtype=np.float32) / 255.0,
                                       config.blur_sigma_px)

    def silhouette(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Filled outline of the mesh at pose (R, t) as a float mask in [0, 1].

        PIL fills every pixel a polygon touches, which grows a shape by about half a pixel
        per side; at this resolution that made fitted objects ~1% too far away. Drawing
        ``supersample`` times larger and averaging down cuts that bias by a factor of four.
        """
        cam = self.triangles @ np.asarray(R).T + np.asarray(t)
        in_front = (cam[:, :, 2] > 1e-6).all(axis=1)
        cam = cam[in_front]
        factor = max(int(self.config.supersample), 1)
        width, height = self.size
        image = Image.new("L", (width * factor, height * factor), 0)
        if len(cam):
            fx, fy, cx, cy = (self.camera_K[0, 0], self.camera_K[1, 1],
                              self.camera_K[0, 2], self.camera_K[1, 2])
            # +0.5: PIL addresses pixel corners, the intrinsics pixel centres.
            u = (fx * cam[:, :, 0] / cam[:, :, 2] + cx + 0.5) * factor
            v = (fy * cam[:, :, 1] / cam[:, :, 2] + cy + 0.5) * factor
            draw = ImageDraw.Draw(image)
            for tri_u, tri_v in zip(u, v):
                draw.polygon([(tri_u[0], tri_v[0]), (tri_u[1], tri_v[1]), (tri_u[2], tri_v[2])],
                             fill=255)
        if factor > 1:
            image = image.resize((width, height), Image.BOX)
        return np.asarray(image, dtype=np.float32) / 255.0

    def iou(self, R: np.ndarray, t: np.ndarray) -> float:
        """Soft intersection-over-union between the mesh outline at (R, t) and the target."""
        rendered = gaussian_filter(self.silhouette(R, t), self.config.blur_sigma_px)
        union = np.maximum(rendered, self._target).sum()
        return float(np.minimum(rendered, self._target).sum() / union) if union > 0 else 0.0


def refine_pose(matcher: SilhouetteMatcher, R0: np.ndarray, t0: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Moves a pose locally to maximise outline overlap. Returns ``(R, t, iou)``.

    Rotation is a small camera-frame rotation applied to ``R0``; translation is a shift of
    ``t0`` scaled by the object distance. Both are bounded by the config, so refinement
    cannot wander off to a different pose.
    """
    config = matcher.config
    R0, t0 = np.asarray(R0, dtype=np.float64), np.asarray(t0, dtype=np.float64).reshape(3)
    shift_scale = np.linalg.norm(t0)
    rotation_limit = np.radians(config.max_rotation_deg)

    def pose_from(params: np.ndarray):
        return (Rotation.from_rotvec(params[:3]).as_matrix() @ R0,
                t0 + params[3:] * shift_scale)

    limits = np.array([rotation_limit] * 3 + [config.max_translation_frac] * 3)

    def loss(params: np.ndarray) -> float:
        return 1.0 - matcher.iou(*pose_from(params))

    result = minimize(loss, np.zeros(6), method="Powell", bounds=Bounds(-limits, limits),
                      options={"maxiter": config.max_iterations, "xtol": 1e-3, "ftol": 1e-5})
    R, t = pose_from(result.x)
    return R, t, matcher.iou(R, t)


def select_best_pose(candidates: Sequence[Candidate], triangles: np.ndarray, mask: np.ndarray,
                     camera_K: np.ndarray, config: RefinementConfig = RefinementConfig()
                     ) -> Tuple[Optional[ScoredPose], List[ScoredPose]]:
    """Refines every candidate, scores it by outline overlap, and returns the best.

    The ranking score is ``iou + config.inlier_weight * inliers / max_inliers``. The outline
    alone cannot separate a pose from its flipped twin when their silhouettes are nearly
    identical, and it misleads when the mesh is not exact; FoundPose's own matched-point
    count (appearance) is independent of both, so a small weight guards against such
    choices. With an exact mesh it costs almost nothing; with a mesh missing parts it cut
    gross failures from 17.5% to 2.5% in testing. ``inlier_weight = 0`` ranks by outline only.

    Returns ``(best, all_scored)``; ``best`` is None when there are no candidates.
    """
    if not len(candidates):
        return None, []
    matcher = SilhouetteMatcher(triangles, mask, camera_K, config)
    most_inliers = max(max(c.inliers for c in candidates), 1.0)
    scored = []
    for candidate in candidates:
        R, t, iou = refine_pose(matcher, candidate.R, candidate.t)
        score = iou + config.inlier_weight * candidate.inliers / most_inliers
        scored.append(ScoredPose(R, t, iou, candidate.inliers, score=score))
    return max(scored, key=lambda pose: pose.score), scored


def decimate_triangles(vertices: np.ndarray, faces: np.ndarray, cells: int = 40
                       ) -> np.ndarray:
    """Cheap mesh simplification by vertex clustering, for large (e.g. generated) meshes.

    Vertices are snapped to a ``cells``-per-axis grid over the bounding box and merged;
    triangles that collapse are dropped. Returns triangles (T, 3, 3). The outline is kept
    to about one grid cell, which is plenty for silhouette matching.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    step = max((high - low).max() / cells, 1e-12)
    keys = np.floor((vertices - low) / step).astype(np.int64)
    _, first_index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    cell_centres = np.zeros((len(first_index), 3))
    counts = np.bincount(inverse, minlength=len(first_index))
    for axis in range(3):
        cell_centres[:, axis] = np.bincount(inverse, weights=vertices[:, axis],
                                            minlength=len(first_index)) / counts
    merged = inverse[np.asarray(faces)]
    keep = (merged[:, 0] != merged[:, 1]) & (merged[:, 1] != merged[:, 2]) \
        & (merged[:, 0] != merged[:, 2])
    return cell_centres[merged[keep]]
