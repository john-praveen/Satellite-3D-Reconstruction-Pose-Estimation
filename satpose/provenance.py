"""Contribution 2 -- observation-aware correspondence weighting.

A generated mesh is only trustworthy where a camera has actually looked at it; everywhere
else the generative prior guessed. This module keeps a per-vertex *observation strength*
that grows every time a vertex is seen in a posed frame, turns it into a trust weight
between ``hallucination_weight`` (never seen) and 1 (well observed), and uses those
weights to down-weight 2D-3D correspondences that land on guessed geometry.

Visibility is computed per vertex, without rendering: a vertex counts as seen when it is
inside the image (and the object mask, if given), faces the camera, and is not hidden
behind a nearer vertex in a coarse depth buffer.

Pose convention matches the rest of the package: ``x_cam = R @ x_obj + t``.

The pose refinement here starts from a pose you already have (for example FoundPose's
PnP-RANSAC result) and polishes it with the trust weights. Hooking it into the official
FoundPose ``infer.py`` needs that script's correspondences, which it does not export by
default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ObservationConfig:
    """Tunable parameters of the observation model."""
    hallucination_weight: float = 0.2   # trust in a vertex that was never observed
    saturation: float = 1.0             # accumulated strength at which trust reaches ~63%
    min_incidence_cos: float = 0.15     # ignore surfaces seen at a grazing angle
    depth_tolerance: float = 0.03       # occlusion slack, as a fraction of the object radius
    zbuffer_cell_px: int = 4            # depth-buffer cell size in pixels


class ObservationMap:
    """Per-vertex record of how well each part of the mesh has been observed."""

    def __init__(self, vertices: np.ndarray, normals: np.ndarray,
                 config: ObservationConfig = ObservationConfig()):
        vertices = np.asarray(vertices, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or normals.shape != vertices.shape:
            raise ValueError("vertices and normals must both have shape (N, 3)")
        self.vertices = vertices
        self.normals = normals
        self.config = config
        self.strength = np.zeros(len(vertices))
        self._radius = float(np.linalg.norm(vertices, axis=1).max())
        self._tree: Optional[cKDTree] = None

    # -- observation -----------------------------------------------------------------

    def visible_vertices(self, R: np.ndarray, t: np.ndarray, camera_K: np.ndarray,
                         image_size: Tuple[int, int], mask: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns ``(visible, incidence_cos)`` for every vertex.

        ``image_size`` is ``(width, height)``; ``mask`` is an optional boolean (H, W) array
        marking object pixels, so vertices projecting onto background are not counted.
        """
        cfg = self.config
        width, height = image_size
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)

        cam = self.vertices @ R.T + t
        depth = cam[:, 2]
        in_front = depth > 1e-9
        safe_depth = np.where(in_front, depth, 1.0)
        pixels = cam @ np.asarray(camera_K, dtype=np.float64).T
        u = pixels[:, 0] / safe_depth
        v = pixels[:, 1] / safe_depth
        inside = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)

        to_camera = -cam / np.maximum(np.linalg.norm(cam, axis=1, keepdims=True), 1e-9)
        incidence = np.einsum("ij,ij->i", self.normals @ R.T, to_camera)
        facing = incidence > cfg.min_incidence_cos

        # Coarse depth buffer: a vertex is hidden if a clearly nearer one shares its cell.
        cell = cfg.zbuffer_cell_px
        cols = (width + cell - 1) // cell
        rows = (height + cell - 1) // cell
        cell_index = (np.clip(v, 0, height - 1) // cell).astype(np.int64) * cols \
            + (np.clip(u, 0, width - 1) // cell).astype(np.int64)
        nearest = np.full(rows * cols, np.inf)
        candidates = inside & facing
        np.minimum.at(nearest, cell_index[candidates], depth[candidates])
        unoccluded = depth <= nearest[cell_index] + cfg.depth_tolerance * self._radius

        visible = candidates & unoccluded
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            ui = np.clip(u, 0, width - 1).astype(np.int64)
            vi = np.clip(v, 0, height - 1).astype(np.int64)
            visible &= mask[vi, ui]
        return visible, incidence

    def update(self, R: np.ndarray, t: np.ndarray, camera_K: np.ndarray,
               image_size: Tuple[int, int], mask: Optional[np.ndarray] = None) -> int:
        """Records one posed frame. Returns the number of vertices it observed."""
        visible, incidence = self.visible_vertices(R, t, camera_K, image_size, mask)
        self.strength[visible] += incidence[visible]
        return int(visible.sum())

    # -- trust weights ---------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        """Trust per vertex, from ``hallucination_weight`` (unseen) up towards 1."""
        floor = self.config.hallucination_weight
        return floor + (1.0 - floor) * (1.0 - np.exp(-self.strength / self.config.saturation))

    def observed_fraction(self, threshold: float = 0.5) -> float:
        """Fraction of vertices whose trust is at least ``threshold`` (a coverage measure)."""
        return float(np.mean(self.weights >= threshold))

    def weights_at(self, points_3d: np.ndarray) -> np.ndarray:
        """Trust weight for arbitrary object-frame points, taken from the nearest vertex."""
        if self._tree is None:
            self._tree = cKDTree(self.vertices)
        _, nearest = self._tree.query(np.asarray(points_3d, dtype=np.float64))
        return self.weights[nearest]


def refine_pose_weighted(points_3d: np.ndarray, points_2d: np.ndarray, weights: np.ndarray,
                         camera_K: np.ndarray, R_init: np.ndarray, t_init: np.ndarray,
                         huber_px: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Refines a pose by minimising weighted, robust reprojection error.

    Each correspondence's residual is scaled by ``sqrt(weight)``, so matches on guessed
    geometry pull on the solution less than matches on observed geometry. A Huber loss
    keeps remaining outliers from dominating. Returns ``(R, t)``.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    points_2d = np.asarray(points_2d, dtype=np.float64)
    sqrt_w = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
    camera_K = np.asarray(camera_K, dtype=np.float64)
    if len(points_3d) < 4:
        raise ValueError("need at least 4 correspondences to refine a pose")

    def residuals(params: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(params[:3]).as_matrix()
        cam = points_3d @ rotation.T + params[3:]
        projected = cam @ camera_K.T
        pixels = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-9)
        return ((pixels - points_2d) * sqrt_w).ravel()

    x0 = np.concatenate([Rotation.from_matrix(np.asarray(R_init)).as_rotvec(),
                         np.asarray(t_init, dtype=np.float64).reshape(3)])
    solution = least_squares(residuals, x0, loss="huber", f_scale=huber_px)
    return Rotation.from_rotvec(solution.x[:3]).as_matrix(), solution.x[3:]

