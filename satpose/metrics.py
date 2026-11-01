"""Pose-error metrics for evaluating estimated poses against ground truth.

Rotation is the trustworthy output of this pipeline (the reconstructed mesh has an
arbitrary scale), so rotation and pointing error are the headline metrics; translation
error is only meaningful when the mesh scale is known.
"""
from __future__ import annotations

import numpy as np


def rotation_error_deg(R_estimated: np.ndarray, R_truth: np.ndarray) -> float:
    """Geodesic angle in degrees between two rotation matrices."""
    relative = np.asarray(R_estimated, dtype=np.float64).reshape(3, 3).T \
        @ np.asarray(R_truth, dtype=np.float64).reshape(3, 3)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pointing_error_deg(t_estimated: np.ndarray, t_truth: np.ndarray) -> float:
    """Angle in degrees between the estimated and true line of sight to the target.

    Unlike translation error, this does not depend on the (unknown) mesh scale.
    """
    a = np.asarray(t_estimated, dtype=np.float64).reshape(3)
    b = np.asarray(t_truth, dtype=np.float64).reshape(3)
    cosine = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def translation_error(t_estimated: np.ndarray, t_truth: np.ndarray) -> float:
    """Euclidean distance between translations, in the units of the inputs."""
    return float(np.linalg.norm(np.asarray(t_estimated, dtype=np.float64).reshape(3)
                                - np.asarray(t_truth, dtype=np.float64).reshape(3)))
