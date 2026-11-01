"""Contribution 1 -- viewpoint bookkeeping for the pose-guided closed loop.

While the chaser moves around the target, every estimated pose tells us *which side* of
the target a frame shows. This module turns poses into viewing directions, decides which
frames are worth keeping as keyframes (they show a side we have not covered yet), and maps
the keyframes onto the named input slots of Hunyuan3D-2mv (front / left / back / right).

Conventions
-----------
* A pose (R, t) maps object coordinates to camera coordinates: ``x_cam = R @ x_obj + t``.
  This is the convention FoundPose and ``satpose.visualize`` use.
* The object origin is the mesh centroid, so the camera direction in the object frame is
  simply the camera centre ``-R.T @ t``, normalised.
* The *first* accepted keyframe defines the "front" of the object (azimuth 0), because the
  initial single-view mesh is generated from that frame. Azimuth grows by the right-hand
  rule about ``up_axis``, so +90 deg is the object's "left" side as seen by the camera.
* ``up_axis`` is the mesh's vertical axis. The default (+Y) is the usual glTF convention;
  check it against a real Hunyuan3D mesh before trusting the slot assignment.

This module is pure numpy and has no dependency on the reconstruction or pose back-ends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

# Hunyuan3D-2mv input slots and the azimuth (deg) each one expects, relative to "front".
VIEW_SLOT_AZIMUTH_DEG: Dict[str, float] = {
    "front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0,
}
# The slots that satpose.reconstruction.multiview.reconstruct_multiview requires.
REQUIRED_SLOTS: Tuple[str, ...] = ("front", "left", "back")

DEFAULT_UP_AXIS: Tuple[float, float, float] = (0.0, 1.0, 0.0)

# Outcomes of KeyframeSelector.consider().
ADDED = "added"
REPLACED = "replaced"
REJECTED = "rejected"

_EPS = 1e-9


def _unit(vector: Sequence[float]) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < _EPS:
        raise ValueError("cannot normalise a zero-length vector")
    return vector / norm


def camera_direction_in_object(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Unit vector from the object origin towards the camera, in object coordinates."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return _unit(-R.T @ t)


def angular_distance_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """Angle in degrees between two directions."""
    cosine = float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def azimuth_elevation_deg(direction: Sequence[float], reference_direction: Sequence[float],
                          up_axis: Sequence[float] = DEFAULT_UP_AXIS) -> Tuple[float, float]:
    """Azimuth in [0, 360) relative to ``reference_direction`` and elevation in [-90, 90].

    Azimuth is the signed angle, about ``up_axis``, from the reference direction to
    ``direction`` (right-hand rule). Elevation is measured from the horizontal plane. When
    ``direction`` points straight along the up axis the azimuth is undefined and 0 is
    returned.
    """
    up = _unit(up_axis)
    direction = _unit(direction)
    reference = _unit(reference_direction)

    ref_flat = reference - np.dot(reference, up) * up
    if np.linalg.norm(ref_flat) < 1e-6:
        raise ValueError("reference direction is parallel to the up axis")
    elevation = float(np.degrees(np.arcsin(np.clip(np.dot(direction, up), -1.0, 1.0))))

    dir_flat = direction - np.dot(direction, up) * up
    if np.linalg.norm(dir_flat) < 1e-6:
        return 0.0, elevation
    azimuth = np.degrees(np.arctan2(np.dot(up, np.cross(ref_flat, dir_flat)),
                                    np.dot(ref_flat, dir_flat)))
    return float(azimuth % 360.0), elevation


def circular_difference_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees, in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass(eq=False)
class Keyframe:
    """A frame kept for reconstruction, with its pose and where it looks from."""
    frame_id: Hashable
    R: np.ndarray
    t: np.ndarray
    quality: float
    direction: np.ndarray
    azimuth_deg: float
    elevation_deg: float
    payload: Any = None  # whatever the reconstruction needs, e.g. an image path or array


@dataclass
class KeyframeSelector:
    """Keeps a small set of keyframes that cover the target from well-separated sides.

    * A frame is **added** when its viewing direction is at least ``min_separation_deg`` away
      from every stored keyframe.
    * Otherwise it **replaces** the nearest keyframe if its quality is better by more than
      ``replace_margin``. The first keyframe (the "front" reference) is never replaced, so
      the object frame stays consistent between mesh rebuilds.
    * Frames with ``quality < min_quality`` are **rejected**, so a bad pose estimate can
      not pollute the reconstruction.
    """
    min_separation_deg: float = 30.0
    min_quality: float = 0.0
    replace_margin: float = 0.0
    up_axis: Tuple[float, float, float] = DEFAULT_UP_AXIS
    keyframes: List[Keyframe] = field(default_factory=list)

    def consider(self, frame_id: Hashable, R: np.ndarray, t: np.ndarray, quality: float,
                 payload: Any = None) -> str:
        """Offers a posed frame. Returns ADDED, REPLACED or REJECTED."""
        if quality < self.min_quality:
            return REJECTED

        direction = camera_direction_in_object(R, t)
        if not self.keyframes:
            self.keyframes.append(self._make_keyframe(frame_id, R, t, quality, direction,
                                                     payload, reference=direction))
            return ADDED

        reference = self.keyframes[0].direction
        nearest_index, nearest_distance = self._nearest(direction)
        keyframe = self._make_keyframe(frame_id, R, t, quality, direction, payload, reference)

        if nearest_distance >= self.min_separation_deg:
            self.keyframes.append(keyframe)
            return ADDED
        if (nearest_index != 0
                and quality > self.keyframes[nearest_index].quality + self.replace_margin):
            self.keyframes[nearest_index] = keyframe
            return REPLACED
        return REJECTED

    def _make_keyframe(self, frame_id, R, t, quality, direction, payload, reference) -> Keyframe:
        azimuth, elevation = azimuth_elevation_deg(direction, reference, self.up_axis)
        return Keyframe(frame_id, np.array(R, dtype=np.float64), np.array(t, dtype=np.float64),
                        float(quality), direction, azimuth, elevation, payload)

    def _nearest(self, direction: np.ndarray) -> Tuple[int, float]:
        distances = [angular_distance_deg(direction, k.direction) for k in self.keyframes]
        index = int(np.argmin(distances))
        return index, distances[index]

    def novelty_deg(self, R: np.ndarray, t: np.ndarray) -> float:
        """Angular distance from a pose's viewing direction to the closest keyframe."""
        if not self.keyframes:
            return 180.0
        return self._nearest(camera_direction_in_object(R, t))[1]


def assign_slots(keyframes: Sequence[Keyframe],
                 tolerance_deg: float = 45.0) -> Dict[str, Keyframe]:
    """Maps keyframes to Hunyuan3D-2mv slots by closest azimuth.

    Each keyframe fills at most one slot and each slot at most one keyframe; the best
    (smallest-error) pairs are matched first. Slots with no keyframe within
    ``tolerance_deg`` of their expected azimuth stay empty.
    """
    candidates = []
    for slot, slot_azimuth in VIEW_SLOT_AZIMUTH_DEG.items():
        for keyframe in keyframes:
            error = circular_difference_deg(keyframe.azimuth_deg, slot_azimuth)
            if error <= tolerance_deg:
                candidates.append((error, slot, id(keyframe), keyframe))
    candidates.sort(key=lambda item: item[:3])

    assignment: Dict[str, Keyframe] = {}
    used = set()
    for _, slot, keyframe_key, keyframe in candidates:
        if slot not in assignment and keyframe_key not in used:
            assignment[slot] = keyframe
            used.add(keyframe_key)
    return assignment


def missing_slots(assignment: Dict[str, Keyframe],
                  required: Sequence[str] = REQUIRED_SLOTS) -> List[str]:
    """Required slots that have no keyframe yet."""
    return [slot for slot in required if slot not in assignment]


def azimuth_coverage_deg(keyframes: Sequence[Keyframe]) -> float:
    """Degrees of azimuth spanned by the keyframes: 360 minus the largest empty gap."""
    if len(keyframes) < 2:
        return 0.0
    azimuths = np.sort([k.azimuth_deg for k in keyframes])
    gaps = np.diff(np.append(azimuths, azimuths[0] + 360.0))
    return float(360.0 - gaps.max())
