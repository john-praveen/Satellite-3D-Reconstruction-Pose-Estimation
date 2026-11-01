"""Shared test utilities: synthetic poses and meshes."""
import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def look_at_pose(direction, distance=2.0):
    """Pose (R, t) of a camera sitting at ``direction * distance`` looking at the origin.

    Uses the package convention ``x_cam = R @ x_obj + t`` with an OpenCV camera
    (x right, y down, z forward).
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    forward = -direction
    right = np.cross(forward, UP)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    t = -R @ (direction * distance)
    return R, t


def pose_from_azimuth(azimuth_deg, elevation_deg=10.0, distance=2.0):
    """Pose of a camera at the given azimuth about +Y (0 deg = along +Z, 90 deg = +X)."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    direction = [np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)]
    return look_at_pose(direction, distance)


def fibonacci_sphere(count=4000):
    """Roughly uniform points on the unit sphere (they double as outward normals)."""
    index = np.arange(count) + 0.5
    phi = np.arccos(1 - 2 * index / count)
    theta = np.pi * (1 + 5 ** 0.5) * index
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
