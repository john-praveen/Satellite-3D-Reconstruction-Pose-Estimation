"""A synthetic box-built satellite with known ground-truth poses, for end-to-end checks.

The satellite is deliberately asymmetric (solar panels of different lengths, a dish on one
face, a mast, small units on other faces) so that exactly one orientation matches a view.
This module builds the geometry, writes it as a binary STL, renders it with a small numpy
z-buffer rasteriser (no GPU, no pyrender), and draws wireframe overlays of a pose.

Pose convention (same as the rest of the package): ``x_cam = R @ x_obj + t`` with an OpenCV
camera (x right, y down, z forward). Lengths are in arbitrary "mesh units".
"""
from __future__ import annotations

import itertools
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

# (centre, size) of every box that makes up the satellite, before centring.
_BOXES: Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]], ...] = (
    ((0.00, 0.00, 0.00), (1.00, 0.80, 1.20)),    # bus
    ((0.60, 0.00, 0.00), (0.20, 0.06, 0.06)),    # right panel arm
    ((1.10, 0.00, 0.00), (0.80, 0.60, 0.04)),    # right solar panel (long)
    ((-0.60, 0.00, 0.00), (0.20, 0.06, 0.06)),   # left panel arm
    ((-0.95, 0.00, 0.00), (0.50, 0.60, 0.04)),   # left solar panel (short)
    ((0.15, 0.10, 0.725), (0.30, 0.30, 0.25)),   # dish base on the +Z face
    ((-0.20, 0.65, 0.10), (0.06, 0.50, 0.06)),   # mast on the +Y face
    ((-0.25, -0.15, -0.68), (0.20, 0.20, 0.16)), # thruster on the -Z face
    ((0.55, -0.25, 0.30), (0.10, 0.18, 0.18)),   # camera on the +X face
    ((0.20, -0.45, -0.20), (0.25, 0.10, 0.25)),  # sensor on the -Y face
)


@dataclass(frozen=True)
class Satellite:
    """Triangle soup with per-face normals, plus the box edges for wireframe drawing."""
    triangles: np.ndarray   # (T, 3, 3) vertex positions
    normals: np.ndarray     # (T, 3) outward unit normals
    edges: np.ndarray       # (E, 2, 3) box edge end points

    @property
    def radius(self) -> float:
        return float(np.linalg.norm(self.triangles.reshape(-1, 3), axis=1).max())


def _box_faces_and_edges(center: np.ndarray, size: np.ndarray):
    half = size / 2.0
    triangles, normals = [], []
    for axis in range(3):
        b, c = (axis + 1) % 3, (axis + 2) % 3  # b x c = axis, so (-,-),(+,-),(+,+),(-,+) is CCW
        for sign in (1.0, -1.0):
            corners = []
            for sb, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = center.copy()
                p[axis] += sign * half[axis]
                p[b] += sb * half[b]
                p[c] += sc * half[c]
                corners.append(p)
            if sign < 0:
                corners.reverse()
            normal = np.zeros(3)
            normal[axis] = sign
            for tri in ((0, 1, 2), (0, 2, 3)):
                triangles.append([corners[i] for i in tri])
                normals.append(normal)
    edges = []
    signs = list(itertools.product((-1, 1), repeat=3))
    for (i, a), (j, b_) in itertools.combinations(enumerate(signs), 2):
        if sum(x != y for x, y in zip(a, b_)) == 1:
            edges.append([center + np.array(a) * half, center + np.array(b_) * half])
    return triangles, normals, edges


def build_satellite() -> Satellite:
    """Builds the satellite, centred on the middle of its bounding box."""
    triangles, normals, edges = [], [], []
    for center, size in _BOXES:
        t, n, e = _box_faces_and_edges(np.array(center, float), np.array(size, float))
        triangles += t
        normals += n
        edges += e
    triangles = np.array(triangles)
    edges = np.array(edges)
    low, high = triangles.reshape(-1, 3).min(axis=0), triangles.reshape(-1, 3).max(axis=0)
    shift = (low + high) / 2.0
    return Satellite(triangles - shift, np.array(normals), edges - shift)


def write_binary_stl(path: str, triangles: np.ndarray, normals: Optional[np.ndarray] = None) -> None:
    """Writes triangles (T, 3, 3) as a binary STL file."""
    triangles = np.asarray(triangles, dtype=np.float32)
    if normals is None:
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as f:
        f.write(b"synthetic satellite".ljust(80, b" "))
        f.write(struct.pack("<I", len(triangles)))
        for normal, tri in zip(np.asarray(normals, dtype=np.float32), triangles):
            f.write(struct.pack("<12fH", *normal, *tri.reshape(-1), 0))


def read_binary_stl(path: str) -> np.ndarray:
    """Reads a binary STL into triangles (T, 3, 3)."""
    with open(path, "rb") as f:
        f.read(80)
        (count,) = struct.unpack("<I", f.read(4))
        data = np.frombuffer(f.read(count * 50), dtype=np.dtype(
            [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return data["v"].astype(np.float64)


# -- poses -----------------------------------------------------------------------------

def look_at_pose(direction: Sequence[float], distance: float, roll_deg: float = 0.0,
                 up: Sequence[float] = (0.0, 1.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Pose of a camera at ``direction * distance`` looking at the origin, rolled about its axis."""
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    forward = -direction
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(forward, right), forward])
    angle = np.radians(roll_deg)
    roll = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                     [np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])
    R = roll @ R
    return R, -R @ (direction * distance)


def sample_poses(count: int = 12, distance: float = 4.0, max_elevation_deg: float = 65.0):
    """Well-spread viewing poses. The first is a plain front view; the rest sweep the sphere
    (azimuth by the golden angle, elevation alternating above and below) with varied roll."""
    poses = [look_at_pose([0.25, 0.2, 1.0], distance)]
    golden = np.pi * (3 - 5 ** 0.5)
    for i in range(1, count):
        azimuth = i * golden * 2.0
        elevation = np.radians(max_elevation_deg) * np.sin(i * 1.7)
        direction = [np.sin(azimuth) * np.cos(elevation), np.sin(elevation),
                     np.cos(azimuth) * np.cos(elevation)]
        poses.append(look_at_pose(direction, distance, roll_deg=30.0 * np.sin(i * 2.3)))
    return poses


# -- rendering -------------------------------------------------------------------------

def render(satellite: Satellite, R: np.ndarray, t: np.ndarray, camera_K: np.ndarray,
           image_size: Tuple[int, int]):
    """Rasterises the satellite with flat grey shading.

    Returns ``(rgb, mask, depth)``: a (H, W, 3) uint8 image on a dark background, a boolean
    object mask, and a float depth map (``inf`` where there is no surface).
    """
    width, height = image_size
    R, t = np.asarray(R, float), np.asarray(t, float)
    cam = satellite.triangles @ R.T + t
    normals_cam = satellite.normals @ R.T
    light = np.array([-0.3, -0.4, -1.0])
    light /= np.linalg.norm(light)

    fx, fy, cx, cy = camera_K[0, 0], camera_K[1, 1], camera_K[0, 2], camera_K[1, 2]
    depth = np.full((height, width), np.inf)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = (18, 20, 28)

    for tri, normal in zip(cam, normals_cam):
        if np.dot(normal, tri.mean(axis=0)) >= 0 or (tri[:, 2] <= 1e-3).any():
            continue  # facing away, or behind the camera
        x = fx * tri[:, 0] / tri[:, 2] + cx
        y = fy * tri[:, 1] / tri[:, 2] + cy
        x0, x1 = max(int(np.floor(x.min())), 0), min(int(np.ceil(x.max())) + 1, width)
        y0, y1 = max(int(np.floor(y.min())), 0), min(int(np.ceil(y.max())) + 1, height)
        if x0 >= x1 or y0 >= y1:
            continue
        px, py = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        if abs(area) < 1e-9:
            continue
        w1 = ((px - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (py - y[0])) / area
        w2 = ((x[1] - x[0]) * (py - y[0]) - (px - x[0]) * (y[1] - y[0])) / area
        w0 = 1.0 - w1 - w2
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        pixel_depth = 1.0 / (w0 / tri[0, 2] + w1 / tri[1, 2] + w2 / tri[2, 2])
        window = depth[y0:y1, x0:x1]
        closer = inside & (pixel_depth < window)
        window[closer] = pixel_depth[closer]
        shade = 0.30 + 0.70 * max(0.0, float(np.dot(normal, light)))
        value = int(round(255 * 0.78 * shade))
        rgb[y0:y1, x0:x1][closer] = value

    return rgb, np.isfinite(depth), depth


def draw_wireframe(image: Image.Image, satellite: Satellite, R: np.ndarray, t: np.ndarray,
                   camera_K: np.ndarray, depth: Optional[np.ndarray], color: Tuple[int, int, int],
                   width: int = 3, hidden_color: Optional[Tuple[int, int, int]] = None,
                   samples: int = 40) -> None:
    """Draws the satellite's box edges for a pose onto ``image`` (in place).

    Edges are sampled along their length and compared to ``depth`` (the rendered depth map
    of the view): visible parts are drawn solid, hidden parts thin in ``hidden_color`` (or
    skipped if it is None). With ``depth=None`` every edge is drawn solid.
    """
    draw = ImageDraw.Draw(image)
    R, t = np.asarray(R, float), np.asarray(t, float)
    fx, fy, cx, cy = camera_K[0, 0], camera_K[1, 1], camera_K[0, 2], camera_K[1, 2]
    height, w = image.height, image.width
    fractions = np.linspace(0.0, 1.0, samples)[:, None]
    for start, end in satellite.edges:
        points = (start + (end - start) * fractions) @ R.T + t
        if (points[:, 2] <= 1e-3).any():
            continue
        u = fx * points[:, 0] / points[:, 2] + cx
        v = fy * points[:, 1] / points[:, 2] + cy
        if depth is None:
            visible = np.ones(samples, dtype=bool)
        else:
            ui, vi = np.clip(u.astype(int), 0, w - 1), np.clip(v.astype(int), 0, height - 1)
            visible = points[:, 2] <= depth[vi, ui] + 0.03 + 0.01 * points[:, 2]
        for k in range(samples - 1):
            segment = [(u[k], v[k]), (u[k + 1], v[k + 1])]
            if visible[k] and visible[k + 1]:
                draw.line(segment, fill=color, width=width)
            elif hidden_color is not None:
                draw.line(segment, fill=hidden_color, width=1)
