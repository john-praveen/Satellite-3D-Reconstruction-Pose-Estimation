"""A small set of synthetic test objects, with and without surface texture.

All are built from boxes, cylinders and cones, as triangle soups with outward normals plus the
edges needed for wireframe overlays (see ``satpose.synthetic.Satellite``). They exist to check
that the pose pipeline generalises beyond one satellite:

* ``satellite``           -- the original grey-shaded test satellite.
* ``satellite_textured``  -- the same geometry, photographed with texture (gold foil body,
                             blue solar cells). FoundPose still matches against the plain grey
                             mesh, as it would with a generated (untextured) mesh, so this
                             isolates the effect of texture and lighting.
* ``rocket_body``         -- a nearly rotation-symmetric stage with a few asymmetric fittings.
* ``debris``              -- irregular: bent plates, rods and a tank.

``get_object(name)`` returns ``(geometry, texture_function_or_None)``.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from satpose.synthetic import Satellite, _box_faces_and_edges, build_satellite

Texture = Callable[[np.ndarray, np.ndarray], np.ndarray]


# -- geometry builders -----------------------------------------------------------------

class _Assembly:
    """Collects parts, then centres the result on its bounding box."""

    def __init__(self):
        self.triangles: List[np.ndarray] = []
        self.normals: List[np.ndarray] = []
        self.edges: List[np.ndarray] = []
        self.ids: List[int] = []

    def add(self, triangles, normals, edges, part: int) -> None:
        self.triangles += list(triangles)
        self.normals += list(normals)
        self.edges += list(edges)
        self.ids += [part] * len(triangles)

    def finish(self) -> Satellite:
        triangles, edges = np.array(self.triangles), np.array(self.edges)
        points = triangles.reshape(-1, 3)
        shift = (points.min(axis=0) + points.max(axis=0)) / 2.0
        return Satellite(triangles - shift, np.array(self.normals), edges - shift,
                         parts=np.array(self.ids))


def _unit(vector: Sequence[float]) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _outward_triangle(a, b, c, outward: np.ndarray):
    """Orders a triangle's vertices so its winding agrees with ``outward``; returns it with
    its unit normal."""
    normal = np.cross(b - a, c - a)
    if np.dot(normal, outward) < 0:
        b, c = c, b
        normal = -normal
    return np.array([a, b, c]), normal / np.linalg.norm(normal)


def frustum(base: Sequence[float], axis: Sequence[float], radius_start: float,
            radius_end: float, length: float, segments: int = 24):
    """A cylinder (equal radii) or cone (different radii) with flat caps.

    Returns ``(triangles, normals, edges)``; edges are the two end circles and four lines.
    """
    base, axis = np.asarray(base, dtype=np.float64), _unit(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(axis, helper))
    v = np.cross(axis, u)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    radial = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    ring0 = base + radius_start * radial
    ring1 = base + axis * length + radius_end * radial
    top_centre = base + axis * length

    triangles, normals = [], []
    for i in range(segments):
        j = (i + 1) % segments
        outward = radial[i] + radial[j]
        for tri in ((ring0[i], ring0[j], ring1[j]), (ring0[i], ring1[j], ring1[i])):
            t, n = _outward_triangle(*tri, outward)
            triangles.append(t)
            normals.append(n)
        t, n = _outward_triangle(base, ring0[j], ring0[i], -axis)
        triangles.append(t)
        normals.append(n)
        t, n = _outward_triangle(top_centre, ring1[i], ring1[j], axis)
        triangles.append(t)
        normals.append(n)

    edges = [[ring0[i], ring0[(i + 1) % segments]] for i in range(segments)]
    edges += [[ring1[i], ring1[(i + 1) % segments]] for i in range(segments)]
    edges += [[ring0[k], ring1[k]] for k in range(0, segments, segments // 4)]
    return triangles, normals, np.array(edges)


def tilted_box(center: Sequence[float], size: Sequence[float], rotation: np.ndarray):
    """A box turned by ``rotation`` about its own centre."""
    triangles, normals, edges = _box_faces_and_edges(np.zeros(3), np.asarray(size, float))
    rotation, center = np.asarray(rotation, float), np.asarray(center, float)
    return ([np.asarray(t) @ rotation.T + center for t in triangles],
            [rotation @ n for n in normals],
            [np.asarray(e) @ rotation.T + center for e in edges])


def _rotation(axis: Sequence[float], degrees: float) -> np.ndarray:
    axis, angle = _unit(axis), np.radians(degrees)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


# Part ids used by the textures below.
TUBE, NOZZLE, BAND, FITTING = 0, 1, 2, 3
PLATE, ROD, TANK, BLOCK = 0, 1, 2, 3


def build_rocket_body() -> Satellite:
    """A stage along the y axis. Nearly rotation-symmetric; the side fittings are what make
    its orientation unambiguous, so the outline alone is a weak cue."""
    a = _Assembly()
    a.add(*frustum((0, -1.0, 0), (0, 1, 0), 0.35, 0.35, 1.8), TUBE)
    a.add(*frustum((0, 0.8, 0), (0, 1, 0), 0.35, 0.06, 0.5), TUBE)          # nose cone
    a.add(*frustum((0, -1.0, 0), (0, -1, 0), 0.12, 0.30, 0.35), NOZZLE)
    a.add(*frustum((0, -0.25, 0), (0, 1, 0), 0.39, 0.39, 0.12), BAND)
    a.add(*tilted_box((0.42, 0.3, 0.0), (0.18, 0.35, 0.22), np.eye(3)), FITTING)
    a.add(*frustum((-0.30, 0.10, 0.20), (-0.6, 0.5, 0.2), 0.025, 0.025, 0.6, segments=12), FITTING)
    a.add(*tilted_box((-0.30, -0.6, -0.25), (0.14, 0.14, 0.14), np.eye(3)), FITTING)
    return a.finish()


def build_debris() -> Satellite:
    """Irregular debris: two bent plates, three rods of different lengths, a tank, a block."""
    a = _Assembly()
    a.add(*tilted_box((0.0, 0.0, 0.0), (1.6, 0.9, 0.05), np.eye(3)), PLATE)
    a.add(*tilted_box((0.9, 0.1, 0.35), (1.0, 0.7, 0.05), _rotation((0, 1, 0), 35)), PLATE)
    a.add(*frustum((-0.4, 0.0, 0.1), (0.2, 1.0, 0.1), 0.03, 0.03, 1.2, segments=12), ROD)
    a.add(*frustum((0.2, -0.2, 0.0), (1.0, 0.3, -0.2), 0.03, 0.03, 1.0, segments=12), ROD)
    a.add(*frustum((-0.6, 0.3, -0.1), (-0.3, -1.0, 0.4), 0.025, 0.025, 0.7, segments=12), ROD)
    a.add(*frustum((-0.2, -0.35, -0.3), (1, 0, 0), 0.18, 0.18, 0.5), TANK)
    a.add(*tilted_box((-0.7, -0.3, 0.05), (0.3, 0.25, 0.2), _rotation((1, 1, 0), 20)), BLOCK)
    return a.finish()


# -- textures ------------------------------------------------------------------------

def _hash01(points: np.ndarray, scale: float) -> np.ndarray:
    """Deterministic pseudo-random value in [0, 1) per cell of a 3D grid."""
    cells = np.floor(points * scale)
    value = np.sin(cells @ np.array([12.9898, 78.233, 37.719])) * 43758.5453
    return value - np.floor(value)


def _foil(points: np.ndarray) -> np.ndarray:
    base = np.array([0.82, 0.62, 0.22]) * (0.65 + 0.55 * _hash01(points, 5.0))[:, None]
    seams = np.abs(np.sin(points[:, 0] * 20.0)) < 0.05
    base[seams] *= 0.55
    return base


def _solar(points: np.ndarray, spacing: float = 10.0) -> np.ndarray:
    cells = np.array([0.08, 0.18, 0.50]) * (0.8 + 0.4 * _hash01(points, spacing))[:, None]
    lines = (np.mod(points[:, 0] * spacing, 1.0) < 0.07) | (np.mod(points[:, 1] * spacing, 1.0) < 0.07)
    cells[lines] = 0.78
    return cells


def _metal(points: np.ndarray) -> np.ndarray:
    return np.array([0.70, 0.70, 0.72]) * (0.8 + 0.2 * _hash01(points, 12.0))[:, None]


def satellite_texture(points: np.ndarray, parts: np.ndarray) -> np.ndarray:
    """Gold foil on the bus (box 0), blue solar cells on the panels (boxes 2 and 4), metal else."""
    out = _metal(points)
    bus, panels = parts == 0, (parts == 2) | (parts == 4)
    out[bus] = _foil(points[bus])
    out[panels] = _solar(points[panels])
    return out


def rocket_texture(points: np.ndarray, parts: np.ndarray) -> np.ndarray:
    """White body with black rings, an orange band, dark copper nozzle, grey fittings."""
    out = _metal(points)
    tube = parts == TUBE
    body = np.full((tube.sum(), 3), 0.9)
    body[np.mod(points[tube][:, 1] * 3.0, 1.0) < 0.12] = 0.08
    out[tube] = body
    out[parts == BAND] = np.array([0.9, 0.40, 0.10])
    out[parts == NOZZLE] = np.array([0.35, 0.25, 0.20]) * (0.7 + 0.6 * _hash01(points[parts == NOZZLE], 9.0))[:, None]
    return out


def debris_texture(points: np.ndarray, parts: np.ndarray) -> np.ndarray:
    """Solar-cell plates, metal rods, a white striped tank, a dark block."""
    out = _metal(points)
    out[parts == PLATE] = _solar(points[parts == PLATE], spacing=8.0)
    tank = parts == TANK
    stripes = np.full((tank.sum(), 3), 0.88)
    stripes[np.mod(points[tank][:, 0] * 6.0, 1.0) < 0.2] = 0.2
    out[tank] = stripes
    out[parts == BLOCK] = 0.15
    return out


# -- meshes from files -----------------------------------------------------------------

def crease_edges(triangles: np.ndarray, angle_deg: float = 40.0) -> np.ndarray:
    """Edges worth drawing as a wireframe: those where neighbouring faces meet at more than
    ``angle_deg``, plus open (boundary) edges. Returns (E, 2, 3) end points."""
    triangles = np.asarray(triangles, dtype=np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    vertices, inverse = np.unique(np.round(triangles.reshape(-1, 3), 6), axis=0, return_inverse=True)
    faces = np.asarray(inverse).reshape(-1, 3)
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(len(faces)), 3)
    edge, edge_of = np.unique(np.sort(pairs, axis=1), axis=0, return_inverse=True)
    edge_of = np.asarray(edge_of).reshape(-1)
    order = np.argsort(edge_of, kind="stable")
    sorted_edge, sorted_owner = edge_of[order], owner[order]
    first = np.full(len(edge), -1)
    second = np.full(len(edge), -1)
    starts = np.r_[True, sorted_edge[1:] != sorted_edge[:-1]]
    first[sorted_edge[starts]] = sorted_owner[starts]
    second[sorted_edge[~starts]] = sorted_owner[~starts]
    paired = second >= 0
    sharp = np.ones(len(edge), dtype=bool)                       # boundary edges are kept
    sharp[paired] = (normals[first[paired]] * normals[second[paired]]).sum(axis=1) \
        < np.cos(np.radians(angle_deg))
    return vertices[edge[sharp]]


def load_mesh_object(path: str, max_triangles: int = 6000, crease_deg: float = 70.0) -> Satellite:
    """A mesh file (stl, or glb/obj/ply via trimesh) as a test object.

    The mesh is simplified to ``max_triangles`` or fewer where its coarsest grid (16 cells
    along the longest side) allows (rendering here is pure Python), centred
    on its bounding box, given outward-facing normals, and its creases become the wireframe.
    Generated meshes are far too detailed to render or draw as they are.
    """
    from satpose.refinement import decimate_triangles
    from satpose.synthetic import read_binary_stl

    if path.lower().endswith(".stl"):
        soup = read_binary_stl(path).astype(np.float64)
        vertices, inverse = np.unique(soup.reshape(-1, 3), axis=0, return_inverse=True)
        faces = np.asarray(inverse).reshape(-1, 3)
    else:
        import trimesh

        mesh = trimesh.load(path, force="mesh")
        vertices, faces = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)
    triangles = vertices[faces]
    for cells in (160, 120, 90, 70, 55, 45, 36, 28, 22, 16):
        triangles = decimate_triangles(vertices, faces, cells)
        if len(triangles) <= max_triangles:
            break

    volume = np.einsum("ij,ij->", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) / 6.0
    if volume < 0:                       # inside-out winding: make the normals point outwards
        triangles = triangles[:, [0, 2, 1]]
    points = triangles.reshape(-1, 3)
    triangles = triangles - (points.min(axis=0) + points.max(axis=0)) / 2.0
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return Satellite(triangles, normals, crease_edges(triangles, crease_deg),
                     parts=np.zeros(len(triangles), dtype=int))


# -- registry --------------------------------------------------------------------------

_OBJECTS = {
    "satellite": (build_satellite, None),
    "satellite_textured": (build_satellite, satellite_texture),
    "rocket_body": (build_rocket_body, rocket_texture),
    "debris": (build_debris, debris_texture),
}
OBJECT_NAMES = tuple(_OBJECTS)


def get_object(name: str) -> Tuple[Satellite, Optional[Texture]]:
    """Geometry and texture function of a named test object, or of a mesh file (given by its
    path, rendered untextured)."""
    if name not in _OBJECTS and os.path.isfile(name):
        return load_mesh_object(name), None
    if name not in _OBJECTS:
        raise ValueError(f"unknown object {name!r}; choose from {OBJECT_NAMES}")
    build, texture = _OBJECTS[name]
    return build(), texture
