import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]


def obb_corners_object_space(mesh_path: str, mesh_centroid: np.ndarray) -> np.ndarray:
    mesh = trimesh.load(mesh_path, force="mesh")
    obb = mesh.bounding_box_oriented
    corners = obb.vertices - mesh_centroid  # same centered frame used during template rendering
    return corners


def draw_wireframe(raw_image_path: str, mesh_path: str, mesh_centroid, R, t,
                    camera_K: np.ndarray, out_path: str, color=(0, 255, 0), width=4):
    """Draws the object's oriented bounding box, projected via the estimated pose. Quick
    to compute, but a box looks similar under several wrong rotations (rotational
    symmetry) -- prefer draw_axes or render_mask_overlay to actually judge correctness."""
    corners = obb_corners_object_space(mesh_path, np.asarray(mesh_centroid))
    R = np.asarray(R)
    t = np.asarray(t).reshape(3)

    cam_points = (R @ corners.T).T + t
    pixels = (camera_K @ cam_points.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]

    img = Image.open(raw_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for a, b in BOX_EDGES:
        pa, pb = tuple(pixels[a]), tuple(pixels[b])
        draw.line([pa, pb], fill=color, width=width)

    img.save(out_path)
    return out_path


# OpenCV (solvePnP) camera convention is X-right/Y-down/Z-forward-positive; pyrender/OpenGL
# is X-right/Y-up/Z-backward-positive. Flipping Y and Z converts between them.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0])


def render_mask_overlay(raw_image_path: str, mesh_path: str, mesh_centroid, R, t,
                         camera_K: np.ndarray, image_width: int, image_height: int,
                         out_path: str, fill_color=(0, 220, 255), fill_alpha=0.35,
                         contour_color=(0, 220, 255), contour_width=3):
    """Re-renders the actual reconstructed mesh at the estimated pose, from the real
    camera's viewpoint, and overlays its silhouette (translucent fill + contour) on the
    real photo. This is the most trustworthy visual sanity check: a box or axis triad can
    look plausible even when wrong, but the true mesh silhouette only lines up with the
    photo if the pose -- and the reconstruction's shape -- are actually both right."""
    import cv2

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.vertices = mesh.vertices - np.asarray(mesh_centroid)

    R = np.asarray(R)
    t = np.asarray(t).reshape(3)
    R_gl = _CV_TO_GL @ R
    t_gl = _CV_TO_GL @ t
    mesh_pose = np.eye(4)
    mesh_pose[:3, :3] = R_gl
    mesh_pose[:3, 3] = t_gl

    distance = np.linalg.norm(t_gl)
    fx, fy, cx, cy = camera_K[0, 0], camera_K[1, 1], camera_K[0, 2], camera_K[1, 2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                        znear=max(distance * 0.05, 1e-4), zfar=distance * 20)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.75, 0.75, 0.78, 1.0])

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False), pose=mesh_pose)
    scene.add(camera, pose=np.eye(4))
    scene.add(light, pose=mesh_pose)

    renderer = pyrender.OffscreenRenderer(image_width, image_height)
    _, depth = renderer.render(scene)
    renderer.delete()
    mask = depth > 0

    img = np.array(Image.open(raw_image_path).convert("RGB")).astype(np.float32)
    color_arr = np.array(fill_color, dtype=np.float32)
    img[mask] = img[mask] * (1 - fill_alpha) + color_arr * fill_alpha

    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    img_u8 = img.astype(np.uint8)
    cv2.drawContours(img_u8, contours, -1, contour_color, contour_width)

    Image.fromarray(img_u8).save(out_path)
    return out_path


def draw_axes(raw_image_path: str, mesh_path: str, mesh_centroid, R, t,
              camera_K: np.ndarray, out_path: str, width=6, origin_radius=8):
    """Draws the object's local X (red), Y (green), Z (blue) axes projected into the
    image via the estimated pose -- easier to sanity-check by eye than a symmetric box,
    since a box looks similar under several wrong rotations but an axis triad doesn't."""
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh_centroid = np.asarray(mesh_centroid)
    radius = np.linalg.norm(mesh.vertices - mesh_centroid, axis=1).max()
    axis_length = radius * 0.8

    axes_3d = np.array([
        [0.0, 0.0, 0.0],
        [axis_length, 0.0, 0.0],
        [0.0, axis_length, 0.0],
        [0.0, 0.0, axis_length],
    ])
    R = np.asarray(R)
    t = np.asarray(t).reshape(3)

    cam_points = (R @ axes_3d.T).T + t
    pixels = (camera_K @ cam_points.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    origin, x_end, y_end, z_end = [tuple(p) for p in pixels]

    img = Image.open(raw_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.line([origin, z_end], fill=(50, 50, 255), width=width)   # Z drawn first, under X/Y
    draw.line([origin, y_end], fill=(0, 255, 0), width=width)
    draw.line([origin, x_end], fill=(255, 0, 0), width=width)
    r = origin_radius
    draw.ellipse([origin[0] - r, origin[1] - r, origin[0] + r, origin[1] + r], fill=(255, 255, 0))

    img.save(out_path)
    return out_path
