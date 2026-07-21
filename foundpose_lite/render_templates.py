import json
import os

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

# On headless Linux, set PYOPENGL_PLATFORM=egl in the shell *before* launching
# python -- must happen before pyrender/OpenGL is imported anywhere in the
# process. On Windows/macOS, leave unset (pyrender picks a windowed backend).
import pyrender  # noqa: E402


def load_centered_mesh(mesh_path: str):
    mesh = trimesh.load(mesh_path, force="mesh")
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    radius = np.linalg.norm(mesh.vertices, axis=1).max()
    return mesh, centroid, radius


def render_templates(mesh_path: str, out_dir: str, num_templates: int, template_size: int,
                      delta: float, seed: int = 0):
    """Renders `num_templates` synthetic views of the mesh, sampled ~uniformly over SO(3),
    on a black background with 3-point lighting. This is the "coarse pose" template bank
    used later for retrieval + matching. More templates = better SO(3) coverage but
    linearly slower to render and slower to build the object representation from --
    300-800 is a reasonable range; 800 is the default used throughout testing."""
    os.makedirs(out_dir, exist_ok=True)
    mesh, centroid, radius = load_centered_mesh(mesh_path)
    diameter = 2.0 * radius

    f_template = float(template_size)
    cx = cy = template_size / 2.0
    distance = f_template * diameter / (delta * template_size)
    znear, zfar = distance * 0.1, distance * 10.0

    camera = pyrender.IntrinsicsCamera(fx=f_template, fy=f_template, cx=cx, cy=cy,
                                        znear=znear, zfar=zfar)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = distance

    # 3-point lighting attached to the camera frame (so it's viewpoint-consistent across
    # templates) but offset in angle -- a single light glued directly behind the camera
    # gives flat, gradient-free shading, which is the worst case for DINOv2 to read 3D
    # shape from an untextured surface.
    def light_pose(yaw_deg, pitch_deg):
        r = Rotation.from_euler("yx", [yaw_deg, pitch_deg], degrees=True).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = r
        return camera_pose @ pose

    key_light = (pyrender.DirectionalLight(color=np.ones(3), intensity=5.0), light_pose(35, -25))
    fill_light = (pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), light_pose(-40, 15))
    rim_light = (pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), light_pose(160, 10))

    renderer = pyrender.OffscreenRenderer(template_size, template_size)
    rotations = Rotation.random(num_templates, random_state=seed)

    metadata = {
        "mesh_path": mesh_path,
        "centroid": centroid.tolist(),
        "radius": float(radius),
        "distance": float(distance),
        "camera": {"fx": f_template, "fy": f_template, "cx": cx, "cy": cy,
                    "size": template_size},
        "templates": [],
    }

    trimesh_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.75, 0.75, 0.78, 1.0], metallicFactor=0.05, roughnessFactor=0.8)

    for i, rot in enumerate(rotations):
        R_obj = rot.as_matrix()
        rotated_mesh = mesh.copy()
        rotated_mesh.vertices = rotated_mesh.vertices @ R_obj.T

        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.15, 0.15, 0.15])
        pyrender_mesh = pyrender.Mesh.from_trimesh(rotated_mesh, material=trimesh_material,
                                                     smooth=False)
        scene.add(pyrender_mesh)
        scene.add(camera, pose=camera_pose)
        for light, pose in (key_light, fill_light, rim_light):
            scene.add(light, pose=pose)

        color, depth = renderer.render(scene)

        name = f"template_{i:04d}"
        from PIL import Image
        Image.fromarray(color).save(os.path.join(out_dir, f"{name}_rgb.png"))
        np.save(os.path.join(out_dir, f"{name}_depth.npy"), depth.astype(np.float32))

        metadata["templates"].append({
            "id": i,
            "rgb": f"{name}_rgb.png",
            "depth": f"{name}_depth.npy",
            "R_obj_to_cam": R_obj.tolist(),
            "t_obj_to_cam": [0.0, 0.0, -distance],
        })

    renderer.delete()
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    return metadata
