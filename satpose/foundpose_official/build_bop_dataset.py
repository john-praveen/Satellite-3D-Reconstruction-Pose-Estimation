"""Builds a minimal fake "BOP dataset" around ONE custom object + a handful of query
photos, so the official FoundPose repo's scripts (which are hard-wired to load data via
bop_toolkit's BOP dataset format) can run against it. Registers as the "custom" dataset
name (see the `elif dataset_name == "custom":` branch patched into
external/bop_toolkit/bop_toolkit_lib/dataset_params.py) -- that branch reads its
per-object parameters (image size, depth range) from dataset_meta.json, written here, so
a NEW object never requires editing bop_toolkit code again.
"""
import json
import os

import numpy as np
import trimesh
from PIL import Image


def mask_bbox_from_nobg(nobg_path: str):
    alpha = np.array(Image.open(nobg_path).convert("RGBA"))[:, :, 3]
    ys, xs = np.where(alpha > 10)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    mask = (alpha > 10).astype(np.uint8) * 255
    return mask, [x0, y0, x1 - x0, y1 - y0]


def build_custom_bop_dataset(mesh_path: str, image_nobg_paths: list, image_raw_paths: list,
                              camera_K: np.ndarray, image_width: int, image_height: int,
                              bop_datasets_root: str, mm_scale: float = 500.0,
                              depth_range_mm: tuple = (800.0, 3000.0)):
    """image_nobg_paths / image_raw_paths must be same length and same order -- one raw
    photo + its background-removed RGBA version per query image. mm_scale: no real
    calibration for the reconstructed mesh's absolute size, so this is an arbitrary but
    consistent scale factor (translation/depth from PnP will inherit this same
    arbitrariness -- rotation is unaffected, see README's "Known limitations")."""
    bop_root = os.path.join(bop_datasets_root, "custom")
    os.makedirs(bop_root, exist_ok=True)

    # -- 3D model, in mm --
    mesh = trimesh.load(mesh_path, force="mesh")
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices = (mesh.vertices - centroid) * mm_scale
    # Extra per-face data (e.g. the "stl" value trimesh keeps from an STL file) would be
    # written into the PLY, and bop_toolkit's PLY reader cannot skip unknown properties:
    # it misreads every face after the first.
    mesh.face_attributes.clear()
    models_dir = os.path.join(bop_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    mesh.export(os.path.join(models_dir, "obj_000001.ply"))

    bounds = mesh.bounds
    radius = float(np.linalg.norm(mesh.vertices, axis=1).max())
    models_info = {"1": {
        "min_x": float(bounds[0, 0]), "min_y": float(bounds[0, 1]), "min_z": float(bounds[0, 2]),
        "size_x": float(bounds[1, 0] - bounds[0, 0]),
        "size_y": float(bounds[1, 1] - bounds[0, 1]),
        "size_z": float(bounds[1, 2] - bounds[0, 2]),
        "diameter": 2.0 * radius,
    }}
    with open(os.path.join(models_dir, "models_info.json"), "w") as f:
        json.dump(models_info, f)

    # -- top-level camera.json --
    cam = {"width": image_width, "height": image_height,
           "fx": float(camera_K[0, 0]), "fy": float(camera_K[1, 1]),
           "cx": float(camera_K[0, 2]), "cy": float(camera_K[1, 2]), "depth_scale": 1.0}
    with open(os.path.join(bop_root, "camera.json"), "w") as f:
        json.dump(cam, f)

    # -- dataset_meta.json (read by the generic "custom" branch in dataset_params.py) --
    with open(os.path.join(bop_root, "dataset_meta.json"), "w") as f:
        json.dump({"im_size": [image_width, image_height], "scene_ids": [0],
                    "depth_range": list(depth_range_mm)}, f)

    # -- one scene with all query images --
    scene_dir = os.path.join(bop_root, "test", "000000")
    for sub in ["rgb", "depth", "mask_visib"]:
        os.makedirs(os.path.join(scene_dir, sub), exist_ok=True)

    cam_K_flat = camera_K.flatten().tolist()
    scene_camera, scene_gt, scene_gt_info, targets = {}, {}, {}, []
    for im_id, (raw_path, nobg_path) in enumerate(zip(image_raw_paths, image_nobg_paths)):
        Image.open(raw_path).convert("RGB").save(os.path.join(scene_dir, "rgb", f"{im_id:06d}.png"))
        depth_dummy = np.zeros((image_height, image_width), dtype=np.uint16)
        Image.fromarray(depth_dummy).save(os.path.join(scene_dir, "depth", f"{im_id:06d}.png"))

        mask, bbox = mask_bbox_from_nobg(nobg_path)
        Image.fromarray(mask).save(os.path.join(scene_dir, "mask_visib", f"{im_id:06d}_000000.png"))

        scene_camera[str(im_id)] = {"cam_K": cam_K_flat, "depth_scale": 1.0,
                                     "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                     "cam_t_w2c": [0.0, 0.0, 0.0]}
        scene_gt[str(im_id)] = [{"obj_id": 1, "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                   "cam_t_m2c": [0.0, 0.0, 1000.0]}]
        scene_gt_info[str(im_id)] = [{"bbox_obj": bbox, "visib_fract": 1.0}]
        targets.append({"scene_id": 0, "im_id": im_id, "obj_id": 1, "inst_count": 1})

    with open(os.path.join(scene_dir, "scene_camera.json"), "w") as f:
        json.dump(scene_camera, f)
    with open(os.path.join(scene_dir, "scene_gt.json"), "w") as f:
        json.dump(scene_gt, f)
    with open(os.path.join(scene_dir, "scene_gt_info.json"), "w") as f:
        json.dump(scene_gt_info, f)
    with open(os.path.join(bop_root, "test_targets_bop19.json"), "w") as f:
        json.dump(targets, f)

    print(f"built custom BOP dataset at {bop_root} with {len(image_raw_paths)} image(s)")
    return bop_root
