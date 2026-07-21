import json
import pickle

import cv2
import numpy as np
from PIL import Image

from foundpose_lite.cyclic_buddies import cyclic_buddies_matching
from foundpose_lite.dino_features import extract_patch_descriptors
from foundpose_lite.nn_utils import batched_soft_assign


def crop_on_black_bg(nobg_rgba_path: str, template_size: int, delta: float):
    img = Image.open(nobg_rgba_path).convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        raise ValueError("no foreground pixels found in nobg image alpha channel")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx_box, cy_box = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    longer_side = max(x1 - x0, y1 - y0)
    crop_size = longer_side / delta

    black_bg = Image.new("RGB", img.size, (0, 0, 0))
    black_bg.paste(img, mask=img.split()[3])

    box = (cx_box - crop_size / 2, cy_box - crop_size / 2,
           cx_box + crop_size / 2, cy_box + crop_size / 2)
    crop = black_bg.crop(box).resize((template_size, template_size), Image.BILINEAR)
    alpha_img = Image.fromarray(alpha)
    mask_crop = alpha_img.crop(box).resize((template_size, template_size), Image.NEAREST)

    scale = template_size / crop_size
    # original_pixel = (crop_pixel / scale) + box_top_left
    transform = {"box_x0": box[0], "box_y0": box[1], "scale": scale}
    return np.array(crop), np.array(mask_crop), transform


def crop_to_original(u_c, v_c, transform):
    u = transform["box_x0"] + u_c / transform["scale"]
    v = transform["box_y0"] + v_c / transform["scale"]
    return u, v


def soft_assign_bow(reduced_descs: np.ndarray, words: np.ndarray, idf: np.ndarray,
                     soft_k: int, sigma: float):
    nearest, weights = batched_soft_assign(reduced_descs, words, soft_k, sigma)
    n_i = np.zeros(len(words), dtype=np.float64)
    for p in range(len(reduced_descs)):
        for k in range(soft_k):
            n_i[nearest[p, k]] += weights[p, k]
    n_t = n_i.sum()
    if n_t == 0:
        return np.zeros(len(words), dtype=np.float32)
    return ((n_i / n_t) * idf).astype(np.float32)


def estimate_pose_from_crop(repr_data: dict, crop_rgb: np.ndarray, valid_flat: np.ndarray,
                             query_points_2d: np.ndarray, camera_K: np.ndarray,
                             dino_model: str, layer: int, patch_size: int, top_k: int,
                             pnp_iters: int, pnp_thresh_px: float, device: str,
                             match_top_k_buddies: int = 60):
    """Core retrieval + PnP-RANSAC, decoupled from where crop_rgb/mask came from."""
    if valid_flat.sum() < 4:
        raise RuntimeError(f"only {valid_flat.sum()} foreground patches in crop -- "
                            f"check the nobg image / mask")

    feats = extract_patch_descriptors(crop_rgb, dino_model, layer, patch_size, device)
    raw_descs = feats.reshape(-1, feats.shape[-1])[valid_flat]
    reduced = (raw_descs - repr_data["pca_mean"]) @ repr_data["pca_components"].T
    reduced = reduced.astype(np.float32)

    query_bow = soft_assign_bow(reduced, repr_data["words"], repr_data["idf"],
                                 repr_data["soft_k"], repr_data["sigma"])

    template_ids = sorted(repr_data["per_template"].keys())
    template_bows = np.stack([repr_data["per_template"][t]["bow"] for t in template_ids])
    query_norm = query_bow / (np.linalg.norm(query_bow) + 1e-8)
    tmpl_norms = template_bows / (np.linalg.norm(template_bows, axis=1, keepdims=True) + 1e-8)
    sims = tmpl_norms @ query_norm
    retrieved = [template_ids[i] for i in np.argsort(-sims)[:top_k]]
    print(f"crop: {valid_flat.sum()} foreground patches; "
          f"top-{top_k} retrieved templates: {retrieved} sims: {sims[np.argsort(-sims)[:top_k]]}")

    best = None
    for t_id in retrieved:
        tmpl = repr_data["per_template"][t_id]
        if len(tmpl["reduced_descs"]) == 0:
            continue
        query_ids, tmpl_ids = cyclic_buddies_matching(
            query_points_2d, reduced, tmpl["reduced_descs"], match_top_k_buddies)
        object_points = tmpl["object_points"][tmpl_ids]

        image_points = query_points_2d[query_ids].astype(np.float64)
        obj_points = object_points.astype(np.float64)

        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_points, image_points, camera_K, None,
                iterationsCount=pnp_iters, reprojectionError=pnp_thresh_px,
                flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error as e:
            print(f"  template {t_id}: cv2 error: {e}")
            ok, inliers = False, None
        print(f"  template {t_id}: ok={ok} inliers={0 if inliers is None else len(inliers)}")
        if not ok or inliers is None:
            continue
        num_inliers = len(inliers)
        if best is None or num_inliers > best["num_inliers"]:
            best = {
                "template_id": int(t_id),
                "num_inliers": int(num_inliers),
                "rvec": rvec,
                "tvec": tvec,
                "inlier_obj_points": obj_points[inliers.flatten()],
                "inlier_image_points": image_points[inliers.flatten()],
            }

    if best is None:
        raise RuntimeError("PnP-RANSAC failed on every retrieved template")

    # continuous refinement: minimize reprojection error over the inlier set via
    # Levenberg-Marquardt, polishing the pose beyond the nearest discrete template's
    # exact orientation (same idea DreamSat-Pose's pipeline uses after its own PnP-RANSAC).
    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
        best["inlier_obj_points"], best["inlier_image_points"], camera_K, None,
        best["rvec"], best["tvec"])
    R_refined, _ = cv2.Rodrigues(rvec_refined)
    best["R"] = R_refined.tolist()
    best["t"] = tvec_refined.flatten().tolist()
    del best["rvec"], best["tvec"], best["inlier_obj_points"], best["inlier_image_points"]

    best["retrieved_templates"] = retrieved
    best["similarities"] = sims[np.argsort(-sims)[:top_k]].tolist()
    best["mesh_centroid"] = repr_data["mesh_centroid"].tolist()
    best["mesh_path"] = repr_data["mesh_path"]
    return best


def estimate_pose(repr_path: str, nobg_image_path: str, camera_K: np.ndarray,
                   dino_model: str, layer: int, patch_size: int, template_size: int,
                   delta: float, top_k: int, pnp_iters: int, pnp_thresh_px: float,
                   device: str, match_top_k_buddies: int = 60):
    with open(repr_path, "rb") as f:
        repr_data = pickle.load(f)

    crop_rgb, mask_crop, transform = crop_on_black_bg(nobg_image_path, template_size, delta)
    grid = template_size // patch_size

    valid = np.zeros((grid, grid), dtype=bool)
    for row in range(grid):
        for col in range(grid):
            v = row * patch_size + patch_size // 2
            u = col * patch_size + patch_size // 2
            valid[row, col] = mask_crop[v, u] > 10
    valid_flat = valid.reshape(-1)

    query_points_2d_full = np.zeros((grid, grid, 2))
    for row in range(grid):
        for col in range(grid):
            u_c = col * patch_size + patch_size // 2
            v_c = row * patch_size + patch_size // 2
            query_points_2d_full[row, col] = crop_to_original(u_c, v_c, transform)
    query_points_2d = query_points_2d_full.reshape(-1, 2)[valid_flat]

    return estimate_pose_from_crop(repr_data, crop_rgb, valid_flat, query_points_2d, camera_K,
                                    dino_model, layer, patch_size, top_k, pnp_iters,
                                    pnp_thresh_px, device, match_top_k_buddies)


def save_pose_result(result: dict, out_path: str):
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
