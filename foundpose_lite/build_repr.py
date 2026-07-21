import json
import os
import pickle

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from foundpose_lite.dino_features import extract_patch_descriptors
from foundpose_lite.nn_utils import batched_soft_assign


def lift_patch_to_object_space(u, v, depth, cam, R_obj, t_obj_to_cam):
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    x_c = (u - cx) * depth / fx
    y_c = -(v - cy) * depth / fy
    z_c = -depth
    point_cam = np.array([x_c, y_c, z_c])
    R_obj = np.asarray(R_obj)
    t = np.asarray(t_obj_to_cam)
    return R_obj.T @ (point_cam - t)


def collect_template_patches(templates_dir: str, metadata: dict, dino_model: str, layer: int,
                              patch_size: int, device: str):
    cam = metadata["camera"]
    grid = metadata["camera"]["size"] // patch_size
    records = []  # list of dict(raw_desc, object_point, template_id)

    for tmpl in metadata["templates"]:
        rgb = np.array(Image.open(os.path.join(templates_dir, tmpl["rgb"])).convert("RGB"))
        depth = np.load(os.path.join(templates_dir, tmpl["depth"]))
        feats = extract_patch_descriptors(rgb, dino_model, layer, patch_size, device)

        for row in range(grid):
            for col in range(grid):
                v = row * patch_size + patch_size // 2
                u = col * patch_size + patch_size // 2
                d = float(depth[v, u])
                if d <= 0:
                    continue
                point_obj = lift_patch_to_object_space(
                    u, v, d, cam, tmpl["R_obj_to_cam"], tmpl["t_obj_to_cam"])
                records.append({
                    "raw_desc": feats[row, col],
                    "object_point": point_obj,
                    "template_id": tmpl["id"],
                })
    return records


def build_object_representation(templates_dir: str, dino_model: str, layer: int,
                                 patch_size: int, pca_dim: int, num_words: int,
                                 soft_k: int, sigma: float, device: str, out_path: str):
    with open(os.path.join(templates_dir, "metadata.json")) as f:
        metadata = json.load(f)

    records = collect_template_patches(templates_dir, metadata, dino_model, layer,
                                        patch_size, device)
    print(f"collected {len(records)} valid patches across {len(metadata['templates'])} templates")

    raw_descs = np.stack([r["raw_desc"] for r in records]).astype(np.float32)
    pca = PCA(n_components=min(pca_dim, raw_descs.shape[1]))
    reduced = pca.fit_transform(raw_descs).astype(np.float32)

    n_clusters = min(num_words, len(records))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=3)
    kmeans.fit(reduced)
    words = kmeans.cluster_centers_.astype(np.float32)

    # soft-assign every patch to its `soft_k` nearest words (batched to bound memory)
    nearest, weights = batched_soft_assign(reduced, words, soft_k, sigma)

    num_templates = len(metadata["templates"])
    template_ids = np.array([r["template_id"] for r in records])

    n_i = np.zeros(n_clusters, dtype=np.float64)          # weighted occurrences of word i, all templates
    n_it = np.zeros((num_templates, n_clusters), dtype=np.float64)
    for p in range(len(records)):
        t_id = template_ids[p]
        for k in range(soft_k):
            w = nearest[p, k]
            wt = weights[p, k]
            n_i[w] += wt
            n_it[t_id, w] += wt
    n_t = n_it.sum(axis=1)  # total weighted word count per template

    idf = np.log(num_templates / np.clip(n_i, 1e-6, None))
    bow_vectors = np.zeros((num_templates, n_clusters), dtype=np.float32)
    for t_id in range(num_templates):
        if n_t[t_id] > 0:
            bow_vectors[t_id] = (n_it[t_id] / n_t[t_id]) * idf

    per_template = {t_id: {"reduced_descs": [], "object_points": []} for t_id in range(num_templates)}
    for p, rec in enumerate(records):
        per_template[rec["template_id"]]["reduced_descs"].append(reduced[p])
        per_template[rec["template_id"]]["object_points"].append(rec["object_point"])
    for t_id in per_template:
        per_template[t_id]["reduced_descs"] = np.stack(per_template[t_id]["reduced_descs"]) \
            if per_template[t_id]["reduced_descs"] else np.zeros((0, reduced.shape[1]), dtype=np.float32)
        per_template[t_id]["object_points"] = np.stack(per_template[t_id]["object_points"]) \
            if per_template[t_id]["object_points"] else np.zeros((0, 3), dtype=np.float32)
        per_template[t_id]["bow"] = bow_vectors[t_id]
        per_template[t_id]["R_obj_to_cam"] = np.array(metadata["templates"][t_id]["R_obj_to_cam"])

    repr_data = {
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "words": words,
        "soft_k": soft_k,
        "sigma": sigma,
        "num_templates": num_templates,
        "idf": idf.astype(np.float32),
        "per_template": per_template,
        "mesh_centroid": np.array(metadata["centroid"]),
        "mesh_path": metadata["mesh_path"],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(repr_data, f)
    print(f"saved object representation to {out_path}")
    return repr_data
