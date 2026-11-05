"""Shared logic for running the ACTUAL official FoundPose (facebookresearch/foundpose,
ECCV 2024) pose-estimation algorithm against a reconstructed mesh -- used by both
run_single_view_pipeline.py and run_multiview_pipeline.py. Once you have a mesh (from
either single- or multi-view reconstruction), the rest of the pipeline (build a minimal
fake BOP dataset around it, run the official repo's gen_templates.py -> gen_repre.py ->
infer.py in ITS OWN conda env, then render our mask+axes visualization on the resulting
poses) is identical. See README.md "Official pipeline setup" for the two-conda-env setup
this needs.
"""
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

from satpose import config
from satpose.foundpose_official.build_bop_dataset import build_custom_bop_dataset
from satpose.foundpose_official.patch_bop_toolkit import apply as patch_bop_toolkit
from satpose.postprocess import improve_poses, load_mesh_triangles, save_poses
from satpose.visualize import draw_axes, render_mask_overlay

_remover = None
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv")


def _get_background_remover():
    """Loads Hunyuan3D's background remover on first use, so importing this module does
    not require the Hunyuan3D-2 repo (e.g. when cut-outs are supplied by the caller)."""
    global _remover
    if _remover is None:
        sys.path.insert(0, config.HY3DGEN_PATH)
        from hy3dgen.rembg import BackgroundRemover
        _remover = BackgroundRemover()
    return _remover


def remove_background(image_path: str, out_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_nobg = _get_background_remover()(image)
    image_nobg.save(out_path)
    return out_path


def resolve_query_images(source, out_dir: str, video_frame_stride: int = 1):
    """Turns a query source (config.QUERY_SOURCE or equivalent) into a concrete list
    of image paths. `source` may be: None (caller supplies its own default), a single
    image path, a list of image paths, a folder (every image file inside it,
    non-recursive), or a video file (frames extracted to `<out_dir>/video_frames/`,
    one per `video_frame_stride`-th frame)."""
    if source is None:
        return None
    if isinstance(source, (list, tuple)):
        return [os.path.abspath(p) for p in source]

    source = os.path.abspath(source)
    if os.path.isdir(source):
        images = sorted(
            os.path.join(source, f) for f in os.listdir(source)
            if f.lower().endswith(_IMAGE_EXTENSIONS)
        )
        if not images:
            raise ValueError(f"no image files found in {source}")
        return images

    if os.path.splitext(source)[1].lower() in _VIDEO_EXTENSIONS:
        return _extract_video_frames(source, out_dir, video_frame_stride)

    return [source]


def _extract_video_frames(video_path: str, out_dir: str, stride: int):
    import cv2

    frames_dir = os.path.join(out_dir, "video_frames")
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    frame_paths = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame_path = os.path.join(frames_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        idx += 1
    cap.release()

    if not frame_paths:
        raise ValueError(f"no frames extracted from {video_path}")
    print(f"extracted {len(frame_paths)} frame(s) (every {stride}) from {video_path} "
          f"-> {frames_dir}")
    return frame_paths


def _rotation_matrix_to_quaternion(R):
    """3x3 rotation matrix -> (w, x, y, z) quaternion, via Shepperd's method."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, \
            (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, \
            (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, \
            (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, \
            (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return w, x, y, z


def save_poses_csv(poses: list, query_images: list, out_dir: str):
    """Writes one row per estimated pose to `<out_dir>/estimated_poses.csv` -- rotation
    as both a flattened 3x3 matrix and a quaternion, translation, and match-quality
    fields, keyed by the query photo's filename."""
    csv_path = os.path.join(out_dir, "estimated_poses.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "img_id",
            "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22",
            "quat_w", "quat_x", "quat_y", "quat_z",
            "tx", "ty", "tz", "inliers_est", "score",
        ])
        for pose in poses:
            im_id = int(pose["img_id"])
            image_name = os.path.basename(query_images[im_id])
            R = pose["R"]
            t = [v[0] if isinstance(v, list) else v for v in pose["t"]]
            qw, qx, qy, qz = _rotation_matrix_to_quaternion(R)
            writer.writerow([
                image_name, im_id,
                R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2],
                R[2][0], R[2][1], R[2][2],
                qw, qx, qy, qz,
                t[0], t[1], t[2],
                pose.get("inliers_est"), pose.get("score"),
            ])
    print(f"saved {csv_path}")
    return csv_path


def _official_inference_dir(bop_root: str, infer_config: str) -> str:
    """Folder the official infer.py writes to: ``outputs/inference/custom_<version>/1``.

    Its ``version`` setting also selects which object representation is loaded, so configs
    that share a representation share this folder; ``_snapshot_inference`` keeps each run.
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foundpose_official",
                               "configs", "infer", f"{infer_config}.json")
    with open(config_path) as f:
        version = json.load(f)["infer_opts"]["version"]
    return os.path.join(bop_root, "outputs", "inference", f"custom_{version}", "1")


def _inference_dir(bop_root: str, infer_config: str) -> str:
    """Where the results of one inference config are kept (a copy of the official output)."""
    return os.path.join(bop_root, "outputs", "runs", infer_config)


def _snapshot_inference(bop_root: str, infer_config: str) -> str:
    """Copies the official output to this config's own folder, replacing an older copy."""
    destination = _inference_dir(bop_root, infer_config)
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(_official_inference_dir(bop_root, infer_config), destination)
    return destination


def _conda_env_args(env: str) -> list:
    """`-n <name>` for an environment name, `-p <path>` when `env` is a directory path."""
    is_path = os.sep in env or "/" in env
    return ["-p" if is_path else "-n", env]


def _run_official_script(script_name: str, config_name: str, bop_root: str):
    env = os.environ.copy()
    env["REPO_PATH"] = config.FOUNDPOSE_OFFICIAL_REPO_PATH
    env["BOP_PATH"] = bop_root
    env["PYTHONPATH"] = os.pathsep.join([
        config.FOUNDPOSE_OFFICIAL_REPO_PATH,
        os.path.join(config.FOUNDPOSE_OFFICIAL_REPO_PATH, "external", "bop_toolkit"),
        os.path.join(config.FOUNDPOSE_OFFICIAL_REPO_PATH, "external", "dinov2"),
    ])
    opts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foundpose_official",
                              "configs", script_name.replace(".py", ""), f"{config_name}.json")
    cmd = [os.environ.get("CONDA_EXE", "conda"), "run", "--no-capture-output",
           *_conda_env_args(config.FOUNDPOSE_OFFICIAL_CONDA_ENV), "python",
           os.path.join("scripts", script_name), "--opts-path", opts_path]
    print(f"== running official {script_name} ==")
    result = subprocess.run(cmd, cwd=config.FOUNDPOSE_OFFICIAL_REPO_PATH, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed (exit code {result.returncode}) -- "
                            f"see README.md if this is an environment/setup issue")


def run_official_foundpose(mesh_path: str, query_images: list, out_dir: str,
                            skip_templates: bool = False, save_mask: bool = True,
                            save_axes: bool = True, nobg_paths: list = None,
                            infer_config: str = None, refinement: str = None,
                            template_config: str = "custom", repre_config: str = "custom"):
    """End-to-end: remove the background from each query photo, build a minimal BOP
    dataset around `mesh_path`, run the official FoundPose repo's
    gen_templates.py -> gen_repre.py -> infer.py (in its own conda env), save the
    estimated poses (JSON + CSV), and -- unless disabled -- render a mask/axes
    visualization for each query photo. Returns the list of estimated poses
    ({img_id, R, t, inliers_est, score}, one per query image).

    `nobg_paths`: optional RGBA cut-outs, one per query image and in the same order. When
    given, the background-removal step is skipped (used for synthetic renders, which
    already have an exact alpha mask).

    `infer_config`, `template_config`, `repre_config`: names of the FoundPose configs under
    satpose/foundpose_official/configs/{infer,gen_templates,gen_repre}/ (defaults:
    config.INFER_CONFIG, "custom", "custom").
    A variant that changes the templates needs its own matching trio, with a distinct
    `version` in each config.

    `refinement`: how to improve FoundPose's poses with the photo's silhouette (see
    satpose/postprocess.py): "none" or "full". Defaults to
    config.POSE_REFINEMENT."""
    refinement = config.POSE_REFINEMENT if refinement is None else refinement
    infer_config = config.INFER_CONFIG if infer_config is None else infer_config
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    query_images = [os.path.abspath(q) for q in query_images]
    bop_root = os.path.join(out_dir, "bop_datasets")

    if patch_bop_toolkit(config.FOUNDPOSE_OFFICIAL_REPO_PATH):
        print("== registered the 'custom' dataset in the FoundPose repo's bop_toolkit ==")

    for q in query_images:
        actual_size = Image.open(q).size
        expected_size = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
        if actual_size != expected_size:
            raise ValueError(
                f"{q} is {actual_size[0]}x{actual_size[1]}, but satpose/config.py's "
                f"IMAGE_WIDTH/IMAGE_HEIGHT is set to {expected_size[0]}x{expected_size[1]}. "
                f"CAMERA_K is derived from those, so a mismatch feeds wrong intrinsics into "
                f"the official pipeline (not just the mask+axes visualization) -- update "
                f"IMAGE_WIDTH/IMAGE_HEIGHT (and CAMERA_K, unless you're using real "
                f"calibration for this exact resolution) in satpose/config.py to match your photos "
                f"before running.")

    if nobg_paths is None:
        nobg_paths = []
        for q in query_images:
            q_stem = os.path.splitext(os.path.basename(q))[0]
            nobg_path = os.path.join(out_dir, f"{q_stem}_nobg.png")
            remove_background(q, nobg_path)
            nobg_paths.append(nobg_path)
    elif len(nobg_paths) != len(query_images):
        raise ValueError("nobg_paths must have one entry per query image")

    if not skip_templates:
        build_custom_bop_dataset(mesh_path, nobg_paths, query_images, config.CAMERA_K,
                                  config.IMAGE_WIDTH, config.IMAGE_HEIGHT, bop_root)
        _run_official_script("gen_templates.py", template_config, bop_root)
        _run_official_script("gen_repre.py", repre_config, bop_root)
    else:
        print("== skipping template generation, reusing existing BOP dataset/templates ==")
    stale_candidates = os.path.join(_official_inference_dir(bop_root, infer_config), "candidates")
    if os.path.exists(stale_candidates):
        shutil.rmtree(stale_candidates)  # never mix candidates from an earlier config
    _run_official_script("infer.py", infer_config, bop_root)

    inference_dir = _snapshot_inference(bop_root, infer_config)
    with open(os.path.join(inference_dir, "estimated-poses.json")) as f:
        poses = json.load(f)

    mesh_ply = os.path.join(bop_root, "custom", "models", "obj_000001.ply")
    if refinement != "none":
        print(f"== improving {len(poses)} pose(s) with silhouette matching ({refinement}) ==")
        poses = improve_poses(poses, os.path.join(inference_dir, "candidates"), nobg_paths,
                              load_mesh_triangles(mesh_ply), config.CAMERA_K, refinement)
        save_poses(poses, os.path.join(inference_dir, "estimated-poses-refined.json"))

    save_poses_csv(poses, query_images, out_dir)

    for pose in poses:
        im_id = int(pose["img_id"])
        raw_path = query_images[im_id]
        q_stem = os.path.splitext(os.path.basename(raw_path))[0]

        mask_path = os.path.join(out_dir, f"{q_stem}_mask.png")
        result_path = "(no visualization saved, see estimated_poses.csv/.json)"
        if save_mask:
            render_mask_overlay(raw_path, mesh_ply, [0.0, 0.0, 0.0], pose["R"], pose["t"],
                                 config.CAMERA_K, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                 mask_path)
            result_path = mask_path
        if save_axes:
            # draws on top of the mask overlay if we made one, otherwise on the raw photo
            axes_base = mask_path if save_mask else raw_path
            combined_path = os.path.join(
                out_dir, f"{q_stem}_mask_axes.png" if save_mask else f"{q_stem}_axes.png")
            draw_axes(axes_base, mesh_ply, [0.0, 0.0, 0.0], pose["R"], pose["t"],
                      config.CAMERA_K, combined_path)
            result_path = combined_path
        print(f"{raw_path}: inliers={pose.get('inliers_est')} score={pose.get('score')} "
              f"-> {result_path}")

    return poses
