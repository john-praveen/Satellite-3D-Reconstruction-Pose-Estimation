"""Tracking demo: a video of a turning test object with the estimated pose drawn on every frame,
and how fast the pipeline processes frames.

    python scripts/make_tracking_video.py generate --frames 48    # turntable renders + ground truth
    python scripts/make_tracking_video.py infer                   # FoundPose on every frame   (GPU)
    python scripts/make_tracking_video.py video                   # time each variant, write the videos

``generate --object`` also takes the path of a mesh file (e.g. a generated .glb), and ``video
--thin`` draws thin lines for meshes with many edges. ``infer`` and ``video`` need the FoundPose
environment's Python (see scripts/evaluate_synthetic.py). Left panel:
FoundPose as shipped; right panel: our pipeline (20 candidates + silhouette refinement).
Green is the true wireframe, red the estimate (rotation and translation). The frame rate is
how many frames per second each variant processes *one frame at a time*, measured on this
machine, not the playback rate of the video. Run it on an otherwise idle machine, or the
timings will be too slow.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from satpose import config, evaluation, verification  # noqa: E402
from satpose.objects import get_object  # noqa: E402
from satpose.postprocess import improve_poses, load_mesh_triangles  # noqa: E402
from satpose.synthetic import draw_wireframe, render, turntable_poses  # noqa: E402

BASELINE_CONFIG = evaluation.BASELINE_CONFIG   # FoundPose as shipped: 5 templates, best pose only
PANEL = 640
PLAYBACK_FPS = 8
STAGES = ("prep", "feat_extract", "grid_sample", "proj", "corresp", "pose_coarse", "final_select")


def foundpose_seconds(pose: dict) -> float:
    """Time FoundPose itself spent on this image (all its stages)."""
    return float(sum(v for v in pose.get("time", {}).values() if isinstance(v, (int, float))))


def load_poses(out_dir: str, infer_config: str) -> List[dict]:
    with open(os.path.join(evaluation.inference_dir(out_dir, infer_config), "estimated-poses.json")) as f:
        return sorted(json.load(f), key=lambda p: int(p["img_id"]))


def run_variants(out_dir: str, truth: dict) -> Dict[str, dict]:
    """Processes every frame with each variant, timing it. Returns per variant the poses
    (FoundPose format), seconds per frame and the pose errors."""
    camera_K = np.array(truth["camera_K"])
    ply = os.path.join(evaluation.foundpose_dir(out_dir), "bop_datasets", "custom", "models",
                       "obj_000001.ply")
    triangles = load_mesh_triangles(ply)
    cutouts = [os.path.join(out_dir, v["cutout"]) for v in truth["views"]]

    shipped = load_poses(out_dir, BASELINE_CONFIG)
    top20 = load_poses(out_dir, config.INFER_CONFIG)
    candidates = os.path.join(evaluation.inference_dir(out_dir, config.INFER_CONFIG), "candidates")
    variants = {
        "FoundPose as shipped": dict(poses=shipped, base=[foundpose_seconds(p) for p in shipped],
                                     mode=None),
        "our pipeline (with refinement)": dict(poses=top20, base=[foundpose_seconds(p) for p in top20],
                                 mode=config.POSE_REFINEMENT),
    }
    results = {}
    for name, spec in variants.items():
        poses, seconds = [], []
        for pose, base in zip(spec["poses"], spec["base"]):
            started = time.perf_counter()
            if spec["mode"] is None:
                improved = [pose]
            else:
                improved = improve_poses([pose], candidates, cutouts, triangles, camera_K, spec["mode"])
            seconds.append(base + time.perf_counter() - started)
            poses.append(improved[0])
        results[name] = {"poses": poses, "seconds": seconds,
                         "errors": evaluation.pose_errors(out_dir, poses)}
        print(f"{name}: {np.mean(seconds):.2f} s/frame -> {1 / np.mean(seconds):.2f} fps", flush=True)
    return results


def panel(out_dir: str, truth: dict, index: int, pose: dict, error: dict, seconds: float,
          title: str, satellite, centroid: np.ndarray, widths=(4, 3)) -> Image.Image:
    view = truth["views"][index]
    camera_K, size = np.array(truth["camera_K"]), tuple(truth["image_size"])
    R_true, t_true = np.array(view["R"]), np.array(view["t"])
    _, _, depth = render(satellite, R_true, t_true, camera_K, size)
    image = Image.open(os.path.join(out_dir, view["image"])).convert("RGB")
    draw_wireframe(image, satellite, R_true, t_true, camera_K, depth, verification.GT_COLOR,
                   width=widths[0], hidden_color=verification.GT_HIDDEN_COLOR)
    R_est = np.array(pose["R"], dtype=np.float64).reshape(3, 3)
    centre = np.array(pose["t"], dtype=np.float64).reshape(3) / evaluation.MM_SCALE
    draw_wireframe(image, satellite, R_est, centre - R_est @ centroid, camera_K, None,
                   verification.ESTIMATE_COLOR, width=widths[1])
    text = ImageDraw.Draw(image)
    font = verification._font(34)
    text.text((16, 12), title, fill=(255, 255, 255), font=font)
    text.text((16, 54), f"rotation error {error['rotation_deg']:.1f} deg, "
                        f"position error {100 * error['relative']:.1f}%", fill=(255, 255, 255), font=font)
    text.text((16, 96), f"{seconds * 1000:.0f} ms/frame = {1 / seconds:.2f} fps",
              fill=(255, 220, 120), font=font)
    text.text((16, size[1] - 50), f"frame {index + 1}", fill=(200, 200, 200), font=font)
    return image.resize((PANEL, PANEL), Image.LANCZOS)


def write_videos(out_dir: str, truth: dict, results: Dict[str, dict], widths=(4, 3)) -> None:
    import cv2
    import imageio

    satellite, _ = get_object(truth.get("object", "satellite"))
    centroid = verification.mesh_centroid(os.path.join(out_dir, "satellite.stl"))
    left, right = "FoundPose as shipped", "our pipeline (with refinement)"
    frames = []
    for index in range(len(truth["views"])):
        tiles = [panel(out_dir, truth, index, results[name]["poses"][index],
                       results[name]["errors"][index], results[name]["seconds"][index],
                       name, satellite, centroid, widths) for name in (left, right)]
        sheet = Image.new("RGB", (2 * PANEL, PANEL))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (i * PANEL, 0))
        frames.append(np.array(sheet))

    mp4 = os.path.join(out_dir, "tracking.mp4")
    writer = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), PLAYBACK_FPS, (2 * PANEL, PANEL))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    gif = os.path.join(out_dir, "tracking.gif")
    small = [np.array(Image.fromarray(f).resize((PANEL, PANEL // 2), Image.LANCZOS)) for f in frames]
    imageio.mimsave(gif, small, duration=1000 // PLAYBACK_FPS, loop=0)
    print(f"videos: {mp4}\n        {gif}")


def report(out_dir: str, results: Dict[str, dict], shipped_stages: List[dict]) -> str:
    lines = [f"{'variant':<24}{'ms/frame (mean)':>17}{'fps':>8}{'rot med':>9}{'rot mean':>10}"
             f"{'<=10deg':>9}{'pos med':>9}", "-" * 86]
    for name, r in results.items():
        rot = np.array([e["rotation_deg"] for e in r["errors"]])
        rel = np.array([100 * e["relative"] for e in r["errors"]])
        lines.append(f"{name:<24}{1000 * np.mean(r['seconds']):17.0f}{1 / np.mean(r['seconds']):8.2f}"
                     f"{np.median(rot):8.2f}d{rot.mean():9.1f}d{100 * np.mean(rot <= 10):8.0f}%"
                     f"{np.median(rel):8.2f}%")
    lines += ["", "FoundPose time per frame by stage (mean, ms):"]
    for stage in STAGES:
        values = [p["time"].get(stage, 0.0) for p in shipped_stages]
        lines.append(f"  {stage:<14}{1000 * np.mean(values):8.0f}")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "fps_report.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "frame_errors.json"), "w") as f:   # per frame, for finding failures
        json.dump({name: {"seconds": r["seconds"],
                          "rotation_deg": [e["rotation_deg"] for e in r["errors"]],
                          "position_relative": [e["relative"] for e in r["errors"]]}
                   for name, r in results.items()}, f, indent=1)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["generate", "infer", "video"])
    parser.add_argument("--out-dir", default=os.path.join(config.OUTPUT_DIR, "turntable_synthetic"))
    parser.add_argument("--frames", type=int, default=48, help="generate: frames in one full turn")
    parser.add_argument("--object", default="satellite", dest="object_name",
                        help="generate: a test object name, or the path of a mesh file")
    parser.add_argument("--thin", action="store_true",
                        help="video: thin wireframe lines, for meshes with many edges")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)

    if args.command == "generate":
        verification.generate(out_dir, object_name=args.object_name,
                              poses=turntable_poses(args.frames))
        print(f"generated {args.frames} turntable frames in {out_dir}")
    elif args.command == "infer":
        evaluation.run_inference(out_dir, (BASELINE_CONFIG, config.INFER_CONFIG))
    else:
        truth = verification.load_ground_truth(out_dir)
        results = run_variants(out_dir, truth)
        print(report(out_dir, results, load_poses(out_dir, BASELINE_CONFIG)))
        write_videos(out_dir, truth, results, (2, 1) if args.thin else (4, 3))


if __name__ == "__main__":
    main()
