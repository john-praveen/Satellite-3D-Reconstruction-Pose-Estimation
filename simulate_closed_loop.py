"""Runs the closed loop (Contributions 1 and 3) on a simulated fly-around, with fake stages.

SIMULATION ONLY: the pose "estimator" returns the true pose plus noise that shrinks as the
model improves, and the "rebuild" just sleeps. It demonstrates the scheduling -- keyframe
selection, the background rebuild, the atomic model swap, and a fast loop that never
waits -- not the accuracy of Hunyuan3D or FoundPose.

Usage:
    python simulate_closed_loop.py
    python simulate_closed_loop.py --rebuild-seconds 5 --frames 120
"""
import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from satpose.metrics import rotation_error_deg
from satpose.runtime import ClosedLoopPipeline, PoseEstimate
from satpose.viewpoints import KeyframeSelector


def pose_at_azimuth(azimuth_deg: float, elevation_deg: float = 10.0, distance: float = 2.0):
    """True pose of a camera circling the target about +Y, looking at the origin."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    direction = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    forward = -direction
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(forward, right), forward])
    return R, -R @ (direction * distance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=72, help="frames in one 360 degree orbit")
    parser.add_argument("--frame-seconds", type=float, default=0.05,
                        help="simulated pose-estimation time per frame")
    parser.add_argument("--rebuild-seconds", type=float, default=1.5,
                        help="simulated model rebuild time (about 180 on a Jetson Orin AGX)")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    truth = {}

    def estimate_pose(frame_id, model_quality):
        """Fake FoundPose: true pose plus noise that falls as the model improves."""
        time.sleep(args.frame_seconds)
        R, t = truth[frame_id]
        noise_deg = 1.0 + 12.0 * (1.0 - model_quality)
        noisy = Rotation.from_rotvec(rng.normal(0, np.radians(noise_deg) / 3, 3)).as_matrix() @ R
        return PoseEstimate(noisy, t, quality=1.0)

    def rebuild_model(slot_images):
        """Fake Hunyuan3D-2mv: more filled view slots -> better model."""
        time.sleep(args.rebuild_seconds)
        return len(slot_images) / 4.0

    pipeline = ClosedLoopPipeline(
        estimate_pose=lambda frame_id, model: estimate_pose(frame_id, model),
        rebuild_model=rebuild_model,
        initial_model=0.25,  # a single-view mesh: guessed everywhere but the first side
        selector=KeyframeSelector(min_separation_deg=25.0))

    print("SIMULATION: fake pose estimator and fake rebuild; shows scheduling, not accuracy.\n")
    print(f"{'frame':>5} {'azimuth':>11} {'model':>6} {'rot.err':>10}  event")
    for index in range(args.frames):
        azimuth = 360.0 * index / args.frames
        truth[index] = pose_at_azimuth(azimuth)
        result = pipeline.process_frame(index, index)
        error = rotation_error_deg(result.pose.R, truth[index][0])
        event = "<- rebuild started in background" if result.rebuild_started else ""
        if index % 4 == 0 or event:
            print(f"{index:5d} {azimuth:7.0f} deg v{result.model_version:<5d} {error:7.2f} deg  {event}")

    pipeline.wait_for_rebuild()
    stats = pipeline.stats.summary()
    print(f"\nframes: {stats['frames']}   rebuilds: {stats['rebuilds']} "
          f"(mean {stats['rebuild_mean_s']:.1f}s)")
    print(f"fast-loop latency: mean {stats['frame_latency_mean_s'] * 1000:.0f} ms, "
          f"max {stats['frame_latency_max_s'] * 1000:.0f} ms "
          f"(a rebuild takes {args.rebuild_seconds:.1f}s, yet no frame waited for one)")


if __name__ == "__main__":
    main()
