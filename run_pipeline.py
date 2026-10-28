"""The single interactive entry point for the whole pipeline. Run it, answer one
prompt (single-view or multi-view) -- or pre-set config.PIPELINE_MODE to skip the
prompt entirely -- and everything else (reconstruction photo(s), camera intrinsics,
where query photos come from, whether to render mask/axes visualizations) comes
straight from satpose/config.py. No CLI flags to remember; meant for handing this project
off to someone who just wants to edit a few variables and run one script.

Usage:
    python run_pipeline.py

(For scripted/CLI use with flags instead of editing satpose/config.py, see
run_single_view_pipeline.py / run_multiview_pipeline.py -- run_pipeline.py is a
thin, config-driven wrapper around the same underlying functions.)
"""
import itertools
import os
import sys
import threading
import time

from satpose import config
from satpose.reconstruction.multiview import reconstruct_multiview
from satpose.pipeline import resolve_query_images, run_official_foundpose
from satpose.reconstruction.single_view import reconstruct_single_view


class _Spinner:
    """Prints a rotating spinner + elapsed time while a blocking call (reconstruction)
    runs, so the terminal never looks stuck -- Hunyuan3D-2 also prints its own
    step-by-step progress bars underneath this, this is just a "still alive" heartbeat
    wrapped around the whole call."""

    def __init__(self, message: str):
        self.message = message
        self._stop = threading.Event()
        self._thread = None
        self._start = None

    def __enter__(self):
        print(self.message)
        self._start = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self):
        for ch in itertools.cycle("|/-\\"):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r  {ch} {time.time() - self._start:.0f}s elapsed")
            sys.stdout.flush()
            time.sleep(0.2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self._thread.join()
        sys.stdout.write(f"\r  done ({time.time() - self._start:.0f}s){' ' * 20}\n")
        sys.stdout.flush()


def _get_mode() -> str:
    if config.PIPELINE_MODE in ("single", "multi"):
        print(f"== mode (from config.PIPELINE_MODE): {config.PIPELINE_MODE}-view ==")
        return config.PIPELINE_MODE
    while True:
        answer = input("Single view or multi view? [s/m]: ").strip().lower()
        if answer in ("s", "single"):
            return "single"
        if answer in ("m", "multi"):
            return "multi"
        print("Please enter 's' (single) or 'm' (multi).")


def main():
    mode = _get_mode()

    if mode == "single":
        image = config.SINGLE_VIEW_IMAGE
        stem = os.path.splitext(os.path.basename(image))[0]
        out_dir = os.path.abspath(os.path.join(config.OUTPUT_DIR, stem))
        os.makedirs(out_dir, exist_ok=True)
        mesh_path = os.path.join(out_dir, f"{stem}_mesh.glb")
        recon_sources = [image]
        with _Spinner(f"== reconstructing 3D mesh from {image} (Hunyuan3D-2) =="):
            reconstruct_single_view(image, mesh_path)
    else:
        front, back, left, right = (config.FRONT_IMAGE, config.BACK_IMAGE,
                                     config.LEFT_IMAGE, config.RIGHT_IMAGE)
        missing = [name for name, val in
                   [("FRONT_IMAGE", front), ("BACK_IMAGE", back), ("LEFT_IMAGE", left)]
                   if not val]
        if missing:
            raise ValueError(f"multi-view mode needs satpose/config.py's {', '.join(missing)} set "
                              f"(RIGHT_IMAGE is optional)")
        stem = os.path.splitext(os.path.basename(front))[0]
        out_dir = os.path.abspath(os.path.join(config.OUTPUT_DIR, f"{stem}_multiview"))
        os.makedirs(out_dir, exist_ok=True)
        mesh_path = os.path.join(out_dir, f"{stem}_mesh_mv.glb")
        recon_sources = [front, back, left] + ([right] if right else [])
        with _Spinner(f"== reconstructing 3D mesh from {', '.join(recon_sources)} "
                      f"(Hunyuan3D-2 multiview) =="):
            reconstruct_multiview(front, back, left, mesh_path, right_path=right)

    query_images = resolve_query_images(
        config.QUERY_SOURCE, out_dir, config.VIDEO_FRAME_STRIDE) or recon_sources

    print(f"== estimating pose on {len(query_images)} photo(s), using CAMERA_K from "
          f"satpose/config.py ==")
    run_official_foundpose(mesh_path, query_images, out_dir,
                            save_mask=config.SAVE_MASK, save_axes=config.SAVE_AXES)

    print(f"\nDone. Results saved under {out_dir}:")
    print(f"  - estimated_poses.csv, and the raw estimated-poses.json under bop_datasets/outputs/inference/")
    if config.SAVE_MASK or config.SAVE_AXES:
        print(f"  - *_mask.png / *_mask_axes.png -- ALWAYS check these before trusting a pose")
    else:
        print(f"  - (SAVE_MASK/SAVE_AXES are both off in satpose/config.py -- no visualization images "
              f"were rendered, only the CSV/JSON numbers)")


if __name__ == "__main__":
    main()
