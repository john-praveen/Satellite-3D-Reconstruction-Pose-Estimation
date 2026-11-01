"""Checks the pose pipeline end to end on a synthetic box satellite with known orientations.

Steps (run them in order, or just `run`):
    python verify_orientation.py generate    # satellite STL + 12 renders + ground-truth wireframes
    python verify_orientation.py run         # generate, run FoundPose on it, compare, draw overlays
    python verify_orientation.py evaluate    # compare an existing FoundPose result with truth
    python verify_orientation.py selftest    # same report with FAKE estimates (no FoundPose needed)

Everything is written under output/synthetic_satellite/ (change with --out-dir). Open
`gt_contact_sheet.png` to see the ground-truth wireframes, and `wireframe_eval/contact_sheet.png`
to see the estimated orientation (red) against the truth (green).

`run` needs the FoundPose setup from README.md ("Official pipeline setup"). Hunyuan3D is NOT
needed: the satellite's own STL is used as the mesh, so this tests the pose stage alone.
"""
import argparse
import os

from satpose import config
from satpose import verification as v


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["generate", "run", "evaluate", "selftest"])
    parser.add_argument("--out-dir", default=os.path.join(config.OUTPUT_DIR, "synthetic_satellite"))
    parser.add_argument("--views", type=int, default=12, help="number of rendered views")
    parser.add_argument("--threshold-deg", type=float, default=10.0,
                        help="rotation error counted as a success in the summary")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)

    if args.command in ("generate", "run") or not os.path.exists(v._paths(out_dir)["gt"]):
        v.generate(out_dir, num_views=args.views)
        print(f"generated satellite, renders and ground truth in {out_dir}")
    if args.command == "generate":
        print(f"ground-truth wireframes: {out_dir}{os.sep}gt_contact_sheet.png")
        return

    if args.command == "run":
        poses_path = v.run_foundpose(out_dir)
        summary = v.evaluate(out_dir, poses_path, threshold_deg=args.threshold_deg)
    elif args.command == "evaluate":
        summary = v.evaluate(out_dir, threshold_deg=args.threshold_deg)
    else:
        print("SELFTEST: these estimates are fake (truth + noise, one view flipped).")
        summary = v.selftest(out_dir)
    print(v.format_summary(summary))


if __name__ == "__main__":
    main()
