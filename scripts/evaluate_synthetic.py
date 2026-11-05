"""Scores the pose pipeline on random synthetic views whose true poses are known.

    python scripts/evaluate_synthetic.py generate --out-dir output/synthetic_eval --seed 99
    python scripts/evaluate_synthetic.py infer    --out-dir output/synthetic_eval   # FoundPose (GPU, ~10 min)
    python scripts/evaluate_synthetic.py overlays --out-dir output/synthetic_eval   # wireframes: pipeline vs as shipped
    python scripts/evaluate_synthetic.py accuracy --out-dirs output/synthetic_eval  # rotation + translation table

``--object`` picks the test object for ``generate`` (satellite, satellite_textured, rocket_body,
debris, or the path of a mesh file). ``infer`` and ``overlays`` must be started with the FoundPose
environment's Python, e.g.
    set CONDA_EXE=%USERPROFILE%\\anaconda3\\Scripts\\conda.exe
    D:\\conda_envs\\foundpose_repo\\python.exe scripts/evaluate_synthetic.py infer
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from satpose import config, evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["generate", "infer", "overlays", "accuracy"])
    parser.add_argument("--out-dir", default=os.path.join(config.OUTPUT_DIR, "synthetic_eval"))
    parser.add_argument("--out-dirs", nargs="+", default=None,
                        help="accuracy: several evaluated sets to tabulate together")
    parser.add_argument("--object", default="satellite", dest="object_name",
                        help="generate: test object name, or the path of a mesh file")
    parser.add_argument("--seed", type=int, default=evaluation.SEED,
                        help="generate: random seed; a new seed gives an independent view set")
    parser.add_argument("--views", type=int, default=evaluation.NUM_VIEWS,
                        help="generate: number of views")
    parser.add_argument("--reuse-templates", action="store_true",
                        help="infer: keep the templates/representation from an earlier run")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)

    if args.command == "generate":
        evaluation.generate(out_dir, args.seed, args.views, args.object_name)
        print(f"generated {args.views} views in {out_dir}")
    elif args.command == "infer":
        evaluation.run_inference(out_dir, reuse_templates=args.reuse_templates)
    elif args.command == "overlays":
        for label, summary in evaluation.write_overlays(out_dir).items():
            print(f"OVERLAYS {label:<10} median {summary['median_deg']:.2f} deg, "
                  f"{summary['within_threshold']}/{summary['views']} within 10 deg -> {summary['eval_dir']}")
    else:   # needs `overlays` to have saved the refined poses
        tables = {}
        for folder in args.out_dirs or [out_dir]:
            name = os.path.basename(os.path.abspath(folder))
            for label, errors in evaluation.write_accuracy(os.path.abspath(folder)).items():
                tables[f"{name} / {label}"] = errors
        text = "\n".join(evaluation.accuracy_table(tables))
        print(text)
        with open(os.path.join(config.OUTPUT_DIR, "accuracy.txt"), "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
