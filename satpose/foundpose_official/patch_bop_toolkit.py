"""Adapts the official FoundPose repo's bop_toolkit to the "custom" dataset this project builds.

Two small, idempotent patches (applied automatically by ``satpose.pipeline``):

* ``dataset_params.py`` only knows the standard BOP benchmarks. It gets a generic "custom"
  dataset, whose image size and depth range are read from the ``dataset_meta.json`` written
  by ``build_bop_dataset.py``, so a new object never needs another patch.
* ``config.py`` ships a placeholder ``output_path`` (``/path/to/output/folder``), so templates
  and results would land in a stray folder. It now points at ``<BOP_PATH>/outputs``, which is
  where ``satpose.pipeline`` looks for the estimated poses.
* ``scripts/infer.py`` computes a coarse pose for each of its top matched templates but keeps
  only the one with the most inliers. It now also saves all of them (``candidates/<im_id>.json``
  next to the results) so ``satpose.refinement`` can re-rank and refine them.
"""
import os

MARKER = "satpose: custom dataset"
_TOOLKIT_DIR = os.path.join("external", "bop_toolkit", "bop_toolkit_lib")
_DATASET_PARAMS = os.path.join(_TOOLKIT_DIR, "dataset_params.py")
_CONFIG = os.path.join(_TOOLKIT_DIR, "config.py")
_INFER = os.path.join("scripts", "infer.py")

_SPLIT_BRANCH = '''    # satpose: custom dataset -- one reconstructed object; the image size and depth range
    # come from dataset_meta.json (written by satpose/foundpose_official/build_bop_dataset.py).
    elif dataset_name == "custom":
        import json

        with open(join(datasets_path, dataset_name, "dataset_meta.json")) as f:
            meta = json.load(f)
        p["scene_ids"] = meta["scene_ids"]
        p["im_size"] = tuple(meta["im_size"])
        if split == "test":
            p["depth_range"] = tuple(meta["depth_range"])
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)

'''

_UNKNOWN_DATASET = '    else:\n        raise ValueError("Unknown BOP dataset ({}).".format(dataset_name))'

_OUTPUT_PATH = 'output_path = r"/path/to/output/folder"'

_COARSE_ANCHOR = '                times["pose_coarse"] = timer.elapsed("Time for coarse pose")'
_SAVE_CANDIDATES = '''

                # satpose: custom dataset support -- save every coarse pose candidate (one per
                # matched template) so they can be re-ranked and refined outside this script.
                import json as _satpose_json

                _satpose_dir = os.path.join(output_dir, "candidates")
                os.makedirs(_satpose_dir, exist_ok=True)
                with open(os.path.join(_satpose_dir, f"{bop_im_id:06d}.json"), "w") as _satpose_f:
                    _satpose_json.dump(
                        [
                            {
                                "R": np.asarray(_p["R_m2c"]).tolist(),
                                "t": np.asarray(_p["t_m2c"]).reshape(-1).tolist(),
                                "inliers": _p["quality"],
                            }
                            for _p in coarse_poses
                        ],
                        _satpose_f,
                    )'''

_PATCHES = (
    (_DATASET_PARAMS, (
        ('        "handal": list(range(1, 41)),\n    }[dataset_name]',
         f'        "handal": list(range(1, 41)),\n        "custom": [1],  # {MARKER}\n    }}[dataset_name]'),
        ('        "handal": [26, 35, 36, 37, 38, 39, 40],\n    }[dataset_name]',
         f'        "handal": [26, 35, 36, 37, 38, 39, 40],\n        "custom": [],  # {MARKER}\n    }}[dataset_name]'),
        (_UNKNOWN_DATASET, _SPLIT_BRANCH + _UNKNOWN_DATASET),
    )),
    (_CONFIG, (
        (_OUTPUT_PATH,
         f'# {MARKER}: keep results next to the dataset when BOP_PATH is set.\n'
         f'output_path = (os.path.join(datasets_path, "outputs") if "BOP_PATH" in os.environ\n'
         f'               else r"/path/to/output/folder")'),
    )),
    (_INFER, (
        (_COARSE_ANCHOR, _COARSE_ANCHOR + _SAVE_CANDIDATES),
    )),
)


def _patch_file(path: str, replacements) -> bool:
    with open(path, newline="", encoding="utf-8") as f:
        original = f.read()
    if MARKER in original:
        return False

    uses_crlf = "\r\n" in original
    text = original.replace("\r\n", "\n")
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError(f"{path} does not have the expected layout; cannot patch it "
                               f"automatically")
        text = text.replace(old, new)
    if uses_crlf:
        text = text.replace("\n", "\r\n")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(text)
    return True


def apply(foundpose_repo_path: str) -> bool:
    """Patches the repo's bop_toolkit. Returns True if any file changed.

    Raises RuntimeError if a file does not look like the expected bop_toolkit version.
    """
    changed = False
    for relative_path, replacements in _PATCHES:
        changed |= _patch_file(os.path.join(foundpose_repo_path, relative_path), replacements)
    return changed
