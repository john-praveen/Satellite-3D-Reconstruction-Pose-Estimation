"""Registers a generic "custom" dataset in the official FoundPose repo's bop_toolkit.

The official scripts look up datasets by name in ``bop_toolkit_lib/dataset_params.py`` and
only know the standard BOP benchmarks. ``build_bop_dataset.py`` packages one reconstructed
object as dataset "custom"; this patch teaches bop_toolkit about it, reading the image size
and depth range from the ``dataset_meta.json`` that ``build_bop_dataset.py`` writes, so a
new object never needs another patch.

``apply`` is idempotent and is called automatically by ``satpose.pipeline``.
"""
import os

MARKER = "satpose: custom dataset"
_RELATIVE_PATH = os.path.join("external", "bop_toolkit", "bop_toolkit_lib", "dataset_params.py")

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

_REPLACEMENTS = (
    ('        "handal": list(range(1, 41)),\n    }[dataset_name]',
     f'        "handal": list(range(1, 41)),\n        "custom": [1],  # {MARKER}\n    }}[dataset_name]'),
    ('        "handal": [26, 35, 36, 37, 38, 39, 40],\n    }[dataset_name]',
     f'        "handal": [26, 35, 36, 37, 38, 39, 40],\n        "custom": [],  # {MARKER}\n    }}[dataset_name]'),
    (_UNKNOWN_DATASET, _SPLIT_BRANCH + _UNKNOWN_DATASET),
)


def apply(foundpose_repo_path: str) -> bool:
    """Patches the repo's dataset_params.py. Returns True if it changed the file.

    Raises RuntimeError if the file does not look like the expected bop_toolkit version.
    """
    path = os.path.join(foundpose_repo_path, _RELATIVE_PATH)
    with open(path, newline="", encoding="utf-8") as f:
        original = f.read()
    if MARKER in original:
        return False

    uses_crlf = "\r\n" in original
    text = original.replace("\r\n", "\n")
    for old, new in _REPLACEMENTS:
        if text.count(old) != 1:
            raise RuntimeError(f"{path} does not have the expected layout; cannot add the "
                               f"'custom' dataset automatically")
        text = text.replace(old, new)
    if uses_crlf:
        text = text.replace("\n", "\r\n")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(text)
    return True
