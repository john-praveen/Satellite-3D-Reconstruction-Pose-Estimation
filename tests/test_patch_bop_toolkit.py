import os
import tempfile
import unittest

from satpose.foundpose_official import patch_bop_toolkit as patch

MINIMAL_DATASET_PARAMS = '''import math
from os.path import join


def get_model_params(datasets_path, dataset_name):
    obj_ids = {
        "lm": list(range(1, 16)),
        "handal": list(range(1, 41)),
    }[dataset_name]
    symmetric_obj_ids = {
        "lm": [3, 7, 10, 11],
        "handal": [26, 35, 36, 37, 38, 39, 40],
    }[dataset_name]
    return obj_ids, symmetric_obj_ids


def get_split_params(datasets_path, dataset_name, split):
    p = {}
    if dataset_name == "lm":
        p["scene_ids"] = list(range(1, 16))
    else:
        raise ValueError("Unknown BOP dataset ({}).".format(dataset_name))
    return p
'''


MINIMAL_CONFIG = '''import os

if "BOP_PATH" in os.environ:
    datasets_path = os.environ["BOP_PATH"]
else:
    datasets_path = r"/path/to/bop/datasets"

output_path = r"/path/to/output/folder"
'''


MINIMAL_INFER = '''import os

import numpy as np


def run(output_dir, bop_im_id, coarse_poses, timer, times):
    for _ in [0]:
        for _ in [0]:
            for _ in [0]:
                times["pose_coarse"] = timer.elapsed("Time for coarse pose")
    return times
'''


def make_repo(folder, text, newline="\n"):
    """Writes a miniature official repo (the three patched files); returns dataset_params."""
    target = os.path.join(folder, patch._DATASET_PARAMS)
    os.makedirs(os.path.dirname(target))
    os.makedirs(os.path.join(folder, os.path.dirname(patch._INFER)))
    with open(target, "w", newline="") as f:
        f.write(text.replace("\n", newline))
    with open(os.path.join(folder, patch._CONFIG), "w", newline="") as f:
        f.write(MINIMAL_CONFIG.replace("\n", newline))
    with open(os.path.join(folder, patch._INFER), "w", newline="") as f:
        f.write(MINIMAL_INFER.replace("\n", newline))
    return target


def load_module(path):
    namespace = {"__name__": "dataset_params"}
    with open(path, newline="") as f:
        exec(compile(f.read(), path, "exec"), namespace)
    return namespace


class PatchTests(unittest.TestCase):
    def test_custom_dataset_is_registered_and_reads_its_meta_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = make_repo(folder, MINIMAL_DATASET_PARAMS)
            self.assertTrue(patch.apply(folder))
            module = load_module(target)
            self.assertEqual(module["get_model_params"]("", "custom"), ([1], []))

            datasets = os.path.join(folder, "datasets")
            os.makedirs(os.path.join(datasets, "custom"))
            with open(os.path.join(datasets, "custom", "dataset_meta.json"), "w") as f:
                f.write('{"im_size": [1024, 1024], "scene_ids": [0], "depth_range": [800, 3000]}')
            params = module["get_split_params"](datasets, "custom", "test")
            self.assertEqual(params["im_size"], (1024, 1024))
            self.assertEqual(params["depth_range"], (800, 3000))
            self.assertEqual(params["scene_ids"], [0])

    def test_output_path_follows_bop_path_when_it_is_set(self):
        with tempfile.TemporaryDirectory() as folder:
            make_repo(folder, MINIMAL_DATASET_PARAMS)
            patch.apply(folder)
            config_path = os.path.join(folder, patch._CONFIG)
            previous = os.environ.get("BOP_PATH")
            try:
                os.environ["BOP_PATH"] = os.path.join(folder, "bop")
                self.assertEqual(load_module(config_path)["output_path"],
                                 os.path.join(folder, "bop", "outputs"))
                del os.environ["BOP_PATH"]
                self.assertEqual(load_module(config_path)["output_path"],
                                 "/path/to/output/folder")
            finally:
                if previous is None:
                    os.environ.pop("BOP_PATH", None)
                else:
                    os.environ["BOP_PATH"] = previous

    def test_infer_saves_every_coarse_candidate_and_still_returns_normally(self):
        import json

        import numpy as np

        class Timer:
            def elapsed(self, message):
                return 0.5

        with tempfile.TemporaryDirectory() as folder:
            make_repo(folder, MINIMAL_DATASET_PARAMS)
            patch.apply(folder)
            module = load_module(os.path.join(folder, patch._INFER))
            candidates = [
                {"R_m2c": np.eye(3), "t_m2c": np.array([[1.0], [2.0], [3.0]]), "quality": 40.0},
                {"R_m2c": -np.eye(3), "t_m2c": np.array([4.0, 5.0, 6.0]), "quality": 12.0},
            ]
            times = module["run"](folder, 3, candidates, Timer(), {})
            self.assertEqual(times, {"pose_coarse": 0.5})  # original behaviour intact
            with open(os.path.join(folder, "candidates", "000003.json")) as f:
                saved = json.load(f)
            self.assertEqual([c["inliers"] for c in saved], [40.0, 12.0])
            self.assertEqual(saved[0]["t"], [1.0, 2.0, 3.0])
            self.assertEqual(saved[1]["R"][0][0], -1.0)

    def test_other_datasets_still_work_and_unknown_ones_still_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            target = make_repo(folder, MINIMAL_DATASET_PARAMS)
            patch.apply(folder)
            module = load_module(target)
            self.assertEqual(module["get_split_params"]("", "lm", "test")["scene_ids"],
                             list(range(1, 16)))
            with self.assertRaises(ValueError):
                module["get_split_params"]("", "nonsense", "test")

    def test_applying_twice_changes_nothing_the_second_time(self):
        with tempfile.TemporaryDirectory() as folder:
            target = make_repo(folder, MINIMAL_DATASET_PARAMS)
            patch.apply(folder)
            with open(target, newline="") as f:
                once = f.read()
            self.assertFalse(patch.apply(folder))
            with open(target, newline="") as f:
                self.assertEqual(f.read(), once)

    def test_windows_line_endings_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            target = make_repo(folder, MINIMAL_DATASET_PARAMS, newline="\r\n")
            self.assertTrue(patch.apply(folder))
            with open(target, newline="") as f:
                text = f.read()
            self.assertEqual(text.count("\n"), text.count("\r\n"))  # no bare LF line endings

    def test_unexpected_file_layout_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            make_repo(folder, "def nothing():\n    pass\n")
            with self.assertRaises(RuntimeError):
                patch.apply(folder)


if __name__ == "__main__":
    unittest.main()
