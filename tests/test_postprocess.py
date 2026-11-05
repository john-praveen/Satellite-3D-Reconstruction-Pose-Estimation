import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from satpose import config
from satpose.metrics import rotation_error_deg
from satpose.postprocess import (
    improve_poses, load_candidates, load_mask, save_poses,
)
from satpose.synthetic import build_satellite, random_poses, render

K = config.CAMERA_K
SIZE = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
FLIP_Y = np.diag([-1.0, 1.0, -1.0])


def entry(R, t, inliers):
    return {"img_id": "0", "R": np.asarray(R).tolist(), "t": [[float(v)] for v in t],
            "score": "0.3", "inliers_est": inliers}


class PostprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.satellite = build_satellite()
        cls.R, cls.t = random_poses(1, seed=5)[0]
        rgb, mask, _ = render(cls.satellite, cls.R, cls.t, K, SIZE)
        cls.folder = tempfile.TemporaryDirectory()
        cls.cutout = os.path.join(cls.folder.name, "view_cutout.png")
        Image.fromarray(np.dstack([rgb, mask.astype(np.uint8) * 255]), "RGBA").save(cls.cutout)
        cls.candidates_dir = os.path.join(cls.folder.name, "candidates")
        os.makedirs(cls.candidates_dir)
        slightly_off = Rotation.from_rotvec([0.05, -0.08, 0.04]).as_matrix() @ cls.R
        cls.right = (slightly_off, cls.t * 1.02)          # fewer inliers, right side
        cls.wrong = (cls.R @ FLIP_Y, cls.t)               # more inliers, flipped
        with open(os.path.join(cls.candidates_dir, "000000.json"), "w") as f:
            json.dump([{"R": cls.wrong[0].tolist(), "t": cls.wrong[1].tolist(), "inliers": 60},
                       {"R": cls.right[0].tolist(), "t": cls.right[1].tolist(), "inliers": 25}], f)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def run_mode(self, mode):
        # FoundPose's own pick is the flipped candidate: it has the most inliers.
        poses = [entry(*self.wrong, 60)]
        return improve_poses(poses, self.candidates_dir, [self.cutout],
                             self.satellite.triangles, K, mode)[0]

    def error(self, pose):
        return rotation_error_deg(np.array(pose["R"]), self.R)

    def test_none_keeps_foundposes_pick(self):
        self.assertGreater(self.error(self.run_mode("none")), 170)

    def test_full_selects_the_right_candidate_and_tightens_it(self):
        result = self.run_mode("full")
        self.assertLess(self.error(result), 3.0)
        self.assertLess(self.error(result), rotation_error_deg(self.right[0], self.R))
        self.assertGreater(result["silhouette_iou"], 0.93)

    def test_output_keeps_the_input_schema(self):
        result = self.run_mode("full")
        self.assertEqual(np.array(result["R"]).shape, (3, 3))
        self.assertEqual(np.array(result["t"]).shape, (3, 1))
        self.assertEqual(result["img_id"], "0")
        self.assertEqual(result["score"], "0.3")

    def test_an_image_without_candidates_refines_the_original_pose(self):
        with tempfile.TemporaryDirectory() as empty:
            pose = entry(*self.right, 25)
            result = improve_poses([pose], empty, [self.cutout], self.satellite.triangles,
                                   K, "full")[0]
            self.assertLess(self.error(result), rotation_error_deg(self.right[0], self.R))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            improve_poses([], self.candidates_dir, [], self.satellite.triangles, K, "magic")

    def test_loaders(self):
        self.assertEqual(len(load_candidates(self.candidates_dir, 0)), 2)
        self.assertEqual(load_candidates(self.candidates_dir, 7), [])
        mask = load_mask(self.cutout)
        self.assertEqual(mask.shape, (SIZE[1], SIZE[0]))
        self.assertGreater(mask.sum(), 20000)

    def test_save_poses_round_trips(self):
        path = os.path.join(self.folder.name, "out.json")
        save_poses([entry(self.R, self.t, 1)], path)
        with open(path) as f:
            self.assertEqual(json.load(f)[0]["img_id"], "0")


if __name__ == "__main__":
    unittest.main()
