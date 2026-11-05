import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from satpose.metrics import (
    pointing_error_deg, rotation_error_deg, speed_score, translation_components, translation_error,
)


class MetricTests(unittest.TestCase):
    def test_rotation_error_matches_the_rotation_angle(self):
        R = Rotation.from_euler("z", 25, degrees=True).as_matrix()
        self.assertAlmostEqual(rotation_error_deg(np.eye(3), R), 25.0, places=6)
        self.assertAlmostEqual(rotation_error_deg(R, R), 0.0, places=6)

    def test_pointing_error_ignores_scale(self):
        self.assertAlmostEqual(pointing_error_deg([0, 0, 5], [0, 0, 1]), 0.0, places=6)
        self.assertAlmostEqual(pointing_error_deg([1, 0, 0], [0, 1, 0]), 90.0, places=6)

    def test_translation_error_is_euclidean(self):
        self.assertAlmostEqual(translation_error([0, 0, 0], [3, 4, 0]), 5.0)

    def test_translation_is_split_into_lateral_and_depth_and_made_relative(self):
        parts = translation_components([0.3, 0.4, 4.0 + 1.2], [0.0, 0.0, 4.0])
        self.assertAlmostEqual(parts["lateral"], 0.5)
        self.assertAlmostEqual(parts["depth"], 1.2)
        self.assertAlmostEqual(parts["total"], np.hypot(0.5, 1.2))
        self.assertAlmostEqual(parts["relative"], 1.3 / 4.0)       # 1.3 off at a distance of 4
        self.assertAlmostEqual(translation_components([0, 0, 2.8], [0, 0, 4.0])["depth"], 1.2)

    def test_speed_score_adds_radians_and_relative_translation(self):
        self.assertAlmostEqual(speed_score(0.0, 0.0), 0.0)
        self.assertAlmostEqual(speed_score(180.0, 0.05), np.pi + 0.05)


if __name__ == "__main__":
    unittest.main()
