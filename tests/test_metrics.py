import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from satpose.metrics import pointing_error_deg, rotation_error_deg, translation_error


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


if __name__ == "__main__":
    unittest.main()
