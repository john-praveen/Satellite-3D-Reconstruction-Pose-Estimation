import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from helpers import fibonacci_sphere
from satpose.metrics import rotation_error_deg
from satpose.provenance import ObservationMap, refine_pose_weighted

K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
SIZE = (640, 480)
# Identity rotation with the object 3 units in front of the camera: the camera sits at
# z = -3 in object coordinates, so it sees the z < 0 side of the object.
R_FRONT, T_FRONT = np.eye(3), np.array([0.0, 0.0, 3.0])


class ObservationMapTests(unittest.TestCase):
    def setUp(self):
        self.points = fibonacci_sphere()
        self.map = ObservationMap(self.points, self.points)

    def test_only_the_camera_facing_side_of_a_sphere_is_observed(self):
        seen = self.map.update(R_FRONT, T_FRONT, K, SIZE)
        self.assertGreater(seen, 0)
        front_side = self.points[:, 2] < -0.6  # well inside the visible cap
        back_side = self.points[:, 2] > 0.3
        self.assertGreater(self.map.strength[front_side].min(), 0.0)
        self.assertEqual(self.map.strength[back_side].max(), 0.0)

    def test_unseen_vertices_get_the_floor_weight_and_seen_ones_more(self):
        self.map.update(R_FRONT, T_FRONT, K, SIZE)
        weights = self.map.weights
        floor = self.map.config.hallucination_weight
        self.assertAlmostEqual(weights[self.points[:, 2] > 0.3].max(), floor)
        self.assertGreater(weights[self.points[:, 2] < -0.6].min(), floor)

    def test_weights_grow_with_repeated_observation_and_stay_below_one(self):
        self.map.update(R_FRONT, T_FRONT, K, SIZE)
        once = self.map.weights.max()
        for _ in range(20):
            self.map.update(R_FRONT, T_FRONT, K, SIZE)
        self.assertGreater(self.map.weights.max(), once)
        self.assertLessEqual(self.map.weights.max(), 1.0)

    def test_observing_the_far_side_raises_overall_coverage(self):
        self.map.update(R_FRONT, T_FRONT, K, SIZE)
        half = self.map.observed_fraction()
        R_back = np.diag([-1.0, 1.0, -1.0])  # 180 degrees about Y
        self.map.update(R_back, T_FRONT, K, SIZE)
        self.assertGreater(self.map.observed_fraction(), half + 0.1)

    def test_a_nearer_surface_hides_a_farther_one(self):
        axis = np.linspace(-0.5, 0.5, 60)
        grid = np.array([[x, y] for x in axis for y in axis])
        near = np.column_stack([grid, np.zeros(len(grid))])
        far = np.column_stack([grid, np.full(len(grid), 0.5)])
        vertices = np.vstack([near, far])
        normals = np.tile([0.0, 0.0, -1.0], (len(vertices), 1))
        visible, _ = ObservationMap(vertices, normals).visible_vertices(
            R_FRONT, T_FRONT, K, SIZE)
        self.assertTrue(visible[:len(near)].all())
        self.assertFalse(visible[len(near):].any())

    def test_background_pixels_in_the_mask_are_not_counted(self):
        empty_mask = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
        self.assertEqual(self.map.update(R_FRONT, T_FRONT, K, SIZE, mask=empty_mask), 0)

    def test_weights_at_uses_the_nearest_vertex(self):
        self.map.update(R_FRONT, T_FRONT, K, SIZE)
        probe = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        seen_weight, unseen_weight = self.map.weights_at(probe)
        self.assertGreater(seen_weight, unseen_weight)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            ObservationMap(np.zeros((5, 3)), np.zeros((4, 3)))


class WeightedRefinementTests(unittest.TestCase):
    """Half the model is 'hallucinated': its geometry is shifted, so its matches are biased."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.R_true = Rotation.from_euler("xyz", [10, -20, 15], degrees=True).as_matrix()
        self.t_true = np.array([0.1, -0.05, 4.0])
        true_points = rng.uniform(-1, 1, (300, 3))
        cam = true_points @ self.R_true.T + self.t_true
        projected = cam @ K.T
        self.pixels = projected[:, :2] / projected[:, 2:3]

        self.guessed = np.arange(len(true_points)) % 2 == 0
        self.model_points = true_points.copy()
        self.model_points[self.guessed] += [0.25, 0.0, 0.0]  # wrong geometry where guessed
        self.R_start = Rotation.from_euler("xyz", [5, -10, 8], degrees=True).as_matrix()
        self.t_start = np.array([0.0, 0.0, 4.5])

    def refine(self, weights):
        return refine_pose_weighted(self.model_points, self.pixels, weights, K,
                                    self.R_start, self.t_start)

    def test_perfect_data_is_recovered_exactly(self):
        points = self.model_points.copy()
        points[self.guessed] -= [0.25, 0.0, 0.0]
        R, t = refine_pose_weighted(points, self.pixels, np.ones(len(points)), K,
                                    self.R_start, self.t_start)
        self.assertLess(rotation_error_deg(R, self.R_true), 1e-3)
        np.testing.assert_allclose(t, self.t_true, atol=1e-3)

    def test_down_weighting_guessed_geometry_improves_the_pose(self):
        uniform = np.ones(len(self.model_points))
        trusted = np.where(self.guessed, 0.02, 1.0)
        error_uniform = rotation_error_deg(self.refine(uniform)[0], self.R_true)
        error_trusted = rotation_error_deg(self.refine(trusted)[0], self.R_true)
        self.assertLess(error_trusted, error_uniform)
        self.assertLess(error_trusted, 0.5)

    def test_too_few_correspondences_are_rejected(self):
        with self.assertRaises(ValueError):
            refine_pose_weighted(self.model_points[:3], self.pixels[:3], np.ones(3), K,
                                 self.R_start, self.t_start)


if __name__ == "__main__":
    unittest.main()
