import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from satpose import config
from satpose.metrics import rotation_error_deg
from satpose.refinement import (
    Candidate, RefinementConfig, SilhouetteMatcher, decimate_triangles, refine_pose,
    select_best_pose,
)
from satpose.synthetic import build_satellite, random_poses, render

K = config.CAMERA_K
SIZE = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
FLIP_Y = np.diag([-1.0, 1.0, -1.0])  # 180 degrees about the satellite's vertical axis


def no_refinement():
    """Swaps the optimiser for one that keeps the pose, to test the candidate scoring alone."""
    def keep(matcher, R, t):
        R, t = np.asarray(R, float), np.asarray(t, float).reshape(3)
        return R, t, matcher.iou(R, t)
    return mock.patch("satpose.refinement.refine_pose", side_effect=keep)


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.satellite = build_satellite()
        cls.R, cls.t = random_poses(1, seed=5)[0]
        _, cls.mask, _ = render(cls.satellite, cls.R, cls.t, K, SIZE)
        cls.matcher = SilhouetteMatcher(cls.satellite.triangles, cls.mask, K)

    def perturbed(self, degrees=8.0, axis=(1.0, 2.0, -1.0), shift=0.04):
        axis = np.asarray(axis) / np.linalg.norm(axis)
        R = Rotation.from_rotvec(axis * np.radians(degrees)).as_matrix() @ self.R
        return R, self.t * (1.0 + shift)

    def test_the_true_pose_matches_its_own_mask_almost_perfectly(self):
        self.assertGreater(self.matcher.iou(self.R, self.t), 0.93)

    def test_the_overlap_peaks_at_the_true_distance_and_position(self):
        # A matcher that draws shapes slightly too big (or off-centre) would be best-fitted by
        # an object that is too far away (or shifted), a bias refinement would pass on to the
        # translation of every estimate.
        scales = np.linspace(0.98, 1.02, 41)
        best = scales[int(np.argmax([self.matcher.iou(self.R, self.t * s) for s in scales]))]
        self.assertAlmostEqual(best, 1.0, delta=0.004)
        shifts = np.linspace(-0.02, 0.02, 41) * np.linalg.norm(self.t)
        best_shift = shifts[int(np.argmax([self.matcher.iou(self.R, self.t + [d, 0.0, 0.0])
                                           for d in shifts]))]
        self.assertAlmostEqual(best_shift / np.linalg.norm(self.t), 0.0, delta=0.0012)

    def test_a_pose_looking_from_the_other_side_matches_clearly_worse(self):
        wrong = self.matcher.iou(self.R @ FLIP_Y, self.t)
        self.assertLess(wrong, self.matcher.iou(self.R, self.t) - 0.1)

    def test_a_pose_behind_the_camera_scores_zero(self):
        self.assertEqual(self.matcher.iou(self.R, np.array([0.0, 0.0, -4.0])), 0.0)

    def test_refinement_pulls_a_perturbed_pose_back_to_the_truth(self):
        R0, t0 = self.perturbed()
        R, t, iou = refine_pose(self.matcher, R0, t0)
        self.assertLess(rotation_error_deg(R0, self.R), 8.5)
        self.assertLess(rotation_error_deg(R, self.R), 4.0)
        self.assertGreater(iou, self.matcher.iou(R0, t0) + 0.05)
        self.assertLess(np.linalg.norm(t - self.t) / np.linalg.norm(self.t), 0.04)

    def test_refinement_stays_within_its_rotation_limit(self):
        R0 = Rotation.from_rotvec([0.0, np.radians(60), 0.0]).as_matrix() @ self.R
        R, _, _ = refine_pose(self.matcher, R0, self.t)
        moved = rotation_error_deg(R, R0)
        self.assertLessEqual(moved, self.matcher.config.max_rotation_deg * 1.8 + 1e-6)

    def test_the_best_candidate_wins_even_if_a_wrong_one_has_more_inliers(self):
        R_good, t_good = self.perturbed(degrees=5.0)
        # The flipped candidate has 50% more matched points, yet the outline clearly prefers the
        # other. (With a far larger gap, e.g. 4x the points, the inlier bonus can overrule it.)
        candidates = [Candidate(self.R @ FLIP_Y, self.t, inliers=60.0),
                      Candidate(R_good, t_good, inliers=40.0)]
        best, scored = select_best_pose(candidates, self.satellite.triangles, self.mask, K)
        self.assertEqual(len(scored), 2)
        self.assertLess(rotation_error_deg(best.R, self.R), 4.0)
        self.assertEqual(best.inliers, 40.0)
        self.assertEqual(best.iou, max(p.iou for p in scored))

    def test_matched_points_break_a_near_tie_in_outline_fit_only_when_weighted(self):
        slightly_worse = Rotation.from_rotvec([0.0, 0.0, np.radians(3.0)]).as_matrix() @ self.R
        candidates = [Candidate(self.R, self.t, inliers=10.0),
                      Candidate(slightly_worse, self.t, inliers=40.0)]
        args = (candidates, self.satellite.triangles, self.mask, K)
        with no_refinement():
            outline_only, scored = select_best_pose(*args, config=RefinementConfig(inlier_weight=0.0))
            weighted, _ = select_best_pose(*args, config=RefinementConfig(inlier_weight=0.2))
        self.assertLess(scored[1].iou, scored[0].iou)          # near-tie, outline prefers #0
        self.assertEqual(outline_only.inliers, 10.0)
        self.assertEqual(weighted.inliers, 40.0)               # appearance breaks the tie
        self.assertAlmostEqual(weighted.score, weighted.iou + 0.2)

    def test_matched_points_are_weighted_by_default(self):
        self.assertEqual(RefinementConfig().inlier_weight, 0.2)

    def test_a_large_outline_gap_is_not_overruled_by_matched_points(self):
        candidates = [Candidate(self.R, self.t, inliers=5.0),
                      Candidate(self.R @ FLIP_Y, self.t, inliers=50.0)]
        with no_refinement():
            best, _ = select_best_pose(candidates, self.satellite.triangles, self.mask, K,
                                       RefinementConfig(inlier_weight=0.1))
        self.assertEqual(best.inliers, 5.0)

    def test_no_candidates_gives_no_pose(self):
        self.assertEqual(select_best_pose([], self.satellite.triangles, self.mask, K), (None, []))


class DecimationTests(unittest.TestCase):
    def test_a_dense_grid_is_simplified_and_keeps_its_extent(self):
        n = 40
        xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        vertices = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])
        index = np.arange(n * n).reshape(n, n)
        faces = np.concatenate([
            np.stack([index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, :-1].ravel()], 1),
            np.stack([index[:-1, 1:].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()], 1)])
        triangles = decimate_triangles(vertices, faces, cells=8)
        self.assertLess(len(triangles), len(faces) / 10)
        self.assertGreater(len(triangles), 20)
        points = triangles.reshape(-1, 3)
        self.assertLess(abs(points[:, 0].min()), 0.1)
        self.assertGreater(points[:, 0].max(), 0.9)
        for tri in triangles:  # no triangle collapsed to a point or a line
            self.assertGreater(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])), 1e-9)


if __name__ == "__main__":
    unittest.main()
