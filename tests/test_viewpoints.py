import unittest

import numpy as np

from helpers import look_at_pose, pose_from_azimuth
from satpose.viewpoints import (
    ADDED, REJECTED, REPLACED, KeyframeSelector, angular_distance_deg, assign_slots,
    azimuth_coverage_deg, azimuth_elevation_deg, camera_direction_in_object,
    circular_difference_deg, missing_slots,
)

FRONT = np.array([0.0, 0.0, 1.0])


class GeometryTests(unittest.TestCase):
    def test_camera_direction_is_recovered_from_pose(self):
        direction = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])
        R, t = look_at_pose(direction, distance=5.0)
        self.assertAlmostEqual(np.linalg.det(R), 1.0)
        np.testing.assert_allclose(camera_direction_in_object(R, t), direction, atol=1e-9)

    def test_azimuth_follows_right_hand_rule_about_up(self):
        expected = {(0, 0, 1): 0, (1, 0, 0): 90, (0, 0, -1): 180, (-1, 0, 0): 270}
        for direction, azimuth in expected.items():
            self.assertAlmostEqual(azimuth_elevation_deg(direction, FRONT)[0], azimuth)

    def test_elevation_is_measured_from_the_horizontal_plane(self):
        _, elevation = azimuth_elevation_deg([0, 1, 1], FRONT)
        self.assertAlmostEqual(elevation, 45.0)

    def test_straight_up_has_zero_azimuth_and_full_elevation(self):
        azimuth, elevation = azimuth_elevation_deg([0, 1, 0], FRONT)
        self.assertEqual(azimuth, 0.0)
        self.assertAlmostEqual(elevation, 90.0)

    def test_reference_along_up_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            azimuth_elevation_deg([1, 0, 0], [0, 1, 0])

    def test_angle_helpers(self):
        self.assertAlmostEqual(angular_distance_deg([1, 0, 0], [0, 1, 0]), 90.0)
        self.assertAlmostEqual(circular_difference_deg(350, 10), 20.0)
        self.assertAlmostEqual(circular_difference_deg(0, 180), 180.0)


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.selector = KeyframeSelector(min_separation_deg=30.0, min_quality=0.5,
                                         replace_margin=0.1)

    def offer(self, azimuth, quality, frame_id=None):
        R, t = pose_from_azimuth(azimuth)
        return self.selector.consider(frame_id if frame_id is not None else azimuth, R, t,
                                      quality)

    def test_first_frame_becomes_the_front_reference(self):
        self.assertEqual(self.offer(40, 1.0), ADDED)
        self.assertAlmostEqual(self.selector.keyframes[0].azimuth_deg, 0.0)

    def test_novel_viewpoint_is_added_and_near_duplicate_rejected(self):
        self.offer(0, 1.0)
        self.assertEqual(self.offer(90, 1.0), ADDED)
        self.assertEqual(self.offer(100, 1.0), REJECTED)
        self.assertEqual(len(self.selector.keyframes), 2)

    def test_low_quality_frame_is_rejected_even_if_novel(self):
        self.offer(0, 1.0)
        self.assertEqual(self.offer(120, 0.1), REJECTED)

    def test_better_frame_replaces_a_nearby_keyframe_but_never_the_reference(self):
        self.offer(0, 0.6)
        self.offer(90, 0.6, frame_id="old")
        self.assertEqual(self.offer(95, 0.9, frame_id="new"), REPLACED)
        self.assertEqual([k.frame_id for k in self.selector.keyframes][1], "new")
        self.assertEqual(self.offer(5, 0.99), REJECTED)  # reference stays put

    def test_novelty_is_distance_to_nearest_keyframe(self):
        self.assertEqual(self.selector.novelty_deg(*pose_from_azimuth(0)), 180.0)
        self.offer(0, 1.0, frame_id="a")
        R, t = pose_from_azimuth(60, elevation_deg=10.0)
        self.assertAlmostEqual(self.selector.novelty_deg(R, t), 60.0 * np.cos(np.radians(10)),
                               delta=1.0)


class SlotTests(unittest.TestCase):
    def keyframes_at(self, azimuths):
        selector = KeyframeSelector(min_separation_deg=10.0)
        for azimuth in azimuths:
            selector.consider(azimuth, *pose_from_azimuth(azimuth), quality=1.0)
        return selector.keyframes

    def test_four_views_fill_all_four_slots(self):
        assignment = assign_slots(self.keyframes_at([0, 95, 175, 265]))
        self.assertEqual({s: k.frame_id for s, k in assignment.items()},
                         {"front": 0, "left": 95, "back": 175, "right": 265})
        self.assertEqual(missing_slots(assignment), [])

    def test_slots_without_a_nearby_view_stay_empty(self):
        assignment = assign_slots(self.keyframes_at([0, 90]), tolerance_deg=30.0)
        self.assertEqual(sorted(assignment), ["front", "left"])
        self.assertEqual(missing_slots(assignment), ["back"])

    def test_a_keyframe_fills_only_one_slot(self):
        assignment = assign_slots(self.keyframes_at([0, 135]), tolerance_deg=60.0)
        self.assertEqual(len({id(k) for k in assignment.values()}), len(assignment))

    def test_coverage_is_360_minus_largest_gap(self):
        self.assertAlmostEqual(azimuth_coverage_deg(self.keyframes_at([0, 90, 180, 270])), 270.0)
        self.assertAlmostEqual(azimuth_coverage_deg(self.keyframes_at([0, 90])), 90.0)
        self.assertEqual(azimuth_coverage_deg(self.keyframes_at([0])), 0.0)


if __name__ == "__main__":
    unittest.main()
