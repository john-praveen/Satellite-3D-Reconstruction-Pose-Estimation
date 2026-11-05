import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation

from satpose import config, verification
from satpose.metrics import rotation_error_deg
from satpose.synthetic import (
    build_satellite, draw_wireframe, look_at_pose, read_binary_stl, render, sample_poses,
    turntable_poses, write_binary_stl,
)

K = config.CAMERA_K
SIZE = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.satellite = build_satellite()

    def test_geometry_is_centred_and_fits_the_frame_at_the_default_distance(self):
        points = self.satellite.triangles.reshape(-1, 3)
        np.testing.assert_allclose((points.min(0) + points.max(0)) / 2, 0.0, atol=1e-9)
        pixel_radius = K[0, 0] * self.satellite.radius / 4.0
        self.assertLess(pixel_radius, SIZE[0] / 2)

    def test_winding_agrees_with_the_stored_outward_normals(self):
        tri = self.satellite.triangles
        geometric = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.assertTrue((np.einsum("ij,ij->i", geometric, self.satellite.normals) > 0).all())
        np.testing.assert_allclose(np.linalg.norm(self.satellite.normals, axis=1), 1.0)

    def test_stl_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sat.stl")
            write_binary_stl(path, self.satellite.triangles, self.satellite.normals)
            np.testing.assert_allclose(read_binary_stl(path), self.satellite.triangles,
                                       atol=1e-6)

    def test_poses_are_proper_rotations_at_the_requested_distance(self):
        for R, t in sample_poses(12, distance=4.0):
            self.assertAlmostEqual(np.linalg.det(R), 1.0)
            self.assertAlmostEqual(np.linalg.norm(t), 4.0)


class TurntableTests(unittest.TestCase):
    def test_consecutive_views_turn_by_the_same_small_angle_and_the_orbit_closes(self):
        poses = turntable_poses(24, distance=4.0)
        steps = [rotation_error_deg(poses[(i + 1) % 24][0], poses[i][0]) for i in range(24)]
        np.testing.assert_allclose(steps, steps[0], atol=1e-6)      # constant, including the wrap
        self.assertAlmostEqual(steps[0], 15.0, delta=1.0)          # a full turn in 24 steps
        for R, t in poses:
            self.assertAlmostEqual(np.linalg.norm(t), 4.0)
            self.assertAlmostEqual(np.linalg.det(R), 1.0)

    def test_generate_uses_the_given_poses(self):
        poses = turntable_poses(3)
        with tempfile.TemporaryDirectory() as folder:
            verification.generate(folder, poses=poses)
            truth = verification.load_ground_truth(folder)
            self.assertEqual(len(truth["views"]), 3)
            for view, (R, t) in zip(truth["views"], poses):
                np.testing.assert_allclose(view["R"], R)
                np.testing.assert_allclose(view["t"], t)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.satellite = build_satellite()
        self.R, self.t = look_at_pose([0.25, 0.2, 1.0], 4.0)

    def test_front_view_is_centred_in_the_image(self):
        _, mask, _ = render(self.satellite, self.R, self.t, K, SIZE)
        ys, xs = np.nonzero(mask)
        self.assertGreater(mask.sum(), 20000)
        self.assertAlmostEqual((xs.min() + xs.max()) / 2, SIZE[0] / 2, delta=80)
        self.assertAlmostEqual((ys.min() + ys.max()) / 2, SIZE[1] / 2, delta=80)

    def test_wireframe_lies_on_the_rendered_silhouette(self):
        rgb, mask, depth = render(self.satellite, self.R, self.t, K, SIZE)
        wire = Image.fromarray(rgb)
        draw_wireframe(wire, self.satellite, self.R, self.t, K, depth, (0, 255, 0), width=1)
        drawn = (np.array(wire) != rgb).any(axis=2)
        self.assertGreater(drawn.sum(), 500)
        near_object = binary_dilation(mask, iterations=4)
        self.assertGreater((drawn & near_object).sum() / drawn.sum(), 0.97)

    def test_the_satellite_looks_different_when_turned_around(self):
        flip = Rotation.from_euler("y", 180, degrees=True).as_matrix()
        a, _, _ = render(self.satellite, self.R, self.t, K, SIZE)
        b, _, _ = render(self.satellite, self.R @ flip, self.t, K, SIZE)
        self.assertGreater(np.abs(a.astype(int) - b.astype(int)).mean(), 2.0)


class VerificationTests(unittest.TestCase):
    def test_selftest_reports_small_errors_and_catches_the_flipped_view(self):
        with tempfile.TemporaryDirectory() as folder:
            verification.generate(folder, num_views=6)
            for name in ("satellite.stl", "ground_truth.json", "gt_contact_sheet.png"):
                self.assertTrue(os.path.exists(os.path.join(folder, name)), name)
            summary = verification.selftest(folder, noise_deg=3.0, flipped_view=5)
            errors = summary["errors_deg"]
            self.assertAlmostEqual(errors[5], 180.0, places=3)
            self.assertLess(max(e for i, e in errors.items() if i != 5), 6.0)
            self.assertEqual(summary["within_threshold"], 5)
            self.assertTrue(os.path.exists(os.path.join(summary["eval_dir"],
                                                        "contact_sheet.png")))

    def test_perfect_estimates_report_zero_error_and_draw_over_the_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            verification.generate(folder, num_views=2)
            truth = verification.load_ground_truth(folder)
            poses = os.path.join(folder, "perfect.json")
            with open(poses, "w") as f:
                json.dump([{"img_id": i, "R": v["R"], "t": v["t"]}
                           for i, v in enumerate(truth["views"])], f)
            summary = verification.evaluate(folder, poses)
            self.assertAlmostEqual(max(summary["errors_deg"].values()), 0.0, places=4)

            # Where every true edge should be, drawn on a blank canvas.
            view = truth["views"][0]
            expected = Image.new("RGB", tuple(truth["image_size"]))
            draw_wireframe(expected, build_satellite(), np.array(view["R"]), np.array(view["t"]),
                           np.array(truth["camera_K"]), None, (255, 255, 255), width=1)
            near_expected = binary_dilation(np.array(expected).any(axis=2), iterations=4)

            overlay = np.array(Image.open(os.path.join(summary["eval_dir"], "view_00.png")))
            red = (overlay == verification.ESTIMATE_COLOR).all(axis=2)
            self.assertGreater(red.sum(), 500)
            self.assertGreater((red & near_expected).sum() / red.sum(), 0.99)

    def test_the_mesh_centroid_stays_put_however_wrong_the_estimated_rotation_is(self):
        # FoundPose rotates about the mesh centroid, so the overlay must too; otherwise a
        # rotation error would also appear as a spurious shift of the whole object.
        centroid = np.array([-0.09, -0.21, -0.0205])
        R_true, t_true = look_at_pose([0.3, 0.4, 1.0], 4.0)
        for angle in (0.0, 10.0, 90.0, 180.0):
            R_est = R_true @ Rotation.from_euler("xyz", [angle, angle / 2, 0], degrees=True).as_matrix()
            t_draw = verification.overlay_translation(R_true, t_true, R_est, centroid)
            np.testing.assert_allclose(R_est @ centroid + t_draw, R_true @ centroid + t_true,
                                       atol=1e-12)
        np.testing.assert_allclose(
            verification.overlay_translation(R_true, t_true, R_true, centroid), t_true)

    def test_centroid_matches_the_mean_of_the_distinct_vertices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sat.stl")
            satellite = build_satellite()
            write_binary_stl(path, satellite.triangles, satellite.normals)
            vertices = np.unique(satellite.triangles.reshape(-1, 3).astype(np.float32), axis=0)
            self.assertEqual(len(vertices), 80)
            np.testing.assert_allclose(verification.mesh_centroid(path), vertices.mean(axis=0),
                                       atol=1e-6)

    def test_large_sets_also_get_a_sheet_of_the_worst_views(self):
        with tempfile.TemporaryDirectory() as folder:
            verification.generate(folder, num_views=18, seed=4)
            truth = verification.load_ground_truth(folder)
            flip = Rotation.from_euler("y", 180, degrees=True).as_matrix()
            spoiled = {3, 11, 17}                         # three views with a 180 degree error
            poses = [{"img_id": i, "t": v["t"],
                      "R": (np.array(v["R"]) @ flip if i in spoiled else np.array(v["R"])).tolist()}
                     for i, v in enumerate(truth["views"])]
            path = os.path.join(folder, "poses.json")
            with open(path, "w") as f:
                json.dump(poses, f)
            summary = verification.evaluate(folder, path)
            for name in ("contact_sheet.png", "worst_views_contact_sheet.png"):
                self.assertTrue(os.path.exists(os.path.join(summary["eval_dir"], name)), name)
            worst = sorted(summary["errors_deg"].items(), key=lambda kv: -kv[1])[:3]
            self.assertEqual({i for i, _ in worst}, spoiled)

    def test_missing_estimates_are_reported_not_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            verification.generate(folder, num_views=2)
            poses = os.path.join(folder, "poses.json")
            with open(poses, "w") as f:
                f.write("[]")
            summary = verification.evaluate(folder, poses)
            self.assertEqual(summary["estimated"], 0)
            self.assertIsNone(summary["median_deg"])


if __name__ == "__main__":
    unittest.main()
