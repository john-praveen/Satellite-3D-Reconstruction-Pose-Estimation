import os
import tempfile
import unittest

import numpy as np

from satpose import config, verification
from satpose.objects import (
    OBJECT_NAMES, build_rocket_body, crease_edges, frustum, get_object, load_mesh_object,
)
from satpose.synthetic import (
    _box_faces_and_edges, look_at_pose, random_poses, render, write_binary_stl,
)

K = config.CAMERA_K
SIZE = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)


class GeometryTests(unittest.TestCase):
    def test_every_object_has_consistent_outward_geometry_that_fits_the_frame(self):
        for name in OBJECT_NAMES:
            geometry, _ = get_object(name)
            tri, normals = geometry.triangles, geometry.normals
            winding = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            self.assertTrue((np.einsum("ij,ij->i", winding, normals) > 0).all(), name)
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)
            self.assertEqual(len(geometry.parts), len(tri), name)
            self.assertLess(K[0, 0] * geometry.radius / 4.0, SIZE[0] / 2, name)  # fits at 4.0
            points = tri.reshape(-1, 3)
            np.testing.assert_allclose((points.min(0) + points.max(0)) / 2, 0.0, atol=1e-9)

    def test_a_cylinder_is_closed_and_its_normals_point_away_from_the_axis(self):
        triangles, normals, edges = frustum((0, 0, 0), (0, 1, 0), 0.5, 0.5, 2.0, segments=16)
        triangles = np.array(triangles)
        centroids = triangles.mean(axis=1)
        sides = np.abs(np.abs(np.array(normals)[:, 1]) - 1.0) > 1e-6          # not the flat caps
        radial = centroids[sides] * [1, 0, 1]
        self.assertTrue((np.einsum("ij,ij->i", radial, np.array(normals)[sides]) > 0).all())
        self.assertEqual(len(edges), 16 + 16 + 4)

    def test_unknown_object_is_rejected(self):
        with self.assertRaises(ValueError):
            get_object("teapot")


class MeshFileTests(unittest.TestCase):
    def test_only_the_edges_of_a_cube_are_creases_not_its_face_diagonals(self):
        triangles, _, _ = _box_faces_and_edges(np.zeros(3), np.ones(3))
        edges = crease_edges(np.array(triangles))
        self.assertEqual(len(edges), 12)
        lengths = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1)
        np.testing.assert_allclose(lengths, 1.0)

    def test_a_flat_open_sheet_keeps_its_boundary(self):
        sheet = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 0, 0], [1, 1, 0], [0, 1, 0]]], float)
        self.assertEqual(len(crease_edges(sheet)), 4)             # four sides, no diagonal

    def test_a_mesh_file_loads_centred_with_outward_normals_even_when_inside_out(self):
        original = build_rocket_body()
        for inside_out in (False, True):
            triangles = original.triangles + [5.0, -2.0, 1.0]       # off-centre on purpose
            normals = original.normals
            if inside_out:
                triangles, normals = triangles[:, [0, 2, 1]], -normals
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "body.stl")
                write_binary_stl(path, triangles, normals)
                loaded = load_mesh_object(path)
            points = loaded.triangles.reshape(-1, 3)
            np.testing.assert_allclose((points.min(0) + points.max(0)) / 2, 0.0, atol=1e-5)
            volume = np.einsum("ij,ij->", loaded.triangles[:, 0],
                               np.cross(loaded.triangles[:, 1], loaded.triangles[:, 2])) / 6.0
            self.assertGreater(volume, 0.0, f"inside_out={inside_out}")
            winding = np.cross(loaded.triangles[:, 1] - loaded.triangles[:, 0],
                               loaded.triangles[:, 2] - loaded.triangles[:, 0])
            self.assertTrue((np.einsum("ij,ij->i", winding, loaded.normals) > 0).all())
            self.assertLessEqual(len(loaded.triangles), 6000)
            self.assertGreater(len(loaded.edges), 20)

    def test_get_object_accepts_a_path_and_a_big_mesh_is_simplified(self):
        big = np.array(build_rocket_body().triangles)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "big.stl")
            write_binary_stl(path, big, np.zeros((len(big), 3)))
            geometry, texture = get_object(path)
            self.assertIsNone(texture)
            self.assertEqual(len(geometry.parts), len(geometry.triangles))
            small = load_mesh_object(path, max_triangles=300)
            self.assertLessEqual(len(small.triangles), 300)
            self.assertLess(len(small.triangles), len(big))


class TextureTests(unittest.TestCase):
    def test_texture_changes_the_colours_but_not_the_outline(self):
        geometry, texture = get_object("satellite_textured")
        R, t = look_at_pose([0.3, 0.25, 1.0], 4.0)
        grey, grey_mask, grey_depth = render(geometry, R, t, K, SIZE)
        coloured, mask, depth = render(geometry, R, t, K, SIZE, texture=texture)
        np.testing.assert_array_equal(mask, grey_mask)
        np.testing.assert_allclose(depth[mask], grey_depth[mask])
        self.assertGreater(np.abs(coloured.astype(int) - grey.astype(int)).mean(), 10)
        spread = coloured[mask].std(axis=0)
        self.assertTrue((spread > 15).all())             # colourful, not a flat tint

    def test_every_texture_returns_valid_colours(self):
        for name in OBJECT_NAMES:
            geometry, texture = get_object(name)
            if texture is None:
                continue
            points = geometry.triangles.reshape(-1, 3)[::7]
            parts = np.repeat(geometry.parts, 3)[::7]
            colours = texture(points, parts)
            self.assertEqual(colours.shape, (len(points), 3))
            self.assertTrue(((colours >= 0) & (colours <= 1)).all(), name)


class PipelineIntegrationTests(unittest.TestCase):
    def test_generation_records_the_object_and_evaluation_draws_it(self):
        with tempfile.TemporaryDirectory() as folder:
            verification.generate(folder, num_views=2, seed=1, object_name="debris")
            truth = verification.load_ground_truth(folder)
            self.assertEqual(truth["object"], "debris")
            perfect = [{"img_id": i, "R": v["R"], "t": v["t"]}
                       for i, v in enumerate(truth["views"])]
            import json
            poses = os.path.join(folder, "perfect.json")
            with open(poses, "w") as f:
                json.dump(perfect, f)
            summary = verification.evaluate(folder, poses)
            self.assertAlmostEqual(max(summary["errors_deg"].values()), 0.0, places=4)

    def test_random_views_of_each_object_stay_inside_the_image(self):
        for name in OBJECT_NAMES:
            geometry, texture = get_object(name)
            for R, t in random_poses(3, seed=2):
                _, mask, _ = render(geometry, R, t, K, SIZE, texture=texture)
                ys, xs = np.nonzero(mask)
                self.assertGreater(mask.sum(), 5000, name)
                self.assertGreater(xs.min(), 2, name)
                self.assertLess(xs.max(), SIZE[0] - 3, name)
                self.assertGreater(ys.min(), 2, name)
                self.assertLess(ys.max(), SIZE[1] - 3, name)


if __name__ == "__main__":
    unittest.main()
