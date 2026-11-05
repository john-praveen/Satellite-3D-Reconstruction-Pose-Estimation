import os
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from satpose import verification
from satpose.evaluation import MM_SCALE, accuracy_table, pose_errors


def turned(degrees):
    return Rotation.from_euler("z", degrees, degrees=True).as_matrix()


class PoseErrorTests(unittest.TestCase):
    """Poses are built the way FoundPose reports them: the mesh centroid's position, in mm."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        verification.generate(self.folder.name, num_views=3, seed=5)
        self.truth = verification.load_ground_truth(self.folder.name)
        self.centroid = verification.mesh_centroid(os.path.join(self.folder.name, "satellite.stl"))

    def poses(self, depth_shift=0.0, sideways=0.0, rotation=np.eye(3)):
        poses = []
        for i, view in enumerate(self.truth["views"]):
            R, t = np.array(view["R"]), np.array(view["t"])
            centre = R @ self.centroid + t + [sideways, 0.0, depth_shift]
            poses.append({"img_id": str(i), "R": (R @ rotation).tolist(),
                          "t": [[v * MM_SCALE] for v in centre]})
        return poses

    def test_a_pose_at_the_true_centroid_has_no_error(self):
        for e in pose_errors(self.folder.name, self.poses()):
            self.assertAlmostEqual(e["rotation_deg"], 0.0, places=4)
            self.assertAlmostEqual(e["total"], 0.0, places=9)
            self.assertAlmostEqual(e["speed_score"], 0.0, places=4)

    def test_a_shift_along_the_line_of_sight_is_a_depth_error_only(self):
        for e in pose_errors(self.folder.name, self.poses(depth_shift=0.2)):
            self.assertAlmostEqual(e["depth"], 0.2)
            self.assertAlmostEqual(e["lateral"], 0.0, places=9)
            self.assertAlmostEqual(e["relative"], 0.2 / 4.0, delta=0.01)   # distance is about 4

    def test_a_sideways_shift_is_a_lateral_error_and_rotation_is_scored_separately(self):
        errors = pose_errors(self.folder.name, self.poses(sideways=0.1, rotation=turned(10)))
        for e in errors:
            self.assertAlmostEqual(e["lateral"], 0.1)
            self.assertAlmostEqual(e["rotation_deg"], 10.0, places=4)
            self.assertAlmostEqual(e["speed_score"], np.radians(10) + e["relative"])

    def test_a_missing_estimate_is_none_and_is_left_out_of_the_table(self):
        errors = pose_errors(self.folder.name, self.poses()[:2])
        self.assertIsNone(errors[2])
        table = "\n".join(accuracy_table({"set / pipeline": errors, "empty": [None]}))
        self.assertIn("set / pipeline", table)
        self.assertIn("no estimates", table)


if __name__ == "__main__":
    unittest.main()
