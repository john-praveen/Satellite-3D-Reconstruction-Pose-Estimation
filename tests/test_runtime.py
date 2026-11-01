import threading
import time
import unittest

from helpers import pose_from_azimuth
from satpose.runtime import (
    ClosedLoopPipeline, ModelStore, PoseEstimate, RebuildPolicy, RebuildWorker, RuntimeStats,
)
from satpose.viewpoints import ADDED, KeyframeSelector

REBUILD_SECONDS = 0.4


def tracking_stage(frame, model):
    """Fake pose estimator: a frame is just its true azimuth in degrees."""
    R, t = pose_from_azimuth(frame)
    return PoseEstimate(R, t, quality=1.0)


class ModelStoreTests(unittest.TestCase):
    def test_swap_bumps_the_version_and_get_returns_a_consistent_pair(self):
        store = ModelStore("v0")
        self.assertEqual(store.get(), ("v0", 0))
        self.assertEqual(store.swap("v1"), 1)
        self.assertEqual(store.get(), ("v1", 1))


class RebuildWorkerTests(unittest.TestCase):
    def make_worker(self, rebuild):
        store, stats = ModelStore("old"), RuntimeStats()
        return RebuildWorker(rebuild, store, stats), store, stats

    def test_successful_rebuild_swaps_the_model_in(self):
        worker, store, stats = self.make_worker(lambda images: f"built from {sorted(images)}")
        self.assertTrue(worker.submit({"front": 1, "left": 2}))
        self.assertTrue(worker.wait(2.0))
        self.assertEqual(store.get(), ("built from ['front', 'left']", 1))
        self.assertEqual(len(stats.rebuild_durations_s), 1)

    def test_failed_rebuild_keeps_the_old_model_and_is_counted(self):
        def broken(_):
            raise RuntimeError("out of memory")
        worker, store, stats = self.make_worker(broken)
        with self.assertLogs("satpose.runtime", level="ERROR"):
            worker.submit({})
            worker.wait(2.0)
        self.assertEqual(store.get(), ("old", 0))
        self.assertEqual(stats.rebuild_failures, 1)
        self.assertFalse(worker.busy)

    def test_a_second_submit_is_refused_while_busy(self):
        release = threading.Event()
        worker, _, _ = self.make_worker(lambda _: release.wait(2.0) and "new")
        self.assertTrue(worker.submit({}))
        self.assertTrue(worker.busy)
        self.assertFalse(worker.submit({}))
        release.set()
        self.assertTrue(worker.wait(2.0))


class ClosedLoopTests(unittest.TestCase):
    def make_pipeline(self, rebuild=None, policy=None):
        def default_rebuild(images):
            time.sleep(REBUILD_SECONDS)
            return f"model-from-{len(images)}-views"
        return ClosedLoopPipeline(
            tracking_stage, rebuild or default_rebuild, initial_model="single-view",
            selector=KeyframeSelector(min_separation_deg=20.0), policy=policy)

    def orbit(self, pipeline, azimuths):
        return [pipeline.process_frame(i, az) for i, az in enumerate(azimuths)]

    def test_no_rebuild_until_front_left_and_back_have_been_seen(self):
        pipeline = self.make_pipeline()
        results = self.orbit(pipeline, [0, 30, 60])
        self.assertFalse(any(r.rebuild_started for r in results))
        self.assertEqual(results[0].keyframe_decision, ADDED)

    def test_rebuild_starts_once_coverage_is_sufficient_and_the_fast_loop_never_waits(self):
        pipeline = self.make_pipeline()
        results = self.orbit(pipeline, range(0, 200, 10))
        self.assertEqual(sum(r.rebuild_started for r in results), 1)
        self.assertLess(max(r.latency_s for r in results), REBUILD_SECONDS / 2)
        self.assertTrue(all(r.model_version == 0 for r in results))  # still the old model

    def test_new_model_is_used_after_the_rebuild_finishes(self):
        pipeline = self.make_pipeline()
        self.orbit(pipeline, range(0, 200, 10))
        self.assertTrue(pipeline.wait_for_rebuild(5.0))
        result = pipeline.process_frame("later", 100)
        self.assertEqual(result.model_version, 1)
        self.assertEqual(pipeline.store.get()[0], "model-from-3-views")
        self.assertEqual(pipeline.stats.summary()["rebuilds"], 1)

    def test_only_one_rebuild_runs_at_a_time(self):
        started, release = [], threading.Event()

        def slow(images):
            started.append(sorted(images))
            release.wait(5.0)
            return "model"

        pipeline = self.make_pipeline(rebuild=slow)
        self.orbit(pipeline, range(0, 360, 10))  # many new viewpoints while busy
        self.assertEqual(len(started), 1)
        release.set()
        pipeline.wait_for_rebuild(5.0)

    def test_second_rebuild_follows_once_new_viewpoints_arrive_after_the_first(self):
        pipeline = self.make_pipeline(policy=RebuildPolicy(min_new_keyframes=2))
        self.orbit(pipeline, range(0, 200, 10))
        pipeline.wait_for_rebuild(5.0)
        results = self.orbit(pipeline, range(200, 360, 10))
        pipeline.wait_for_rebuild(5.0)
        self.assertTrue(any(r.rebuild_started for r in results))
        self.assertEqual(pipeline.stats.summary()["rebuilds"], 2)

    def test_failed_pose_estimation_is_reported_without_crashing(self):
        pipeline = ClosedLoopPipeline(lambda frame, model: None, lambda images: "m", "m0")
        result = pipeline.process_frame(0, 0)
        self.assertIsNone(result.pose)
        self.assertIsNone(result.keyframe_decision)
        self.assertFalse(result.rebuild_started)


if __name__ == "__main__":
    unittest.main()
