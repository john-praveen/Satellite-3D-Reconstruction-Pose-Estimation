"""Contribution 3 -- asynchronous dual-rate closed loop.

Two loops run at different speeds and never wait for each other:

* **Fast loop** (``ClosedLoopPipeline.process_frame``): estimates the pose of every frame
  against the *current* model and offers the posed frame to the keyframe selector.
* **Slow loop** (``RebuildWorker``): when enough new viewpoints have been collected, rebuilds
  the model from the keyframes in a background thread (minutes on an embedded GPU) and swaps
  the result in atomically when it is ready.

The reconstruction and pose back-ends are injected as plain callables, so this module has no
dependency on Hunyuan3D, FoundPose, or any GPU, and can be tested with fake stages:

    estimate_pose(frame, model) -> PoseEstimate | None
    rebuild_model(slot_images)  -> model        # slot_images: {"front": payload, ...}
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from satpose.viewpoints import (
    ADDED, REQUIRED_SLOTS, KeyframeSelector, assign_slots, missing_slots,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoseEstimate:
    """A pose (``x_cam = R @ x_obj + t``) and a quality score, higher is better."""
    R: np.ndarray
    t: np.ndarray
    quality: float


class ModelStore:
    """Holds the current model and its version; reads and swaps are thread-safe."""

    def __init__(self, model: Any = None):
        self._lock = threading.Lock()
        self._model = model
        self._version = 0

    def get(self) -> Tuple[Any, int]:
        """Returns ``(model, version)`` as one consistent pair."""
        with self._lock:
            return self._model, self._version

    def swap(self, model: Any) -> int:
        """Replaces the model and returns the new version number."""
        with self._lock:
            self._model = model
            self._version += 1
            return self._version


@dataclass
class RuntimeStats:
    """Timing of both loops, to show the fast loop is never blocked by the slow one."""
    frame_latencies_s: List[float] = field(default_factory=list)
    rebuild_durations_s: List[float] = field(default_factory=list)
    rebuild_failures: int = 0

    def summary(self) -> Dict[str, float]:
        frames = self.frame_latencies_s
        return {
            "frames": len(frames),
            "frame_latency_mean_s": float(np.mean(frames)) if frames else 0.0,
            "frame_latency_max_s": float(np.max(frames)) if frames else 0.0,
            "rebuilds": len(self.rebuild_durations_s),
            "rebuild_mean_s": (float(np.mean(self.rebuild_durations_s))
                               if self.rebuild_durations_s else 0.0),
            "rebuild_failures": self.rebuild_failures,
        }


class RebuildWorker:
    """Runs model rebuilds in a background thread, one at a time.

    A failed rebuild is logged and counted, and the old model stays in place: the slow loop
    can never take the fast loop down with it.
    """

    def __init__(self, rebuild_model: Callable[[Mapping[str, Any]], Any], store: ModelStore,
                 stats: RuntimeStats):
        self._rebuild_model = rebuild_model
        self._store = store
        self._stats = stats
        self._idle = threading.Event()
        self._idle.set()
        self._start_lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    def submit(self, slot_images: Mapping[str, Any]) -> bool:
        """Starts a rebuild unless one is already running. Never blocks; returns True if started."""
        with self._start_lock:
            if self.busy:
                return False
            self._idle.clear()
        threading.Thread(target=self._run, args=(dict(slot_images),), daemon=True).start()
        return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Blocks until no rebuild is running. Returns False if the timeout expired."""
        return self._idle.wait(timeout)

    def _run(self, slot_images: Dict[str, Any]) -> None:
        started = time.perf_counter()
        try:
            model = self._rebuild_model(slot_images)
            version = self._store.swap(model)
            self._stats.rebuild_durations_s.append(time.perf_counter() - started)
            logger.info("model rebuilt in %.1fs -> version %d",
                        time.perf_counter() - started, version)
        except Exception:
            self._stats.rebuild_failures += 1
            logger.exception("model rebuild failed; keeping the previous model")
        finally:
            self._idle.set()


@dataclass
class RebuildPolicy:
    """When the slow loop is worth starting.

    A rebuild needs every required slot filled, at least ``min_new_keyframes`` viewpoints
    added since the last rebuild, and ``min_interval_s`` seconds since the last one started.
    """
    min_new_keyframes: int = 1
    required_slots: Sequence[str] = REQUIRED_SLOTS
    slot_tolerance_deg: float = 45.0
    min_interval_s: float = 0.0


@dataclass(frozen=True)
class FrameResult:
    """Outcome of the fast loop for one frame."""
    frame_id: Hashable
    pose: Optional[PoseEstimate]
    model_version: int
    keyframe_decision: Optional[str]
    rebuild_started: bool
    latency_s: float


class ClosedLoopPipeline:
    """Pose-guided closed loop: track every frame, rebuild the model in the background."""

    def __init__(self,
                 estimate_pose: Callable[[Any, Any], Optional[PoseEstimate]],
                 rebuild_model: Callable[[Mapping[str, Any]], Any],
                 initial_model: Any,
                 selector: Optional[KeyframeSelector] = None,
                 policy: Optional[RebuildPolicy] = None):
        self._estimate_pose = estimate_pose
        self.selector = selector or KeyframeSelector()
        self.policy = policy or RebuildPolicy()
        self.store = ModelStore(initial_model)
        self.stats = RuntimeStats()
        self._worker = RebuildWorker(rebuild_model, self.store, self.stats)
        self._new_keyframes = 0
        self._last_rebuild_start = -float("inf")

    def process_frame(self, frame_id: Hashable, frame: Any) -> FrameResult:
        """Fast loop. Estimates the pose on the current model; never waits for a rebuild."""
        started = time.perf_counter()
        model, version = self.store.get()
        pose = self._estimate_pose(frame, model)

        decision = None
        rebuild_started = False
        if pose is not None:
            decision = self.selector.consider(frame_id, pose.R, pose.t, pose.quality,
                                              payload=frame)
            if decision == ADDED:
                self._new_keyframes += 1
            rebuild_started = self._maybe_start_rebuild()

        latency = time.perf_counter() - started
        self.stats.frame_latencies_s.append(latency)
        return FrameResult(frame_id, pose, version, decision, rebuild_started, latency)

    def wait_for_rebuild(self, timeout: Optional[float] = None) -> bool:
        """Blocks until the background rebuild (if any) has finished."""
        return self._worker.wait(timeout)

    def _maybe_start_rebuild(self) -> bool:
        policy = self.policy
        if self._new_keyframes < policy.min_new_keyframes or self._worker.busy:
            return False
        if time.perf_counter() - self._last_rebuild_start < policy.min_interval_s:
            return False

        assignment = assign_slots(self.selector.keyframes, policy.slot_tolerance_deg)
        if missing_slots(assignment, policy.required_slots):
            return False

        slot_images = {slot: keyframe.payload for slot, keyframe in assignment.items()}
        if not self._worker.submit(slot_images):
            return False
        self._new_keyframes = 0
        self._last_rebuild_start = time.perf_counter()
        return True
