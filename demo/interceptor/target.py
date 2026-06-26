"""Tracking-target abstraction: what point in the frame the drone tracks.
Each Target carries visibility rule, point extractor, target_y_ratio. Active Target picked via the TARGET constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

from interceptor.perception import Person, KEYPOINT_VISIBILITY_THRESHOLD
from interceptor.constants import (
    BBOX_HEIGHT_RATIO_SETPOINT,
    BBOX_WIDTH_RATIO_SETPOINT,
    FRAME_HEIGHT_PIXELS,
    FRAME_WIDTH_PIXELS,
    LOCK_EMBEDDING_EMA_ALPHA,
    LOCK_MATCH_THRESHOLD,
    NOSE_CONFIDENCE_THRESHOLD,
    SHOULDER_WIDTH_RATIO_SETPOINT,
    YOLO_CONFIDENCE_MIN,
)

# Torch only needed for LockState embedding + locked-selector cosine distance.
if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Target:
    """Tracking-target spec.
        name: HUD + log label
        target_y_ratio: vertical setpoint in frame (0=top, 1=bottom). Drives altitude PID.
        is_visible: gate predicate on Person — required keypoints/conf pass
        point: Person -> (x_px, y_px) extractor
        closeness: Person -> [0,1] proxy for distance (bigger = closer). bbox pitch PID input.
        closeness_setpoint: target closeness — pitch PID drives forward until reached.
    """

    name: str
    target_y_ratio: float
    is_visible: Callable[[Person], bool]
    point: Callable[[Person], tuple[int, int]]
    closeness: Callable[[Person], float]
    closeness_setpoint: float


def _nose_visible(person: Person) -> bool:
    """Nose conf >= NOSE_CONFIDENCE_THRESHOLD."""
    return person.nose is not None and person.nose.confidence >= NOSE_CONFIDENCE_THRESHOLD


def _both_eyes_visible(person: Person) -> bool:
    """Both eye keypoints above KEYPOINT_VISIBILITY_THRESHOLD."""
    if person.left_eye is None or person.right_eye is None:
        return False
    return (
        person.left_eye.confidence >= KEYPOINT_VISIBILITY_THRESHOLD
        and person.right_eye.confidence >= KEYPOINT_VISIBILITY_THRESHOLD
    )

def _both_shoulders_visible(person: Person) -> bool:
    """Both shoulder keypoints above KEYPOINT_VISIBILITY_THRESHOLD."""
    if person.left_shoulder is None or person.right_shoulder is None:
        return False
    return (
        person.left_shoulder.confidence >= KEYPOINT_VISIBILITY_THRESHOLD
        and person.right_shoulder.confidence >= KEYPOINT_VISIBILITY_THRESHOLD
    )

def _bbox_visible(person: Person) -> bool:
    """Bbox conf >= YOLO threshold (always true for detector output)."""
    return person.bbox_confidence >= YOLO_CONFIDENCE_MIN



def _nose_point(person: Person) -> tuple[int, int]:
    return person.nose.x, person.nose.y


def _eyes_midpoint(person: Person) -> tuple[int, int]:
    mx = (person.left_eye.x + person.right_eye.x) // 2
    my = (person.left_eye.y + person.right_eye.y) // 2
    return mx, my


def _shoulders_midpoint(person: Person) -> tuple[int, int]:
    mx = (person.left_shoulder.x + person.right_shoulder.x) // 2
    my = (person.left_shoulder.y + person.right_shoulder.y) // 2
    return mx, my


def _bbox_center(person: Person) -> tuple[int, int]:
    return person.bbox_center_x, person.bbox_center_y


def _bbox_height_ratio(person: Person) -> float:
    """Person bbox height divided by frame height. Bigger = closer."""
    return person.bbox_height_pixels / FRAME_HEIGHT_PIXELS


def _shoulder_width_ratio(person: Person) -> float:
    """Shoulder pixel distance / frame width."""
    return abs(person.left_shoulder.x - person.right_shoulder.x) / FRAME_WIDTH_PIXELS


def _bbox_width_ratio(person: Person) -> float:
    """Bbox width / frame width. Bigger = closer."""
    return person.bbox_width_pixels / FRAME_WIDTH_PIXELS


NOSE_TARGET = Target(
    name="nose",
    target_y_ratio=0.2,
    is_visible=_nose_visible,
    point=_nose_point,
    closeness=_bbox_height_ratio,
    closeness_setpoint=BBOX_HEIGHT_RATIO_SETPOINT,
)

SHOULDERS_MIDPOINT_TARGET = Target(
    name="shoulders_midpoint",
    target_y_ratio=0.25,
    is_visible=_both_shoulders_visible,
    point=_shoulders_midpoint,
    closeness=_bbox_width_ratio,
    closeness_setpoint=BBOX_WIDTH_RATIO_SETPOINT,
)

EYES_MIDPOINT_TARGET = Target(
    name="eyes_midpoint",
    target_y_ratio=0.2,
    is_visible=_both_eyes_visible,
    point=_eyes_midpoint,
    closeness=_bbox_height_ratio,
    closeness_setpoint=BBOX_HEIGHT_RATIO_SETPOINT,
)

BBOX_CENTER_TARGET = Target(
    name="bbox_center",
    target_y_ratio=0.5,
    is_visible=_bbox_visible,
    point=_bbox_center,
    closeness=_bbox_height_ratio,
    closeness_setpoint=BBOX_HEIGHT_RATIO_SETPOINT,
)


def select_best_person(
    persons: list[Person],
    target: "Target",
    last_point_px: tuple[float, float] | None = None,
) -> "Person | None":
    """Pick one Person to track from detections.
    Priority:
        1. last_point_px given + visible pool non-empty -> nearest in pixels
        2. else visible pool -> largest bbox area
        3. else all -> largest bbox area
        4. empty -> None
    """
    if not persons:
        return None

    visible = [p for p in persons if target.is_visible(p)]
    pool = visible if visible else persons

    if last_point_px is not None and visible:
        last_x, last_y = last_point_px
        def _dist_sq(p: Person) -> float:
            px, py = target.point(p)
            return (px - last_x) ** 2 + (py - last_y) ** 2
        return min(visible, key=_dist_sq)

    return max(pool, key=lambda p: p.bbox_area_pixels)


# Multi-target lock selection
@dataclass
class LockState:
    """Locked person's ReID embedding + last-known location.
        embedding: 1-D L2-normalized tensor on embedder device. None = unlocked (biggest-bbox fallback).
        last_point_px: last (x, y) of locked target's tracking point. Spatial prior + search-FSM yaw recovery.
    """

    embedding: Optional["torch.Tensor"] = None
    last_point_px: Optional[tuple[float, float]] = None

    @property
    def is_locked(self) -> bool:
        return self.embedding is not None

    def clear(self) -> None:
        """Reset to unlocked. Selector falls back to biggest-bbox."""
        self.embedding = None
        self.last_point_px = None

    def lock_from(self, embedding: "torch.Tensor",
                  point_px: Optional[tuple[float, float]] = None) -> None:
        """Init lock from fresh detection embedding (must be L2-normalized).
        detach+clone so batch updates don't mutate the stored vector.
        """
        self.embedding = embedding.detach().clone()
        self.last_point_px = point_px

    def update(self, new_embedding: "torch.Tensor",
               new_point_px: Optional[tuple[float, float]] = None,
               alpha: float = LOCK_EMBEDDING_EMA_ALPHA) -> None:
        """EMA-blend locked embedding toward fresh match. Tracks slow drift (pose/lighting) without erasing identity.
        Caller MUST only invoke on confirmed match — mismatched update corrupts the fingerprint.
        """
        if self.embedding is None:
            return
        # Local import — keeps module importable when torch absent (tests on unlocked path).
        from interceptor.reid import update_embedding_ema
        self.embedding = update_embedding_ema(self.embedding, new_embedding, alpha)
        if new_point_px is not None:
            self.last_point_px = new_point_px


def select_target_person(
    persons: list[Person],
    target: "Target",
    lock: LockState,
    embeddings: Optional["torch.Tensor"] = None,
    last_smoothed_point_px: Optional[tuple[float, float]] = None,
    match_threshold: float = LOCK_MATCH_THRESHOLD,
) -> tuple[Optional[Person], Optional[int], float]:
    """Pick active tracking target from persons. Pure inspection (lock not mutated here).
    Unlocked -> delegate to select_best_person.
    Locked   -> min cosine distance to lock.embedding; None if above match_threshold.
    Returns (chosen_person, chosen_index, match_distance). distance = -1.0 in unlocked path.
    """
    if not persons:
        return None, None, -1.0

    if not lock.is_locked:
        best = select_best_person(persons, target, last_smoothed_point_px)
        if best is None:
            return None, None, -1.0
        return best, persons.index(best), -1.0

    # Locked regime — must have embeddings to match against.
    if embeddings is None or embeddings.shape[0] != len(persons):
        return None, None, -1.0

    import torch  # local import — only paid on locked path
    locked_vec = lock.embedding.flatten()
    # Vectorized cosine: (N, 512) @ (512,) = (N,)
    sims = embeddings @ locked_vec
    dists = 1.0 - sims  # cosine distances in [0, 2]
    best_idx_t = torch.argmin(dists)
    best_idx = int(best_idx_t.item())
    best_dist = float(dists[best_idx_t].item())

    if best_dist > match_threshold:
        return None, None, best_dist
    return persons[best_idx], best_idx, best_dist



# Webapp dropdown keys -> Target. Add new instances here.
AVAILABLE_TARGETS: dict[str, Target] = {
    "nose":               NOSE_TARGET,
    "eyes_midpoint":      EYES_MIDPOINT_TARGET,
    "shoulders_midpoint": SHOULDERS_MIDPOINT_TARGET,
    "bbox_center":        BBOX_CENTER_TARGET,
}


class _ActiveTarget:
    """Mutable holder around the active Target.
    Wrapper preserves object identity for `from interceptor.target import TARGET` callers across runtime swaps.
    Rebinding the module attribute would not propagate to existing `from` imports — swap `_inner` instead.
    """

    def __init__(self, t: Target) -> None:
        self._inner = t

    def set(self, t: Target) -> None:
        self._inner = t

    @property
    def inner(self) -> Target:
        return self._inner

    def __getattr__(self, name: str):
        # Only fires for names not on self — _inner/set/inner go via normal lookup.
        return getattr(self._inner, name)


# Startup default. Webapp /cmd/set_target swaps at runtime.
TARGET = _ActiveTarget(SHOULDERS_MIDPOINT_TARGET)


def set_active_target(name: str) -> bool:
    """Swap active target by name. True on success, False if unknown."""
    t = AVAILABLE_TARGETS.get(name)
    if t is None:
        return False
    TARGET.set(t)
    return True
