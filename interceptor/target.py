"""
interceptor/target.py

Tracking-target abstraction. Decouples "what point in the frame the drone tracks"
from "face only". Each Target carries its own visibility rule, point extractor
and target_y_ratio (frame Y framing). The orchestrator picks one Target via the
constants.TARGET constant and the perception/HUD/PID pipeline stays generic.

No drone SDK calls, no drawing — pure data + pure functions on Person.
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

# Torch is only needed for the LockState embedding and the locked-selector
# cosine-distance computation. Tests for the visibility predicates and the
# unlocked selector path do not exercise these.
if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Target:
    """One tracking target definition.

    Attributes:
        name: human-readable label, used in HUD + log columns.
        target_y_ratio: where the target should sit vertically in the frame
            (0.0 = top, 0.5 = center, 1.0 = bottom). Drives altitude PID setpoint.
        is_visible: predicate on a Person — true when the target's required
            keypoints/confidences pass the gate for tracking mode.
        point: extractor on a Person returning (x_px, y_px). Caller MUST guard
            with is_visible(person) first; behavior is undefined otherwise.
        closeness: closeness signal in [0, 1]. Bigger = closer to camera.
            Used by the pitch fallback PID when front ToF is invalid. Each target
            picks the most informative signal for its framing (bbox height for
            face/body targets, bbox width for chest-height tracking).
            Caller MUST guard with is_visible(person) first.
        closeness_setpoint: target value of closeness. Pitch PID drives the drone
            forward until closeness reaches this setpoint.
    """

    name: str
    target_y_ratio: float
    is_visible: Callable[[Person], bool]
    point: Callable[[Person], tuple[int, int]]
    closeness: Callable[[Person], float]
    closeness_setpoint: float


# ---------------------------------------------------------------------------
# Visibility predicates
# ---------------------------------------------------------------------------

def _nose_visible(person: Person) -> bool:
    """Strict gate: nose conf at or above NOSE_CONFIDENCE_THRESHOLD."""
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
    """Bbox confidence at or above YOLO detector threshold (always true for detector output)."""
    return person.bbox_confidence >= YOLO_CONFIDENCE_MIN


# ---------------------------------------------------------------------------
# Point extractors
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Closeness signals for pitch fallback PID (0..1, bigger = closer)
# ---------------------------------------------------------------------------

def _bbox_height_ratio(person: Person) -> float:
    """Person bbox height divided by frame height. Bigger = closer."""
    return person.bbox_height_pixels / FRAME_HEIGHT_PIXELS


def _shoulder_width_ratio(person: Person) -> float:
    """Pixel distance between left/right shoulder keypoints, normalized by frame width.

    Bigger = closer. Useful when drone hovers at chest height — bbox height saturates
    near 1.0 because the full body fills the frame, so it carries no distance info.
    Shoulder spread shrinks linearly with distance from camera.
    """
    return abs(person.left_shoulder.x - person.right_shoulder.x) / FRAME_WIDTH_PIXELS


def _bbox_width_ratio(person: Person) -> float:
    """Person bbox width divided by frame width. Bigger = closer.

    More reliable than shoulder keypoint spread: available whenever a bbox exists,
    no keypoint confidence requirement, and naturally represents person silhouette
    width regardless of shoulder visibility.
    """
    return person.bbox_width_pixels / FRAME_WIDTH_PIXELS


# ---------------------------------------------------------------------------
# Built-in Target instances
# ---------------------------------------------------------------------------

# Face-tracking baseline (Phase 5 behavior). Upper-fifth framing keeps drone
# at face height — matches the original FACE_TARGET_Y_RATIO = 0.2.
NOSE_TARGET = Target(
    name="nose",
    target_y_ratio=0.2,
    is_visible=_nose_visible,
    point=_nose_point,
    closeness=_bbox_height_ratio,
    closeness_setpoint=BBOX_HEIGHT_RATIO_SETPOINT,
)

# Default for Phase 6a: drone hovers chest-height so front ToF cone hits torso.
# Uses bbox width as closeness signal — more reliable than shoulder keypoint spread
# (available whenever bbox exists, no keypoint confidence gate, same scaling).
SHOULDERS_MIDPOINT_TARGET = Target(
    name="shoulders_midpoint",
    target_y_ratio=0.25,
    is_visible=_both_shoulders_visible,
    point=_shoulders_midpoint,
    closeness=_bbox_width_ratio,
    closeness_setpoint=BBOX_WIDTH_RATIO_SETPOINT,
)

# Same framing as nose (face-height) but more robust when nose keypoint
# alone flickers — eyes are usually detected as a pair.
EYES_MIDPOINT_TARGET = Target(
    name="eyes_midpoint",
    target_y_ratio=0.2,
    is_visible=_both_eyes_visible,
    point=_eyes_midpoint,
    closeness=_bbox_height_ratio,
    closeness_setpoint=BBOX_HEIGHT_RATIO_SETPOINT,
)

# Coarse fallback — works even when person is turned away (no face/keypoints).
BBOX_CENTER_TARGET = Target(
    name="bbox_center",
    target_y_ratio=0.5,
    is_visible=_bbox_visible,
    point=_bbox_center,
    closeness=_bbox_height_ratio,
    closeness_setpoint=BBOX_HEIGHT_RATIO_SETPOINT,
)


# ---------------------------------------------------------------------------
# Single-person selection
# ---------------------------------------------------------------------------

def select_best_person(
    persons: list[Person],
    target: "Target",
    last_point_px: tuple[float, float] | None = None,
) -> "Person | None":
    """Pick a single Person to track from a list of detections.

    Identity priority:
        1. last_point_px given AND visible pool non-empty → nearest in pixels.
        2. Else among visible — largest bbox area wins.
        3. Else among all — largest bbox area wins (proxy for "closest").
        4. Empty list → None.
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


# ---------------------------------------------------------------------------
# Multi-target lock — ReID-based identity persistence
# ---------------------------------------------------------------------------

@dataclass
class LockState:
    """Holds the locked person's ReID fingerprint plus last-known location.

    Attributes:
        embedding: 1-D L2-normalized (512,) tensor on the embedder's device.
            None = unlocked (drone uses default biggest-bbox selection).
        last_point_px: last known (x, y) of the locked person's tracking
            target point. Used as a spatial prior when ambiguous matches occur
            and to drive the search-FSM grace-period yaw recovery.
    """

    embedding: Optional["torch.Tensor"] = None
    last_point_px: Optional[tuple[float, float]] = None

    @property
    def is_locked(self) -> bool:
        return self.embedding is not None

    def clear(self) -> None:
        """Reset to unlocked. Selector falls back to biggest-bbox behavior."""
        self.embedding = None
        self.last_point_px = None

    def lock_from(self, embedding: "torch.Tensor",
                  point_px: Optional[tuple[float, float]] = None) -> None:
        """Initialize the lock from a fresh detection's embedding.

        `embedding` must already be L2-normalized — the caller (ReidEmbedder)
        guarantees this. We detach + clone so subsequent batch updates don't
        mutate the stored vector.
        """
        self.embedding = embedding.detach().clone()
        self.last_point_px = point_px

    def update(self, new_embedding: "torch.Tensor",
               new_point_px: Optional[tuple[float, float]] = None,
               alpha: float = LOCK_EMBEDDING_EMA_ALPHA) -> None:
        """EMA-blend the locked embedding toward a freshly matched detection.

        Tracks slow appearance drift (turning around, lighting shift) without
        erasing identity. Re-normalizes so cosine distance stays well-defined.
        Caller MUST only invoke this when a match was confirmed; updating on
        a mismatched detection corrupts the fingerprint and loses the lock.
        """
        if self.embedding is None:
            return
        # Local import keeps the module importable when torch isn't available
        # (tests that exercise only the unlocked path).
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
    """Pick the active tracking target out of `persons`.

    Two regimes:
        1. lock.is_locked = False — fall through to select_best_person:
           nearest-to-last-point if visible, else largest bbox.
        2. lock.is_locked = True — compute cosine distance from lock.embedding
           to each detection's embedding; pick min distance if below
           match_threshold, else return None (drone enters grace+search FSM
           pipeline keyed on the LOCKED target only).

    Args:
        persons: detections this frame.
        target: active Target (defines visibility predicate + point extractor).
        lock: current lock state. Mutated only by the orchestrator, never
            here — this function is pure inspection.
        embeddings: (N, 512) L2-normalized embeddings matching `persons` order,
            or None when lock is unlocked (skips the costly embed step).
        last_smoothed_point_px: tracker's last EMA-smoothed target point,
            used as the nearest-neighbour prior in the unlocked regime.
        match_threshold: cosine-distance cutoff for the locked regime.

    Returns:
        (chosen_person, chosen_index, match_distance):
            chosen_person — Person to track, or None.
            chosen_index — index of chosen person in `persons`, or None.
            match_distance — cosine distance of the chosen match in the
                locked regime; -1.0 when unlocked (no embedding compared).
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

    import torch  # local import — only paid in the locked path
    locked_vec = lock.embedding.flatten()
    # Vectorized cosine: embeddings (N, 512) @ locked_vec (512,) = (N,)
    sims = embeddings @ locked_vec
    dists = 1.0 - sims  # (N,) cosine distances in [0, 2]
    best_idx_t = torch.argmin(dists)
    best_idx = int(best_idx_t.item())
    best_dist = float(dists[best_idx_t].item())

    if best_dist > match_threshold:
        return None, None, best_dist
    return persons[best_idx], best_idx, best_dist


# ---------------------------------------------------------------------------
# Active selection — runtime-swappable
# ---------------------------------------------------------------------------
# AVAILABLE_TARGETS keys are the strings the webapp dropdown sends back to
# switch the active target at runtime. Add new Target instances here too.
AVAILABLE_TARGETS: dict[str, Target] = {
    "nose":               NOSE_TARGET,
    "eyes_midpoint":      EYES_MIDPOINT_TARGET,
    "shoulders_midpoint": SHOULDERS_MIDPOINT_TARGET,
    "bbox_center":        BBOX_CENTER_TARGET,
}


class _ActiveTarget:
    """Mutable holder around the currently active Target.

    Why this wrapper exists: callers do ``from interceptor.target import TARGET``
    and then read ``TARGET.name`` / ``TARGET.target_y_ratio`` / call
    ``select_best_person(..., TARGET, ...)``. If TARGET were a plain module-level
    binding, rebinding ``target.TARGET = NEW`` here would not propagate to
    existing ``from`` imports. The wrapper keeps the same object identity while
    its ``_inner`` attribute is swapped. All attribute access proxies through
    __getattr__ to the inner Target, so existing code keeps working unchanged.
    """

    def __init__(self, t: Target) -> None:
        self._inner = t

    def set(self, t: Target) -> None:
        self._inner = t

    @property
    def inner(self) -> Target:
        return self._inner

    def __getattr__(self, name: str):
        # __getattr__ only fires when name isn't found on self — so _inner /
        # set / inner are reached normally; everything else falls through here.
        return getattr(self._inner, name)


# Default at startup. Webapp /cmd/set_target can swap it at runtime.
TARGET = _ActiveTarget(SHOULDERS_MIDPOINT_TARGET)


def set_active_target(name: str) -> bool:
    """Swap the active target by name. Returns True on success, False if unknown."""
    t = AVAILABLE_TARGETS.get(name)
    if t is None:
        return False
    TARGET.set(t)
    return True
