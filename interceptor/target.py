"""
interceptor/target.py

Tracking-target abstraction. Decouples "what point in the frame the drone tracks"
from "face only". Each Target carries its own visibility rule, point extractor
and target_y_ratio (frame Y framing). The orchestrator picks one Target via the
constants.TARGET constant and the perception/HUD/PID pipeline stays generic.

No drone SDK calls, no drawing — pure data + pure functions on Person.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from interceptor.perception import Person, KEYPOINT_VISIBILITY_THRESHOLD
from interceptor.constants import (
    BBOX_HEIGHT_RATIO_SETPOINT,
    BBOX_WIDTH_RATIO_SETPOINT,
    FRAME_HEIGHT_PIXELS,
    FRAME_WIDTH_PIXELS,
    NOSE_CONFIDENCE_THRESHOLD,
    SHOULDER_WIDTH_RATIO_SETPOINT,
    YOLO_CONFIDENCE_MIN,
)


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
    locked_track_id: int | None = None,
) -> "Person | None":
    """Pick a single Person to track from a list of detections.

    Identity priority:
        1. locked_track_id set → only detections with that track_id are eligible.
           If none match this frame → return None (target invisible this frame).
        2. last_point_px given AND visible pool non-empty → nearest in pixels.
        3. Else among visible — largest bbox area wins.
        4. Else among all — largest bbox area wins (proxy for "closest").
        5. Empty list → None.

    locked_track_id is the strongest constraint. When set (post-enrollment) the
    drone refuses to swap onto any other person regardless of bbox size, distance
    to prior centroid, or visibility class. Caller decides when to drop the lock
    (e.g. on prolonged invisibility).
    """
    if not persons:
        return None

    if locked_track_id is not None:
        locked = [p for p in persons if p.track_id == locked_track_id]
        if not locked:
            return None
        visible_locked = [p for p in locked if target.is_visible(p)]
        pool = visible_locked if visible_locked else locked
        return max(pool, key=lambda p: p.bbox_area_pixels)

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
# Active selection
# ---------------------------------------------------------------------------
# Switch this constant to change tracking behavior at next session start:
#   NOSE_TARGET               — Phase 5 face tracking (drone hovers face-height)
#   SHOULDERS_MIDPOINT_TARGET — chest-height tracking (front ToF hits torso)
#   EYES_MIDPOINT_TARGET      — face-height, more robust than nose alone
#   BBOX_CENTER_TARGET        — works when person is turned away
# Lives here (not constants.py) so target.py stays the single source of target
# definitions without a circular import via perception.py.
TARGET = SHOULDERS_MIDPOINT_TARGET
