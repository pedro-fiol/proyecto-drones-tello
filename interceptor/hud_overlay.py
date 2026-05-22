"""
interceptor/hud_overlay.py

Pure OpenCV drawing functions for the video HUD overlay.
No Tello SDK calls, no threading, no state — just frame annotation.

`draw_person_overlays` selects which Person is the tracking target using the
provided Target instance (see interceptor.target). Tracked person is also
returned so the orchestrator can read its bbox dimensions for pitch fallback.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import TYPE_CHECKING, Optional

from interceptor.constants import FRONT_TOF_WALL_STOP_CM

if TYPE_CHECKING:
    from interceptor.perception import Person
    from interceptor.target import Target


# ---- HUD layout constants ----
HUD_LINE_HEIGHT_PIXELS = 26
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_FACE_BOX = (0, 255, 0)
COLOR_BODY_BOX = (0, 165, 255)
COLOR_CROSSHAIR = (0, 0, 255)
COLOR_STATE_TEXT = (0, 255, 255)
COLOR_TELEMETRY_PRIMARY = (0, 200, 255)
COLOR_TELEMETRY_SECONDARY = (150, 150, 255)
COLOR_MANUAL_BADGE = (0, 220, 255)    # yellow-ish
COLOR_AUTO_BADGE = (180, 180, 180)    # grey
COLOR_TRACK_ID = (255, 200, 0)        # cyan-ish for unlocked track id
COLOR_LOCKED_BOX = (0, 0, 255)        # red for locked target

def draw_all_track_labels(
    frame: np.ndarray,
    persons: "list[Person]",
    locked_track_id: Optional[int] = None,
) -> None:
    """Draw thin bbox + track_id label on every detection.

    Locked target gets a thick red box. Others get thin cyan boxes with
    "ID:N rank:R" — R is bbox-area rank (1=largest). Lets user see all
    candidate IDs at a glance and choose which to lock via the cycle key.
    """
    if not persons:
        return
    sorted_by_area = sorted(persons, key=lambda p: p.bbox_area_pixels, reverse=True)
    rank_by_person = {id(p): i + 1 for i, p in enumerate(sorted_by_area)}
    for person in persons:
        is_locked = (
            locked_track_id is not None
            and person.track_id == locked_track_id
        )
        color = COLOR_LOCKED_BOX if is_locked else COLOR_TRACK_ID
        thickness = 3 if is_locked else 1
        cv2.rectangle(frame, (person.x1, person.y1), (person.x2, person.y2), color, thickness)
        id_text = f"ID:{person.track_id}" if person.track_id is not None else "ID:?"
        rank = rank_by_person[id(person)]
        label = f"{id_text} rank:{rank}"
        if is_locked:
            label = f"[LOCKED] {label}"
        cv2.putText(frame, label, (person.x1, person.y2 + 18),
                    HUD_FONT, 0.55, color, 1)


def draw_person_overlays(
    frame: np.ndarray,
    persons: "list[Person]",
    target: "Target",
) -> tuple:
    """
    Draw bbox + label + target dot for every detected person.

    Returns (target_x, target_y, tracking, tracked_person):
        target_x, target_y — pixel coords of the active target on the chosen
            person, or (None, None) when no person passes target.is_visible.
        tracking — True when at least one person passed target.is_visible.
        tracked_person — the Person whose target is active (for tracking),
            or the first body bbox as fallback (for bbox-PID closeness signal),
            or None if no persons at all.

    Visible-target persons take priority over body-only ones for `tracked_person`.
    """
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    tracking = False
    tracked_person: Optional["Person"] = None

    for person in persons:
        if target.is_visible(person):
            color = COLOR_FACE_BOX
            label = f"{target.name.upper()} {person.bbox_confidence:.2f}"
            target_x, target_y = target.point(person)
            tracking = True
            tracked_person = person
        else:
            color = COLOR_BODY_BOX
            label = f"BACK {person.bbox_confidence:.2f}"
            if tracked_person is None:
                tracked_person = person   # bbox-only fallback for pitch distance

        cv2.rectangle(frame, (person.x1, person.y1), (person.x2, person.y2), color, 2)
        cv2.putText(frame, label, (person.x1, person.y1 - 8),
                    HUD_FONT, 0.55, color, 1)

        if tracking and person is tracked_person and target_x is not None:
            cv2.circle(frame, (target_x, target_y), 6, COLOR_FACE_BOX, -1)

    return target_x, target_y, tracking, tracked_person


def draw_frame_crosshair(frame: np.ndarray, center_x: int, center_y: int) -> None:
    """Draw a small red dot at the frame center as an aiming reference."""
    cv2.circle(frame, (center_x, center_y), 6, COLOR_CROSSHAIR, -1)


def draw_target_y_line(frame: np.ndarray, target_y: int) -> None:
    """Horizontal green line marking the desired target Y position (drives altitude PID setpoint)."""
    h, w = frame.shape[:2]
    cv2.line(frame, (0, target_y), (w, target_y), COLOR_FACE_BOX, 1)


def draw_phantom_badge(frame: np.ndarray) -> None:
    """Big green PHANTOM badge top-right — impossible to miss when dry-running."""
    h, w = frame.shape[:2]
    cv2.putText(frame, "PHANTOM", (w - 220, 50),
                HUD_FONT, 1.0, COLOR_FACE_BOX, 3)


def draw_wall_warning(frame: np.ndarray, front_tof_cm: float, threshold_cm: float = FRONT_TOF_WALL_STOP_CM) -> None:
    """Big red WALL badge centered when front ToF inside threshold. Manual-mode crash aid."""
    if front_tof_cm <= 0 or front_tof_cm > threshold_cm:
        return
    h, w = frame.shape[:2]
    text = f"WALL {int(front_tof_cm)} cm"
    (tw, th), _ = cv2.getTextSize(text, HUD_FONT, 1.2, 4)
    x = (w - tw) // 2
    y = h // 2 + 80
    cv2.rectangle(frame, (x - 14, y - th - 14), (x + tw + 14, y + 14), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), HUD_FONT, 1.2, (0, 0, 255), 4)


def draw_control_mode_badge(
    frame: np.ndarray,
    manual_mode: bool,
    manual_lr: int = 0,
    manual_fb: int = 0,
    manual_ud: int = 0,
    manual_yaw: int = 0,
) -> None:
    """Top-right badge: AUTO (grey) or MANUAL (yellow) + active setpoints when manual."""
    h, w = frame.shape[:2]
    lh = HUD_LINE_HEIGHT_PIXELS

    if manual_mode:
        cv2.putText(frame, "MANUAL", (w - 195, lh * 4),
                    HUD_FONT, 0.9, COLOR_MANUAL_BADGE, 2)
        cv2.putText(frame, f"lr={manual_lr:+d} fb={manual_fb:+d}",
                    (w - 230, lh * 5), HUD_FONT, 0.55, COLOR_MANUAL_BADGE, 1)
        cv2.putText(frame, f"ud={manual_ud:+d} yaw={manual_yaw:+d}",
                    (w - 230, lh * 6), HUD_FONT, 0.55, COLOR_MANUAL_BADGE, 1)
    else:
        cv2.putText(frame, "AUTO", (w - 130, lh * 4),
                    HUD_FONT, 0.9, COLOR_AUTO_BADGE, 1)


def draw_detection_state(
    frame: np.ndarray,
    target_center_x: int | None,
    target_center_y: int | None,
    frame_center_x: int,
    target_y_reference: int,
    tracking_target: bool,
    target_name: str = "TARGET",
) -> None:
    """Draw the tracking state label and pixel error vector at the top of the frame.

    target_y_reference is the desired target Y when tracking_target, frame center Y otherwise.
    """
    if target_center_x is not None:
        err_x = target_center_x - frame_center_x
        err_y = target_center_y - target_y_reference

        state_label = target_name.upper() if tracking_target else "BODY"
        cv2.putText(frame, f"{state_label}  err=({err_x:+d},{err_y:+d}px)",
                    (10, HUD_LINE_HEIGHT_PIXELS), HUD_FONT, 0.65, COLOR_STATE_TEXT, 2)
    else:
        cv2.putText(frame, "NO PERSON - SEARCHING",
                    (10, HUD_LINE_HEIGHT_PIXELS), HUD_FONT, 0.65, COLOR_CROSSHAIR, 2)


def draw_telemetry_strip(frame: np.ndarray, interceptor) -> None:
    """
    Draw the full telemetry readout along the left side of the frame.

    Reads attributes directly off the TelloInterceptor instance via duck typing:
    battery_percent, height_cm, front_tof_cm, down_tof_cm, yaw_deg, pitch_deg, roll_deg,
    x_speed_dm_s, y_speed_dm_s, z_speed_dm_s,
    x_accel_cm_s2, y_accel_cm_s2, z_accel_cm_s2,
    min_temp_C, max_temp_C.
    """
    line_height = HUD_LINE_HEIGHT_PIXELS

    cv2.putText(frame, f"Battery: {interceptor.battery_percent:.0f}%",
                (10, line_height * 2), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)
    cv2.putText(frame, f"Height:  {interceptor.height_cm:.0f} cm",
                (10, line_height * 3), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)
    cv2.putText(frame, f"Front ToF:     {interceptor.front_tof_cm:.0f} cm",
                (10, line_height * 4), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)
    cv2.putText(frame, f"Down ToF:     {interceptor.down_tof_cm:.0f} cm",
                (10, line_height * 5), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)
    cv2.putText(frame, f"Yaw:     {interceptor.yaw_deg:.0f} deg",
                (10, line_height * 6), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)
    cv2.putText(frame, f"Pitch:   {interceptor.pitch_deg:.0f} deg",
                (10, line_height * 7), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)
    cv2.putText(frame, f"Roll:    {interceptor.roll_deg:.0f} deg",
                (10, line_height * 8), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)

    cv2.putText(frame, f"Vx: {interceptor.x_speed_dm_s:.0f} dm/s",
                (10, line_height * 9), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Vy: {interceptor.y_speed_dm_s:.0f} dm/s",
                (10, line_height * 10), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Vz: {interceptor.z_speed_dm_s:.0f} dm/s",
                (10, line_height * 11), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Ax: {interceptor.x_accel_cm_s2:.0f} cm/s2",
                (10, line_height * 12), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Ay: {interceptor.y_accel_cm_s2:.0f} cm/s2",
                (10, line_height * 13), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Az: {interceptor.z_accel_cm_s2:.0f} cm/s2",
                (10, line_height * 14), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Min Temp: {interceptor.min_temp_C:.0f} C",
                (10, line_height * 15), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Max Temp: {interceptor.max_temp_C:.0f} C",
                (200, line_height * 15), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)