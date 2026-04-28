"""
interceptor/hud_overlay.py

Pure OpenCV drawing functions for the video HUD overlay.
No Tello SDK calls, no threading, no state — just frame annotation.

Person tuples follow the format used by TelloInterceptor:
    (x1, y1, x2, y2, conf, face_visible, nose_x, nose_y)
"""

from __future__ import annotations

import cv2
import numpy as np


# ---- HUD layout constants ----
HUD_LINE_HEIGHT_PIXELS = 26
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_FACE_BOX = (0, 255, 0)
COLOR_BODY_BOX = (0, 165, 255)
COLOR_CROSSHAIR = (0, 0, 255)
COLOR_STATE_TEXT = (0, 255, 255)
COLOR_TELEMETRY_PRIMARY = (0, 200, 255)
COLOR_TELEMETRY_SECONDARY = (150, 150, 255)


def draw_person_overlays(frame: np.ndarray, persons: list[tuple]) -> tuple:
    """
    Draw bbox + label + nose dot for every detected person.

    Returns the chosen tracking target as (target_center_x, target_center_y, tracking_face).
    Face-visible persons take priority; otherwise falls back to the first body bbox.
    """
    target_center_x: int | None = None
    target_center_y: int | None = None
    tracking_face = False

    for (x1, y1, x2, y2, conf, face_visible, nose_x, nose_y) in persons:
        if face_visible:
            color = COLOR_FACE_BOX
            label = f"FACE {conf:.2f}"
            target_center_x, target_center_y = nose_x, nose_y
            tracking_face = True
        else:
            color = COLOR_BODY_BOX
            label = f"BACK {conf:.2f}"
            if target_center_x is None:
                target_center_x = (x1 + x2) // 2
                target_center_y = (y1 + y2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    HUD_FONT, 0.55, color, 1)

        if face_visible and nose_x is not None:
            cv2.circle(frame, (nose_x, nose_y), 6, COLOR_FACE_BOX, -1)

    return target_center_x, target_center_y, tracking_face


def draw_frame_crosshair(frame: np.ndarray, center_x: int, center_y: int) -> None:
    """Draw a small red dot at the frame center as an aiming reference."""
    cv2.circle(frame, (center_x, center_y), 6, COLOR_CROSSHAIR, -1)


def draw_detection_state(
    frame: np.ndarray,
    target_center_x: int | None,
    target_center_y: int | None,
    frame_center_x: int,
    frame_center_y: int,
    tracking_face: bool,
) -> None:
    """Draw the tracking state label and pixel error vector at the top of the frame."""
    if target_center_x is not None:
        err_x = target_center_x - frame_center_x
        err_y = target_center_y - frame_center_y
        state_label = "FACE" if tracking_face else "BODY"
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
    lh = HUD_LINE_HEIGHT_PIXELS

    cv2.putText(frame, f"Battery: {interceptor.battery_percent:.0f}%",
                (10, lh * 2), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)
    cv2.putText(frame, f"Height:  {interceptor.height_cm:.0f} cm",
                (10, lh * 3), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)
    cv2.putText(frame, f"Front ToF:     {interceptor.front_tof_cm:.0f} cm",
                (10, lh * 4), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)
    cv2.putText(frame, f"Down ToF:     {interceptor.down_tof_cm:.0f} cm",
                (10, lh * 5), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)
    cv2.putText(frame, f"Yaw:     {interceptor.yaw_deg:.0f} deg",
                (10, lh * 6), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)
    cv2.putText(frame, f"Pitch:   {interceptor.pitch_deg:.0f} deg",
                (10, lh * 7), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)
    cv2.putText(frame, f"Roll:    {interceptor.roll_deg:.0f} deg",
                (10, lh * 8), HUD_FONT, 0.6, COLOR_TELEMETRY_PRIMARY, 1)

    cv2.putText(frame, f"Vx: {interceptor.x_speed_dm_s:.0f} dm/s",
                (10, lh * 9), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Vy: {interceptor.y_speed_dm_s:.0f} dm/s",
                (10, lh * 10), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Vz: {interceptor.z_speed_dm_s:.0f} dm/s",
                (10, lh * 11), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Ax: {interceptor.x_accel_cm_s2:.0f} cm/s2",
                (10, lh * 12), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Ay: {interceptor.y_accel_cm_s2:.0f} cm/s2",
                (10, lh * 13), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Az: {interceptor.z_accel_cm_s2:.0f} cm/s2",
                (10, lh * 14), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Min Temp: {interceptor.min_temp_C:.0f} C",
                (10, lh * 15), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
    cv2.putText(frame, f"Max Temp: {interceptor.max_temp_C:.0f} C",
                (200, lh * 15), HUD_FONT, 0.55, COLOR_TELEMETRY_SECONDARY, 1)
