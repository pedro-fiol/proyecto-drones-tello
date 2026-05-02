import csv
import os
import cv2
import torch
import threading
import time
from djitellopy import Tello, TelloException

from .pid_controller import PIDController
from .perception import Person, PersonDetector


from .constants import (
    FRAME_CENTER_X, FRAME_CENTER_Y,
    FACE_TARGET_Y_RATIO, FACE_LOST_GRACE_S, FACE_Y_EMA_ALPHA,
    GAINS_ALTITUDE_PID, GAINS_ALTITUDE_FACE_PID,
    GEOFENCE_MAX_ALTITUDE_CM, GEOFENCE_MIN_ALTITUDE_CM,
    MIN_TRACKING_ALTITUDE_CM,
    RC_LOOP_INTERVAL_S, TARGET_ALTITUDE_CM, UD_MAX_VELOCITY_CM_S,
    YOLO_FRAME_STRIDE, YOLO_CONFIDENCE_MIN, YOLO_PERSON_CLASS_ID,
    NOSE_KEYPOINT_INDEX, NOSE_CONFIDENCE_THRESHOLD,
)


from .hud_overlay import (
    draw_detection_state,
    draw_face_target_line,
    draw_frame_crosshair,
    draw_person_overlays,
    draw_phantom_badge,
    draw_telemetry_strip,
)

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import logging
logging.getLogger("djitellopy").setLevel(logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Using device: {DEVICE}")


class TelloInterceptor:
    def __init__(self, phantom_mode: bool = False):
        # Phantom: drone never takes off, send_rc_control is no-op. Camera + YOLO + PID + HUD
        # all run normally so logic can be validated without flight risk.
        self.phantom_mode = phantom_mode
        self.tello        = Tello()
        self._sdk_lock    = threading.Lock()
        self._stop_finished = False
        self.video_active = False
        self.front_tof_active   = False
        self.telemetry_active   = False
        self.front_tof_cm    = -1.0
        self.down_tof_cm     = -1.0
        self.height_cm  = -1.0
        self.pitch_deg  = -1.0
        self.roll_deg   = -1.0
        self.yaw_deg    = -1.0
        self.x_speed_dm_s = -1.0
        self.y_speed_dm_s = -1.0
        self.z_speed_dm_s = -1.0
        self.x_accel_cm_s2 = -1.0
        self.y_accel_cm_s2 = -1.0
        self.z_accel_cm_s2 = -1.0
        self.battery_percent = -1.0
        self.min_temp_C = -1.0
        self.max_temp_C = -1.0

        self.person_detector = PersonDetector(device=DEVICE)

        self.rc_loop_active = False
        self.altitude_pid = PIDController(*GAINS_ALTITUDE_PID,
                                          output_min=-UD_MAX_VELOCITY_CM_S, output_max=UD_MAX_VELOCITY_CM_S)
        self.altitude_face_pid = PIDController(*GAINS_ALTITUDE_FACE_PID,
                                               output_min=-UD_MAX_VELOCITY_CM_S, output_max=UD_MAX_VELOCITY_CM_S)
        self.rc = [0, 0, 0, 0]  # lr, fb, ud, yaw

        self._altitude_log: list[tuple] = []
        self._log_start_time: float = 0.0

        # Face-loss hysteresis — last wall-clock time face was detected
        self._last_face_seen_time: float = 0.0

        # EMA-smoothed nose_y for altitude PID (kills YOLO ±5px jitter)
        self._face_y_smoothed: float = 0.0
        self._face_y_ema_initialized: bool = False

        # Mission pad telemetry (downward camera). -1 = no pad in view.
        self.mission_pad_id: int = -1
        self.mission_pad_x_cm: float = 0.0
        self.mission_pad_y_cm: float = 0.0
        self.mission_pad_z_cm: float = 0.0

    def start(self):
        print("[INFO] Connecting to Tello...")
        self.tello.connect()
        print(f"[INFO] Connected. Battery: {self.tello.get_battery()}%")

        # Soft-reset stale state from prior crashed run (no-ops if drone is fresh)
        print("[INFO] Clearing stale state (land + streamoff)...")
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
            self.tello.land()
        except TelloException:
            pass
        try:
            self.tello.streamoff()
        except TelloException:
            pass
        time.sleep(1.5)

        print("[INFO] Starting video stream...")
        self.tello.streamon()

        if self.phantom_mode:
            print("[INFO] PHANTOM mode — skipping takeoff. Logic dry-run only.")
        else:
            print("[INFO] Taking off...")
            # takeoff BEFORE background threads — blocks ~5–20 s, avoids EXT tof? colliding mid-takeoff
            with self._sdk_lock:
                self.tello.takeoff()
            print("[INFO] Airborne.")

        self._log_start_time = time.time()
        self.video_active = True
        self.front_tof_active   = True
        self.telemetry_active   = True
        self.rc_loop_active     = True

        threading.Thread(target=self._update_telemetry, daemon=True).start()
        threading.Thread(target=self._update_front_tof, daemon=True).start()
        threading.Thread(target=self._rc_control_loop, daemon=True).start()

        self._video_loop()

    def stop(self):
        if self._stop_finished:
            return
        # Kill ToF thread FIRST and wait for in-flight EXT tof? (timeout=1s) to drain.
        # Otherwise its response gets queued and 'land'/'streamoff' read it instead of 'ok'.
        self.front_tof_active = False
        self.rc_loop_active = False
        self.video_active = False
        self.telemetry_active = False

        time.sleep(1.5)  # > EXT tof? timeout (1s) so ToF thread fully exits

        if not self.phantom_mode:
            try:
                with self._sdk_lock:
                    self.tello.land()
            except TelloException:
                pass

        time.sleep(0.6)
        try:
            with self._sdk_lock:
                self.tello.streamoff()
        except TelloException:
            pass
        try:
            self.tello.end()
        except Exception:
            pass
        finally:
            self._stop_finished = True
            self._save_altitude_log()

    def _save_altitude_log(self) -> None:
        """Write altitude PID log to logs/altitude_response.csv."""
        if not self._altitude_log:
            return
        os.makedirs("logs", exist_ok=True)
        path = "logs/altitude_response.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "height_cm", "error", "ud_command", "mode"])
            writer.writerows(self._altitude_log)
        print(f"[INFO] Altitude log saved → {path}")

    def _video_loop(self):
        cv2.namedWindow("TelloInterceptor", cv2.WINDOW_AUTOSIZE)
        frame_reader = self.tello.get_frame_read()

        frame_count    = 0
        last_persons: list[Person] = []
        frame_center_x = FRAME_CENTER_X
        frame_center_y = FRAME_CENTER_Y

        while self.video_active:
            frame = frame_reader.frame
            if frame is None:
                continue

            # derive true center from actual frame size
            h, w = frame.shape[:2]
            frame_center_x = w // 2
            frame_center_y = h // 2
            face_target_y  = int(h * FACE_TARGET_Y_RATIO)  # upper-third framing

            frame_count += 1

            # --- Pose detection every YOLO_STRIDE frames ---
            if frame_count % YOLO_FRAME_STRIDE == 0:
                last_persons = self.person_detector.detect_persons_in_frame(frame)



             # --- HUD overlays ---

            target_center_x, target_center_y, tracking_face = draw_person_overlays(frame, last_persons)
            draw_frame_crosshair(frame, frame_center_x, frame_center_y)
            draw_face_target_line(frame, face_target_y)
            target_y_reference = face_target_y if tracking_face else frame_center_y
            draw_detection_state(frame, target_center_x, target_center_y, frame_center_x, target_y_reference, tracking_face)
            draw_telemetry_strip(frame, self)
            if self.phantom_mode:
                draw_phantom_badge(frame)



            # --- Altitude PID with face-loss hysteresis ---
            # Three modes:
            #   FACE   — face visible, park nose at upper-third Y
            #   HOVER  — face lost <FACE_LOST_GRACE_S ago, ud=0 (Tello holds via baro+optical-flow)
            #   BARO   — face lost >grace, search altitude via TARGET_ALTITUDE_CM
            # Hysteresis stops the face/baro PID fight that caused 158 mode switches last run.
            tracking_face = tracking_face and target_center_y is not None
            now = time.time()
            if tracking_face:
                self._last_face_seen_time = now
            since_face_seen = now - self._last_face_seen_time
            in_grace = (not tracking_face) and (since_face_seen < FACE_LOST_GRACE_S) and self._last_face_seen_time > 0

            if tracking_face:
                # EMA smooth nose_y to kill YOLO jitter before differentiating
                if not self._face_y_ema_initialized:
                    self._face_y_smoothed = float(target_center_y)
                    self._face_y_ema_initialized = True
                else:
                    self._face_y_smoothed = (FACE_Y_EMA_ALPHA * target_center_y +
                                             (1.0 - FACE_Y_EMA_ALPHA) * self._face_y_smoothed)
                error_altitude = self._face_y_smoothed - face_target_y  # px, +ve = face below target
                self.altitude_pid.reset_integral()
                ud = self.altitude_face_pid.compute(error_altitude, RC_LOOP_INTERVAL_S)
                mode_label = "face"
            elif in_grace:
                error_altitude = 0.0
                ud = 0.0  # hover — let Tello hold altitude itself
                self.altitude_pid.reset_integral()
                self.altitude_face_pid.reset_integral()
                self._face_y_ema_initialized = False  # next face = fresh EMA
                mode_label = "hover"
            else:
                error_altitude = TARGET_ALTITUDE_CM - self.height_cm  # cm
                self.altitude_face_pid.reset_integral()
                self._face_y_ema_initialized = False
                ud = self.altitude_pid.compute(error_altitude, RC_LOOP_INTERVAL_S)
                mode_label = "baro"

            # Tracking floor — face PID can pull drone DOWN into ground effect / propwash zone.
            # Block descent below MIN_TRACKING_ALTITUDE_CM. Climb still allowed.
            if self.height_cm <= MIN_TRACKING_ALTITUDE_CM and ud < 0:
                ud = 0

            # Geofence clamps (independent of PID)
            if self.height_cm >= GEOFENCE_MAX_ALTITUDE_CM and ud > 0:
                ud = 0
            if 0 < self.height_cm <= GEOFENCE_MIN_ALTITUDE_CM and ud < 0:
                ud = 0

            self.rc[2] = int(round(ud))

            self._altitude_log.append((
                time.time() - self._log_start_time,
                self.height_cm,
                error_altitude,
                ud,
                mode_label,
            ))

           
            cv2.imshow("TelloInterceptor", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                self.video_active = False  # exits loop; threads also see flag before stop().land()
                break

        cv2.destroyAllWindows()

    def _update_front_tof(self):
        """Background thread: read front ToF + down ToF as fast as possible."""
        while self.front_tof_active:
            try:

                # frontward ToF
                with self._sdk_lock:
                    raw = self.tello.send_command_with_return("EXT tof?", timeout=1)
                if raw and raw.strip().startswith("tof "):
                    mm = int(raw.strip().split()[1])
                    self.front_tof_cm = -1.0 if mm >= 8190 else mm / 10.0

                # downward ToF
                state = self.tello.get_current_state()
                if state:
                    self.down_tof_cm = state.get("tof", -1)
                
                # ToF or barometer height?
                baro_height_cm = state.get("h", -1) if state else -1

                if self.down_tof_cm > baro_height_cm: 
                    self.height_cm = self.down_tof_cm
                else:
                    self.height_cm = baro_height_cm

            except Exception:
                pass

    def _update_telemetry(self):
        while self.telemetry_active:
            try:
                state = self.tello.get_current_state()
                if state:
                    self.pitch_deg       = state.get("pitch", -1)
                    self.roll_deg        = state.get("roll",  -1)
                    self.yaw_deg         = state.get("yaw",   -1)
                    self.x_speed_dm_s    = state.get("vgx",  -1)
                    self.y_speed_dm_s    = state.get("vgy",  -1)
                    self.z_speed_dm_s    = state.get("vgz",  -1)
                    self.x_accel_cm_s2   = state.get("agx",  -1)
                    self.y_accel_cm_s2   = state.get("agy",  -1)
                    self.z_accel_cm_s2   = state.get("agz",  -1)
                    self.battery_percent = state.get("bat",  -1)
                    self.min_temp_C = state.get("templ",  -1)
                    self.max_temp_C = state.get("temph",  -1)

                    # Mission pad — `mid` = -1 when no pad detected. x/y/z in cm relative to pad center.
                    self.mission_pad_id   = state.get("mid", -1)
                    self.mission_pad_x_cm = state.get("x",   0)
                    self.mission_pad_y_cm = state.get("y",   0)
                    self.mission_pad_z_cm = state.get("z",   0)


            except Exception:
                pass
            time.sleep(0.2)

    # Thread: runs at 20 Hz, just sends rc[]. No computation here.
    def _rc_control_loop(self) -> None:
        """Send RC commands to drone at 20 Hz independent of video FPS.

        In phantom mode, becomes a pure no-op — rc[] still updates from video loop
        and shows in HUD, but never reaches the drone.
        """
        while self.rc_loop_active:
            if not self.phantom_mode:
                with self._sdk_lock:
                    self.tello.send_rc_control(*self.rc)
            time.sleep(RC_LOOP_INTERVAL_S)  # 0.05 s → 20 Hz