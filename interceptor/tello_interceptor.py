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
    FRAME_CENTER_X, FRAME_CENTER_Y, GAINS_ALTITUDE_PID, GEOFENCE_MAX_ALTITUDE_CM, RC_LOOP_INTERVAL_S, TARGET_ALTITUDE_CM,
    YOLO_FRAME_STRIDE, YOLO_CONFIDENCE_MIN, YOLO_PERSON_CLASS_ID,
    NOSE_KEYPOINT_INDEX, NOSE_CONFIDENCE_THRESHOLD,
)


from .hud_overlay import (
    draw_detection_state,
    draw_frame_crosshair,
    draw_person_overlays,
    draw_telemetry_strip,
)

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import logging
logging.getLogger("djitellopy").setLevel(logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Using device: {DEVICE}")


class TelloInterceptor:
    def __init__(self):
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
        self.altitude_pid = PIDController(*GAINS_ALTITUDE_PID, output_min=-100, output_max=100)
        self.rc = [0, 0, 0, 0]  # lr, fb, ud, yaw

        self._altitude_log: list[tuple] = []
        self._log_start_time: float = 0.0

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
            writer.writerow(["time_s", "height_cm", "error_cm", "ud_command"])
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

            frame_count += 1

            # --- Pose detection every YOLO_STRIDE frames ---
            if frame_count % YOLO_FRAME_STRIDE == 0:
                last_persons = self.person_detector.detect_persons_in_frame(frame)


            # PID control    
            error_altitude = TARGET_ALTITUDE_CM - self.height_cm
            ud = self.altitude_pid.compute(error_altitude, RC_LOOP_INTERVAL_S)
            
            if self.height_cm >= GEOFENCE_MAX_ALTITUDE_CM:
                ud = min(ud, 0)  # only allow downward movement
            
            self.rc[2] = int(round(ud))  # set vertical speed command
            self._altitude_log.append((
                time.time() - self._log_start_time,
                self.height_cm,
                error_altitude,
                ud,
            ))

            # --- Draw detections + HUD ---
            target_center_x, target_center_y, tracking_face = draw_person_overlays(frame, last_persons)
            draw_frame_crosshair(frame, frame_center_x, frame_center_y)
            draw_detection_state(frame, target_center_x, target_center_y,
                                 frame_center_x, frame_center_y, tracking_face)
            draw_telemetry_strip(frame, self)

            cv2.imshow("TelloInterceptor", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                self.video_active = False  # exits loop; threads also see flag before stop().land()
                break

        cv2.destroyAllWindows()

    def _update_front_tof(self):
        """Background thread: read front ToF + down ToF as fast as possible."""
        while self.front_tof_active:
            try:
                with self._sdk_lock:
                    raw = self.tello.send_command_with_return("EXT tof?", timeout=1)

                if raw and raw.strip().startswith("tof "):
                    mm = int(raw.strip().split()[1])
                    self.front_tof_cm = -1.0 if mm >= 8190 else mm / 10.0

                # down ToF lives in the UDP state stream — cheap dict lookup, no SDK call
                state = self.tello.get_current_state()
                if state:
                    self.down_tof_cm = state.get("tof", -1)
            except Exception:
                pass

    def _update_telemetry(self):
        while self.telemetry_active:
            try:
                state = self.tello.get_current_state()
                if state:
                    self.height_cm       = state.get("h",   -1)
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

                 # si no esta sobrevolando ningún obstáculo, usar el down ToF como altura (más confiable que el barómetro)
                if self.down_tof_cm > self.height_cm:
                    self.height_cm = self.down_tof_cm

            except Exception:
                pass
            time.sleep(0.2)

    # Thread: runs at 20 Hz, just sends rc[]. No computation here.
    def _rc_control_loop(self) -> None:
        """Send RC commands to drone at 20 Hz independent of video FPS."""
        while self.rc_loop_active:
            with self._sdk_lock:
                self.tello.send_rc_control(*self.rc)
            time.sleep(RC_LOOP_INTERVAL_S)  # 0.05 s → 20 Hz