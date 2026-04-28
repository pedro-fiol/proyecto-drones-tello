import os
import cv2
import torch
import threading
import time
from djitellopy import Tello, TelloException
from ultralytics import YOLO
from .constants import FRAME_CENTER_X, FRAME_CENTER_Y
from .hud_overlay import (
    draw_detection_state,
    draw_frame_crosshair,
    draw_person_overlays,
    draw_telemetry_strip,
)

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import logging
logging.getLogger("djitellopy").setLevel(logging.WARNING)

DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_STR          = DEVICE.upper()   # "CUDA" or "CPU" for HUD
YOLO_STRIDE         = 2
NOSE_IDX            = 0      # nose = keypoint 0 in COCO format
NOSE_CONF_THRESHOLD = 0.7    # min confidence to consider face visible

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

        self.model_pose = YOLO("yolov8s-pose.pt")   # person + keypoints
        self.model_pose.to(DEVICE)

        # self.model_obj = YOLO("yolov8s.pt")        # objects — uncomment when needed
        # self.model_obj.to(DEVICE)

    def start(self):
        self.tello.connect()
        self.tello.streamon()
        # takeoff BEFORE background threads — blocks ~5–20 s, avoids EXT tof? colliding mid-takeoff
        with self._sdk_lock:
            self.tello.takeoff()

        self.video_active = True
        self.front_tof_active   = True
        self.telemetry_active   = True

        threading.Thread(target=self._update_telemetry, daemon=True).start()
        threading.Thread(target=self._update_front_tof, daemon=True).start()

        self._video_loop()

    def stop(self):
        if self._stop_finished:
            return
        self.front_tof_active = False
        self.video_active = False
        self.telemetry_active = False

        time.sleep(0.2)
        try:
            with self._sdk_lock:
                self.tello.land()
        except TelloException:
            pass

        time.sleep(0.6)
        try:
            self.tello.streamoff()
        except TelloException:
            pass
        try:
            self.tello.end()
        except Exception:
            pass
        finally:
            self._stop_finished = True

    def _video_loop(self):
        cv2.namedWindow("TelloInterceptor", cv2.WINDOW_AUTOSIZE)
        frame_reader = self.tello.get_frame_read()

        frame_count    = 0
        last_persons   = []    # (x1, y1, x2, y2, conf, face_visible, nose_x, nose_y)
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
            if frame_count % YOLO_STRIDE == 0:
                results = self.model_pose(
                    frame,
                    classes=[0],    # person only
                    conf=0.7,
                    verbose=False
                )
                last_persons.clear()
                for r in results:
                    for i, box in enumerate(r.boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])

                        face_visible = False
                        nose_x, nose_y = None, None
                        if r.keypoints is not None and i < len(r.keypoints):
                            kp        = r.keypoints[i]
                            nose_conf = float(kp.conf[0][NOSE_IDX])
                            if nose_conf > NOSE_CONF_THRESHOLD:
                                face_visible = True
                                nose_x = int(kp.xy[0][NOSE_IDX][0])
                                nose_y = int(kp.xy[0][NOSE_IDX][1])

                        last_persons.append((x1, y1, x2, y2, conf, face_visible, nose_x, nose_y))

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
            except Exception:
                pass
            time.sleep(0.2)
