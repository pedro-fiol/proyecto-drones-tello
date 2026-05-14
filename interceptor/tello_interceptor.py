import csv
import ctypes
import ctypes.wintypes
import os
import cv2
import torch
import threading
import time
import logging

from djitellopy import Tello, TelloException
from .pid_controller import PIDController
from .perception import Person, PersonDetector
from .target import TARGET, select_best_person


from .constants import (
    FRONT_TOF_WALL_BACKOFF_CM, TARGET_EMA_ALPHA, FB_MAX_VELOCITY_CM_S, FRAME_CENTER_X, FRAME_CENTER_Y,
    TARGET_LOST_GRACE_S, FRONT_TOF_WALL_STOP_CM,
    GAINS_ALTITUDE_PID, GAINS_ALTITUDE_TARGET_PID,
    GAINS_FORWARD_BACK_TOF_PID, GAINS_FORWARD_BACK_BBOX_PID, GAINS_YAW_PID,
    INTERCEPT_DISTANCE_CM,
    MANUAL_FB_VELOCITY_CM_S, MANUAL_LR_VELOCITY_CM_S,
    MANUAL_UD_VELOCITY_CM_S, MANUAL_YAW_VELOCITY_DEG_S,
    MAX_TRACKING_ALTITUDE_CM, MIN_TRACKING_ALTITUDE_CM,
    RC_LOOP_INTERVAL_S, TARGET_ALTITUDE_CM, UD_MAX_VELOCITY_CM_S, WALL_BACKOFF_VELOCITY_CM_S, YAW_MAX_VELOCITY_CM_S,
    YOLO_FRAME_STRIDE,
)


# ---- Windows keyboard input (manual mode) ----
# GetAsyncKeyState polled each frame for hold-to-move (release = stop).
# PeekMessage drain stops cv2 freeze when key held (OS WM_KEYDOWN flood).
_user32 = ctypes.windll.user32
_VK_W, _VK_S, _VK_A, _VK_D, _VK_Q, _VK_E = 0x57, 0x53, 0x41, 0x44, 0x51, 0x45
_VK_UP, _VK_DOWN = 0x26, 0x28
_VK_SPACE, _VK_M, _VK_ESC = 0x20, 0x4D, 0x1B


def _is_key_pressed(vk: int) -> bool:
    """True when key currently held (Windows VK code)."""
    return _user32.GetAsyncKeyState(vk) & 0x8000 != 0

class _MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd",    ctypes.wintypes.HWND),
        ("message", ctypes.wintypes.UINT),
        ("wParam",  ctypes.wintypes.WPARAM),
        ("lParam",  ctypes.wintypes.LPARAM),
        ("time",    ctypes.wintypes.DWORD),
        ("pt_x",    ctypes.wintypes.LONG),
        ("pt_y",    ctypes.wintypes.LONG),
    ]


_PM_REMOVE = 0x0001
_WM_KEYFIRST, _WM_KEYLAST = 0x0100, 0x0109


def _drain_keyboard_messages() -> None:
    """Flush WM_KEY* from this thread's queue so cv2 render pump keeps up under held keys."""
    msg = _MSG()
    while _user32.PeekMessageW(ctypes.byref(msg), 0, _WM_KEYFIRST, _WM_KEYLAST, _PM_REMOVE):
        pass


from .hud_overlay import (
    draw_control_mode_badge,
    draw_detection_state,
    draw_target_y_line,
    draw_frame_crosshair,
    draw_person_overlays,
    draw_phantom_badge,
    draw_telemetry_strip,
)

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


logging.getLogger("djitellopy").setLevel(logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Using device: {DEVICE}")


class TelloInterceptor:
    # Phantom mode -> Drone does not take off
    def __init__(self, phantom_mode: bool = False):
        self.phantom_mode = phantom_mode
        self.tello        = Tello()
        self._sdk_lock    = threading.Lock() # to avoid UDP socket conflicts with simultaneous SDK commands
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
        
        # Actuation
        self.rc_loop_active = False
        self.altitude_pid = PIDController(*GAINS_ALTITUDE_PID, output_min=-UD_MAX_VELOCITY_CM_S, output_max=UD_MAX_VELOCITY_CM_S)
        self.altitude_target_pid = PIDController(*GAINS_ALTITUDE_TARGET_PID, output_min=-UD_MAX_VELOCITY_CM_S, output_max=UD_MAX_VELOCITY_CM_S)
        self.yaw_pid =  PIDController(*GAINS_YAW_PID, output_min=-YAW_MAX_VELOCITY_CM_S, output_max=YAW_MAX_VELOCITY_CM_S)
        self.pitch_pid =  PIDController(*GAINS_FORWARD_BACK_TOF_PID, output_min=-FB_MAX_VELOCITY_CM_S, output_max=FB_MAX_VELOCITY_CM_S)
        self.pitch_bbox_pid = PIDController(*GAINS_FORWARD_BACK_BBOX_PID, output_min=-FB_MAX_VELOCITY_CM_S, output_max=FB_MAX_VELOCITY_CM_S)
        self.rc = [0, 0, 0, 0]  # left right, forward backward, up down, yaw

        # logging for PID tunning
        self._altitude_log: list[tuple] = []
        self._yaw_log: list[tuple] = []
        self._pitch_log: list[tuple] = []
        self._log_start_time: float = 0.0

        # last time the active target was visible
        # Used to avoid switching to searching mode (no target detected) due to detection jitter
        self._last_target_seen_time: float = 0.0

        # EMA-smoothed target position for PIDs (kills YOLO ±5px jitter)
        self._target_smoothed_x_px: float = 0.0
        self._target_smoothed_y_px: float = 0.0
        self._target_ema_initialized: bool = False

        # EMA-smoothed closeness signal for bbox pitch PID. Same TARGET_EMA_ALPHA as
        # target x/y — closeness is computed from raw detection, jitters per frame.
        # Without smoothing, Kd term spikes on YOLO noise and produces twitchy fb.
        self._closeness_smoothed: float = 0.0
        self._closeness_ema_initialized: bool = False

        # Tracks last pitch PID source ("tof"|"bbox"|"none") to detect source switches.
        # On switch, the incoming PID's error_last is seeded with current error so the
        # first D-term doesn't spike off a stale error from a different unit system.
        self._pitch_source_last: str = "none"

        # Manual keyboard control
        self.is_manual: bool = False
        self.manual_lr: int = 0
        self.manual_fb: int = 0
        self.manual_ud: int = 0
        self.manual_yaw: int = 0
        self._space_was_pressed: bool = False
        self._m_was_pressed: bool = False
        self._esc_was_pressed: bool = False

        self._space_rising_edge: bool = False
        self._m_rising_edge: bool = False
        self._esc_rising_edge: bool = False

        self._is_airborne: bool = False
        self._is_toggling_flight: bool = False


    def start(self):

        print("[INFO] Connecting to Tello...")
        self.tello.connect()
        print(f"[INFO] Connected. Battery: {self.tello.get_battery()}%")

        # Reset drone from previous run just in case
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
        time.sleep(2.5)  # Increased delay to allow drone + SDK socket to reset

        print("[INFO] Starting video stream...")
        self.tello.streamon()

        if self.phantom_mode:
            print("[INFO] PHANTOM mode — skip takeoff. Logic only.")
        else:
            print("[INFO] Auto-takeoff disabled. Press SPACE to take off manually.")
            # try:
            #     self.tello.takeoff()
            #     self._is_airborne = True
            # except TelloException as e:
            #     print(f"[ERROR] Auto-takeoff failed: {e}")
            #     self._is_airborne = False

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

        # Save logs BEFORE drone shutdown — if save raises, drone cleanup must still run
        try:
            self._save_altitude_log()
        except Exception as e:
            print(f"[WARN] altitude log save failed: {e}")
        try:
            self._save_yaw_log()
        except Exception as e:
            print(f"[WARN] yaw log save failed: {e}")
        try:
            self._save_pitch_log()
        except Exception as e:
            print(f"[WARN] pitch log save failed: {e}")

        if not self.phantom_mode and self._is_airborne:
            print("[INFO] Initiating auto-landing on shutdown...")
            try:
                with self._sdk_lock:
                    if 'responses' in self.tello.get_own_udp_object():
                        self.tello.get_own_udp_object()['responses'].clear()
                    self.tello.land()
                    print("[INFO] Auto-landing complete.")
            except TelloException as e:
                print(f"[ERROR] Failed to land during shutdown: {e}")

        # Do not use "end()" from djitellopy, it closes SDK socket and causes errors for subsequent runs.
        time.sleep(0.6)
        try:
            with self._sdk_lock:
                self.tello.streamoff()
        except TelloException:
            pass
        finally:
            self._stop_finished = True


    def _save_csv_safe(self, path: str, header: list, rows: list) -> None:
        """Write CSV. On PermissionError (file locked), fall back to timestamped name."""
        os.makedirs("logs", exist_ok=True)
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            print(f"[INFO] Log saved → {path}")
        except PermissionError:
            ts = time.strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(path)
            fallback = f"{base}_{ts}{ext}"
            with open(fallback, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            print(f"[WARN] {path} locked → saved to {fallback}")


    def _save_yaw_log(self) -> None:
        """Write yaw PID log to logs/yaw_response.csv."""
        if not self._yaw_log:
            return
        self._save_csv_safe(
            "logs/yaw_response.csv",
            ["time_s", "target_x", "error_x_pixels", "yaw_command"],
            self._yaw_log,
        )


    def _save_pitch_log(self) -> None:
        """Write pitch PID log to logs/pitch_response.csv.

        Columns:
            time_s              — seconds since takeoff
            front_tof_cm        — raw front ToF reading (-1 = invalid)
            closeness           — tracked person closeness signal from active TARGET (-1 = no person)
            error               — PID error in active source units (cm for tof, closeness units for bbox)
            fb_command          — commanded fb velocity (cm/s) sent to RC
            source              — "tof" | "bbox" | "none"
        """
        if not self._pitch_log:
            return
        self._save_csv_safe(
            "logs/pitch_response.csv",
            ["time_s", "front_tof_cm", "closeness", "error", "fb_command", "source"],
            self._pitch_log,
        )


    def _save_altitude_log(self) -> None:
        """Write altitude PID log to logs/altitude_response.csv."""
        if not self._altitude_log:
            return
        self._save_csv_safe(
            "logs/altitude_response.csv",
            ["time_s", "height_cm", "error", "ud_command", "mode"],
            self._altitude_log,
        )


    def _video_loop(self):
        cv2.namedWindow("TelloInterceptor", cv2.WINDOW_AUTOSIZE)
        drone_video = self.tello.get_frame_read()

        frame_count    = 0
        consecutive_none_frames = 0
        detected_persons: list[Person] = []
        frame_center_x = FRAME_CENTER_X
        frame_center_y = FRAME_CENTER_Y

        while self.video_active:
            current_frame = drone_video.frame
            if current_frame is None:
                consecutive_none_frames += 1
                # Exit only after a long, sustained outage. H264 decoder hiccups can
                # produce 1-3s bursts of None frames and self-recover; do not pull the
                # plug (and land) on those. 300 ≈ 10s.
                if consecutive_none_frames > 300:
                    print("[ERROR] Video stream dead — H.264 decoder stuck. Exiting.")
                    self.video_active = False
                    break
                # Keep cv2 window pumping and ESC alive even when no frame is ready.
                cv2.waitKey(1)
                self._poll_keyboard()
                if self._esc_rising_edge:
                    self.video_active = False
                    break
                time.sleep(0.01)
                continue
            consecutive_none_frames = 0  # Reset counter on successful frame

            # derive true center from actual frame size
            h, w = current_frame.shape[:2]
            frame_center_x = w // 2
            frame_center_y = h // 2
            target_setpoint_x_px = frame_center_x                       # yaw setpoint: target centered horizontally
            target_setpoint_y_px = int(h * TARGET.target_y_ratio)       # alt setpoint: per-target framing

            frame_count += 1

            # --- Pose detection every YOLO_STRIDE frames ---
            if frame_count % YOLO_FRAME_STRIDE == 0:
                detected_persons = self.person_detector.detect_persons_in_frame(current_frame)

            # --- Single-person filter — Phase 6a is single-target by spec ---
            best_person = select_best_person(detected_persons, TARGET)
            persons_to_draw = [best_person] if best_person is not None else []

            # --- HUD overlays ---
            target_x_px, target_y_px, tracking_target, tracked_person = draw_person_overlays(
                current_frame, persons_to_draw, TARGET
            )
            draw_frame_crosshair(current_frame, frame_center_x, frame_center_y)
            draw_target_y_line(current_frame, target_setpoint_y_px)

            draw_detection_state(current_frame, target_x_px, target_y_px, frame_center_x,
                                 target_setpoint_y_px, tracking_target, target_name=TARGET.name)
            draw_telemetry_strip(current_frame, self)
            if self.phantom_mode:
                draw_phantom_badge(current_frame)
            draw_control_mode_badge(
                current_frame, self.is_manual,
                self.manual_lr, self.manual_fb, self.manual_ud, self.manual_yaw,
            )

            tracking_target = tracking_target and target_y_px is not None and target_x_px is not None


            # --- PID ---
            # Three modes:
            #   INTERCEPTING — target visible, park target at TARGET.target_y_ratio via target PID
            #   HOVER        — target lost < TARGET_LOST_GRACE_S ago, ud=0 (Tello holds via baro+optical-flow)
            #   SEARCHING    — target lost > TARGET_LOST_GRACE_S, hold TARGET_ALTITUDE_CM via baro PID
            #
            # YOLO detection jitter -> drone switches between intercepting and searching modes rapidly -> PID controller oscillates
            # Fix: grace period after target loss where drone enters HOVER mode

            valid_front_tof = self.front_tof_cm > 0

            # default pitch logging fields (filled per branch)
            pitch_source = "none"
            closeness_value = -1.0

            # lr currently unused by auto PIDs (Phase 7+); init 0 so manual mode + rc[0] write are safe.
            lr = 0

            # target visible
            if tracking_target:
                self._last_target_seen_time = time.time()
                mode_label = "intercepting"

                # EMA smooth target position to kill YOLO jitter before differentiating
                if not self._target_ema_initialized:
                    self._target_smoothed_x_px = float(target_x_px)
                    self._target_smoothed_y_px = float(target_y_px)
                    self._target_ema_initialized = True
                else:
                    self._target_smoothed_x_px = (TARGET_EMA_ALPHA * target_x_px +
                                                  (1.0 - TARGET_EMA_ALPHA) * self._target_smoothed_x_px)
                    self._target_smoothed_y_px = (TARGET_EMA_ALPHA * target_y_px +
                                                  (1.0 - TARGET_EMA_ALPHA) * self._target_smoothed_y_px)

                # All errors use convention: error = target - measurement (with sign baked into Kp)
                # --- Target Altitude PID ---
                error_altitude = self._target_smoothed_y_px - target_setpoint_y_px   # px (+ve = target below setpoint in image)
                ud = self.altitude_target_pid.compute(error_altitude, RC_LOOP_INTERVAL_S)

                # --- Yaw PID ---
                error_yaw = self._target_smoothed_x_px - frame_center_x   # px (+ve = target right of center → CW)
                yaw = self.yaw_pid.compute(error_yaw, RC_LOOP_INTERVAL_S)

                # --- Pitch PID ---

                # ToF PID
                if valid_front_tof:
                    pitch_source = "tof"
                    error_pitch = INTERCEPT_DISTANCE_CM - self.front_tof_cm

                    # Switching INTO ToF: seed error_last to current err so D-term
                    # doesn't spike off whatever was last in pitch_pid (possibly stale).
                    if self._pitch_source_last != "tof":
                        self.pitch_pid.error_last = error_pitch

                    fb = self.pitch_pid.compute(error_pitch, RC_LOOP_INTERVAL_S)
                    self.pitch_bbox_pid.reset_integral()


                    # Invalidate closeness EMA so next bbox switch re-inits cleanly
                    # rather than blending fresh detection with frozen stale value.
                    self._closeness_ema_initialized = False

                elif tracked_person is not None and TARGET.is_visible(tracked_person):
                    raw_closeness = TARGET.closeness(tracked_person)

                    # EMA-smooth closeness before PID. Same TARGET_EMA_ALPHA as target x/y EMA.
                    if not self._closeness_ema_initialized:
                        self._closeness_smoothed = float(raw_closeness)
                        self._closeness_ema_initialized = True
                    else:
                        self._closeness_smoothed = (TARGET_EMA_ALPHA * raw_closeness +
                                                    (1.0 - TARGET_EMA_ALPHA) * self._closeness_smoothed)
                    closeness_value = self._closeness_smoothed
                    
                    if closeness_value < TARGET.closeness_setpoint:
                        pitch_source = "bbox"
                        error_pitch = TARGET.closeness_setpoint - closeness_value
                        # Switching INTO bbox: seed error_last (ratio units) to avoid
                        # D-spike off a stale ToF-unit error from previous bbox use.
                        if self._pitch_source_last != "bbox":
                            self.pitch_bbox_pid.error_last = error_pitch
                        fb = self.pitch_bbox_pid.compute(error_pitch, RC_LOOP_INTERVAL_S)
                        self.pitch_pid.reset_integral()
                    else:
                        pitch_source = "none"
                        error_pitch = 0.0
                        fb = 0.0
                        self.pitch_pid.reset_integral()
                        self.pitch_bbox_pid.reset_integral()
                else:
                    pitch_source = "none"
                    error_pitch = 0.0
                    fb = 0.0
                    self.pitch_pid.reset_integral()
                    self.pitch_bbox_pid.reset_integral()


            # target not visible but seen recently -> grace period, hover instead of searching
            elif self._last_target_seen_time > 0 and (time.time() - self._last_target_seen_time) < TARGET_LOST_GRACE_S:
                mode_label = "hover"

                # altitude
                error_altitude = 0.0
                ud = 0.0  # hover — let Tello hold altitude itself
                self.altitude_pid.reset_integral()
                self.altitude_target_pid.reset_integral()

                # yaw
                error_yaw = 0.0
                yaw = 0.0
                self.yaw_pid.reset_integral()

                # pitch
                error_pitch = 0.0
                fb = 0.0
                self.pitch_pid.reset_integral()
                self.pitch_bbox_pid.reset_integral()

                # EMA — next target = fresh start for both position + closeness
                self._target_ema_initialized = False
                self._closeness_ema_initialized = False

            # target not visible and not in grace -> searching mode
            else:
                mode_label = "searching"
                error_altitude = TARGET_ALTITUDE_CM - self.height_cm  # cm (+ve = below target)

                # altitude PID — baro
                self.altitude_target_pid.reset_integral()
                self._target_ema_initialized = False
                self._closeness_ema_initialized = False
                ud = self.altitude_pid.compute(error_altitude, RC_LOOP_INTERVAL_S)

                # yaw
                yaw = 0.0
                error_yaw = 0.0
                self.yaw_pid.reset_integral()

                # pitch — only push back from walls during search
                if valid_front_tof and self.front_tof_cm < INTERCEPT_DISTANCE_CM:
                    pitch_source = "tof"
                    error_pitch = INTERCEPT_DISTANCE_CM - self.front_tof_cm
                    fb = self.pitch_pid.compute(error_pitch, RC_LOOP_INTERVAL_S)
                else:
                    pitch_source = "none"
                    error_pitch = 0.0
                    fb = 0.0
                    self.pitch_pid.reset_integral()
                self.pitch_bbox_pid.reset_integral()



            # Manual keyboard override — replaces PID output before safety clamp.
            # Safety blocks below still apply on top, keyboard cannot bypass them.
            if self.is_manual:
                mode_label = "manual"
                lr = self.manual_lr
                fb = self.manual_fb
                ud = self.manual_ud
                yaw = self.manual_yaw

            # Altitude Safety, dead-reckoning XY dropped (unreliable on Tello)
            if self.height_cm >= MAX_TRACKING_ALTITUDE_CM and ud > 0:
                ud = 0  # Block climbing above ceiling
            elif 0 < self.height_cm <= MIN_TRACKING_ALTITUDE_CM and ud < 0:
                ud = 0  # Block descending below floor

            # Forward Back Safety
            if valid_front_tof:
                if self.front_tof_cm <= FRONT_TOF_WALL_STOP_CM and fb > 0:
                    fb = 0
                    self.pitch_pid.reset_integral()
                    self.pitch_bbox_pid.reset_integral()
                if mode_label == "searching":
                    if self.front_tof_cm <= FRONT_TOF_WALL_BACKOFF_CM:
                        fb = WALL_BACKOFF_VELOCITY_CM_S
                        self.pitch_pid.reset_integral()
                        self.pitch_bbox_pid.reset_integral()
                    if FRONT_TOF_WALL_BACKOFF_CM < self.front_tof_cm and fb < 0:
                        fb = 0
                        self.pitch_pid.reset_integral()
                        self.pitch_bbox_pid.reset_integral()



            self.rc[0] = int(round(lr))
            self.rc[1] = int(round(fb))
            self.rc[2] = int(round(ud))
            self.rc[3] = int(round(yaw))

            # for debug
            #print(self.rc[1], self.front_tof_cm)
            

            self._altitude_log.append((
                time.time() - self._log_start_time,
                self.height_cm,
                error_altitude,
                ud,
                mode_label,
            ))

            self._yaw_log.append((
                time.time() - self._log_start_time,
                self._target_smoothed_x_px if tracking_target else -1,
                error_yaw if tracking_target or mode_label == "hover" else 0.0,
                yaw,
            ))

            self._pitch_log.append((
                time.time() - self._log_start_time,
                self.front_tof_cm,
                closeness_value,
                error_pitch,
                fb,
                pitch_source,
            ))

            # Latch pitch source for next-frame transition detection (D-term seeding).
            self._pitch_source_last = pitch_source

           
            cv2.imshow("TelloInterceptor", current_frame)
            # Pump cv2 render. _poll_keyboard() drains key messages so cv2
            # doesn't freeze when a movement key is held.
            cv2.waitKey(1)
            self._poll_keyboard()

            # ESC — exit always, regardless of mode
            if self._esc_rising_edge:
                self.video_active = False
                break
            # Window-X also exits — useful when terminal has focus and ESC won't fire
            if cv2.getWindowProperty("TelloInterceptor", cv2.WND_PROP_VISIBLE) < 1:
                self.video_active = False
                break

            # SPACE — takeoff / land toggle. Threaded + _sdk_lock to avoid SDK socket clash.
            # Clears djitellopy's response queue first — stale "unknown command: keepalive" or
            # "tof N" responses from prior commands pollute the queue and get popped instead
            # of the real takeoff/land "ok" response, causing djitellopy's 4-retry fail.
            if self._space_rising_edge and not self.phantom_mode:
                if not self._is_toggling_flight:
                    self._is_toggling_flight = True
                    def _toggle_flight():
                        try:
                            # Re-init SDK mode in case the drone rebooted or timed out of SDK mode. 
                            # Do this outside the main lock so we don't hold up other threads on timeout.
                            try:
                                self.tello.send_control_command("command", timeout=1)
                            except Exception:
                                pass # Ignore timeout, standard djitellopy connect() already does this
                                
                            with self._sdk_lock:
                                # Ensure we don't accidentally pop earlier unrelated responses
                                if 'responses' in self.tello.get_own_udp_object():
                                    self.tello.get_own_udp_object()['responses'].clear()
                                
                                if self._is_airborne:
                                    self.tello.land()
                                    self._is_airborne = False
                                else:
                                    self.tello.takeoff()
                                    self._is_airborne = True
                        except TelloException as e:
                            # Do NOT touch _is_airborne here. The successful-path assignment
                            # only runs if the SDK call returned. If it raised, the flag is
                            # already correct (still pre-call value).
                            print(f"[WARN] takeoff/land failed: {e}")
                        finally:
                            self._is_toggling_flight = False
                    threading.Thread(target=_toggle_flight, daemon=True).start()

            # M — toggle manual mode + reset all PIDs to avoid I-term pop on switch
            if self._m_rising_edge:
                self.is_manual = not self.is_manual
                self.manual_lr = self.manual_fb = self.manual_ud = self.manual_yaw = 0
                self.altitude_pid.reset_integral()
                self.altitude_target_pid.reset_integral()
                self.yaw_pid.reset_integral()
                self.pitch_pid.reset_integral()
                self.pitch_bbox_pid.reset_integral()

        cv2.destroyAllWindows()

    def _poll_keyboard(self) -> None:
        _drain_keyboard_messages()

        space_now = _is_key_pressed(_VK_SPACE)
        m_now     = _is_key_pressed(_VK_M)
        esc_now   = _is_key_pressed(_VK_ESC)

        self._space_rising_edge = space_now and not self._space_was_pressed
        self._m_rising_edge     = m_now     and not self._m_was_pressed
        self._esc_rising_edge   = esc_now   and not self._esc_was_pressed

        self._space_was_pressed = space_now
        self._m_was_pressed     = m_now
        self._esc_was_pressed   = esc_now

        if not self.is_manual:
            self.manual_lr = self.manual_fb = self.manual_ud = self.manual_yaw = 0
            return

        self.manual_fb = (
             MANUAL_FB_VELOCITY_CM_S  if _is_key_pressed(_VK_W) else
            -MANUAL_FB_VELOCITY_CM_S  if _is_key_pressed(_VK_S) else 0
        )
        self.manual_lr = (
            -MANUAL_LR_VELOCITY_CM_S  if _is_key_pressed(_VK_A) else
             MANUAL_LR_VELOCITY_CM_S  if _is_key_pressed(_VK_D) else 0
        )
        self.manual_yaw = (
            -MANUAL_YAW_VELOCITY_DEG_S if _is_key_pressed(_VK_Q) else
             MANUAL_YAW_VELOCITY_DEG_S if _is_key_pressed(_VK_E) else 0
        )
        self.manual_ud = (
             MANUAL_UD_VELOCITY_CM_S  if _is_key_pressed(_VK_UP)   else
            -MANUAL_UD_VELOCITY_CM_S  if _is_key_pressed(_VK_DOWN) else 0
        )

    def _update_front_tof(self):
        """Background thread: read front ToF + down ToF at ~5 Hz.

        Sleep matters: spamming `EXT tof?` holds _sdk_lock for ~100ms per call with
        no gap, starves the RC thread, and stresses WiFi → H264 decoder errors
        (`error while decoding MB ...`) → dead frames → auto-land.
        """
        while self.front_tof_active:
            try:
                # frontward ToF — single short critical section, then release lock
                with self._sdk_lock:
                    raw = self.tello.send_command_with_return("EXT tof?", timeout=1)
                if raw and raw.strip().startswith("tof "):
                    mm = int(raw.strip().split()[1])
                    self.front_tof_cm = -1.0 if mm >= 8190 else mm / 10.0

                # downward ToF + baro — read from state listener (no SDK command, no lock)
                state = self.tello.get_current_state()
                if state:
                    self.down_tof_cm = state.get("tof", -1)
                    # Baro is primary altitude per CLAUDE.md. Down ToF is obstacle-warning only.
                    self.height_cm = state.get("h", -1)

            except Exception:
                pass
            time.sleep(0.2)  # 5 Hz — enough for wall avoidance, frees SDK + WiFi

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