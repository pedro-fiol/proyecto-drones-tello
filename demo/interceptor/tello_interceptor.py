import ctypes
import ctypes.wintypes
import os
import cv2
import torch
import threading
import time
import logging
from typing import Optional

from djitellopy import Tello, TelloException
from .pid_controller import PIDController
from .perception import Person, PersonDetector
from .target import (
    TARGET, AVAILABLE_TARGETS, LockState,
    select_best_person, select_target_person, set_active_target,
)
from .reid import ReidEmbedder, crop_bbox

from .constants import (
    GAINS_LEFT_RIGHT_PID, LR_MAX_VELOCITY_CM_S,
    SEARCH_SPIN_VELOCITY_DEG_S,
    SEARCH_ADVANCE_TOLERANCE_CM, SEARCH_ADVANCE_TIMEOUT_S,
    SEARCH_OPEN_SPACE_VELOCITY_CM_S,
    GRACE_RECOVERY_YAW_DEG_S, GRACE_RECOVERY_MIN_OFFSET_PX,
    TARGET_EMA_ALPHA, FB_MAX_VELOCITY_CM_S, FRAME_CENTER_X, FRAME_CENTER_Y,
    TARGET_LOST_GRACE_S, FRONT_TOF_WALL_STOP_CM, FRONT_TOF_DISCONTINUITY_FREEZE_S,
    FRONT_TOF_INVALID_HYSTERESIS_FRAMES, TRACKING_MISS_HYSTERESIS_FRAMES,
    SEARCH_ADVANCE_MAX_DISTANCE_CM,
    SEARCH_ADVANCE_SWEEP_EVERY_S, SEARCH_ADVANCE_SWEEP_YAW_DEG_S,
    SEARCH_ADVANCE_SWEEP_ARC_DEG, SEARCH_ADVANCE_SWEEP_WALL_CM,
    GAINS_ALTITUDE_PID, GAINS_ALTITUDE_TARGET_PID,
    GAINS_FORWARD_BACK_TOF_PID, GAINS_FORWARD_BACK_BBOX_PID, GAINS_YAW_PID,
    INTERCEPT_DISTANCE_CM,
    LOCK_MATCH_THRESHOLD,
    MANUAL_FB_VELOCITY_CM_S, MANUAL_LR_VELOCITY_CM_S,
    MANUAL_UD_VELOCITY_CM_S, MANUAL_YAW_VELOCITY_DEG_S,
    MAX_TRACKING_ALTITUDE_CM, MIN_TRACKING_ALTITUDE_CM,
    RC_LOOP_INTERVAL_S, TARGET_ALTITUDE_CM, UD_MAX_VELOCITY_CM_S, YAW_MAX_VELOCITY_CM_S,
    WEB_MANUAL_HEARTBEAT_GRACE_S,
    YOLO_FRAME_STRIDE,
)

from .hud_overlay import (
    draw_control_mode_badge,
    draw_detection_state,
    draw_target_y_line,
    draw_frame_crosshair,
    draw_person_overlays,
    draw_phantom_badge,
    draw_telemetry_strip,
    draw_wall_warning,
)


os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
logging.getLogger("djitellopy").setLevel(logging.WARNING)





# Windows keyboard input — GetAsyncKeyState for hold-to-move,
# PeekMessage drain to stop cv2 freeze when key held.
_user32 = ctypes.windll.user32
_VK_W, _VK_S, _VK_A, _VK_D, _VK_Q, _VK_E = 0x57, 0x53, 0x41, 0x44, 0x51, 0x45
_VK_UP, _VK_DOWN = 0x26, 0x28
_VK_SPACE, _VK_M, _VK_ESC = 0x20, 0x4D, 0x1B
_VK_I, _VK_C = 0x49, 0x43


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
    """Drain WM_KEY* from queue — cv2 freezes otherwise when key held."""
    msg = _MSG()
    while _user32.PeekMessageW(ctypes.byref(msg), 0, _WM_KEYFIRST, _WM_KEYLAST, _PM_REMOVE):
        pass


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")


class TelloInterceptor:
    """Drone orchestrator — owns Tello SDK, video loop, PIDs, FSM."""

    def __init__(self, phantom_mode: bool = False, headless: bool = False):
        # phantom_mode=True: skip takeoff, run logic dry
        # headless: no cv2 window (dashboard shows latest_frame)
        self.phantom_mode = phantom_mode
        self.headless = headless
        self.tello        = Tello()
        self._sdk_lock    = threading.Lock()  # serialize SDK commands over UDP
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

        # compute device. _pending_device is applied in _video_loop (don't swap mid-predict)
        self.device = DEVICE
        self._pending_device: Optional[str] = None

        self.person_detector = PersonDetector(device=self.device)

        # ReID embedder (OSNet x0_25 + MSMT17 weights from models/). Init fail -> lock disabled, drone still flies.
        try:
            self.reid: Optional[ReidEmbedder] = ReidEmbedder(device=self.device)
        except Exception as exc:
            print(f"[WARN] ReidEmbedder init failed; lock feature disabled: {exc}")
            self.reid = None

        self.lock = LockState()

        # Lock intents, consumed once per video tick.
        #   _at: (x_norm, y_norm) click — lock bbox containing point
        #   _cycle: 'I' key — lock biggest if unlocked, else advance L→R
        #   _clear: 'C' key — drop lock
        self._lock_pending_at: Optional[tuple[float, float]] = None
        self._lock_pending_cycle: bool = False
        self._lock_pending_clear: bool = False

        # Lock diagnostics for telemetry/HUD
        self._last_lock_distance: float = -1.0
        self._last_lock_idx: Optional[int] = None
        self._last_persons_count: int = 0

        # Actuation
        self.rc_loop_active = False
        self.altitude_pid = PIDController(*GAINS_ALTITUDE_PID, output_min=-UD_MAX_VELOCITY_CM_S, output_max=UD_MAX_VELOCITY_CM_S)
        self.altitude_target_pid = PIDController(*GAINS_ALTITUDE_TARGET_PID, output_min=-UD_MAX_VELOCITY_CM_S, output_max=UD_MAX_VELOCITY_CM_S)
        self.yaw_pid =  PIDController(*GAINS_YAW_PID, output_min=-YAW_MAX_VELOCITY_CM_S, output_max=YAW_MAX_VELOCITY_CM_S)
        self.pitch_pid =  PIDController(*GAINS_FORWARD_BACK_TOF_PID, output_min=-FB_MAX_VELOCITY_CM_S, output_max=FB_MAX_VELOCITY_CM_S)
        self.pitch_bbox_pid = PIDController(*GAINS_FORWARD_BACK_BBOX_PID, output_min=-FB_MAX_VELOCITY_CM_S, output_max=FB_MAX_VELOCITY_CM_S)
        self.roll_pid = PIDController(*GAINS_LEFT_RIGHT_PID, output_min=-LR_MAX_VELOCITY_CM_S, output_max=LR_MAX_VELOCITY_CM_S)
        self.rc = [0, 0, 0, 0]  # lr, fb, ud, yaw

        # EMA-smoothed target position — kills YOLO ±5px jitter before differentiating.
        self._target_smoothed_x_px: float = 0.0
        self._target_smoothed_y_px: float = 0.0
        self._target_ema_initialized: bool = False

        # EMA-smoothed closeness for bbox pitch PID. Raw closeness jitters -> Kd spikes -> twitchy fb.
        self._closeness_smoothed: float = 0.0
        self._closeness_ema_initialized: bool = False

        # EMA-smoothed bbox — roll dead-zone needs stable edges, raw bbox flickers.
        self._bbox_center_x_smoothed: float = 0.0
        self._bbox_width_smoothed: float = 0.0
        self._bbox_ema_initialized: bool = False

        # Last pitch source ("tof"|"bbox"|"none"). On switch, seed error_last to dodge D-spike (different units).
        self._pitch_source_last: str = "none"

        # ToF→bbox hysteresis — stay on cached ToF until N invalid frames in a row.
        self._tof_invalid_count: int = 0
        self._last_valid_front_tof_cm: float = -1.0

        # Detection hysteresis — keeps tracking_target True through short YOLO dropouts.
        self._tracking_miss_count: int = 0

        # Manual keyboard control
        self.is_manual: bool = False
        self.manual_lr: int = 0
        self.manual_fb: int = 0
        self.manual_ud: int = 0
        self.manual_yaw: int = 0
        self._space_was_pressed: bool = False
        self._m_was_pressed: bool = False
        self._esc_was_pressed: bool = False
        self._i_was_pressed: bool = False
        self._c_was_pressed: bool = False

        self._space_rising_edge: bool = False
        self._m_rising_edge: bool = False
        self._esc_rising_edge: bool = False
        self._i_rising_edge: bool = False
        self._c_rising_edge: bool = False

        self._is_airborne: bool = False
        self._is_toggling_flight: bool = False

        # Search FSM: "none" | "spin" | "advance". Spin until ToF clears, advance until wall, repeat.
        # Yaw only during spin, fb only during advance. Altitude PID always on.
        self._search_state: str = "none"
        self._search_sub_state_t0: float = 0.0

        # Last raw detection time — gates HOVER grace before SEARCH.
        self._last_target_seen_time: float = 0.0

        # Last side target was on (-1 left, +1 right, 0 centered). Edge-loss recovery + initial spin dir.
        self._last_target_side: int = 0

        # Diagonal-wall guard — on valid→-1 ToF transition, freeze fb to dodge edge-flyby crash.
        self._front_tof_cm_prev: float = -1.0
        self._front_tof_freeze_until: float = 0.0

        # Spin angle accumulator — exits at 360° (full camera sweep).
        self._search_spin_angle_deg: float = 0.0

        # Advance sub-FSM: "moving" | "sweeping". Distance cap + periodic sweep.
        self._advance_phase: str = "moving"
        self._advance_phase_t0: float = 0.0
        self._advance_last_sweep_t: float = 0.0
        self._advance_distance_cm: float = 0.0
        self._advance_sweep_obstacle: bool = False

        # UI control state — _ui_*_pending flags OR'd into keyboard rising edges in _poll_keyboard.
        self.latest_frame = None
        self.current_mode_label: str = "init"
        self._ui_takeoff_land_pending: bool = False
        self._ui_toggle_manual_pending: bool = False
        self._ui_lock_pending: bool = False
        self._ui_clear_lock_pending: bool = False

        # Manual heartbeat — values zero if no fresh ping within WEB_MANUAL_HEARTBEAT_GRACE_S (dead-UI safety).
        self._ui_manual_until: float = 0.0

    # Dashboard controls — set flag, _video_loop consumes next tick.
    def request_takeoff_land(self) -> None:
        """Toggle takeoff/land — same effect as pressing SPACE."""
        self._ui_takeoff_land_pending = True

    def toggle_manual(self) -> None:
        """Toggle manual mode — same effect as pressing M."""
        self._ui_toggle_manual_pending = True

    def track_or_cycle(self) -> None:
        """Track / cycle target — same effect as pressing I.
        First call locks the biggest person; each later call advances to the next."""
        self._ui_lock_pending = True

    def clear_lock(self) -> None:
        """Clear identity lock — same effect as pressing C."""
        self._ui_clear_lock_pending = True

    def lock_at(self, x_norm: float, y_norm: float) -> None:
        """Lock the person under a tap on the video. x, y normalized 0..1."""
        x = max(0.0, min(1.0, float(x_norm)))
        y = max(0.0, min(1.0, float(y_norm)))
        self._lock_pending_at = (x, y)

    def set_device(self, device: str) -> bool:
        """Ask for a cuda/cpu swap (done next video tick). False if invalid or no cuda."""
        device = device.lower()
        if device not in ("cuda", "cpu"):
            return False
        if device == "cuda" and not torch.cuda.is_available():
            return False
        if device != self.device:
            self._pending_device = device
        return True

    def request_stop(self) -> None:
        """Trigger clean shutdown: same effect as pressing ESC."""
        self.video_active = False

    def set_target(self, name: str) -> bool:
        """Swap active target. Returns False on unknown name. Resets PIDs+EMA (new setpoints would spike)."""
        ok = set_active_target(name)
        if not ok:
            return False
        # Wipe PID + EMA so switch doesn't pop.
        self.altitude_target_pid.reset_integral()
        self.yaw_pid.reset_integral()
        self.pitch_pid.reset_integral()
        self.pitch_bbox_pid.reset_integral()
        self.roll_pid.reset_integral()
        self._target_ema_initialized = False
        self._closeness_ema_initialized = False
        self._bbox_ema_initialized = False
        return True

    def set_manual_velocity(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        """Set manual RC velocity from the UI joystick. Needs is_manual=True. Each call refreshes heartbeat."""
        self.manual_lr  = int(lr)
        self.manual_fb  = int(fb)
        self.manual_ud  = int(ud)
        self.manual_yaw = int(yaw)
        self._ui_manual_until = time.time() + WEB_MANUAL_HEARTBEAT_GRACE_S

    def get_telemetry(self) -> dict:
        """Snapshot of all telemetry + RC state. Called by WS push loop."""
        return {
            "mode":            self.current_mode_label,
            "phantom_mode":    self.phantom_mode,
            "device":          self.device,
            "cuda_available":  torch.cuda.is_available(),
            "is_airborne":     self._is_airborne,
            "is_manual":       self.is_manual,
            "battery_percent": self.battery_percent,
            "front_tof_cm":    self.front_tof_cm,
            "down_tof_cm":     self.down_tof_cm,
            "height_cm":       self.height_cm,
            "pitch_deg":       self.pitch_deg,
            "roll_deg":        self.roll_deg,
            "yaw_deg":         self.yaw_deg,
            "vx_dm_s":         self.x_speed_dm_s,
            "vy_dm_s":         self.y_speed_dm_s,
            "vz_dm_s":         self.z_speed_dm_s,
            "ax_cm_s2":        self.x_accel_cm_s2,
            "ay_cm_s2":        self.y_accel_cm_s2,
            "az_cm_s2":        self.z_accel_cm_s2,
            "temp_min_c":      self.min_temp_C,
            "temp_max_c":      self.max_temp_C,
            "rc_lr":           self.rc[0],
            "rc_fb":           self.rc[1],
            "rc_ud":           self.rc[2],
            "rc_yaw":          self.rc[3],
            # 0=locked, None=not. Frontend uses non-null to flip video glow. No per-track ID (lock is by embedding).
            "locked_track_id": 0 if self.lock.is_locked else None,
            "lock_distance":   self._last_lock_distance,
            "lock_threshold":  LOCK_MATCH_THRESHOLD,
            "lock_target_idx": self._last_lock_idx,
            "num_detections":  self._last_persons_count,
            "reid_available":  self.reid is not None,
            "search_state":    self._search_state,
            "target_name":     getattr(TARGET, "name", "—"),
            "available_targets": list(AVAILABLE_TARGETS.keys()),
        }

    def start(self):
        """Connect drone, start streams + threads, run video loop."""
        print("[INFO] Connecting to Tello...")
        last_exc = None
        for attempt in range(1, 4):
            try:
                self.tello.connect()
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                print(f"[WARN] connect attempt {attempt}/3 failed: {exc}")
                time.sleep(2.0)
        if last_exc is not None:
            raise RuntimeError(
                "Tello connect failed after 3 attempts. "
                "Connect your PC's WiFi to the Tello's network (its own AP at 192.168.10.1)."
            ) from last_exc
        print(f"[INFO] Connected. Battery: {self.tello.get_battery()}%")

        # Reset stale state from prior run.
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
        time.sleep(2.5) 

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

        self.video_active = True
        self.front_tof_active   = True
        self.telemetry_active   = True
        self.rc_loop_active     = True


        # Video loop in main thread. RC, telemetry, ToF in daemons.
        threading.Thread(target=self._update_telemetry, daemon=True).start()
        threading.Thread(target=self._update_front_tof, daemon=True).start()
        threading.Thread(target=self._rc_control_loop, daemon=True).start()

        self._video_loop()


    def stop(self):
        """Stop all threads, land drone."""
        if self._stop_finished:
            return

        # Kill ToF thread first — its in-flight "EXT tof?" reply would be popped by land/streamoff otherwise.
        self.front_tof_active = False
        self.rc_loop_active = False
        self.video_active = False
        self.telemetry_active = False

        time.sleep(1.5)  # > EXT tof? timeout (1s) so ToF thread fully exits

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

        # Avoid djitellopy.end() — closes SDK socket and breaks subsequent runs.
        time.sleep(0.6)
        try:
            with self._sdk_lock:
                self.tello.streamoff()
        except TelloException:
            pass
        finally:
            self._stop_finished = True


    def _video_loop(self):
        """Main loop — video, detection, intercept logic."""
        import numpy as np
        if not self.headless:
            cv2.namedWindow("TelloInterceptor", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("TelloInterceptor", np.zeros((720, 960, 3), dtype=np.uint8))
            cv2.waitKey(1)

        drone_video = self.tello.get_frame_read()

        frame_count    = 0
        consecutive_none_frames = 0
        detected_persons: list[Person] = []
        frame_center_x = FRAME_CENTER_X
        frame_center_y = FRAME_CENTER_Y

        while self.video_active:
            # do a pending device swap here, not mid-predict
            if self._pending_device is not None and self._pending_device != self.device:
                new_device = self._pending_device
                try:
                    self.person_detector.set_device(new_device)
                    if self.reid is not None:
                        self.reid.set_device(new_device)
                        if self.lock.embedding is not None:
                            self.lock.embedding = self.lock.embedding.to(new_device)
                    self.device = new_device
                    print(f"[INFO] Compute device -> {new_device}")
                except Exception as exc:
                    print(f"[WARN] device swap to {new_device} failed: {exc}")
                self._pending_device = None

            # leer cada frame del stream de video
            current_frame = drone_video.frame
            # djitellopy returns RGB (PyAV default); cv2 and YOLO expect BGR.
            if current_frame is not None:
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

            # Handle missing frames
            if current_frame is None:
                consecutive_none_frames += 1
                if consecutive_none_frames > 300:
                    print("[ERROR] Video stream dead ")
                    self.video_active = False
                    break
                if not self.headless:
                    cv2.waitKey(1)

                # Keep keyboard responsive even on dead frames
                self._poll_keyboard()
                if self._esc_rising_edge:
                    self.video_active = False
                    break
                
                time.sleep(0.01)
                continue

            consecutive_none_frames = 0


            # Frame center + vertical setpoint (per-target framing).
            h, w = current_frame.shape[:2]
            frame_center_x = w // 2
            frame_center_y = h // 2

            target_setpoint_y_px = int(h * TARGET.target_y_ratio)

            frame_count += 1


            # Pose detection every YOLO_STRIDE frames.
            if frame_count % YOLO_FRAME_STRIDE == 0:
                detected_persons = self.person_detector.detect_persons_in_frame(current_frame)


            # Multi-target lock: embed detections -> apply pending intents -> select target -> EMA smooth.
            embeddings = None
            if self.reid is not None and detected_persons:
                crops = [
                    crop_bbox(current_frame, p.x1, p.y1, p.x2, p.y2)
                    for p in detected_persons
                ]

                # Skip embed if any crop invalid — ReID errors otherwise.
                if all(c is not None for c in crops):
                    try:
                        embeddings = self.reid.embed_batch(crops)
                    except Exception as exc:
                        print(f"[WARN] ReID embed_batch failed; skipping lock match: {exc}")
                        embeddings = None

            # --- Apply pending lock intents ---
            if self._lock_pending_clear:
                self.lock.clear()
                self._lock_pending_clear = False

            if self._lock_pending_at is not None and embeddings is not None and detected_persons:
                x_norm, y_norm = self._lock_pending_at
                click_px = int(x_norm * w)
                click_py = int(y_norm * h)
                hit_idx: Optional[int] = None
                for idx, p in enumerate(detected_persons):
                    if p.x1 <= click_px <= p.x2 and p.y1 <= click_py <= p.y2:
                        hit_idx = idx
                        break
                if hit_idx is not None:
                    center = (float(detected_persons[hit_idx].bbox_center_x),
                              float(detected_persons[hit_idx].bbox_center_y))
                    self.lock.lock_from(embeddings[hit_idx], center)
                    print(f"[INFO] Lock at ({click_px},{click_py}) -> person #{hit_idx}")
                else:
                    print(f"[INFO] Lock click at ({click_px},{click_py}) hit no bbox")
                self._lock_pending_at = None

            if self._lock_pending_cycle and embeddings is not None and detected_persons:
                # L→R order keeps cycling predictable across frames.
                order = sorted(
                    range(len(detected_persons)),
                    key=lambda i: detected_persons[i].bbox_center_x,
                )
                if not self.lock.is_locked:
                    # Unlocked → lock biggest bbox.
                    target_idx = max(
                        range(len(detected_persons)),
                        key=lambda i: detected_persons[i].bbox_area_pixels,
                    )
                else:
                    # Locked → advance to next person L→R.
                    locked_vec = self.lock.embedding.flatten()
                    sims = embeddings @ locked_vec
                    import torch as _torch
                    current_idx = int(_torch.argmax(sims).item())
                    pos_in_order = order.index(current_idx) if current_idx in order else 0
                    target_idx = order[(pos_in_order + 1) % len(order)]
                center = (float(detected_persons[target_idx].bbox_center_x),
                          float(detected_persons[target_idx].bbox_center_y))
                self.lock.lock_from(embeddings[target_idx], center)
                print(f"[INFO] Lock cycle -> person #{target_idx}")
                self._lock_pending_cycle = False

            # --- Selection ---
            last_point_px = (
                (self._target_smoothed_x_px, self._target_smoothed_y_px)
                if self._target_ema_initialized else None
            )
            best_person, best_idx, lock_dist = select_target_person(
                detected_persons, TARGET, self.lock, embeddings, last_point_px,
            )
            self._last_persons_count = len(detected_persons)
            self._last_lock_distance = lock_dist
            self._last_lock_idx = best_idx

            # EMA-update locked embedding only on confirmed match — prevents drift onto false positive.
            if (self.lock.is_locked and best_person is not None
                    and best_idx is not None and embeddings is not None):
                self.lock.update(
                    embeddings[best_idx],
                    (float(best_person.bbox_center_x), float(best_person.bbox_center_y)),
                )

            persons_to_draw = detected_persons  # draw all, mark chosen

            # UI frame copy BEFORE HUD lands on current_frame.
            ui_frame = current_frame.copy()
            draw_person_overlays(ui_frame, persons_to_draw, TARGET,
                                 chosen_idx=best_idx, locked=self.lock.is_locked)

            # HUD overlays
            target_x_px, target_y_px, tracking_target, tracked_person = draw_person_overlays(
                current_frame, persons_to_draw, TARGET,
                chosen_idx=best_idx, locked=self.lock.is_locked,
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

            draw_wall_warning(current_frame, self.front_tof_cm)

            tracking_target_raw = tracking_target and target_y_px is not None and target_x_px is not None

            # Detection hysteresis — short YOLO dips keep tracking_target=True with last smoothed values.
            if tracking_target_raw:
                self._tracking_miss_count = 0
                tracking_target = True
            else:
                self._tracking_miss_count += 1
                tracking_target = (
                    self._tracking_miss_count <= TRACKING_MISS_HYSTERESIS_FRAMES
                    and self._target_ema_initialized
                )


            # Intercept FSM:
            #   INTERCEPTING — target visible, target PID
            #   HOVER        — target lost <TARGET_LOST_GRACE_S ago, ud=0 (Tello holds via baro+optical-flow)
            #   SEARCHING    — lost longer, baro PID + spin/advance
            # Grace HOVER avoids PID flapping on YOLO jitter.
            valid_front_tof = self.front_tof_cm > 0

            # pitch source latched across frames for D-term seeding on transitions
            pitch_source = "none"

            # Default velocities — overwritten by active mode.
            lr = 0
            fb = 0
            ud = 0
            yaw = 0

            error_roll = 0.0

            # Grounded -> skip FSM. Phantom dry-runs FSM, so don't gate on airborne there.
            if not self._is_airborne and not self.phantom_mode:
                mode_label = "grounded"
                error_altitude = 0.0
                error_yaw = 0.0
                error_pitch = 0.0
                self._search_state = "none"
                self.altitude_pid.reset_integral()
                self.altitude_target_pid.reset_integral()
                self.yaw_pid.reset_integral()
                self.pitch_pid.reset_integral()
                self.pitch_bbox_pid.reset_integral()
                self.roll_pid.reset_integral()
                self._target_ema_initialized = False
                self._closeness_ema_initialized = False
                self._bbox_ema_initialized = False

            # Target visible -> INTERCEPTING mode
            elif tracking_target:
                # Refresh only on raw detection so grace timer measures true loss duration.
                if tracking_target_raw:
                    self._last_target_seen_time = time.time()
                mode_label = "intercepting"

                # Reset Search if target re-acquired after loss
                self._search_state = "none"

                # Raw measurements only on fresh detection — hysteresis holds last smoothed values otherwise.
                if tracking_target_raw:
                    target_side = target_x_px - frame_center_x
                    if target_side > 0:
                        self._last_target_side = 1
                    elif target_side < 0:
                        self._last_target_side = -1

                    # EMA smooth target position — kills YOLO jitter before D-term.
                    if not self._target_ema_initialized:
                        self._target_smoothed_x_px = float(target_x_px)
                        self._target_smoothed_y_px = float(target_y_px)
                        self._target_ema_initialized = True
                    else:
                        self._target_smoothed_x_px = (TARGET_EMA_ALPHA * target_x_px +
                                                      (1.0 - TARGET_EMA_ALPHA) * self._target_smoothed_x_px)
                        self._target_smoothed_y_px = (TARGET_EMA_ALPHA * target_y_px +
                                                      (1.0 - TARGET_EMA_ALPHA) * self._target_smoothed_y_px)

                # EMA-smooth closeness (bbox PID consumes it on ToF fallback).
                if tracking_target_raw and tracked_person is not None and TARGET.is_visible(tracked_person):
                    raw_closeness = TARGET.closeness(tracked_person)
                    if not self._closeness_ema_initialized:
                        self._closeness_smoothed = float(raw_closeness)
                        self._closeness_ema_initialized = True
                    else:
                        self._closeness_smoothed = (TARGET_EMA_ALPHA * raw_closeness +
                                                    (1.0 - TARGET_EMA_ALPHA) * self._closeness_smoothed)

                # EMA-smooth bbox — roll dead-zone needs stable edges.
                if tracking_target_raw and tracked_person is not None:
                    raw_center = float(tracked_person.bbox_center_x)
                    raw_width = float(tracked_person.bbox_width_pixels)
                    if not self._bbox_ema_initialized:
                        self._bbox_center_x_smoothed = raw_center
                        self._bbox_width_smoothed = raw_width
                        self._bbox_ema_initialized = True
                    else:
                        self._bbox_center_x_smoothed = (TARGET_EMA_ALPHA * raw_center +
                                                       (1.0 - TARGET_EMA_ALPHA) * self._bbox_center_x_smoothed)
                        self._bbox_width_smoothed = (TARGET_EMA_ALPHA * raw_width +
                                                    (1.0 - TARGET_EMA_ALPHA) * self._bbox_width_smoothed)

                # --- Target Altitude PID ---
                error_altitude = self._target_smoothed_y_px - target_setpoint_y_px
                ud = self.altitude_target_pid.compute(error_altitude, RC_LOOP_INTERVAL_S)


                # --- Yaw PID ---
                error_yaw = self._target_smoothed_x_px - frame_center_x   
                yaw = self.yaw_pid.compute(error_yaw, RC_LOOP_INTERVAL_S)


                # --- Pitch PID ---
                # ToF hysteresis — stay on cached ToF until N invalid frames (stops ToF↔bbox flicker).
                if valid_front_tof:
                    self._tof_invalid_count = 0
                    self._last_valid_front_tof_cm = self.front_tof_cm
                else:
                    self._tof_invalid_count += 1

                if self._last_valid_front_tof_cm > 0 and self._tof_invalid_count < FRONT_TOF_INVALID_HYSTERESIS_FRAMES:
                    pitch_source = "tof"
                    front_tof_for_pid = self.front_tof_cm if valid_front_tof else self._last_valid_front_tof_cm
                    error_pitch = INTERCEPT_DISTANCE_CM - front_tof_for_pid

                    # ToF entry: seed error_last so D-term doesn't spike off stale ratio-unit err.
                    if self._pitch_source_last != "tof":
                        self.pitch_pid.error_last = error_pitch

                    fb = self.pitch_pid.compute(error_pitch, RC_LOOP_INTERVAL_S)
                    self.pitch_bbox_pid.reset_integral()

                elif self._closeness_ema_initialized:
                    pitch_source = "bbox"
                    closeness_value = self._closeness_smoothed
                    error_pitch = TARGET.closeness_setpoint - closeness_value
                    # bbox entry: seed error_last to avoid D-spike off stale cm-unit err.
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


                # --- Roll PID ---
                # EMA-smoothed bbox so dead-zone edges don't flicker per YOLO frame.
                bbox_half = self._bbox_width_smoothed / 2
                bbox_left  = self._bbox_center_x_smoothed - bbox_half
                bbox_right = self._bbox_center_x_smoothed + bbox_half

                if bbox_left <= frame_center_x <= bbox_right:
                    lr = 0
                    error_roll = 0.0
                    self.roll_pid.reset_integral()
                else:
                    error_roll = self._target_smoothed_x_px - frame_center_x
                    lr = self.roll_pid.compute(error_roll, RC_LOOP_INTERVAL_S)


            # Lost target but seen recently -> HOVER grace period
            elif self._last_target_seen_time > 0 and (time.time() - self._last_target_seen_time) < TARGET_LOST_GRACE_S:
                mode_label = "hover"

                # altitude — let Tello hold it via baro+optical-flow
                error_altitude = 0.0
                ud = 0.0
                self.altitude_pid.reset_integral()
                self.altitude_target_pid.reset_integral()

                # yaw — edge-loss recovery. Off-side loss => yaw toward last-seen side. Centered loss => no yaw.
                error_yaw = 0.0
                yaw = self._last_target_side * GRACE_RECOVERY_YAW_DEG_S
                self.yaw_pid.reset_integral()

                # roll
                self.roll_pid.reset_integral()

                # pitch
                error_pitch = 0.0
                fb = 0.0
                self.pitch_pid.reset_integral()
                self.pitch_bbox_pid.reset_integral()

                # Reset EMA — next acquisition is a fresh target
                self._target_ema_initialized = False
                self._closeness_ema_initialized = False
                self._bbox_ema_initialized = False


            # Lost beyond grace -> SEARCH
            else:
                mode_label = "searching"
                error_altitude = TARGET_ALTITUDE_CM - self.height_cm  # cm (+ve = below target)

                # Baro-based altitude hold during search.
                self.altitude_target_pid.reset_integral()
                self._target_ema_initialized = False
                self._closeness_ema_initialized = False
                self._bbox_ema_initialized = False
                ud = self.altitude_pid.compute(error_altitude, RC_LOOP_INTERVAL_S)

                # Roll / bbox-pitch / yaw inactive — reset integrals to avoid wind-up.
                self.roll_pid.reset_integral()
                self.pitch_bbox_pid.reset_integral()
                self.yaw_pid.reset_integral()
                error_yaw = 0.0
                error_pitch = 0.0
                pitch_source = "none"

                # Search FSM
                if self._search_state == "none":
                    self._search_state = "spin"
                    self._search_sub_state_t0 = time.time()
                    self._search_spin_angle_deg = 0.0

                if self._search_state == "spin":
                    # Spin toward last-seen side. Default 0 -> +1 (CW) on fresh boot.
                    spin_sign = self._last_target_side if self._last_target_side != 0 else 1
                    yaw = spin_sign * SEARCH_SPIN_VELOCITY_DEG_S
                    fb  = 0.0

                    # Spin 360° before advancing — full camera sweep.
                    self._search_spin_angle_deg += SEARCH_SPIN_VELOCITY_DEG_S * RC_LOOP_INTERVAL_S

                    if self._search_spin_angle_deg >= 360.0:
                        yaw = 0.0
                        self._search_state = "advance"
                        self._search_sub_state_t0 = time.time()
                        # Fresh PID state — advance is a new control task.
                        self.pitch_pid.reset_integral()
                        self.pitch_pid.error_last = 0.0
                        self._advance_phase = "moving"
                        self._advance_phase_t0 = time.time()
                        self._advance_last_sweep_t = time.time()
                        self._advance_distance_cm = 0.0
                        self._advance_sweep_obstacle = False


                elif self._search_state == "advance":
                    # Advance: "moving" = fb forward. "sweeping" = pause + yaw arc to scan for diagonal walls.
                    # Distance cap forces periodic re-spin.
                    now = time.time()
                    yaw = 0.0

                    if self._advance_phase == "moving":
                        if valid_front_tof:
                            error_pitch = INTERCEPT_DISTANCE_CM - self.front_tof_cm
                            pitch_source = "tof"
                            fb = max(0.0, self.pitch_pid.compute(error_pitch, RC_LOOP_INTERVAL_S))
                        else:
                            # ToF out-of-range = no obstacle = open space ahead.
                            error_pitch = 0.0
                            pitch_source = "none"
                            fb = SEARCH_OPEN_SPACE_VELOCITY_CM_S
                            self.pitch_pid.reset_integral()

                        # Approx distance integration — RC tick interval as dt (conservative).
                        self._advance_distance_cm += fb * RC_LOOP_INTERVAL_S

                        settled = valid_front_tof and abs(INTERCEPT_DISTANCE_CM - self.front_tof_cm) < SEARCH_ADVANCE_TOLERANCE_CM
                        timed_out = (now - self._search_sub_state_t0) > SEARCH_ADVANCE_TIMEOUT_S
                        distance_capped = self._advance_distance_cm >= SEARCH_ADVANCE_MAX_DISTANCE_CM

                        if settled or timed_out or distance_capped:
                            self._search_state = "spin"
                            self._search_sub_state_t0 = now
                            self._search_spin_angle_deg = 0.0
                        elif now - self._advance_last_sweep_t >= SEARCH_ADVANCE_SWEEP_EVERY_S:
                            self._advance_phase = "sweeping"
                            self._advance_phase_t0 = now
                            self._advance_sweep_obstacle = False

                    else:  # "sweeping"
                        fb = 0.0
                        elapsed = now - self._advance_phase_t0
                        leg_s = SEARCH_ADVANCE_SWEEP_ARC_DEG / SEARCH_ADVANCE_SWEEP_YAW_DEG_S
                        # leg1 [0,leg_s):    +arc to the right
                        # leg2 [leg_s,3*leg_s): swing back through 0 to -arc
                        # leg3 [3*leg_s,4*leg_s): return to 0
                        if elapsed < leg_s:
                            yaw = SEARCH_ADVANCE_SWEEP_YAW_DEG_S
                        elif elapsed < 3 * leg_s:
                            yaw = -SEARCH_ADVANCE_SWEEP_YAW_DEG_S
                        elif elapsed < 4 * leg_s:
                            yaw = SEARCH_ADVANCE_SWEEP_YAW_DEG_S
                        else:
                            yaw = 0.0
                            if self._advance_sweep_obstacle:
                                # Diagonal wall seen during sweep -> abort advance, re-spin.
                                self._search_state = "spin"
                                self._search_sub_state_t0 = now
                                self._search_spin_angle_deg = 0.0
                            else:
                                self._advance_phase = "moving"
                                self._advance_last_sweep_t = now

                        # Watch ToF during sweep — close reading = off-axis wall.
                        if valid_front_tof and self.front_tof_cm <= SEARCH_ADVANCE_SWEEP_WALL_CM:
                            self._advance_sweep_obstacle = True

            # Manual override — replaces PID output. Safety blocks below still apply.
            if self.is_manual:
                mode_label = "manual"
                lr = self.manual_lr
                fb = self.manual_fb
                ud = self.manual_ud
                yaw = self.manual_yaw

            # Altitude safety — XY dead-reckoning dropped (unreliable on Tello).
            if self.height_cm >= MAX_TRACKING_ALTITUDE_CM and ud > 0:
                ud = 0  # block climb above ceiling
            elif 0 < self.height_cm <= MIN_TRACKING_ALTITUDE_CM and ud < 0:
                ud = 0  # block descent below floor

            # Fb safety — wall stop forward, no reverse (no rear ToF). Negative fb allowed.
            if valid_front_tof:
                if self.front_tof_cm <= FRONT_TOF_WALL_STOP_CM and fb > 0:
                    fb = 0
                    self.pitch_pid.reset_integral()
                    self.pitch_bbox_pid.reset_integral()

            # Diagonal-wall guard — valid→-1 ToF transition = drone past wall edge. Block fb until freeze ends.
            if time.time() < self._front_tof_freeze_until and fb > 0:
                fb = 0
                self.pitch_pid.reset_integral()
                self.pitch_bbox_pid.reset_integral()

            self.rc[0] = int(round(lr))
            self.rc[1] = int(round(fb))
            self.rc[2] = int(round(ud))
            self.rc[3] = int(round(yaw))

            # debug
            #print(self.rc[1], self.front_tof_cm)

            # Latch pitch source for next-frame transition detection (D-term seeding).
            self._pitch_source_last = pitch_source

            # UI frame = minimal overlays. cv2 shows fully-annotated. Atomic assign, no lock.
            self.latest_frame = ui_frame
            self.current_mode_label = mode_label

            if not self.headless:
                cv2.imshow("TelloInterceptor", current_frame)
                # Pump cv2 render. _poll_keyboard drains key msgs so cv2 doesn't freeze on held key.
                cv2.waitKey(1)
            else:
                time.sleep(0.01)  # pace the loop; no cv2 window to pump
            self._poll_keyboard()

            # ESC -> exit, always.
            if self._esc_rising_edge:
                self.video_active = False
                break
            # Window-X also exits — useful when terminal has focus and ESC won't fire.
            if not self.headless and cv2.getWindowProperty("TelloInterceptor", cv2.WND_PROP_VISIBLE) < 1:
                self.video_active = False
                break

            # SPACE — takeoff/land toggle. Threaded + _sdk_lock to avoid SDK clash.
            # Clears djitellopy response queue first — stale "keepalive"/"tof N" gets popped
            # instead of the real "ok", causing 4-retry fail.
            if self._space_rising_edge and not self.phantom_mode:
                if not self._is_toggling_flight:
                    self._is_toggling_flight = True
                    def _toggle_flight():
                        try:
                            # Re-enter SDK mode in case drone rebooted or timed out. Outside main lock — don't block on timeout.
                            try:
                                self.tello.send_control_command("command", timeout=1)
                            except Exception:
                                pass  # timeout fine, connect() already did this

                            with self._sdk_lock:
                                # Clear stale unrelated responses from queue.
                                if 'responses' in self.tello.get_own_udp_object():
                                    self.tello.get_own_udp_object()['responses'].clear()

                                if self._is_airborne:
                                    # Optimistic: land() executes at hardware level even if SDK raises. Flip flag first.
                                    self._is_airborne = False
                                    self.tello.land()
                                else:
                                    # Optimistic: drone lifts before djitellopy confirms over UDP. Stale queue can raise.
                                    # Flip flag first so RC loop + FSM aren't dead-locked grounded.
                                    self._is_airborne = True
                                    # Zero sticks + reset PIDs — clean start regardless of post-takeoff mode.
                                    self.manual_lr = self.manual_fb = self.manual_ud = self.manual_yaw = 0
                                    self.altitude_pid.reset_integral()
                                    self.altitude_target_pid.reset_integral()
                                    self.yaw_pid.reset_integral()
                                    self.pitch_pid.reset_integral()
                                    self.pitch_bbox_pid.reset_integral()
                                    self.roll_pid.reset_integral()
                                    self.tello.takeoff()
                        except TelloException as e:
                            # Flag already flipped. Hardware likely executed; SDK raised on queue race.
                            # Don't revert — reflects real drone state better.
                            print(f"[WARN] takeoff/land SDK raised (flag kept): {e}")
                        finally:
                            self._is_toggling_flight = False
                    threading.Thread(target=_toggle_flight, daemon=True).start()

            # M — toggle manual + reset PIDs to avoid I-term pop on switch.
            if self._m_rising_edge:
                self.is_manual = not self.is_manual
                self.manual_lr = self.manual_fb = self.manual_ud = self.manual_yaw = 0
                self.altitude_pid.reset_integral()
                self.altitude_target_pid.reset_integral()
                self.yaw_pid.reset_integral()
                self.pitch_pid.reset_integral()
                self.pitch_bbox_pid.reset_integral()
                self.roll_pid.reset_integral()

            # I — lock-cycle (biggest if unlocked, else next person L→R). Applied next tick with fresh embeddings.
            # C — clear lock (back to biggest-bbox default).
            if self._i_rising_edge:
                if self.reid is None:
                    print("[WARN] ReID embedder unavailable; lock disabled.")
                else:
                    self._lock_pending_cycle = True
            if self._c_rising_edge:
                self._lock_pending_clear = True

        cv2.destroyAllWindows()

    def _poll_keyboard(self) -> None:
        _drain_keyboard_messages()

        space_now = _is_key_pressed(_VK_SPACE)
        m_now     = _is_key_pressed(_VK_M)
        esc_now   = _is_key_pressed(_VK_ESC)
        i_now     = _is_key_pressed(_VK_I)
        c_now     = _is_key_pressed(_VK_C)

        self._space_rising_edge = space_now and not self._space_was_pressed
        self._m_rising_edge     = m_now     and not self._m_was_pressed
        self._esc_rising_edge   = esc_now   and not self._esc_was_pressed
        self._i_rising_edge     = i_now     and not self._i_was_pressed
        self._c_rising_edge     = c_now     and not self._c_was_pressed

        self._space_was_pressed = space_now
        self._m_was_pressed     = m_now
        self._esc_was_pressed   = esc_now
        self._i_was_pressed     = i_now
        self._c_was_pressed     = c_now

        # UI overrides — OR pending flags into rising edges, clear after so a held flag doesn't refire.
        if self._ui_takeoff_land_pending:
            self._space_rising_edge = True
            self._ui_takeoff_land_pending = False
        if self._ui_toggle_manual_pending:
            self._m_rising_edge = True
            self._ui_toggle_manual_pending = False
        if self._ui_lock_pending:
            self._i_rising_edge = True
            self._ui_lock_pending = False
        if self._ui_clear_lock_pending:
            self._c_rising_edge = True
            self._ui_clear_lock_pending = False

        if not self.is_manual:
            self.manual_lr = self.manual_fb = self.manual_ud = self.manual_yaw = 0
            return

        # Priority: keyboard > UI joystick > idle. PC operator wins on key press.
        kb_any = any(_is_key_pressed(k) for k in (
            _VK_W, _VK_S, _VK_A, _VK_D, _VK_Q, _VK_E, _VK_UP, _VK_DOWN,
        ))
        if kb_any:
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
        elif time.time() < self._ui_manual_until:
            # Heartbeat fresh — leave manual_* as set_manual_velocity set them.
            pass
        else:
            # Both idle -> stop.
            self.manual_lr = self.manual_fb = self.manual_ud = self.manual_yaw = 0

    def _update_front_tof(self):
    
        while self.front_tof_active:
            try:
                # Front ToF — short critical section, release lock immediately.
                with self._sdk_lock:
                    raw = self.tello.send_command_with_return("EXT tof?", timeout=1)
                if raw and raw.strip().startswith("tof "):
                    mm = int(raw.strip().split()[1])
                    new_front_tof_cm = -1.0 if mm >= 8190 else mm / 10.0
                    # Valid→-1 transition = likely past wall edge. Freeze fb to dodge diagonal crash.
                    if self._front_tof_cm_prev > 0 and new_front_tof_cm == -1.0:
                        self._front_tof_freeze_until = time.time() + FRONT_TOF_DISCONTINUITY_FREEZE_S
                    self._front_tof_cm_prev = new_front_tof_cm
                    self.front_tof_cm = new_front_tof_cm

                # Down ToF + baro — state listener, no SDK command, no lock.
                state = self.tello.get_current_state()
                if state:
                    self.down_tof_cm = state.get("tof", -1)
                    # Tello reports OOR (often 6553) when down ToF out-of-range. >=400 cm = invalid.
                    # Unfiltered spike would pollute height_cm -> altitude PID saturates -> drone slams down.
                    if self.down_tof_cm >= 400:
                        self.down_tof_cm = -1
                    baro_cm = state.get("h", -1)

                    if self.down_tof_cm > 0 and baro_cm > 0:
                        self.height_cm = max(self.down_tof_cm, baro_cm)
                    elif self.down_tof_cm > 0:
                        self.height_cm = self.down_tof_cm
                    else:
                        self.height_cm = baro_cm

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

            except Exception:
                pass
            time.sleep(0.2)

    def _rc_control_loop(self) -> None:
        """Send RC at 20 Hz, decoupled from video FPS. Phantom mode = no-op (rc[] still updates for HUD)."""
        while self.rc_loop_active:
            if not self.phantom_mode and self._is_airborne:
                with self._sdk_lock:
                    self.tello.send_rc_control(*self.rc)
            time.sleep(RC_LOOP_INTERVAL_S)  # 0.05 s -> 20 Hz