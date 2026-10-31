"""Full-flow simulation of the demo interceptor with a fake Tello. No drone needed (~2 min).

Real YOLO + ReID on real person photos (ultralytics bus.jpg + zidane.jpg) pasted into a 360°
panorama. The fake drone turns with the rc yaw command; the camera view slides over the panorama.

Phases checked:
  A. takeoff -> 360 scan -> gallery -> random target -> chase
  B. change_target -> random other person -> spin until seen -> chase
  D. video + state freeze (WiFi drop) -> mode link_lost, rc 0 0 0 0 sent -> link back -> recovers
  C. everyone leaves -> lost -> one full spin per person -> give up each -> rescan -> empty -> scan again

Run from demo/:
    ..\\.venv\\Scripts\\python.exe sim_flow.py
"""
import os
import sys
import threading
import time

import cv2
import numpy as np
import ultralytics

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(ultralytics.__file__), "assets")   # bus.jpg + zidane.jpg ship with ultralytics
os.chdir(DEMO_DIR)
sys.path.insert(0, DEMO_DIR)

FOV_DEG = 82.6
PANO_W = int(round(960 * 360 / FOV_DEG))
PANO_H = 720


def build_panorama():
    pano = np.full((PANO_H, PANO_W, 3), 120, dtype=np.uint8)
    zidane = cv2.imread(os.path.join(ASSETS, "zidane.jpg"))
    bus = cv2.imread(os.path.join(ASSETS, "bus.jpg"))
    zidane = cv2.resize(zidane, (int(zidane.shape[1] * PANO_H / zidane.shape[0]), PANO_H))
    bus = cv2.resize(bus, (int(bus.shape[1] * PANO_H / bus.shape[0]), PANO_H))
    pano[:, 600:600 + zidane.shape[1]] = zidane
    pano[:, 2600:2600 + bus.shape[1]] = bus
    return pano


class FakeFrameRead:
    def __init__(self, drone):
        self.drone = drone
        self._last = None

    @property
    def frame(self):
        if self.drone.frozen and self._last is not None:
            return self._last          # same object = decoder stalled
        cx = int(self.drone.heading_deg / 360.0 * PANO_W)
        cols = np.arange(cx - 480, cx + 480) % PANO_W
        bgr = np.take(self.drone.pano, cols, axis=1)
        self._last = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._last


class FakeTello:
    def __init__(self):
        self.pano = build_panorama()
        self.empty_pano = np.full_like(self.pano, 120)
        self.heading_deg = 0.0
        self.yaw_rate = 0.0
        self.last_rc = (0, 0, 0, 0)
        self.frozen = False
        self.led_log = []
        self._state = None
        self._udp = {"responses": [], "state": {}}
        threading.Thread(target=self._physics, daemon=True).start()

    def _physics(self):
        last = time.time()
        while True:
            now = time.time()
            self.heading_deg = (self.heading_deg + self.yaw_rate * (now - last)) % 360.0
            last = now
            time.sleep(0.02)

    def connect(self): pass
    def get_battery(self): return 90
    def land(self): self.yaw_rate = 0.0
    def takeoff(self): time.sleep(1.0)
    def streamon(self): pass
    def streamoff(self): pass
    def get_own_udp_object(self): return self._udp
    def send_control_command(self, command, timeout=7): return True
    def get_frame_read(self, *a, **k): return FakeFrameRead(self)

    def send_rc_control(self, lr, fb, ud, yaw):
        self.last_rc = (lr, fb, ud, yaw)
        self.yaw_rate = float(yaw)

    def send_command_without_return(self, command):
        if command.startswith("EXT mled"):
            self.led_log.append(command)

    def send_command_with_return(self, command, timeout=7):
        time.sleep(0.05)
        return "tof 8190"

    def get_current_state(self):
        if self.frozen and self._state is not None:
            return self._state         # same dict = no new state packet
        yaw = (self.heading_deg + 180.0) % 360.0 - 180.0
        self._state = {"yaw": int(round(yaw)), "h": 150, "tof": 140, "bat": 90, "pitch": 0, "roll": 0,
                       "vgx": 0, "vgy": 0, "vgz": 0, "agx": 0, "agy": 0, "agz": -1000, "templ": 40, "temph": 45}
        return self._state


def main():
    import interceptor.tello_interceptor as ti
    from interceptor.tello_interceptor import TelloInterceptor
    from interceptor.led_matrix import led_command_for_mode

    # Global GetAsyncKeyState would read the real keyboard (typing M/C elsewhere toggles modes).
    ti._is_key_pressed = lambda vk: False

    drone = TelloInterceptor(phantom_mode=False, headless=True)
    fake = FakeTello()
    drone.tello = fake
    drone.led_matrix._tello = fake

    scans_started = [0]
    original_start_scan = drone._start_scan

    def counting_start_scan():
        scans_started[0] += 1
        original_start_scan()
    drone._start_scan = counting_start_scan

    threading.Thread(target=drone.start, daemon=True).start()
    t0 = time.time()

    def snap(tag=""):
        t = drone.get_telemetry()
        print(f"{time.time() - t0:6.1f}s {tag:<12} mode={t['mode']:<12} phase={t['mission_phase']:<5} "
              f"scan={t['scan_turned_deg']:5.0f} spin={t['target_spin_turned_deg']:5.0f} "
              f"gallery={t['gallery_count']} target={t['selected_person_id']} "
              f"tried={sorted(drone._tried_person_ids)} heading={fake.heading_deg:5.0f} "
              f"rc={fake.last_rc} d={t['lock_distance']:.2f}", flush=True)
        return t

    def wait_until(pred, timeout_s, tag):
        end = time.time() + timeout_s
        last_print = 0.0
        while time.time() < end:
            t = drone.get_telemetry()
            if time.time() - last_print > 2.0:
                snap(tag)
                last_print = time.time()
            if pred(t):
                snap(tag + " OK")
                return True
            time.sleep(0.1)
        snap(tag + " TIMEOUT")
        return False

    time.sleep(14.0)  # connect + 2.5 s soft reset + model warm-up (a real run warms up before takeoff too)
    drone.request_takeoff_land()

    ok_a = wait_until(lambda t: t["mission_phase"] == "track" and t["mode"] == "intercepting", 60, "A-scan")
    first = drone._selected_person_id
    time.sleep(3.0)

    drone.change_target()
    ok_b1 = wait_until(lambda t: t["selected_person_id"] not in (None, first), 5, "B-change")
    ok_b2 = wait_until(lambda t: t["mode"] == "intercepting" and t["selected_person_id"] != first, 60, "B-find")
    time.sleep(2.0)

    fake.frozen = True             # WiFi drop: video + state stop updating
    ok_d1 = wait_until(lambda t: t["mode"] == "link_lost" and fake.last_rc == (0, 0, 0, 0), 5, "D-freeze")
    time.sleep(1.0)
    ok_d1 = ok_d1 and fake.last_rc == (0, 0, 0, 0)
    fake.frozen = False
    ok_d2 = wait_until(lambda t: t["mode"] != "link_lost", 5, "D-back")

    fake.pano = fake.empty_pano    # everybody leaves the room
    scans_before = scans_started[0]
    ok_c = wait_until(lambda t: scans_started[0] >= scans_before + 2 and t["gallery_count"] == 0
                      and t["mission_phase"] == "scan", 150, "C-empty")

    led_modes = []
    for cmd in fake.led_log:
        for m in ("scanning", "waiting", "intercepting", "searching", "manual", "grounded"):
            if cmd == led_command_for_mode(m):
                if not led_modes or led_modes[-1] != m:
                    led_modes.append(m)
                break
    print("LED sequence:", " -> ".join(led_modes))
    print(f"RESULT A(scan->random chase)={ok_a}  B(change)={ok_b1 and ok_b2}  "
          f"D(link loss->hover->recover)={ok_d1 and ok_d2}  C(give-up->rescan->scan again)={ok_c}")

    drone.request_stop()
    time.sleep(1.0)
    drone.stop()


if __name__ == "__main__":
    main()
