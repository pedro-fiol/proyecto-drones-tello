# Tello Interceptor

Autonomous person-tracking drone built on a **DJI RoboMaster Tello Talent (RMTT)**.
The drone searches an indoor space, locks onto the first person it sees, follows them, and intercepts them at a configurable distance — using only the onboard camera, the front-facing ToF expansion sensor, and a Wi-Fi link to a PC running PyTorch + CUDA.

> 4th-year **Drone Programming** project — Universitat Politècnica de Catalunya (UPC), course 2025–2026.

---

## Demo videos

<!-- Drag-and-drop your .mp4 demos into a GitHub issue/PR, copy the generated
     `https://github.com/user-attachments/...` URL, and paste it below as the
     image source. GitHub will render an inline player. -->

| Search → Lock → Track | Multi-target switching | Web dashboard |
| :---: | :---: | :---: |
| _video placeholder_ | _video placeholder_ | _video placeholder_ |

---

## What it does

1. **Searches** — when no target is visible, spins on the spot, then advances into open space (front ToF clear) and sweeps.
2. **Detects** — YOLOv8s-pose runs every frame at 30 Hz, returning bounding boxes + 17 COCO keypoints per person.
3. **Locks** — extracts an **OSNet ReID embedding** of the chosen person and tracks that embedding across frames, so a different person walking through the view does not steal the lock.
4. **Tracks** — four independent PID loops drive yaw, altitude, pitch (front-back) and roll (left-right) to keep the target centred and at a fixed distance.
5. **Intercepts** — closes in until the front ToF reads ≤ `INTERCEPT_DISTANCE_CM`, then holds station.
6. **Stops safely** — geofence ceiling/floor clamps, front-ToF wall stop, and a diagonal-wall discontinuity freeze prevent the drone from punching through walls.

The operator can override anything at any time via keyboard or the web dashboard.

---

## Hardware

| Component | Notes |
| --- | --- |
| **DJI RoboMaster Tello Talent (RMTT)** | SSID `RMTT-AD3294`, AP IP `192.168.10.1`. Used via `djitellopy` SDK. |
| **RmTTOC ESP32 expansion kit** | Provides the front-facing **VL53Lx ToF** (range ≤ 120 cm) used for intercept distance and wall avoidance. Read via `EXT tof?` SDK command. |
| **PC running PyTorch + CUDA** | Tested on Python 3.12 + CUDA 12.1. Wi-Fi to the drone AP is required during flight (see `switch-tello.ps1`). |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  main.py                                                        │
│    └── TelloInterceptor  (orchestrator, owns Tello + threads)   │
│         ├── video_loop   ─────► perception ─► targets ─► PIDs   │
│         ├── rc_control_loop @ 20 Hz  ─────►  Tello SDK          │
│         └── front_tof_loop                                      │
│                                                                 │
│  webapp/server.py                                               │
│    └── FastAPI + WebSocket dashboard (telemetry + dpad control) │
└─────────────────────────────────────────────────────────────────┘
```

Module map (`interceptor/`):

| File | Responsibility |
| --- | --- |
| `tello_interceptor.py` | Orchestrator. Owns the Tello, the SDK lock and the worker threads. |
| `perception.py` | YOLOv8s-pose → `list[Person]` (bbox + 17 keypoints + visibility). |
| `target.py` | The four selectable targets: `face`, `eyes`, `shoulders`, `bbox`. Each carries its own visibility rule and point extractor. Also holds the ReID `LockState`. |
| `reid.py` | OSNet embedder + cosine matching + EMA smoothing of the locked embedding. |
| `pid_controller.py` | Reusable PID with output clamping and integral reset. |
| `hud_overlay.py` | All OpenCV drawing — crosshair, bbox, target dot, badges, telemetry strip. |
| `constants.py` | Every tunable value (gains, thresholds, distances) with units in the name. **Edit here, nowhere else.** |

---

## Quick start

### 1. Wi-Fi switch (Windows, run as admin)

```powershell
powershell -ExecutionPolicy Bypass -File switch-tello.ps1 tello
```
Connects the PC to `RMTT-AD3294`. Use `lan` at the end of the session to switch back.

### 2. Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> The ReID weights (`models/osnet_x0_25_msmt17.pt`, 9 MB) ship with the repo so the drone is ready to fly without an extra download.

### 3. Run

```powershell
.venv\Scripts\python.exe main.py
```

A live OpenCV window opens with the HUD. The FastAPI dashboard auto-starts on `http://localhost:8000` — open it on a phone on the same Wi-Fi to control the drone from a touch device.

> Set `PHANTOM = True` in `main.py` to run the full perception + control stack **without taking off** — useful for testing PIDs, perception, and HUD on the desk.

---

## Controls

### Keyboard (operator at the PC)

| Key | Action |
| --- | --- |
| `Space` | Takeoff / land |
| `M` | Toggle manual mode |
| `W` / `S` | Forward / backward |
| `A` / `D` | Strafe left / right |
| `Q` / `E` | Yaw left / right |
| `↑` / `↓` | Up / down |
| `I` | Lock onto the closest centred person (ReID) |
| `C` | Clear the current lock |
| `Esc` | Emergency stop |

### Web dashboard

- Touch dpad for manual control (lr/fb, ud/yaw).
- Tap a person in the video feed to lock onto them.
- Buttons for takeoff/land, toggle manual, lock/clear, stop, and target-mode switch (face / eyes / shoulders / bbox).

---

## Features

- **YOLOv8s-pose perception** — 30 Hz, 17 COCO keypoints, runs on CUDA.
- **Four target modes** — `face`, `eyes`, `shoulders`, `bbox`. Each has its own visibility rule, so the controller falls back gracefully when keypoints disappear.
- **OSNet ReID lock** — once locked, the same person stays tracked even if another walks through the frame. EMA-smoothed embeddings + cosine distance threshold.
- **Four independent PIDs** —
  - **Yaw**: nose-x error → yaw rate (gains `0.25 / 0.001 / 0.05`)
  - **Altitude**: target-y / baro → ud (gains `2.0 / 0.05 / 0.2`)
  - **Pitch**: ToF distance + bbox-width fallback → fb
  - **Roll**: target-x error inside bbox dead-zone → lr
- **Search FSM** — spins in place, then advances into open ToF space, then sweeps. Hysteresis on detection loss avoids flicker.
- **Safety stack** — geofence ceiling/floor, front-ToF wall stop, diagonal-wall discontinuity freeze, manual override with priority, web heartbeat watchdog.
- **Web dashboard** — FastAPI + WebSocket, live MJPEG feed, dpad control, target selector. Same intent flags as the keyboard — no duplicated logic.

---

## Tech stack

- **Python 3.12**
- **PyTorch 2.5.1 + CUDA 12.1** — perception inference
- **Ultralytics YOLOv8s-pose** — detection + keypoints
- **torchreid (OSNet x0.25, MSMT17)** — re-identification embeddings
- **djitellopy** — Tello SDK wrapper (video + commands)
- **OpenCV** — frame decode helpers + HUD rendering
- **PyAV** — H.264 stream decode
- **FastAPI + Uvicorn + WebSockets** — web dashboard
- **NumPy** — math + image arrays

---

## Repository contents

| Path | Purpose |
| --- | --- |
| `main.py` | Entry point. |
| `interceptor/` | Core control + perception package. |
| `webapp/` | FastAPI dashboard + static frontend. |
| `models/osnet_x0_25_msmt17.pt` | Vendored ReID weights. |
| `switch-tello.ps1` | Wi-Fi auto-switch helper (admin). |
| `requirements.txt` | Pinned runtime dependencies. |
| `Tello_Interceptor_-_Defensa.pptx` | Project defense slides. |

---

## License & credits

Academic project, UPC Aerospace Engineering 4A — Drone Programming course, 2025/2026.
