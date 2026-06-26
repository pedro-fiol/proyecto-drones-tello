# Tello Interceptor — Demo

Stripped-down version of the main project for public demos. Single Tkinter
window: large video + four big buttons + two joysticks. No cv2 window, no webapp.

## Run

Works with **any** Tello — no drone-specific setup. Every Tello exposes its own
WiFi AP at `192.168.10.1`, so connecting to its network is the only drone step.

1. Power the Tello. Connect your PC's WiFi to the Tello's network
   (e.g. `RMTT-XXXXXX` or `TELLO-XXXXXX` — whatever your drone broadcasts).
2. Open this folder in **VS Code** and press **F5** (Run ▸ Start Debugging) —
   the "Run Tello Demo" config uses the right interpreter and working directory.
   - Or double-click **`run_demo.bat`**.
   - Add `--phantom` (in `launch.json` args, or `run_demo.bat --phantom`) to test
     video + tracking without taking off.

Uses the main project's `.venv` (CUDA PyTorch) — nothing to install. The
`.vscode/` config pins that interpreter, so F5 just works.

## Controls

| Button | Action |
|---|---|
| **DESPEGAR / ATERRIZAR** | Take off / land (label follows flight state) |
| **SEGUIR OBJETIVO** | Lock onto the nearest person |
| **CAMBIAR OBJETIVO** | Cycle the lock to the next person |
| **MODO MANUAL** | Toggle manual flight (lights amber when active) |
| **GPU / CPU** | Switch inference between CUDA GPU and CPU (chip above the buttons). Greyed to `CPU (sin GPU)` when no CUDA device is present. |

**Tap a person on the video** to lock onto them — the most reliable way to pick or switch target. `SEGUIR OBJETIVO` locks the biggest person; `CAMBIAR OBJETIVO` cycles to the next.

Joysticks (active only in manual mode):
- **Left** — ↕ thrust, ↔ yaw
- **Right** — ↕ pitch, ↔ roll

`F11` fullscreen, `Esc` exit fullscreen. Closing the window lands the drone.

## What differs from the main project

- New `demo_dashboard.py` entry point (Tkinter). It drives `TelloInterceptor`
  **directly** — no FastAPI, no webapp.
- The interceptor's control surface was renamed from the webapp-flavoured
  `web_*` names to a neutral dashboard API:
  `request_takeoff_land()`, `track_or_cycle()`, `toggle_manual()`,
  `clear_lock()`, `set_target()`, `set_manual_velocity()`, `get_telemetry()`,
  `request_stop()`. Click-to-lock (`web_lock_at`) was dropped — not needed.
- `interceptor/tello_interceptor.py` gained a `headless=True` flag to skip its
  cv2 preview window. **All flight/tracking logic is byte-for-byte the same.**
- No `webapp/`, no `main.py` flow.
