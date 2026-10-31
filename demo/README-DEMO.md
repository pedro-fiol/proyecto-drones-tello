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
| **DESPEGAR / ATERRIZAR** | Take off / land (label follows flight state). Takeoff starts the 360° scan. |
| **CAMBIAR OBJETIVO** | Chase another scanned person, picked at random (never the current one) |
| **REESCANEAR** | Forget everyone, turn 360° again, pick a new random person |
| **MODO MANUAL** | Toggle manual flight (lights amber when active) |
| **GPU / CPU** | Switch inference between CUDA GPU and CPU (chip above the buttons). Greyed to `CPU (sin GPU)` when no CUDA device is present. |

### Target flow

1. **Takeoff → scan.** The drone turns 360° (40°/s, ~9 s) and remembers every person it
   sees as P1, P2, … in scan order (CW turn = left to right). Only ReID embeddings are
   kept, in RAM — no photos, nothing written to disk. Half bodies at the frame edge and
   tiny far-away detections are not used as references; after the turn, entries that
   turn out to be the same person are merged. Nobody found → the drone scans again.
2. **Random target.** When the turn ends, the drone picks one scanned person at random and chases it.
3. **CAMBIAR OBJETIVO.** Picks another scanned person at random, spins in place until it sees
   that person, then chases it.
4. **Not found.** If one full spin (360°) does not find the chosen person — or the target is
   lost during the chase — the drone picks another random person it has not tried yet and
   spins again. When nobody from the scan is found, it scans the room again.

**Tap a person on the video** to lock onto them directly. The dot-matrix display (RMTT kit)
shows the state: eye = scanning, crosshair = chasing, `?` = spinning to find the target,
`M` = manual, pause = no persons found.

The panel line `OBJETIVO P2 · d 0.18` shows the ReID distance to the target in view: a match
needs `d <= 0.30` (`LOCK_MATCH_THRESHOLD`). Values just above 0.30 while the person is clearly
in view mean the threshold is too tight for the room's lighting.

**Link safety.** If the drone's video or telemetry stops updating for 1 s while flying (WiFi
drop), the drone is commanded to hover (`rc 0 0 0 0`) and the panel shows `SEÑAL PERDIDA`.
Tello keeps flying its last stick values otherwise.

### Checks without flying

- **LED kit:** drone on the ground, PC on its WiFi, then from this folder
  `..\.venv\Scripts\python.exe led_test.py`. Every `EXT mled` line must answer `mled ok`
  and the matrix must show the icon. Any other answer = the kit firmware does not run the
  matrix commands.
- **Full flow, no drone:** `..\.venv\Scripts\python.exe sim_flow.py` (~2 min). Real YOLO +
  ReID on sample photos with a fake drone: scan → random target → change → link loss →
  empty room. Ends with `RESULT ... True` for each phase.

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
