"""
FastAPI server for Tello Interceptor operator console.

Three IO surfaces:
  GET  /              static index.html
  GET  /video         multipart MJPEG stream of last annotated frame
  WS   /ws/telemetry  20 Hz JSON push of all sensor + control state
  POST /cmd/*         operator commands (takeoff_land, toggle_manual,
                      enroll, clear_lock, stop)

Designed to run on localhost only — PC is on the Tello WiFi AP (no internet),
so no STUN/TURN, no HTTPS, no auth. If you ever expose this to a WAN, add a
tunnel (cloudflared) and a token check on every endpoint first.

start_in_thread(interceptor) launches uvicorn in a daemon thread so the
TelloInterceptor.start() main-thread loop (cv2.imshow) is unaffected.
"""
import asyncio
import threading
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn


# JPEG quality for MJPEG stream. 80 = good quality, ~40 KB/frame at 960x720.
# Drop to 60 if LAN bandwidth limited.
_JPEG_QUALITY = 80

# Telemetry push interval. 20 Hz matches RC_LOOP_INTERVAL_S so the dashboard
# sees the same cadence the drone runs at.
_TELEMETRY_INTERVAL_S = 0.05

# MJPEG frame pacing. 30 Hz target — video loop produces ~30 FPS too.
_MJPEG_INTERVAL_S = 1.0 / 30.0


_STATIC_DIR = Path(__file__).parent / "static"

# Manual RC velocity clamp. Same caps the djitellopy SDK enforces (-100..100),
# but tighter on yaw to match MANUAL_YAW_VELOCITY_DEG_S keyboard default.
_MANUAL_MAX = 100


class ManualCommand(BaseModel):
    """Manual RC velocities from web dpad. Units: cm/s (lr/fb/ud), deg/s (yaw).

    All four required so partial updates can't leave a stale axis spinning.
    Pydantic enforces the range — anything outside gets a 422 from FastAPI.
    """
    lr:  int = Field(..., ge=-_MANUAL_MAX, le=_MANUAL_MAX)
    fb:  int = Field(..., ge=-_MANUAL_MAX, le=_MANUAL_MAX)
    ud:  int = Field(..., ge=-_MANUAL_MAX, le=_MANUAL_MAX)
    yaw: int = Field(..., ge=-_MANUAL_MAX, le=_MANUAL_MAX)


class TargetCommand(BaseModel):
    """Switch active tracking target by name. Names come from AVAILABLE_TARGETS."""
    name: str = Field(..., min_length=1, max_length=64)


def build_app(interceptor) -> FastAPI:
    """Wire all routes against a live TelloInterceptor instance."""
    app = FastAPI(title="Tello Interceptor")

    @app.get("/")
    def index():
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/video")
    def video():
        """Multipart MJPEG. Browser opens it in an <img src> tag."""
        return StreamingResponse(
            _mjpeg_generator(interceptor),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.websocket("/ws/telemetry")
    async def telemetry_ws(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                await ws.send_json(interceptor.web_get_telemetry())
                await asyncio.sleep(_TELEMETRY_INTERVAL_S)
        except WebSocketDisconnect:
            return
        except Exception:
            # Any send error = client gone. Drop quietly.
            return

    @app.post("/cmd/takeoff_land")
    def cmd_takeoff_land():
        interceptor.web_request_takeoff_land()
        return {"ok": True}

    @app.post("/cmd/toggle_manual")
    def cmd_toggle_manual():
        interceptor.web_request_toggle_manual()
        return {"ok": True}

    @app.post("/cmd/enroll")
    def cmd_enroll():
        interceptor.web_request_enroll()
        return {"ok": True}

    @app.post("/cmd/clear_lock")
    def cmd_clear_lock():
        interceptor.web_request_clear_lock()
        return {"ok": True}

    @app.post("/cmd/stop")
    def cmd_stop():
        interceptor.web_request_stop()
        return {"ok": True}

    @app.post("/cmd/set_target")
    def cmd_set_target(cmd: TargetCommand):
        """Swap active tracking target by name (e.g. nose, shoulders_midpoint).

        Returns 400 on unknown name so the frontend can revert the dropdown.
        """
        ok = interceptor.web_set_target(cmd.name)
        if not ok:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=f"unknown target: {cmd.name}")
        return {"ok": True, "name": cmd.name}

    @app.post("/cmd/manual_set")
    def cmd_manual_set(cmd: ManualCommand):
        """Set manual RC velocity. Frontend dpad calls this every ~100 ms
        while a button is held, with zeros on release. Requires manual mode
        active (toggle via /cmd/toggle_manual first) — values are written but
        ignored by the RC loop until is_manual is True.
        """
        interceptor.web_set_manual_velocity(cmd.lr, cmd.fb, cmd.ud, cmd.yaw)
        return {"ok": True}

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    return app


def _mjpeg_generator(interceptor):
    """Yield JPEG-encoded frames as multipart parts.

    Reads interceptor.latest_frame each tick. None = drone not yet streaming;
    yield a 1-frame black placeholder so the browser doesn't show broken-image.
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
    while True:
        frame = interceptor.latest_frame
        if frame is None:
            time.sleep(_MJPEG_INTERVAL_S)
            continue
        ok, jpg = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            time.sleep(_MJPEG_INTERVAL_S)
            continue
        chunk = jpg.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(chunk)).encode() + b"\r\n\r\n" +
            chunk + b"\r\n"
        )
        time.sleep(_MJPEG_INTERVAL_S)


def start_in_thread(interceptor, host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Start uvicorn on a daemon thread.

    Daemon so process exits when TelloInterceptor.start() returns. Returns the
    thread so caller can keep a handle if needed.
    """
    app = build_app(interceptor)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _run():
        server.run()

    t = threading.Thread(target=_run, daemon=True, name="webapp-uvicorn")
    t.start()
    print(f"[INFO] Webapp listening on http://{host}:{port}")
    return t
