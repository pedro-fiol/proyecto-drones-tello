# Silence PyAV / FFmpeg H264 decoder warnings
import av.logging
av.logging.set_level(av.logging.PANIC)
from interceptor.tello_interceptor import TelloInterceptor
from webapp.server import start_in_thread
import traceback

# powershell -ExecutionPolicy Bypass -File switch-tello.ps1 tello
# powershell -ExecutionPolicy Bypass -File switch-tello.ps1 dual
# powershell -ExecutionPolicy Bypass -File switch-tello.ps1 lan



# PHANTOM = True → drone never takes off, all logic in HUD. Use for testing PID + YOLO.
PHANTOM = False

def main():
    dron = TelloInterceptor(phantom_mode=PHANTOM)
    start_in_thread(dron, host="0.0.0.0", port=8000)
    try:
        dron.start()
    except Exception:
        print("[ERROR] start() failed:")
        traceback.print_exc()
    finally:
        dron.stop()

if __name__ == "__main__":
    main()

# .venv\Scripts\python.exe main.py
