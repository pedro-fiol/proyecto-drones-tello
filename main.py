from interceptor.tello_interceptor import TelloInterceptor

# powershell -ExecutionPolicy Bypass -File switch-tello.ps1 tello
# powershell -ExecutionPolicy Bypass -File switch-tello.ps1 lan

import traceback

# PHANTOM = True → drone never takes off, all logic dry-runs in HUD. Use for testing PID + YOLO.
PHANTOM = False

def main():
    dron = TelloInterceptor(phantom_mode=PHANTOM)
    try:
        dron.start()
    except Exception:
        print("[ERROR] start() failed:")
        traceback.print_exc()
    finally:
        dron.stop()

if __name__ == "__main__":
    main()
