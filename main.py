from TelloInterceptor import TelloInterceptor

# powershell -ExecutionPolicy Bypass -File switch-tello.ps1 tello
# powershell -ExecutionPolicy Bypass -File switch-tello.ps1 lan

def main():
    dron = TelloInterceptor()
    try:
        dron.start()
    finally:
        dron.stop()

if __name__ == "__main__":
    main()
