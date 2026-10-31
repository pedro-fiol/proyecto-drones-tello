"""Check the RMTT expansion kit (ESP32) answers EXT commands and drives the dot-matrix display.

Drone on the ground — no takeoff. Connect the PC to the drone WiFi, then from demo/:
    ..\\.venv\\Scripts\\python.exe led_test.py

The drone also emits unsolicited "unknown command: keepalive" replies, which shift djitellopy's
reply-per-command pairing by one. So this script clears the reply queue, sends each command
without waiting, and prints EVERY reply that arrives in the next 1.5 s.

Expected per line:
    EXT version?   -> version esp32v...
    EXT led ...    -> led ok          (top RGB LED red, then off)
    EXT mled ...   -> matrix ok       (matrix shows the icon for ~2 s)
Watch the matrix while it runs: the icon must change on each mled line.
"""

import time

from djitellopy import Tello

from interceptor.led_matrix import LED_CLEAR_COMMAND, led_command_for_mode

REPLY_WINDOW_S = 1.5
HOLD_S = 1.0


def main():
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%\n")
    responses = tello.get_own_udp_object()["responses"]

    steps = [
        ("EXT version?", "ESP32 firmware version"),
        ("EXT led 255 0 0", "top LED red"),
        ("EXT mled s r A", "matrix: letter A, red"),
        (led_command_for_mode("scanning"), "matrix: eye, purple (SCANNING)"),
        (led_command_for_mode("intercepting"), "matrix: crosshair, red (TRACKING)"),
        (led_command_for_mode("searching"), "matrix: ?, purple (SEARCHING)"),
        (LED_CLEAR_COMMAND, "matrix: all off"),
        ("EXT led 0 0 0", "top LED off"),
    ]
    for command, what in steps:
        responses.clear()
        tello.send_command_without_return(command)
        time.sleep(REPLY_WINDOW_S)
        replies = [r.decode("utf-8", "replace").strip() for r in list(responses)]
        own = [r for r in replies if "keepalive" not in r]
        shown = command if len(command) < 40 else command[:37] + "..."
        print(f"{what:<36} {shown:<42} -> {own or 'NO REPLY'}"
              + (f"   (+{len(replies) - len(own)} keepalive)" if len(replies) != len(own) else ""))
        time.sleep(HOLD_S)


if __name__ == "__main__":
    main()
