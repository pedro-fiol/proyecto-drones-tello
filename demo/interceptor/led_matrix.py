"""Status icon on the RMTT expansion-kit dot matrix (8x8, red / blue / purple LEDs).

One icon per drone mode. Colour code: red = hunting, purple = looking, blue = idle or manual.
SDK 3.0 command: "EXT mled g <max 64 chars>", one char per LED, row by row, 'r' 'b' 'p' or '0' (off).
Blank matrix = all-'0' pattern. ("EXT mled sc" clears the BOOT animation, not the display.)

Fire-and-forget: the sender never waits for the "mled ok" reply. Waiting held _sdk_lock for up to
1 s per try, which blocked the RC loop — and the drone keeps flying its last rc values meanwhile.
The ToF poller clears stale replies before each query, so the unread "mled ok" cannot desync it.
A dropped UDP packet is covered by re-sending the current icon every LED_RESEND_S.

LedMatrix runs its own daemon thread. show(mode) never blocks the video loop.

Public:
    build_pattern(rows, color) -> str
    led_command_for_mode(mode) -> str
    LedMatrix(tello, sdk_lock): start, show, stop
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from interceptor.constants import LED_RESEND_S


LED_CLEAR_COMMAND = "EXT mled g " + "0" * 64

# '#' = LED on, '.' = LED off. 8 rows x 8 columns.
_EYE_ROWS = [
    "........",
    "..####..",
    ".#....#.",
    "#..##..#",
    "#..##..#",
    ".#....#.",
    "..####..",
    "........",
]
_PAUSE_ROWS = [
    "........",
    ".##..##.",
    ".##..##.",
    ".##..##.",
    ".##..##.",
    ".##..##.",
    ".##..##.",
    "........",
]
_CROSSHAIR_ROWS = [
    "...##...",
    "...##...",
    "........",
    "##.##.##",
    "##.##.##",
    "........",
    "...##...",
    "...##...",
]
_QUESTION_ROWS = [
    "..####..",
    ".##..##.",
    ".....##.",
    "....##..",
    "...##...",
    "...##...",
    "........",
    "...##...",
]
_MANUAL_ROWS = [
    "#......#",
    "##....##",
    "#.#..#.#",
    "#..##..#",
    "#......#",
    "#......#",
    "#......#",
    "#......#",
]


def build_pattern(rows: list[str], color: str) -> str:
    """Turn 8 rows of '#'/'.' into the 64-char mled string. color is 'r', 'b' or 'p'."""
    if len(rows) != 8 or any(len(r) != 8 for r in rows):
        raise ValueError("LED icon must be 8 rows of 8 chars")
    if color not in ("r", "b", "p"):
        raise ValueError(f"unknown LED colour: {color}")
    return "".join(color if ch == "#" else "0" for row in rows for ch in row)


# Mode label (TelloInterceptor.current_mode_label) -> mled command.
# Unknown modes and "grounded" clear the matrix.
_MODE_TO_COMMAND: dict[str, str] = {
    "scanning":     "EXT mled g " + build_pattern(_EYE_ROWS, "p"),
    "waiting":      "EXT mled g " + build_pattern(_PAUSE_ROWS, "b"),
    "intercepting": "EXT mled g " + build_pattern(_CROSSHAIR_ROWS, "r"),
    "hover":        "EXT mled g " + build_pattern(_QUESTION_ROWS, "p"),   # target lost, grace period
    "searching":    "EXT mled g " + build_pattern(_QUESTION_ROWS, "p"),
    "manual":       "EXT mled g " + build_pattern(_MANUAL_ROWS, "b"),
}


def led_command_for_mode(mode: str) -> str:
    """SDK command that shows the icon for this mode. Grounded / unknown -> clear."""
    return _MODE_TO_COMMAND.get(mode, LED_CLEAR_COMMAND)


class LedMatrix:
    """Background sender for the status icon. Owns no SDK state; shares the orchestrator's _sdk_lock."""

    def __init__(self, tello, sdk_lock: threading.Lock) -> None:
        self._tello = tello
        self._sdk_lock = sdk_lock
        self._wanted_command: str = LED_CLEAR_COMMAND
        self._shown_command: Optional[str] = None
        self._last_send_time: float = 0.0
        self._wake = threading.Event()
        self._active = False
        self._warned = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the sender thread."""
        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="led-matrix")
        self._thread.start()

    def show(self, mode: str) -> None:
        """Request the icon for this mode. Cheap; call every video tick."""
        command = led_command_for_mode(mode)
        if command != self._wanted_command:
            self._wanted_command = command
            self._wake.set()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Clear the matrix, then end the thread."""
        self._wanted_command = LED_CLEAR_COMMAND
        self._active = False
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def _send(self, command: str) -> None:
        """One UDP send, no reply wait. Lock held only for the sendto."""
        try:
            with self._sdk_lock:
                self._tello.send_command_without_return(command)
        except Exception as exc:
            if not self._warned:
                print(f"[WARN] LED matrix send failed: {exc}")
                self._warned = True

    def _loop(self) -> None:
        while True:
            self._wake.wait(timeout=LED_RESEND_S)
            self._wake.clear()
            command = self._wanted_command
            resend_due = (time.time() - self._last_send_time) >= LED_RESEND_S
            if command != self._shown_command or resend_due:
                self._send(command)
                self._shown_command = command
                self._last_send_time = time.time()
            if not self._active:
                return
