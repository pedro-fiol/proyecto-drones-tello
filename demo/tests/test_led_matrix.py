"""
tests/test_led_matrix.py

Unit tests for interceptor/led_matrix.py. Fake Tello records commands. No drone.
"""

import threading
import time

import pytest

import interceptor.led_matrix as led_matrix
from interceptor.led_matrix import (
    LED_CLEAR_COMMAND,
    LedMatrix,
    build_pattern,
    led_command_for_mode,
)


class FakeTello:
    """Records fire-and-forget commands. Any wait-for-reply call is a test failure."""

    def __init__(self, fail=False):
        self.commands: list[str] = []
        self.fail = fail

    def send_command_without_return(self, command):
        if self.fail:
            raise OSError("socket down")
        self.commands.append(command)

    def send_command_with_return(self, command, timeout=1):
        raise AssertionError("LED sender must never wait for a reply (blocks the RC loop)")


def _wait_for(predicate, timeout_s=2.0):
    end = time.time() + timeout_s
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_build_pattern_length_and_charset():
    rows = ["#......."] + ["........"] * 7
    pattern = build_pattern(rows, "r")
    assert len(pattern) == 64
    assert pattern[0] == "r"
    assert set(pattern[1:]) == {"0"}


def test_build_pattern_rejects_bad_shape_and_colour():
    with pytest.raises(ValueError):
        build_pattern(["#"] * 8, "r")
    with pytest.raises(ValueError):
        build_pattern(["........"] * 8, "g")


@pytest.mark.parametrize("mode,colour", [
    ("scanning", "p"),
    ("waiting", "b"),
    ("intercepting", "r"),
    ("hover", "p"),
    ("searching", "p"),
    ("manual", "b"),
])
def test_every_mode_has_a_single_colour_64_led_icon(mode, colour):
    command = led_command_for_mode(mode)
    assert command.startswith("EXT mled g ")
    pattern = command.split()[-1]
    assert len(pattern) == 64
    assert set(pattern) == {colour, "0"}


def test_grounded_unknown_and_link_lost_clear():
    assert led_command_for_mode("grounded") == LED_CLEAR_COMMAND
    assert led_command_for_mode("init") == LED_CLEAR_COMMAND
    assert led_command_for_mode("link_lost") == LED_CLEAR_COMMAND
    assert LED_CLEAR_COMMAND == "EXT mled g " + "0" * 64


def test_icons_are_distinct_per_state_group():
    icons = {led_command_for_mode(m) for m in ("scanning", "waiting", "intercepting", "searching", "manual")}
    assert len(icons) == 5


def test_sends_on_change_without_waiting_and_clears_on_stop():
    tello = FakeTello()
    led = LedMatrix(tello, threading.Lock())
    led.start()
    for _ in range(50):
        led.show("scanning")
    assert _wait_for(lambda: led_command_for_mode("scanning") in tello.commands)
    led.show("waiting")
    assert _wait_for(lambda: led_command_for_mode("waiting") in tello.commands)
    led.stop()
    assert tello.commands.count(led_command_for_mode("scanning")) == 1
    assert tello.commands[-1] == LED_CLEAR_COMMAND


def test_resends_current_icon_periodically(monkeypatch):
    monkeypatch.setattr(led_matrix, "LED_RESEND_S", 0.15)
    tello = FakeTello()
    led = LedMatrix(tello, threading.Lock())
    led.start()
    led.show("intercepting")
    assert _wait_for(lambda: tello.commands.count(led_command_for_mode("intercepting")) >= 3, 2.0)
    led.stop()


def test_lock_held_only_for_the_send():
    lock = threading.Lock()
    tello = FakeTello()
    led = LedMatrix(tello, lock)
    led.start()
    led.show("searching")
    assert _wait_for(lambda: len(tello.commands) >= 1)
    t0 = time.time()
    with lock:
        waited = time.time() - t0
    led.stop()
    assert waited < 0.05


def test_send_failure_does_not_kill_thread():
    tello = FakeTello(fail=True)
    led = LedMatrix(tello, threading.Lock())
    led.start()
    led.show("manual")
    time.sleep(0.2)
    assert led._thread.is_alive()
    tello.fail = False
    led.show("scanning")
    assert _wait_for(lambda: led_command_for_mode("scanning") in tello.commands)
    led.stop()
