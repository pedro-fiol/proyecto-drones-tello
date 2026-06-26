"""Tkinter dashboard for the Tello demo. Big video + a few buttons."""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk

import cv2
from PIL import Image, ImageTk


# manual speed caps (cm/s, deg/s) - kept gentle for a room full of people
LR_MAX_CM_S   = 35
FB_MAX_CM_S   = 35
UD_MAX_CM_S   = 40
YAW_MAX_DEG_S = 60

VIDEO_REFRESH_MS     = 33
TELEMETRY_REFRESH_MS = 150
MANUAL_SEND_MS       = 100   # stick heartbeat, must be < drone's 0.6s grace

FRAME_W, FRAME_H = 960, 720

# same colors as the web console
BG       = "#ececee"
SURFACE  = "#ffffff"
SURFACE2 = "#f6f6f7"
LINE     = "#e3e3e6"
INK      = "#0b0b0d"
INK2     = "#4a4a52"
INK3     = "#8b8b93"
INK4     = "#b5b5bc"
OK       = "#00b65a"
WARN     = "#ff8a00"
CRIT     = "#ff2d2d"
VIDEO_BG = "#050608"

FONT       = "Segoe UI"
BTN_FONT   = (FONT, 22, "bold")
CHIP_FONT  = (FONT, 14, "bold")
STAT_FONT  = (FONT, 13)
STAT_BIG   = (FONT, 30, "bold")
LABEL_FONT = (FONT, 11, "bold")

MODE_ES = {
    "intercepting": "siguiendo", "searching": "buscando", "hover": "esperando",
    "grounded": "en suelo", "manual": "manual", "init": "iniciando",
}


class Joystick(tk.Canvas):
    """Self-centering stick. value() gives (x, y) in -1..1, y up."""

    def __init__(self, master, size: int = 190):
        super().__init__(master, width=size, height=size,
                         bg=BG, highlightthickness=0, bd=0)
        self.size = size
        self.centre = size / 2
        self.knob_r = size * 0.16
        self.max_radius = self.centre - self.knob_r - 4
        self._x = 0.0
        self._y = 0.0
        self._enabled = False

        c, r = self.centre, self.max_radius
        self.create_oval(c - r, c - r, c + r, c + r, outline=LINE, width=2, fill=SURFACE2)
        self.create_line(c - r, c, c + r, c, fill=LINE)
        self.create_line(c, c - r, c, c + r, fill=LINE)
        self.knob = self.create_oval(0, 0, 0, 0, fill=INK4, outline="")
        self._move_knob(0.0, 0.0)

        self.bind("<Button-1>", self._on_drag)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _move_knob(self, x, y):
        self._x, self._y = x, y
        px = self.centre + x * self.max_radius
        py = self.centre - y * self.max_radius
        kr = self.knob_r
        self.coords(self.knob, px - kr, py - kr, px + kr, py + kr)

    def _on_drag(self, event):
        if not self._enabled:
            return
        dx = event.x - self.centre
        dy = event.y - self.centre
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > self.max_radius and dist > 0:
            dx *= self.max_radius / dist
            dy *= self.max_radius / dist
        self._move_knob(dx / self.max_radius, -dy / self.max_radius)

    def _on_release(self, _event):
        self._move_knob(0.0, 0.0)

    def value(self):
        if not self._enabled:
            return 0.0, 0.0
        return self._x, self._y

    def set_enabled(self, enabled):
        self._enabled = enabled
        self.itemconfig(self.knob, fill=INK if enabled else INK4)
        if not enabled:
            self._move_knob(0.0, 0.0)


class DemoDashboard(tk.Tk):
    """Main window. Talks to a TelloInterceptor (or anything with the same methods)."""

    def __init__(self, drone):
        super().__init__()
        self.drone = drone
        self.is_manual = False
        self.device_current = "cpu"
        self.cuda_available = False
        self._closing = False
        self._video_item = None

        self.title("Tello Interceptor")
        self.configure(bg=BG)
        self.geometry("1280x800")
        try:
            self.state("zoomed")
        except tk.TclError:
            pass
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))

        self._build_video()
        self._build_panel()

        self._update_video()
        self._poll_telemetry()
        self._manual_tick()

    def _build_video(self):
        self.video = tk.Canvas(self, bg=VIDEO_BG, highlightthickness=0, bd=0)
        self.video.pack(side="left", fill="both", expand=True)
        self._msg = self.video.create_text(10, 10, text="Conectando con el dron...",
                                           anchor="center", fill=INK4, font=(FONT, 24, "bold"))

    def _build_panel(self):
        panel = tk.Frame(self, bg=BG, width=400)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        pad = dict(padx=20)

        # status card
        card = tk.Frame(panel, bg=SURFACE, highlightthickness=1, highlightbackground=LINE)
        card.pack(fill="x", pady=(20, 14), **pad)
        self.lbl_battery = tk.Label(card, text="-- %", bg=SURFACE, fg=INK, font=STAT_BIG)
        self.lbl_battery.pack(anchor="w", padx=16, pady=(12, 0))
        tk.Label(card, text="BATERÍA", bg=SURFACE, fg=INK3, font=LABEL_FONT).pack(
            anchor="w", padx=16, pady=(0, 8))
        self.lbl_mode = tk.Label(card, text="MODO  —", bg=SURFACE, fg=INK2, font=STAT_FONT)
        self.lbl_mode.pack(anchor="w", padx=16)
        self.lbl_people = tk.Label(card, text="PERSONAS  0", bg=SURFACE, fg=INK2, font=STAT_FONT)
        self.lbl_people.pack(anchor="w", padx=16, pady=(0, 12))

        # cpu/gpu toggle
        self.btn_device = tk.Button(
            panel, text="—", command=self._toggle_device, font=CHIP_FONT,
            bg=SURFACE2, fg=INK3, activebackground="#eeeef0", activeforeground=INK,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=LINE,
            cursor="hand2", pady=8)
        self.btn_device.pack(fill="x", pady=(0, 10), **pad)

        # main buttons
        self.btn_takeoff = self._make_button(panel, "DESPEGAR", self.drone.request_takeoff_land,
                                             INK, "#ffffff", "#222222", INK)
        self.btn_takeoff.pack(fill="x", pady=8, **pad)

        self.btn_track = self._make_button(panel, "SEGUIR OBJETIVO", self.drone.track_or_cycle,
                                           SURFACE2, INK, "#eeeef0", LINE)
        self.btn_track.pack(fill="x", pady=8, **pad)

        self.btn_switch = self._make_button(panel, "CAMBIAR OBJETIVO", self.drone.track_or_cycle,
                                            SURFACE2, INK, "#eeeef0", LINE)
        self.btn_switch.pack(fill="x", pady=8, **pad)

        self.btn_manual = self._make_button(panel, "MODO MANUAL", self._toggle_manual,
                                            SURFACE2, INK, "#eeeef0", LINE)
        self.btn_manual.pack(fill="x", pady=8, **pad)

        # joysticks
        sticks = tk.Frame(panel, bg=BG)
        sticks.pack(side="bottom", fill="x", pady=18, **pad)
        left_col = tk.Frame(sticks, bg=BG)
        left_col.pack(side="left", expand=True)
        self.left_stick = Joystick(left_col)
        self.left_stick.pack()
        tk.Label(left_col, text="↕ THRUST   ↔ YAW", bg=BG, fg=INK3, font=LABEL_FONT).pack(pady=(6, 0))
        right_col = tk.Frame(sticks, bg=BG)
        right_col.pack(side="right", expand=True)
        self.right_stick = Joystick(right_col)
        self.right_stick.pack()
        tk.Label(right_col, text="↕ PITCH   ↔ ROLL", bg=BG, fg=INK3, font=LABEL_FONT).pack(pady=(6, 0))

    def _make_button(self, parent, text, command, bg, fg, active_bg, border):
        return tk.Button(parent, text=text, command=command, font=BTN_FONT,
                         bg=bg, fg=fg, activebackground=active_bg, activeforeground=fg,
                         relief="flat", bd=0, highlightthickness=1, highlightbackground=border,
                         cursor="hand2", pady=16, wraplength=340)

    def _toggle_manual(self):
        self.drone.toggle_manual()

    def _toggle_device(self):
        if not self.cuda_available:
            return
        target = "cpu" if self.device_current == "cuda" else "cuda"
        self.drone.set_device(target)

    def _toggle_fullscreen(self, _event=None):
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))

    def _update_video(self):
        if self._closing:
            return
        frame = getattr(self.drone, "latest_frame", None)
        cw = max(self.video.winfo_width(), 1)
        ch = max(self.video.winfo_height(), 1)
        if frame is None:
            self.video.itemconfig(self._msg, state="normal")
            self.video.coords(self._msg, cw / 2, ch / 2)
        else:
            self.video.itemconfig(self._msg, state="hidden")
            scale = min(cw / FRAME_W, ch / FRAME_H)
            new_w = max(int(FRAME_W * scale), 1)
            new_h = max(int(FRAME_H * scale), 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(img)   # keep ref or Tk drops it
            if self._video_item is None:
                self._video_item = self.video.create_image(cw / 2, ch / 2,
                                                           image=self._photo, anchor="center")
            else:
                self.video.coords(self._video_item, cw / 2, ch / 2)
                self.video.itemconfig(self._video_item, image=self._photo)
        self.after(VIDEO_REFRESH_MS, self._update_video)

    def _poll_telemetry(self):
        if self._closing:
            return
        try:
            t = self.drone.get_telemetry()
        except Exception:
            t = {}

        bat = t.get("battery_percent", -1)
        if bat is None or bat < 0:
            self.lbl_battery.config(text="-- %", fg=INK3)
        else:
            bat = int(bat)
            col = CRIT if bat < 20 else WARN if bat < 40 else OK
            self.lbl_battery.config(text=f"{bat} %", fg=col)
        mode = t.get("mode", "—")
        self.lbl_mode.config(text=f"MODO  {MODE_ES.get(mode, mode)}")
        self.lbl_people.config(text=f"PERSONAS  {t.get('num_detections', 0)}")

        if t.get("is_airborne"):
            self.btn_takeoff.config(text="ATERRIZAR", bg=CRIT, fg="#ffffff",
                                    activebackground="#e02525", highlightbackground=CRIT)
        else:
            self.btn_takeoff.config(text="DESPEGAR", bg=INK, fg="#ffffff",
                                    activebackground="#222222", highlightbackground=INK)

        self.is_manual = bool(t.get("is_manual"))
        if self.is_manual:
            self.btn_manual.config(text="MODO MANUAL ●", bg=WARN, fg="#ffffff",
                                   activebackground="#e67d00", highlightbackground=WARN)
        else:
            self.btn_manual.config(text="MODO MANUAL", bg=SURFACE2, fg=INK,
                                   activebackground="#eeeef0", highlightbackground=LINE)
        self.left_stick.set_enabled(self.is_manual)
        self.right_stick.set_enabled(self.is_manual)

        self.device_current = t.get("device", "cpu")
        self.cuda_available = bool(t.get("cuda_available", False))
        if self.device_current == "cuda":
            self.btn_device.config(text="GPU (CUDA)", fg=OK)
        elif self.cuda_available:
            self.btn_device.config(text="CPU", fg=INK2)
        else:
            self.btn_device.config(text="CPU (sin GPU)", fg=INK4)

        self.after(TELEMETRY_REFRESH_MS, self._poll_telemetry)

    def _manual_tick(self):
        if self._closing:
            return
        if self.is_manual:
            lx, ly = self.left_stick.value()    # turn, throttle
            rx, ry = self.right_stick.value()   # strafe, forward
            self.drone.set_manual_velocity(
                int(round(rx * LR_MAX_CM_S)),
                int(round(ry * FB_MAX_CM_S)),
                int(round(ly * UD_MAX_CM_S)),
                int(round(lx * YAW_MAX_DEG_S)),
            )
        self.after(MANUAL_SEND_MS, self._manual_tick)

    def _on_close(self):
        self._closing = True
        try:
            self.drone.request_stop()
        except Exception:
            pass
        self.destroy()


def _run_drone(drone):
    # land on any loop exit, not only window close
    try:
        drone.start()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        drone.stop()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    import av.logging
    av.logging.set_level(av.logging.PANIC)

    from interceptor.tello_interceptor import TelloInterceptor

    phantom = "--phantom" in sys.argv
    drone = TelloInterceptor(phantom_mode=phantom, headless=True)

    threading.Thread(target=_run_drone, args=(drone,), daemon=True).start()

    app = DemoDashboard(drone)
    app.mainloop()
    drone.stop()


if __name__ == "__main__":
    main()
