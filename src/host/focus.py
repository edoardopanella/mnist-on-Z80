"""
focus.py - Focus on hover: the part under the mouse stays sharp, the others fade.

Tk cannot fade a widget, so each part gets a veil: a borderless, semi-transparent
window in the background colour, placed over it. The focused part gets a blue
border made of four thin windows (Tk on macOS draws no highlight ring on frames).

The mouse is polled every POLL_MS, because <Enter>/<Leave> also fire on child
widgets and on the veils, which makes the focus jump.
    over a part                    that part in focus, the others veiled
    in a gap between parts         focus stays where it was
    outside the window, or
    another app in front           no veils
    window moving                  no veils until it is still for SETTLE_MS
Veils hide while the window moves because Tk reports its position too rarely
for them to follow smoothly.
"""

import time
import tkinter as tk

VEIL_ALPHA = 0.75           # 1 hides the other parts, 0 does not fade them
POLL_MS = 80
SETTLE_MS = 250             # veils come back this long after the window stopped moving
BORDER = 2                  # px of the focus ring
ACCENT = "#2060d0"


class HoverFocus:

    def __init__(self, root, parts):
        """parts: the widgets that can be in focus (frames, usually LabelFrames)."""
        self.root = root
        self.parts = list(parts)
        self.enabled = True
        self.current = None                     # the part in focus, or None
        self.shown = None                       # (current, rectangles) the veils were placed for
        self.bg = root.cget("bg")
        self.veils = [self._window(self.bg, VEIL_ALPHA) for _ in self.parts]
        self.ring = [self._window(ACCENT, 1.0) for _ in range(4)]     # top, bottom, left, right
        self.root_pos = None                    # where the window was last seen
        self.moved_at = 0.0                     # time.monotonic() of its last move
        root.bind("<Configure>", self._on_configure, add="+")
        root.after(POLL_MS, self._tick)

    def _window(self, colour, alpha):
        """A borderless window above root, used for a veil or one side of the ring."""
        w = tk.Toplevel(self.root, bg=colour)
        w.overrideredirect(True)
        w.transient(self.root)          # stays above the window when a click raises it
        w.attributes("-alpha", alpha)
        w.withdraw()
        return w

    def set_enabled(self, on):
        self.enabled = on
        if not on:
            self._set(None)

    # --- the window and the mouse ------------------------------------------------

    def _on_configure(self, event):
        if event.widget is self.root:           # skip the children's <Configure> events
            self._check_moved()

    def _check_moved(self):
        """Hide everything if the window moved. Returns True while it is still settling."""
        pos = (self.root.winfo_rootx(), self.root.winfo_rooty())
        if pos != self.root_pos:
            if self.root_pos is not None:
                self.moved_at = time.monotonic()
                self._hide()
            self.root_pos = pos
        return time.monotonic() - self.moved_at < SETTLE_MS / 1000

    def _tick(self):
        try:
            if self.enabled and not self._check_moved():
                x, y = self.root.winfo_pointerxy()
                if not self._app_active() or not self._inside(self.root, x, y):
                    self._set(None)
                else:
                    hit = next((p for p in self.parts if self._inside(p, x, y)), self.current)
                    self._set(hit)
        finally:
            self.root.after(POLL_MS, self._tick)

    def _app_active(self):
        """False while another app is in front (Tk then has no focus widget)."""
        try:
            return self.root.focus_get() is not None
        except KeyError:                        # focus is on a widget Tk cannot name, like a dialog
            return True

    @staticmethod
    def _inside(widget, x, y):
        if not widget.winfo_ismapped():
            return False
        x0, y0 = widget.winfo_rootx(), widget.winfo_rooty()
        return x0 <= x < x0 + widget.winfo_width() and y0 <= y < y0 + widget.winfo_height()

    # --- veils and ring ----------------------------------------------------------

    def _hide(self):
        """Hide veils and ring but keep the focus. The next _set() shows them again."""
        for w in self.veils + self.ring:
            w.withdraw()
        self.shown = None

    def _set(self, part):
        """Focus part, or nothing if None. Only moves the veils when something changed."""
        rects = tuple((p.winfo_rootx(), p.winfo_rooty(), p.winfo_width(), p.winfo_height())
                      for p in self.parts) if part is not None else ()
        if (part, rects) == self.shown:
            return
        self.current = part
        self.shown = (part, rects)
        for k, (p, veil) in enumerate(zip(self.parts, self.veils)):
            if part is None or p is part or not p.winfo_ismapped():
                veil.withdraw()
            else:
                x, y, w, h = rects[k]
                veil.geometry(f"{w}x{h}+{x}+{y}")
                veil.deiconify()
                veil.lift()
        if part is None:
            for side in self.ring:
                side.withdraw()
            return
        x, y, w, h = rects[self.parts.index(part)]
        b = BORDER
        for side, (sx, sy, sw, sh) in zip(self.ring, ((x, y, w, b), (x, y + h - b, w, b),
                                                      (x, y, b, h), (x + w - b, y, b, h))):
            side.geometry(f"{sw}x{sh}+{sx}+{sy}")
            side.deiconify()
            side.lift()
