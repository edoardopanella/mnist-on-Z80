"""
focus.py - Focus on hover: the part of the window under the mouse stays sharp,
the others fade behind a veil.

Tk cannot blur or fade a widget, so each part gets its own veil: a borderless
window, VEIL_ALPHA opaque, in the window's background colour, laid exactly
over the part. The parts keep updating under it. The focused part also gets a
blue border: four thin windows of the same kind along its edges, inside it,
so the layout does not move (Tk on macOS does not draw a frame's highlight ring).

HoverFocus checks the mouse every POLL_MS instead of using <Enter>/<Leave>:
those also fire on every child widget and on the veils themselves, which would
make the focus jump. The rules:
    mouse over a part             that part in focus, veils on all the others
    mouse in a gap between parts  the focus stays where it was (no flicker)
    mouse outside the window,
    or another app in front       no veils
    the window moves              no veils, until it has been still for SETTLE_MS
The veils are separate windows, and while the window is dragged Tk learns its
new position only now and then: veils moved along would trail behind it. So a
move (a <Configure> of the window, or a new position seen by the poll) hides
them at once, and the poll brings them back when the window stops.
A veil takes the click meant for the part under it, but the mouse focuses a
part as soon as it moves over it, so a click almost never meets a veil.
"""

import time
import tkinter as tk

VEIL_ALPHA = 0.75           # 1 = the faded parts disappear, 0 = no fading
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
        """A borderless window that stays above root: a veil or a side of the ring."""
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
        if event.widget is self.root:           # the toplevel's tag also gets its children's
            self._check_moved()

    def _check_moved(self):
        """Hide everything if the window moved since the last look. True while it is settling."""
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
        """False while another application is in front (Tk then has no focus widget)."""
        try:
            return self.root.focus_get() is not None
        except KeyError:                        # focus on a widget Tk cannot name (a dialog)
            return True

    @staticmethod
    def _inside(widget, x, y):
        if not widget.winfo_ismapped():
            return False
        x0, y0 = widget.winfo_rootx(), widget.winfo_rooty()
        return x0 <= x < x0 + widget.winfo_width() and y0 <= y < y0 + widget.winfo_height()

    # --- veils and ring ----------------------------------------------------------

    def _hide(self):
        """Veils and ring off, focus kept: the next _set() places them again."""
        for w in self.veils + self.ring:
            w.withdraw()
        self.shown = None

    def _set(self, part):
        """Focus part (None: no focus). Moves the veils only when something changed."""
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
