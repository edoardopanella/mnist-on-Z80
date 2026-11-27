"""
network_view.py - The network's values for one digit, shown under the drawing.

    Layer 1   one cell per hidden neuron, shaded by h_q. Clicking a cell
              calls on_neuron(j), which shows that neuron's weights.
    Layer 2   the 10 scores as bars, the answer in green.

The values come from the D reply. set_result() stores them from the worker
thread and refresh() draws them in Tk's thread. In live mode, show_step() is
called 20 times a second and highlights the neuron being computed.
"""

import tkinter as tk
from tkinter import font as tkfont

from z80debug import N_HID, N_IN, N_OUT

WIDTH = 560                         # px, about the drawing + preview
CELL, CELL_GAP = 16, 1              # hidden neuron cells
CELL_H = 26
ROW_H = 18                          # score rows
CURRENT = "#c0c000"                 # being computed (the memory map's "read" colour)
DONE = "#808080"                    # computed, value not received yet
EMPTY = "#909090"                   # outline of a cell not computed yet
BEST, OTHER = "#20a040", "#9090a0"  # score bars
GREY_TEXT = "#808080"


def neuron_fill(h, top):
    """Grey for h, from near black at 0 to near white at top."""
    v = 28 + (240 - 28) * max(0, min(top, h)) // max(1, top)
    return "#%02x%02x%02x" % (v, v, v)


class NetworkView(tk.Frame):

    def __init__(self, master, on_neuron=None):
        super().__init__(master)
        self.on_neuron = on_neuron
        self.values = None          # (acc1, h_q, acc2, digit) on screen
        self.pending = None         # set by set_result() in the worker thread
        self.running = False        # between start() and refresh()
        self.step = None            # z80debug.infer_step() while running (live only)
        self.lit = 0                # lit pixels in the current digit
        base = tkfont.nametofont("TkDefaultFont")
        self.bold = base.copy()
        self.bold.config(weight="bold")
        self.big = base.copy()
        self.big.config(weight="bold", size=base.actual("size") + 6)
        self.caption = base.copy()
        self.caption.config(size=max(9, base.actual("size") - 2))
        self._build()
        self._draw()

    def _build(self):
        # Two parts for focus.HoverFocus. The headline belongs to layer 1.
        l1 = tk.Frame(self)
        l1.pack(fill="x")
        self.headline = tk.Label(l1, anchor="w", justify="left", wraplength=WIDTH, font=self.bold)
        self.headline.pack(fill="x", pady=(0, 4))
        tk.Label(l1, text="Layer 1 · 32 hidden neurons", anchor="w", font=self.bold).pack(fill="x")
        tk.Label(l1, anchor="w", justify="left", wraplength=WIDTH, font=self.caption, fg=GREY_TEXT,
                 text="Each neuron has one weight per pixel and adds up the weights of the lit pixels. "
                      "ReLU turns a negative sum into 0, then the sum is scaled to 0..127. "
                      "Black is 0, white is the strongest. Click a neuron to see its weights in Memory.").pack(fill="x")
        self.hidden = tk.Canvas(l1, width=WIDTH, height=CELL_H + 16, highlightthickness=0,
                                cursor="hand2")
        self.hidden.pack(anchor="w", pady=(4, 0))
        self.hover = tk.Label(l1, anchor="w", font=self.caption, width=1)     # width=1 so it never widens the window
        self.hover.pack(fill="x")
        self.text_fg = self.hover.cget("fg")
        self.hidden.bind("<Motion>", self._on_motion)
        self.hidden.bind("<Leave>", lambda _e: self._show_hover(None))
        self.hidden.bind("<Button-1>", self._on_click)

        l2 = tk.Frame(self)
        l2.pack(fill="x", pady=(6, 0))
        tk.Label(l2, text="Layer 2 · 10 scores, one per digit", anchor="w", font=self.bold).pack(fill="x")
        tk.Label(l2, anchor="w", justify="left", wraplength=WIDTH, font=self.caption, fg=GREY_TEXT,
                 text="Each score is the sum of 32 weights times the 32 hidden values. "
                      "The highest score is the answer.").pack(fill="x")
        self.scores = tk.Canvas(l2, width=WIDTH, height=N_OUT * ROW_H + 4, highlightthickness=0)
        self.scores.pack(anchor="w", pady=(4, 0))
        self.parts = [l1, l2]

    def focus_parts(self):
        """The parts focus.HoverFocus can focus: layer 1 (with the headline) and layer 2."""
        return self.parts

    # --- called by DrawingApp and DebugPanel ------------------------------------

    def start(self, x):
        """A digit is going to the Z80."""
        self.values = self.pending = None
        self.running = True
        self.step = None
        self.lit = int(sum(x))
        self._draw()

    def set_result(self, acc1, h_q, acc2, digit):
        """Store the D reply. Safe to call from the worker thread."""
        self.pending = (list(acc1), list(h_q), list(acc2), digit)

    def refresh(self):
        """Show the values stored by set_result()."""
        if self.pending is not None:
            self.values, self.pending = self.pending, None
        self.running = False
        self.step = None
        self._draw()

    def clear(self):
        self.values = self.pending = None
        self.running = False
        self.step = None
        self._draw()

    def show_step(self, step):
        """Live mode: step is z80debug.infer_step(), or None outside nn_infer."""
        if not self.running:
            return
        # Outside nn_infer the Z80 is either unpacking the image or sending the results
        if step is None:
            after = self.step is not None and self.step[0] in ("layer2", "trace", "send")
            step = ("send", None) if after else ("before", None)
        if step == self.step:
            return
        self.step = step
        self._draw()

    # --- drawing ----------------------------------------------------------------

    def _draw(self):
        self.headline.config(text=self._headline(), font=self.big if self.values else self.bold)
        self._draw_hidden()
        self._draw_scores()
        self._show_hover(None)

    def _headline(self):
        if self.values is not None:
            acc2, d = self.values[2], self.values[3]
            r = max((k for k in range(N_OUT) if k != d), key=lambda k: acc2[k])
            return f"Answer: {d}    (score {acc2[d]:,}; next best: {r}, {acc2[r]:,})"
        if not self.running:
            return "Draw a digit and press Recognize. The Z80 computes the two layers below."
        kind, n = self.step or ("", None)
        if kind == "layer1":
            return f"Layer 1, neuron #{n}: adding its weights for the {self.lit} lit pixels"
        if kind == "layer2":
            return f"Layer 2, digit {n}: 32 weights times the hidden values"
        if kind == "trace":
            return "Network done, copying the results into trace"
        if kind == "send":
            return "Network done, sending the 74 results to the laptop as text"
        if kind == "before":
            return f"Unpacking the image: {N_IN // 8} bytes into {N_IN} pixels"
        return "The Z80 is computing…"

    def _progress(self):
        """(hidden neurons done, current neuron or None, current score or None)."""
        kind, n = self.step or ("", None)
        if kind == "layer1":
            return n, n, None
        if kind == "layer2":
            return N_HID, None, n
        if kind in ("trace", "send"):
            return N_HID, None, None
        return 0, None, None

    def _draw_hidden(self):
        c = self.hidden
        c.delete("all")
        done, current, _ = self._progress() if self.running else (0, None, None)
        top = max(self.values[1]) if self.values is not None else 0
        for j in range(N_HID):
            x0 = j * (CELL + CELL_GAP)
            if self.values is not None:
                fill, outline, width = neuron_fill(self.values[1][j], top), "", 0
            elif j == current:
                fill, outline, width = DONE, CURRENT, 2
            elif j < done:
                fill, outline, width = DONE, "", 0
            else:
                fill, outline, width = "", EMPTY, 1
            c.create_rectangle(x0 + 1, 1, x0 + CELL - 1, CELL_H - 1, fill=fill, outline=outline, width=width)
        for j in (0, 8, 16, 24, 31):
            c.create_text(j * (CELL + CELL_GAP) + CELL // 2, CELL_H + 2, text=f"#{j}", anchor="n",
                          font=self.caption, fill=GREY_TEXT)

    def _draw_scores(self):
        c = self.scores
        c.delete("all")
        _, _, current = self._progress() if self.running else (0, None, None)
        x_name, x0, width = 4, 22, WIDTH - 120
        acc2 = self.values[2] if self.values is not None else None
        if acc2 is not None:
            lo, hi = min(acc2 + [0]), max(acc2 + [0])
            span = max(1, hi - lo)
            zero = x0 + (0 - lo) * width // span
            c.create_line(zero, 0, zero, N_OUT * ROW_H + 2, fill="#606060")
        for d in range(N_OUT):
            y = 2 + d * ROW_H
            if d == current:
                c.create_rectangle(1, y - 1, WIDTH - 2, y + ROW_H - 2, outline=CURRENT, width=2)
            c.create_text(x_name, y, text=str(d), anchor="nw", font=self.bold)
            if acc2 is None:
                continue
            a, b = sorted((zero, x0 + (acc2[d] - lo) * width // span))
            c.create_rectangle(a, y + 2, max(a + 1, b), y + ROW_H - 4, width=0,
                               fill=BEST if d == self.values[3] else OTHER)
            c.create_text(x0 + width + 8, y, text=f"{acc2[d]:,}", anchor="nw", font=self.caption)

    # --- mouse ------------------------------------------------------------------

    def _neuron_at(self, x):
        j = x // (CELL + CELL_GAP)
        return j if 0 <= j < N_HID else None

    def _show_hover(self, j):
        if j is None:
            if self.values is not None:
                active = sum(1 for h in self.values[1] if h > 0)
                text = f"{active} of 32 neurons active, the other {32 - active} are 0 after ReLU"
            else:
                text = ""
            self.hover.config(text=text, fg=GREY_TEXT)
            return
        if self.values is not None:
            s, h = self.values[0][j], self.values[1][j]
            text = f"Neuron #{j}: sum {s:,} → ReLU {max(0, s):,} → rescaled {h}"
        else:
            text = f"Neuron #{j}"
        self.hover.config(text=text + f" · click to see its {N_IN} weights", fg=self.text_fg)

    def _on_motion(self, event):
        self._show_hover(self._neuron_at(event.x))

    def _on_click(self, event):
        j = self._neuron_at(event.x)
        if j is not None and self.on_neuron is not None:
            self.on_neuron(j)
