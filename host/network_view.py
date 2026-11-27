"""
network_view.py - The network, layer by layer, under the drawing.

What the Z80 computes for one digit, in the order it computes it:

    Layer 1   32 cells, one per hidden neuron: its value h_q (0..127),
              black = 0 (cut by ReLU) to white = the strongest of this digit
              (the values are small next to 127). Hover: the sum (acc1),
              ReLU and the rescaled value. Click: on_neuron(j), the debug
              panel then shows that neuron's 784 weights as a 28x28 image.
    Layer 2   the 10 scores (acc2), the highest in green: the answer.

The headline says what happens now, and then the answer, in large type: with
this view DrawingApp does not show its own result label.

The values come from the D reply (z80nn.Z80NN.classify_debug): set_result()
stores them in the worker thread, refresh() shows them in Tk's thread. In the
live mode the debug panel calls show_step() 20 times a second with
z80debug.infer_step(): the neuron being computed is outlined in the "read"
colour, the ones already done turn grey (their values arrive with the reply).
Outside nn_infer the step is None: before layer 1 the Z80 is unpacking the
image ("before"), after layer 2 it writes the 74 results as decimal text and
sends them ("send": with 32-bit divisions, about a third of the time).

DrawingApp calls start(x) when a digit goes to the Z80, refresh() after the
answer, clear() on Clear and after Stop.
"""

import tkinter as tk
from tkinter import font as tkfont

from z80debug import N_HID, N_IN, N_OUT

WIDTH = 560                         # px, about the drawing + preview
CELL, CELL_GAP = 16, 1              # hidden neuron cells
CELL_H = 26
ROW_H = 18                          # score rows
CURRENT = "#c0c000"                 # being computed: the "read" colour of the memory map
DONE = "#808080"                    # computed, value not back yet
EMPTY = "#909090"                   # outline of a cell not computed yet
BEST, OTHER = "#20a040", "#9090a0"  # score bars
GREY_TEXT = "#808080"


def neuron_fill(h, top):
    """h_q 0..top -> grey from almost black to almost white."""
    v = 28 + (240 - 28) * max(0, min(top, h)) // max(1, top)
    return "#%02x%02x%02x" % (v, v, v)


class NetworkView(tk.Frame):

    def __init__(self, master, on_neuron=None):
        super().__init__(master)
        self.on_neuron = on_neuron
        self.values = None          # (acc1, h_q, acc2, digit) shown
        self.pending = None         # set by set_result() in the worker thread
        self.running = False        # between start() and refresh()
        self.step = None            # z80debug.infer_step() while running (live only)
        self.lit = 0                # lit pixels of the digit being computed
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
        # two parts, for focus.HoverFocus: the headline goes with layer 1
        l1 = tk.Frame(self)
        l1.pack(fill="x")
        self.headline = tk.Label(l1, anchor="w", justify="left", wraplength=WIDTH, font=self.bold)
        self.headline.pack(fill="x", pady=(0, 4))
        tk.Label(l1, text="Layer 1 · 32 hidden neurons", anchor="w", font=self.bold).pack(fill="x")
        tk.Label(l1, anchor="w", justify="left", wraplength=WIDTH, font=self.caption, fg=GREY_TEXT,
                 text="Each neuron adds up its weights at the lit pixels (it has one weight per pixel); "
                      "ReLU turns a negative sum into 0, then it is rescaled to 0..127. "
                      "Black = 0, white = the strongest. Click one: its weights in Memory.").pack(fill="x")
        self.hidden = tk.Canvas(l1, width=WIDTH, height=CELL_H + 16, highlightthickness=0,
                                cursor="hand2")
        self.hidden.pack(anchor="w", pady=(4, 0))
        self.hover = tk.Label(l1, anchor="w", font=self.caption, width=1)     # never widens the window
        self.hover.pack(fill="x")
        self.text_fg = self.hover.cget("fg")
        self.hidden.bind("<Motion>", self._on_motion)
        self.hidden.bind("<Leave>", lambda _e: self._show_hover(None))
        self.hidden.bind("<Button-1>", self._on_click)

        l2 = tk.Frame(self)
        l2.pack(fill="x", pady=(6, 0))
        tk.Label(l2, text="Layer 2 · 10 scores, one per digit", anchor="w", font=self.bold).pack(fill="x")
        tk.Label(l2, anchor="w", justify="left", wraplength=WIDTH, font=self.caption, fg=GREY_TEXT,
                 text="Each score adds 32 weights × the 32 hidden values. "
                      "The highest score is the answer.").pack(fill="x")
        self.scores = tk.Canvas(l2, width=WIDTH, height=N_OUT * ROW_H + 4, highlightthickness=0)
        self.scores.pack(anchor="w", pady=(4, 0))
        self.parts = [l1, l2]

    def focus_parts(self):
        """The parts focus.HoverFocus can put in focus: layer 1 (with the headline), layer 2."""
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
        """The D reply. May be called in the worker thread: only stores it."""
        self.pending = (list(acc1), list(h_q), list(acc2), digit)

    def refresh(self):
        """After the answer: show the values of the reply."""
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
        """Live: where nn_infer is (z80debug.infer_step), or None outside it."""
        if not self.running:
            return
        if step is None:            # outside nn_infer: before it, or after it
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
            return "Draw a digit and press Recognize: the Z80 computes the two layers below."
        kind, n = self.step or ("", None)
        if kind == "layer1":
            return f"Layer 1, neuron #{n}: adding its weights at the {self.lit} lit pixels"
        if kind == "layer2":
            return f"Layer 2, digit {n}: its 32 weights × the hidden values"
        if kind == "trace":
            return "Network done: copying the results into trace"
        if kind == "send":
            return "Network done: writing the 74 results as text for the laptop"
        if kind == "before":
            return f"Unpacking the image it received: {N_IN // 8} bytes → {N_IN} pixels"
        return "The Z80 is computing…"

    def _progress(self):
        """(hidden neurons done, current one or None, current score or None) while running."""
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
                text = f"{active} of 32 neurons active; the other {32 - active} are 0 after ReLU"
            else:
                text = ""
            self.hover.config(text=text, fg=GREY_TEXT)
            return
        if self.values is not None:
            s, h = self.values[0][j], self.values[1][j]
            text = f"Neuron #{j}: sum {s:,} → ReLU {max(0, s):,} → rescaled {h}"
        else:
            text = f"Neuron #{j}"
        self.hover.config(text=text + f" · click: its {N_IN} weights in Memory", fg=self.text_fg)

    def _on_motion(self, event):
        self._show_hover(self._neuron_at(event.x))

    def _on_click(self, event):
        j = self._neuron_at(event.x)
        if j is not None and self.on_neuron is not None:
            self.on_neuron(j)
