"""
debug_panel.py - DeZog-style debug view, shown next to the drawing window.

DebugPanel shows one z80debug.Snapshot:

    Visual memory   0x0000-0xFFFF as a bar, 128 bytes per pixel, in the colours
                    of DeZog's zsim: read yellow, write red, prog blue. Under it,
                    the regions of the map file with their start address.
                    Hover: address and symbol. Click: show that address below.
    Registers       red = changed since the previous snapshot; flags decoded
    Call stack      PC, then the return addresses of the IX frame chain
    Stack           words from SP; code addresses named, frame returns in blue
    Memory          16 x 16 bytes + ASCII from a variable or an address,
                    each byte coloured by its access
    Scores          the 10 output scores (acc2), the winner in green

Two modes, from the source:
    snapshot  (SnapshotSource) after each result. The panel never talks to the
              Z80 itself: it calls capture(view_addr) through run_in_background,
              so the serial line is used by one thread only.
    live      (LiveSource, patched ticks) every LIVE_MS in Tk's thread, also while
              the Z80 works. Accesses fade out in about half a second; the speed
              of the emulated clock and Pause are under the memory view.
"""

import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from z80debug import PROG, READ, VARIABLES, VIEW_BYTES, WRITE, flags

BAR_W, BAR_H = 512, 18
BYTES_PER_PX = 65536 // BAR_W           # 128
LABEL_ROW = 15                          # px per row of region labels
BAR_BG = (40, 40, 40)
# DeZog zsim visual memory palette (VISUAL_MEM_COL_READ/WRITE/PROG)
RGB = {READ: (192, 192, 0), WRITE: (192, 0, 0), PROG: (0, 0, 192)}
HEX = {k: "#%02x%02x%02x" % v for k, v in RGB.items()}
PALETTE = np.zeros((PROG + 1, 3), dtype=np.float32)   # access kind -> colour
PALETTE[:] = BAR_BG
for _kind, _rgb in RGB.items():
    PALETTE[_kind] = _rgb
LIVE_MS = 50                            # live refresh period
FADE = 0.8                              # per refresh: 0.8 ** 10 = 0.1 after 0.5 s
SHOWN = 0.25                            # memory view colours a byte while age > SHOWN
SPEEDS = [("100 kHz", 100_000), ("1 MHz", 1_000_000), ("3.5 MHz", 3_500_000),
          ("7.37 MHz (board)", 7_372_800), ("max", 0)]
REG_ROWS = [("AF", "af"), ("BC", "bc"), ("DE", "de"), ("HL", "hl"),
            ("IX", "ix"), ("IY", "iy"), ("SP", "sp"), ("PC", "pc"),
            ("AF'", "af2"), ("BC'", "bc2"), ("DE'", "de2"), ("HL'", "hl2")]


def access_kinds(activity):
    """Access bits -> the kind whose colour wins: write, then prog, then read (0 = none)."""
    return np.where(activity & WRITE, WRITE,
                    np.where(activity & PROG, PROG, np.where(activity & READ, READ, 0))).astype(np.uint8)


def bar_image(kind, age):
    """Per-address kind and age (1 = just now, 0 = faded) -> BAR_W x BAR_H PhotoImage.

    Each pixel covers 128 bytes and shows its most recent access.
    """
    ages = age.reshape(BAR_W, BYTES_PER_PX)
    newest = ages.argmax(axis=1)
    a = ages[np.arange(BAR_W), newest][:, None]
    k = kind.reshape(BAR_W, BYTES_PER_PX)[np.arange(BAR_W), newest]
    rgb = PALETTE[0] + (PALETTE[k] - PALETTE[0]) * a
    img = Image.fromarray(rgb.astype(np.uint8).reshape(1, BAR_W, 3), "RGB")
    return ImageTk.PhotoImage(img.resize((BAR_W, BAR_H), Image.NEAREST))


class DebugPanel(tk.Frame):

    def __init__(self, master, symbols, capture, run_in_background, note="", live=None):
        """live: a z80debug.LiveSource for the live mode (speed and pause), or None."""
        super().__init__(master)
        self.symbols = symbols
        self.capture = capture
        self.run_in_background = run_in_background
        self.live = live
        self.view_addr = symbols["_x"] if "_x" in symbols.addr else symbols.data[0]
        self.snap = None
        self.prev_regs = None
        self.busy = False
        self.kind = np.zeros(65536, dtype=np.uint8)     # last access kind per address
        self.age = np.zeros(65536, dtype=np.float32)    # 1 = just accessed, fades to 0
        self.labels_key = None
        self.last_rate = None                           # (wall time, T-states) for the MHz
        self.mono = tkfont.nametofont("TkFixedFont")
        self.small = tkfont.Font(family=self.mono.actual("family"), size=9)
        self._build(note)
        if live is not None:
            self.after(LIVE_MS, self._live_tick)

    # --- layout ---------------------------------------------------------------

    def _build(self, note):
        bar = tk.LabelFrame(self, text="Visual memory")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.bar = tk.Canvas(bar, width=BAR_W, height=BAR_H + 4 * LABEL_ROW,
                             highlightthickness=0, cursor="crosshair")
        self.bar.pack(padx=4, pady=(4, 0))
        legend = tk.Frame(bar)
        legend.pack(fill="x", padx=4, pady=2)
        for kind, name in ((READ, "read"), (WRITE, "write"), (PROG, "prog")):
            tk.Label(legend, text="  ", bg=HEX[kind]).pack(side="left")
            tk.Label(legend, text=name).pack(side="left", padx=(2, 10))
        self.hover = tk.Label(legend, font=self.small, anchor="e")
        self.hover.pack(side="right")
        self.bar.bind("<Motion>", self._on_bar_motion)
        self.bar.bind("<Leave>", lambda _e: self.hover.config(text=""))
        self.bar.bind("<Button-1>", self._on_bar_click)

        left = tk.Frame(self)
        left.grid(row=1, column=0, sticky="nw", padx=(0, 8))
        regs = tk.LabelFrame(left, text="Registers")
        regs.pack(fill="x")
        self.reg_labels = {}
        for k, (title, key) in enumerate(REG_ROWS):
            r, c = k % 4, k // 4
            tk.Label(regs, text=title, font=self.mono).grid(row=r, column=2 * c, sticky="e", padx=(6, 2))
            self.reg_labels[key] = tk.Label(regs, text="----", font=self.mono)
            self.reg_labels[key].grid(row=r, column=2 * c + 1, sticky="w")
        self.ir_label = tk.Label(regs, font=self.mono, anchor="w")
        self.ir_label.grid(row=4, column=0, columnspan=6, sticky="w", padx=6)
        self.flag_label = tk.Label(regs, font=self.mono, anchor="w")
        self.flag_label.grid(row=5, column=0, columnspan=6, sticky="w", padx=6, pady=(0, 4))

        calls = tk.LabelFrame(left, text="Call stack")
        calls.pack(fill="x", pady=(6, 0))
        self.calls = tk.Text(calls, width=30, height=4, font=self.mono, relief="flat",
                             bg=self.cget("bg"), state="disabled")
        self.calls.pack(padx=4, pady=2)

        stack = tk.LabelFrame(left, text="Stack")
        stack.pack(fill="x", pady=(6, 0))
        self.stack = tk.Text(stack, width=30, height=9, font=self.mono, state="disabled")
        scroll = tk.Scrollbar(stack, command=self.stack.yview)
        self.stack.config(yscrollcommand=scroll.set)
        self.stack.pack(side="left", padx=(4, 0), pady=4)
        scroll.pack(side="left", fill="y", pady=4)
        self.stack.tag_config("ret", foreground="#2060d0")
        self.stack.tag_config("w", foreground=HEX[WRITE])

        right = tk.Frame(self)
        right.grid(row=1, column=1, sticky="nw")
        mem = tk.LabelFrame(right, text="Memory")
        mem.pack(fill="x")
        nav = tk.Frame(mem)
        nav.pack(fill="x", padx=4, pady=2)
        choices = [n.lstrip("_") for n in VARIABLES if n in self.symbols.addr] + ["SP", "PC"]
        self.goto = ttk.Combobox(nav, values=choices, width=9, state="readonly")
        self.goto.pack(side="left")
        self.goto.bind("<<ComboboxSelected>>", self._on_goto)
        self.addr_entry = tk.Entry(nav, width=6, font=self.mono)
        self.addr_entry.pack(side="left", padx=4)
        self.addr_entry.bind("<Return>", self._on_addr_entry)
        tk.Button(nav, text="Refresh", command=self.refresh).pack(side="right")
        self.mem = tk.Text(mem, width=6 + 3 * 16 + 17, height=16, font=self.mono, state="disabled")
        self.mem.pack(padx=4, pady=(0, 4))
        self.mem.tag_config("addr", foreground="#808080")
        for kind in (READ, WRITE, PROG):
            self.mem.tag_config(kind, background=HEX[kind], foreground="white" if kind != READ else "black")

        scores = tk.LabelFrame(right, text="Scores (acc2)")
        scores.pack(fill="x", pady=(6, 0))
        self.scores = tk.Canvas(scores, width=6 + 3 * 16 * self.mono.measure("0"), height=10 * 14 + 4,
                                highlightthickness=0)
        self.scores.pack(padx=4, pady=2)

        if self.live is not None:
            run = tk.Frame(right)
            run.pack(fill="x", pady=(6, 0))
            tk.Label(run, text="Clock").pack(side="left")
            self.speed = ttk.Combobox(run, values=[n for n, _ in SPEEDS], width=15, state="readonly")
            current = [hz for _, hz in SPEEDS].index(self.live.speed) if self.live.speed in \
                [hz for _, hz in SPEEDS] else len(SPEEDS) - 1
            self.speed.current(current)
            self.speed.bind("<<ComboboxSelected>>", self._on_speed)
            self.speed.pack(side="left", padx=4)
            self.pause_button = tk.Button(run, text="Pause", width=7, command=self._on_pause)
            self.pause_button.pack(side="left", padx=4)
            self.clock = tk.Label(run, font=self.small, anchor="e")
            self.clock.pack(side="right")

        self.status = tk.Label(self, text=note, font=self.small, anchor="w", justify="left")
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self._draw_labels(None)

    # --- refresh --------------------------------------------------------------

    def refresh(self):
        """Capture in the background (it talks to the Z80), then show. Live: now."""
        if self.live is not None:
            self.show(self.capture(self.view_addr))
            return
        if self.busy:
            return
        self.busy = True
        addr = self.view_addr
        self.run_in_background(lambda: self.capture(addr), self._on_captured)

    def _on_captured(self, snap, error):
        self.busy = False
        if error is not None:
            self.status.config(text=f"capture failed: {error}")
            return
        self.show(snap)

    def _live_tick(self):
        try:
            self.show(self.capture(self.view_addr))
        finally:
            self.after(LIVE_MS, self._live_tick)

    def _on_speed(self, _event):
        self.live.speed = SPEEDS[self.speed.current()][1]
        self.last_rate = None

    def _on_pause(self):
        self.live.paused = not self.live.paused
        self.pause_button.config(text="Resume" if self.live.paused else "Pause")
        self.last_rate = None

    def _update_access(self, activity, live):
        """Fold new access bits into kind/age. Live: older accesses fade. Snapshot: only these."""
        new = access_kinds(activity)
        hit = new != 0
        if live:
            self.age *= FADE
        else:
            self.age[:] = 0
        self.age[hit] = 1.0
        self.kind[hit] = new[hit]

    def _accessed(self, addr):
        """The kind to colour addr with, or 0."""
        a = addr & 0xFFFF
        return int(self.kind[a]) if self.age[a] > SHOWN else 0

    def show(self, snap):
        self.snap = snap
        self._update_access(snap.activity, snap.live)
        self.bar_photo = bar_image(self.kind, self.age)    # keep a reference for Tk
        self.bar.delete("bar")
        self.bar.create_image(0, 0, image=self.bar_photo, anchor="nw", tags="bar")
        self.bar.tag_lower("bar")
        self._draw_labels(snap)
        self._show_regs(snap.regs)
        self._show_calls(snap.frames)
        self._show_stack(snap)
        self._show_memory(snap)
        self._show_scores(snap.acc2)
        if snap.tstates is not None and self.live is not None:
            self._show_clock(snap.tstates)

    def _show_clock(self, tstates):
        now = time.monotonic()
        text = f"T-states {tstates:,}"
        if self.last_rate is not None and now - self.last_rate[0] > 0.4:
            mhz = (tstates - self.last_rate[1]) / (now - self.last_rate[0]) / 1e6
            self.rate_text = f"   {mhz:.2f} MHz"
            self.last_rate = (now, tstates)
        elif self.last_rate is None:
            self.last_rate = (now, tstates)
            self.rate_text = ""
        self.clock.config(text=text + getattr(self, "rate_text", ""))

    def _draw_labels(self, snap):
        """Region brackets under the bar, packed into rows so texts do not overlap."""
        c = self.bar
        key = None if snap is None else (snap.regs["sp"], snap.regs["pc"] // BYTES_PER_PX, snap.view_addr)
        if key is not None and key == self.labels_key:
            return                      # live: nothing moved, keep the canvas items
        self.labels_key = key
        c.delete("label")
        if snap is None:
            c.create_rectangle(0, 0, BAR_W, BAR_H, fill="#%02x%02x%02x" % BAR_BG, width=0, tags="label")
        regions = self.symbols.regions(snap.regs["sp"] if snap else None)
        row_end = []                    # per row: x where the last text ends
        first_var_row = None
        for label, start, end, level in sorted(regions, key=lambda r: (r[3], r[1])):
            if level == 1 and first_var_row is None:
                first_var_row = len(row_end)
            x0 = start // BYTES_PER_PX
            x1 = max(x0 + 2, end // BYTES_PER_PX)
            text = f"{label} {start:04X}"
            tx = min(x0, BAR_W - self.small.measure(text))
            lowest = first_var_row if level == 1 else 0
            row = next((r for r in range(lowest, len(row_end)) if row_end[r] + 6 <= min(x0, tx)), None)
            if row is None:
                row = len(row_end)
                row_end.append(0)
            row_end[row] = max(x1, tx + self.small.measure(text))
            y = BAR_H + 3 + row * LABEL_ROW
            colour = "#a0a0a0" if level == 0 else "#606060"
            c.create_line(x0, y, x0, y + 4, x1, y + 4, x1, y, fill=colour, tags="label")
            c.create_text(tx, y + 4, text=text, anchor="nw", font=self.small, tags="label")
        if snap is not None:
            x = snap.regs["pc"] // BYTES_PER_PX
            c.create_line(x, 0, x, BAR_H, fill="white", width=1, tags="label")
            v0 = snap.view_addr // BYTES_PER_PX
            c.create_rectangle(v0, 0, v0 + max(1, VIEW_BYTES // BYTES_PER_PX), BAR_H,
                               outline="#00c000", tags="label")
        c.config(height=BAR_H + 4 + max(1, len(row_end)) * LABEL_ROW + 6)

    def _show_regs(self, regs):
        for key, label in self.reg_labels.items():
            changed = self.prev_regs is not None and self.prev_regs[key] != regs[key]
            label.config(text=f"{regs[key]:04X}", fg="red" if changed else "black")
        self.ir_label.config(text=f"I {regs['i']:02X}   R {regs['r']:02X}")
        self.flag_label.config(text=f"F  {flags(regs['af'])}   F' {flags(regs['af2'])}")
        self.prev_regs = regs

    def _set_text(self, widget, lines):
        """lines: list of [(text, tags), ...]."""
        widget.config(state="normal")
        widget.delete("1.0", "end")
        for k, parts in enumerate(lines):
            for text, tags in parts:
                widget.insert("end", text, tags)
            if k < len(lines) - 1:
                widget.insert("end", "\n")
        widget.config(state="disabled")

    def _show_calls(self, frames):
        self._set_text(self.calls, [[(f"{a:04X}  {name or '?'}", ())] for a, name in frames])

    def _show_stack(self, snap):
        rets = {a for a, _ in snap.frames[1:]}
        lines = []
        for addr, word in snap.stack:
            name = self.symbols.nearest(word) if self.symbols.in_code(word) else ""
            changed = WRITE in (self._accessed(addr), self._accessed(addr + 1))
            tags = ("ret",) if word in rets else ("w",) if changed else ()
            lines.append([(f"{addr & 0xFFFF:04X}  {word:04X}  {name or ''}", tags)])
        self._set_text(self.stack, lines)

    def _show_memory(self, snap):
        lines = []
        for row in range(0, VIEW_BYTES, 16):
            base = (snap.view_addr + row) & 0xFFFF
            parts = [(f"{base:04X}  ", ("addr",))]
            chunk = snap.view[row:row + 16]
            for k, b in enumerate(chunk):
                kind = self._accessed(base + k)
                parts.append((f"{b:02X}", (kind,) if kind else ()))
                parts.append((" ", ()))
            parts.append((" " + "".join(chr(b) if 32 <= b < 127 else "." for b in chunk), ()))
            lines.append(parts)
        self._set_text(self.mem, lines)
        if self.focus_get() is not self.addr_entry:     # do not overwrite while typing
            self.addr_entry.delete(0, "end")
            self.addr_entry.insert(0, f"{snap.view_addr:04X}")

    def _show_scores(self, acc2):
        c = self.scores
        c.delete("all")
        if not acc2:
            return
        lo, hi = min(acc2 + [0]), max(acc2 + [0])
        span = max(1, hi - lo)
        best = int(np.argmax(acc2))
        x_name, x0, width = 4, 20, int(c.cget("width")) - 110
        zero = x0 + (0 - lo) * width // span
        for d, v in enumerate(acc2):
            y = 2 + d * 14
            a, b = sorted((zero, x0 + (v - lo) * width // span))
            c.create_text(x_name, y, text=str(d), anchor="nw", font=self.mono)
            c.create_rectangle(a, y + 2, max(a + 1, b), y + 11, width=0,
                               fill="#20a040" if d == best else "#9090a0")
            c.create_text(x0 + width + 8, y, text=f"{v:,}", anchor="nw", font=self.small)
        c.create_line(zero, 0, zero, 10 * 14 + 2, fill="#606060")

    # --- navigation -----------------------------------------------------------

    def set_view(self, addr):
        self.view_addr = addr & 0xFFFF
        self.refresh()

    def _on_goto(self, _event):
        choice = self.goto.get()
        if choice in ("SP", "PC"):
            if self.snap is not None:
                self.set_view(self.snap.regs[choice.lower()])
        else:
            self.set_view(self.symbols["_" + choice])

    def _on_addr_entry(self, _event):
        try:
            self.set_view(int(self.addr_entry.get(), 16))
        except ValueError:
            self.addr_entry.delete(0, "end")

    def _bar_addr(self, event):
        if not 0 <= event.x < BAR_W:
            return None
        return event.x * BYTES_PER_PX

    def _on_bar_motion(self, event):
        addr = self._bar_addr(event)
        if addr is None:
            return
        name = self.symbols.nearest(addr)
        if name is None and self.snap is not None and self.snap.regs["sp"] <= addr + BYTES_PER_PX - 1 \
                and addr < self.symbols.stack_top:
            name = "stack"
        text = f"{addr:04X}-{addr + BYTES_PER_PX - 1:04X}  {name or ''}"
        recent = self.kind[addr:addr + BYTES_PER_PX][self.age[addr:addr + BYTES_PER_PX] > SHOWN]
        text += "  " + "".join(n for k, n in ((READ, "R"), (WRITE, "W"), (PROG, "P")) if (recent == k).any())
        self.hover.config(text=text)

    def _on_bar_click(self, event):
        addr = self._bar_addr(event)
        if addr is not None:
            self.set_view(addr & 0xFFF0)
