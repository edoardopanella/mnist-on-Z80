"""
debug_panel.py - DeZog-style debug view, shown next to the drawing window.

DebugPanel shows one z80debug.Snapshot:

    Memory map      the 64 KB as named blocks, each with a strip that lights
                    up on access (DeZog colours: read yellow, write red,
                    code blue). Click a block to open it in Memory.
    Registers       red if changed since the last snapshot
    Call stack      PC, then the return addresses found through IX
    Stack           words from SP, return addresses in blue
    Memory          any 256 bytes as hex, or as a 28x28 image

Snapshot mode refreshes after each result, through run_in_background so only
one thread talks to the Z80. Live mode refreshes every LIVE_MS in Tk's thread,
also while the Z80 works, and adds clock speed and Pause controls.
"""

import math
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from z80debug import (IMG_W, N_HID, N_IN, PROG, READ, VARIABLES, VIEW_BYTES, WRITE, flags, image_bytes,
                      image_rows, infer_step)

MAP_W = 720                             # memory map width, px
TITLE_H, STRIP_H, PC_H = 32, 22, 9      # heights: block titles and sizes, access strips, PC marker
GAP = 4                                 # px between blocks
FIXED_W = {"Free": 56, "Stack": 72}     # fixed width, their size is not interesting
BAR_BG = (40, 40, 40)
STRIP_KINDS = (WRITE, PROG, READ)       # on equal ages, the first one wins
# What the Z80 is doing, by the innermost of these functions on the call stack
DOING = {"_nn_infer": "classifying the digit", "_read_line": "waiting for the laptop",
         "_cmd_write": "receiving the weights", "_cmd_checksum": "checking the weights",
         "_hex_bytes": "decoding the hex text it received", "_hex_digit": "decoding the hex text it received",
         "_put_i32": "writing the results as decimal text", "_put_list": "writing the results as decimal text",
         "_put_str": "sending text to the laptop",
         "_cmd_infer": "handling the image", "_cmd_regs": "sending its registers",
         "_cmd_memory": "sending memory to this panel",
         "SYSCALL": "talking to the laptop"}    # ticks' console trap, has no IX frame
# DeZog zsim visual memory palette (VISUAL_MEM_COL_READ/WRITE/PROG)
RGB = {READ: (192, 192, 0), WRITE: (192, 0, 0), PROG: (0, 0, 192)}
HEX = {k: "#%02x%02x%02x" % v for k, v in RGB.items()}
PALETTE = np.zeros((PROG + 1, 3), dtype=np.float32)   # access kind -> colour
PALETTE[:] = BAR_BG
for _kind, _rgb in RGB.items():
    PALETTE[_kind] = _rgb
LIVE_MS = 50                            # live refresh period
FADE = 0.8                              # per refresh, so 0.1 after 0.5 s
SHOWN = 0.25                            # a byte stays coloured while age > SHOWN
BOARD_HZ = 7_372_800                    # the board's crystal
IMG_LABEL_W = 44                        # image view: room for the row addresses
IMG_LIT, IMG_DARK, IMG_GRID = (240, 240, 240), (28, 28, 28), (60, 60, 60)
IMG_POS, IMG_NEG = (255, 150, 40), (70, 150, 255)     # signed bytes: positive orange, negative blue
HEX_ROW = 16                            # bytes per row of the hex view
GREY = "#808080"                        # captions and other secondary text
SPEEDS = [("100 kHz", 100_000), ("1 MHz", 1_000_000), ("3.5 MHz", 3_500_000),
          ("7.37 MHz (board)", BOARD_HZ), ("max", 0)]
REG_ROWS = [("AF", "af"), ("BC", "bc"), ("DE", "de"), ("HL", "hl"),
            ("IX", "ix"), ("IY", "iy"), ("SP", "sp"), ("PC", "pc"),
            ("AF'", "af2"), ("BC'", "bc2"), ("DE'", "de2"), ("HL'", "hl2")]


def access_kinds(activity):
    """Access bits to a single kind: write beats prog, prog beats read. 0 if none."""
    return np.where(activity & WRITE, WRITE,
                    np.where(activity & PROG, PROG, np.where(activity & READ, READ, 0))).astype(np.uint8)


def layout_blocks(blocks, min_width):
    """Add pixel columns to the blocks: [(title, start, end, x0, x1)] across MAP_W.

    Free and Stack get a fixed width. The others get min_width(title) plus
    a share of the rest by the square root of their size, so small variables
    stay readable.
    """
    avail = MAP_W - GAP * (len(blocks) - 1)
    base = [FIXED_W.get(t, min_width(t)) for t, _, _ in blocks]
    weight = [0.0 if t in FIXED_W else math.sqrt(e - s) for t, s, e in blocks]
    extra = max(0, avail - sum(base))
    widths = [b + int(extra * w / (sum(weight) or 1)) for b, w in zip(base, weight)]
    widths[weight.index(max(weight))] += max(0, avail - sum(widths))
    out, x = [], 0
    for (title, start, end), w in zip(blocks, widths):
        out.append((title, start, end, x, x + w))
        x += w + GAP
    return out


def strip_row(kind, age, layout, gap_rgb):
    """The access strip as (MAP_W, 3) floats. Each pixel shows the newest access among its bytes."""
    row = np.tile(np.asarray(gap_rgb, dtype=np.float32), (MAP_W, 1))
    for _, start, end, x0, x1 in layout:
        w = x1 - x0
        edges = (np.arange(w) * (end - start)) // w
        k, a = kind[start:end], age[start:end]
        per = np.stack([np.maximum.reduceat(np.where(k == kk, a, 0), edges) for kk in STRIP_KINDS])
        newest = per.argmax(axis=0)
        row[x0:x1] = PALETTE[0] + (PALETTE[np.array(STRIP_KINDS)[newest]] - PALETTE[0]) \
            * per.max(axis=0)[:, None]
    return row


def x_to_addr(layout, x):
    """(block, first address under pixel x), or None between blocks."""
    for block in layout:
        _, start, end, x0, x1 = block
        if x0 <= x < x1:
            return block, start + (x - x0) * (end - start) // (x1 - x0)
    return None


def addr_to_x(layout, addr):
    for _, start, end, x0, x1 in layout:
        if start <= addr < end:
            return x0 + (addr - start) * (x1 - x0) // (end - start)
    return None


def size_text(n):
    return f"{n / 1024:.1f} KB" if n >= 1024 else f"{n} B"


class DebugPanel(tk.Frame):

    def __init__(self, master, symbols, capture, run_in_background, note="", live=None, on_step=None):
        """live is the z80debug.LiveSource in live mode, else None.

        on_step(step), if given, gets the latest z80debug.infer_step() at every capture.
        """
        super().__init__(master)
        self.symbols = symbols
        self.capture = capture
        self.run_in_background = run_in_background
        self.live = live
        self.on_step = on_step
        self.step = None                                # last infer_step() that read weights
        self.view_addr = symbols["_x"] if "_x" in symbols.addr else symbols.data[0]
        self.snap = None
        self.prev_regs = None
        self.busy = False
        self.kind = np.zeros(65536, dtype=np.uint8)     # last access kind per address
        self.age = np.zeros(65536, dtype=np.float32)    # 1 when just accessed, fades to 0
        self.map_key = None                             # blocks the static items were drawn for
        self.layout = []                                # layout_blocks() of the current blocks
        self.title_font = tkfont.nametofont("TkDefaultFont")
        self.last_rate = None                           # (wall time, T-states) for the MHz
        self.stopping = False                           # between fast_forward() and restore_clock()
        self.t_base = None                              # T-states when the current digit started
        self.text_cache = {}                            # widget -> lines shown, to skip rewrites
        self.img_rows = None                            # image view: (addr, value, offset, bits) shown
        self.mono = tkfont.nametofont("TkFixedFont")
        self.small = tkfont.Font(family=self.mono.actual("family"), size=self.mono.actual("size") - 1)
        self.caption_font = self.title_font.copy()
        self.caption_font.config(size=max(9, self.title_font.actual("size") - 2))
        self._build(note)
        if live is not None:
            self.after(LIVE_MS, self._live_tick)

    # --- layout ---------------------------------------------------------------

    def _caption(self, parent, text):
        """A grey caption that wraps to its parent's width. The caller places it."""
        # width=1 so the caption never widens the panel
        label = tk.Label(parent, text=text, font=self.caption_font, fg=GREY, anchor="nw",
                         justify="left", width=1)
        label.bind("<Configure>", lambda e: e.widget.config(wraplength=max(50, e.width - 4)))
        return label

    def _build(self, note):
        bar = tk.LabelFrame(self, text="Memory map")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._caption(bar, "The Z80's 64 KB in address order, split by what the program keeps there. "
                           "A block lights up when the Z80 reads it, writes it or runs code from it. "
                           "Hover for addresses, click to open one in Memory.").pack(fill="x", padx=4)
        self.bar = tk.Canvas(bar, width=MAP_W, height=TITLE_H + STRIP_H + PC_H,
                             highlightthickness=0, cursor="crosshair")
        self.bar.pack(padx=4, pady=(4, 0))
        self.gap_rgb = [v // 257 for v in self.winfo_rgb(self.bar.cget("bg"))]
        # width=1 on lines that change 20 times a second, so they get cut off
        # instead of resizing the window (and moving the focus veils)
        self.now = tk.Label(bar, anchor="w", justify="left", width=1)
        self.now.pack(fill="x", padx=4)
        legend = tk.Frame(bar)
        legend.pack(fill="x", padx=4, pady=2)
        for kind, name in ((READ, "read"), (WRITE, "written"), (PROG, "code running")):
            tk.Label(legend, text="  ", bg=HEX[kind]).pack(side="left")
            tk.Label(legend, text=name).pack(side="left", padx=(2, 10))
        self.hover = tk.Label(legend, font=self.small, anchor="e")
        self.hover.pack(side="right")
        self.bar.bind("<Motion>", self._on_bar_motion)
        self.bar.bind("<Leave>", lambda _e: self.hover.config(text=""))
        self.bar.bind("<Button-1>", self._on_bar_click)

        left = tk.Frame(self)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
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
        self.flag_label.grid(row=5, column=0, columnspan=6, sticky="w", padx=6)
        self._caption(regs, "The CPU's own variables. PC is the next instruction, SP the top of the "
                            "stack, F the flags of the last operation. Red means just changed.").grid(
            row=6, column=0, columnspan=6, sticky="ew", padx=2, pady=(2, 4))

        calls = tk.LabelFrame(left, text="Call stack")
        calls.pack(fill="both", expand=True, pady=(6, 0))
        self._caption(calls, "The function running now (top) and its callers.").pack(
            fill="x", padx=2)
        self.calls = tk.Text(calls, width=30, height=4, font=self.mono, relief="flat",
                             bg=self.cget("bg"), state="disabled", wrap="none")
        self.calls.pack(fill="both", expand=True, padx=4, pady=2)

        stack = tk.LabelFrame(self, text="Stack")
        stack.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(6, 0))
        self._caption(stack, "From SP up. Return addresses in blue, the rest are saved values.").pack(
            fill="x", padx=2)
        stack_body = tk.Frame(stack)
        stack_body.pack(fill="x")
        self.stack = tk.Text(stack_body, width=30, height=7, font=self.mono, state="disabled", wrap="none")
        scroll = tk.Scrollbar(stack_body, command=self.stack.yview)
        self.stack.config(yscrollcommand=scroll.set)
        self.stack.pack(side="left", padx=(4, 0), pady=4)
        scroll.pack(side="left", fill="y", pady=4)
        self.stack.tag_config("ret", foreground="#2060d0")
        self.stack.tag_config("w", foreground=HEX[WRITE])

        right = tk.Frame(self)
        right.grid(row=1, column=1, sticky="nsew")
        mem = tk.LabelFrame(right, text="Memory")
        mem.pack(fill="x")
        self._caption(mem, "Any part of the 64 KB. As an image, x is the digit the network reads, "
                           "and each 28×28 of weights belongs to one neuron.").pack(fill="x", padx=4)
        nav = tk.Frame(mem)
        nav.pack(fill="x", padx=4, pady=(2, 0))
        tk.Label(nav, text="Go to").pack(side="left")
        choices = [n.lstrip("_") for n in VARIABLES if n in self.symbols.addr] + ["SP", "PC"]
        self.goto = ttk.Combobox(nav, values=choices, width=9, state="readonly")
        self.goto.pack(side="left", padx=(4, 10))
        self.goto.bind("<<ComboboxSelected>>", self._on_goto)
        tk.Label(nav, text="Address").pack(side="left")
        self.addr_entry = tk.Entry(nav, width=6, font=self.mono)
        self.addr_entry.pack(side="left", padx=4)
        self.addr_entry.bind("<Return>", self._on_addr_entry)
        tk.Label(nav, text="hex, Enter", font=self.small, fg="#808080").pack(side="left")
        ttk.Button(nav, text="Refresh", command=self.refresh).pack(side="right")
        opts = tk.Frame(mem)
        opts.pack(fill="x", padx=4, pady=(2, 2))
        tk.Label(opts, text="View").pack(side="left")
        self.view_mode = tk.StringVar(value="hex")
        for value, text in (("hex", "Hex"), ("image", "Image 28×28")):
            ttk.Radiobutton(opts, text=text, value=value, variable=self.view_mode,
                            command=self._on_view_mode).pack(side="left", padx=(4, 0))
        tk.Label(opts, text="Pixel").pack(side="left", padx=(12, 0))
        self.pixel = tk.StringVar(value="byte")
        self.pixel_buttons = []
        for value, text in (("byte", "1 byte"), ("bit", "1 bit")):
            b = ttk.Radiobutton(opts, text=text, value=value, variable=self.pixel,
                                command=self._on_pixel, state="disabled")
            b.pack(side="left", padx=(4, 0))
            self.pixel_buttons.append(b)
        body = tk.Frame(mem)
        body.pack(fill="x", padx=4)
        # width=1 so the hint never widens the panel
        self.mem_hint = tk.Label(mem, font=self.small, fg="#808080", anchor="nw", justify="left",
                                 width=1, height=2)
        self.mem_hint.pack(fill="x", padx=4, pady=(0, 2))
        self.mem_hint.bind("<Configure>", lambda e: e.widget.config(wraplength=e.width - 4))
        self.mem_scroll = tk.Scrollbar(body, command=self._on_mem_scroll)
        self.mem_scroll.pack(side="right", fill="y")
        self.mem = tk.Text(body, width=6 + 3 * HEX_ROW + 17, height=VIEW_BYTES // HEX_ROW, font=self.mono,
                           state="disabled", wrap="none", cursor="arrow")
        self.mem.pack(side="left")
        # as tall as the hex view, so switching views does not shift the panel
        hex_h = (VIEW_BYTES // HEX_ROW) * self.mono.metrics("linespace") + 4
        self.cell = max(6, hex_h // IMG_W)
        self.img = tk.Canvas(body, width=IMG_LABEL_W + IMG_W * self.cell, height=hex_h,
                             highlightthickness=0, cursor="crosshair")
        for w in (self.mem, self.img, self.addr_entry):
            w.bind("<MouseWheel>", self._on_mem_wheel)
            w.bind("<Button-4>", self._on_mem_wheel)          # X11 wheel
            w.bind("<Button-5>", self._on_mem_wheel)
            w.bind("<Prior>", lambda _e: self._scroll_pages(-1))
            w.bind("<Next>", lambda _e: self._scroll_pages(1))
        for w in (self.mem, self.img):
            w.bind("<Button-1>", lambda e: e.widget.focus_set())
        self.img.bind("<Motion>", self._on_img_motion)
        self.img.bind("<Leave>", lambda _e: self._show_hint())
        self._show_hint()
        self.mem.tag_config("addr", foreground="#808080")
        for kind in (READ, WRITE, PROG):
            self.mem.tag_config(kind, background=HEX[kind], foreground="white" if kind != READ else "black")

        if self.live is not None:
            clock = tk.LabelFrame(self, text="Clock")
            clock.grid(row=2, column=1, sticky="nsew", pady=(6, 0))
            self._caption(clock, "Clock speed of the emulated Z80 (the board runs at 7.37 MHz). "
                                 "Slow it down to watch each neuron.").pack(fill="x", padx=4)
            run = tk.Frame(clock)
            run.pack(fill="x", padx=4, pady=(2, 0))
            self.speed = ttk.Combobox(run, values=[n for n, _ in SPEEDS], width=15, state="readonly")
            current = [hz for _, hz in SPEEDS].index(self.live.speed) if self.live.speed in \
                [hz for _, hz in SPEEDS] else len(SPEEDS) - 1
            self.speed.current(current)
            self.speed.bind("<<ComboboxSelected>>", self._on_speed)
            self.speed.pack(side="left", padx=4)
            self.pause_button = ttk.Button(run, text="Pause", width=7, command=self._on_pause)
            self.pause_button.pack(side="left", padx=4)
            self.clock = tk.Label(clock, font=self.small, anchor="w", width=1)
            self.clock.pack(fill="x", padx=4, pady=(0, 4))

        self.parts = [bar, regs, calls, stack, mem] + ([clock] if self.live is not None else [])
        self.status = self._caption(self, note)
        self.status.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        if self.view_addr == self.symbols.addr.get("_x"):
            self.view_mode.set("image")     # show x as the digit
            self._apply_view_mode()
        self._draw_map(None)

    def focus_parts(self):
        """The parts focus.HoverFocus can focus."""
        return self.parts

    # --- refresh --------------------------------------------------------------

    def refresh(self):
        """Capture and show. Snapshot mode captures in the background, live mode right away."""
        if self.live is not None:
            self.show(self.capture(self.view_addr, self.view_len()))
            return
        if self.busy:
            return
        self.busy = True
        addr, n = self.view_addr, self.view_len()
        self.run_in_background(lambda: self.capture(addr, n), self._on_captured)

    def _on_captured(self, snap, error):
        self.busy = False
        if error is not None:
            self.status.config(text=f"capture failed: {error}")
            return
        self.show(snap)
        if (snap.view_addr, len(snap.view)) != (self.view_addr, self.view_len()):
            self.refresh()              # the view moved while this capture ran

    def _live_tick(self):
        try:
            self.show(self.capture(self.view_addr, self.view_len()))
        finally:
            self.after(LIVE_MS, self._live_tick)

    def _on_speed(self, _event):
        self.live.speed = SPEEDS[self.speed.current()][1]
        self.last_rate = None

    def _on_pause(self):
        self.live.paused = not self.live.paused
        self.pause_button.config(text="Resume" if self.live.paused else "Pause")
        self.last_rate = None

    def start_digit(self):
        """Start counting T-states for a new digit (live only). Called in the worker thread."""
        self.t_base = self.live.idle_tstates()

    def fast_forward(self):
        """For Stop: finish the running inference at full speed (live only)."""
        self.stopping = True
        self.saved_speed = self.live.speed
        self.live.paused = False
        self.live.speed = 0
        self.pause_button.config(text="Pause", state="disabled")
        self.speed.config(state="disabled")
        self.last_rate = None

    def restore_clock(self):
        """After Stop: restore the clock speed from before."""
        self.stopping = False
        self.live.speed = self.saved_speed
        self.pause_button.config(state="normal")
        self.speed.config(state="readonly")
        self.last_rate = None

    def _update_access(self, activity, live):
        """Merge new accesses into kind/age. Live mode fades old ones, snapshot mode drops them."""
        new = access_kinds(activity)
        hit = new != 0
        if live:
            self.age *= FADE
        else:
            self.age[:] = 0
        self.age[hit] = 1.0
        self.kind[hit] = new[hit]

    def _accessed(self, addr):
        """The access kind to colour addr with, or 0."""
        a = addr & 0xFFFF
        return int(self.kind[a]) if self.age[a] > SHOWN else 0

    def show(self, snap):
        self.snap = snap
        self._update_access(snap.activity, snap.live)
        step = infer_step(snap, self.symbols)
        if step is None or step[0]:         # on ("", None) keep the last step
            self.step = step
        self._draw_map(snap)                # uses self.step
        self._show_regs(snap.regs)
        self._show_calls(snap.frames)
        self._show_stack(snap)
        self._show_memory(snap)
        if self.on_step is not None:
            self.on_step(self.step)
        if snap.tstates is not None and self.live is not None:
            self._show_clock(snap.tstates)

    def _show_clock(self, tstates):
        now = time.monotonic()
        if self.t_base is None:
            text = f"Since start: {tstates:,} clock cycles (T-states)"
        else:
            n = max(0, tstates - self.t_base)
            text = f"This digit: {n:,} cycles · {n / BOARD_HZ:.2f} s on the board"
        if self.last_rate is not None and now - self.last_rate[0] > 0.4:
            mhz = (tstates - self.last_rate[1]) / (now - self.last_rate[0]) / 1e6
            idle = "paused" if self.live.paused else "idle"
            self.rate_text = f" · {mhz:.2f} MHz" if mhz >= 0.005 else f" · {idle}"
            self.last_rate = (now, tstates)
        elif self.last_rate is None:
            self.last_rate = (now, tstates)
            self.rate_text = ""
        self.clock.config(text=text + getattr(self, "rate_text", ""))

    def _draw_map(self, snap):
        """Draw the memory map: blocks, access strip, PC and window marks, status line."""
        c = self.bar
        blocks = self.symbols.blocks(snap.regs["sp"] if snap else None)
        if blocks != self.map_key:      # first time, or SP moved: redraw titles and frames
            self.map_key = blocks
            size_w = self.caption_font.measure("00.0 KB")
            self.layout = layout_blocks(blocks, lambda t: max(self.title_font.measure(t), size_w) + 12)
            c.delete("static")
            y0, y1 = TITLE_H, TITLE_H + STRIP_H
            for title, start, end, x0, x1 in self.layout:
                c.create_text((x0 + x1) // 2, y0 - 15, text=title, anchor="s",
                              font=self.title_font, tags="static")
                c.create_text((x0 + x1) // 2, y0 - 2, text=size_text(end - start), anchor="s",
                              fill=GREY, font=self.caption_font, tags="static")
                c.create_rectangle(x0, y0, x1 - 1, y1 - 1, outline="#909090", tags="static")
        rgb = strip_row(self.kind, self.age, self.layout, self.gap_rgb)
        img = Image.fromarray(rgb.astype(np.uint8).reshape(1, MAP_W, 3), "RGB")
        self.strip_photo = ImageTk.PhotoImage(img.resize((MAP_W, STRIP_H), Image.NEAREST), master=self)
        c.delete("strip", "marks")
        c.create_image(0, TITLE_H, image=self.strip_photo, anchor="nw", tags="strip")
        c.tag_lower("strip")
        if snap is None:
            self.now.config(text="Draw a digit. The blocks light up where the Z80 reads and writes.")
            return
        x = addr_to_x(self.layout, snap.regs["pc"])
        if x is not None:
            y = TITLE_H + STRIP_H
            c.create_line(x, TITLE_H, x, y, fill="white", tags="marks")
            c.create_polygon(x, y + 1, x - 4, y + PC_H - 1, x + 4, y + PC_H - 1, fill="#606060", tags="marks")
        v0 = addr_to_x(self.layout, snap.view_addr)
        v1 = addr_to_x(self.layout, min(0xFFFF, snap.view_addr + len(snap.view) - 1))
        if v0 is not None:
            v1 = v0 + 1 if v1 is None or v1 < v0 else v1
            c.create_rectangle(v0, TITLE_H, max(v0 + 2, v1 + 1), TITLE_H + STRIP_H - 1,
                               outline="#00c000", tags="marks")
        self.now.config(text=self._describe(snap))

    def _describe(self, snap):
        """One line on what the Z80 is doing and which blocks it uses."""
        func = (snap.frames[0][1] or "?").split("+")[0] if snap.frames else "?"
        doing = next((DOING[n.split("+")[0]] for _, n in snap.frames
                      if n and n.split("+")[0] in DOING), None)
        kind, n = self.step or ("", None)
        if snap.live and kind == "layer1":
            doing = f"layer 1, hidden neuron #{n}"
        elif snap.live and kind == "layer2":
            doing = f"layer 2, the score of digit {n}"
        doing = f"{doing} ({func.lstrip('_')})" if doing else f"running {func.lstrip('_')}"
        recent = self.age > SHOWN
        reads, writes = [], []
        for title, start, end, _, _ in self.layout:
            k = self.kind[start:end][recent[start:end]]
            if (k == READ).any() and title not in reads:
                reads.append(title)
            if (k == WRITE).any() and title not in writes:
                writes.append(title)
        if snap.live:
            state = "Stopping, at full speed" if self.stopping else \
                "Paused" if self.live is not None and self.live.paused else "Now"
            text = f"{state}: {doing}"
            if reads:
                text += " · reading " + ", ".join(reads)
            if writes:
                text += " · writing " + ", ".join(writes)
            return text
        text = f"After the last digit: {doing}"
        if writes:
            text += " · changed since the previous look: " + ", ".join(writes)
        return text

    def _show_regs(self, regs):
        for key, label in self.reg_labels.items():
            changed = self.prev_regs is not None and self.prev_regs[key] != regs[key]
            label.config(text=f"{regs[key]:04X}", fg="red" if changed else "black")
        self.ir_label.config(text=f"I {regs['i']:02X}   R {regs['r']:02X}")
        self.flag_label.config(text=f"F  {flags(regs['af'])}   F' {flags(regs['af2'])}")
        self.prev_regs = regs

    def _set_text(self, widget, lines):
        """Fill a Text widget. lines is a list of [(text, tags), ...]. Keeps the scroll position."""
        key = str(widget)
        if self.text_cache.get(key) == lines:
            return
        self.text_cache[key] = lines
        top = int(widget.index("@0,0").split(".")[0])      # first visible line
        widget.config(state="normal")
        widget.delete("1.0", "end")
        for k, parts in enumerate(lines):
            for text, tags in parts:
                widget.insert("end", text, tags)
            if k < len(lines) - 1:
                widget.insert("end", "\n")
        widget.config(state="disabled")
        widget.yview(top - 1)

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
        if self.view_mode.get() == "image":
            self._show_image(snap)
        else:
            self._show_hex(snap)
        n = len(snap.view)
        self.mem_scroll.set(snap.view_addr / 65536, min(1.0, (snap.view_addr + n) / 65536))
        if self.focus_get() is not self.addr_entry:     # do not overwrite while typing
            self.addr_entry.delete(0, "end")
            self.addr_entry.insert(0, f"{snap.view_addr:04X}")
        current = self.goto.get()
        if current in ("SP", "PC") and snap.regs[current.lower()] == snap.view_addr:
            return                      # still at the picked register
        name = self.symbols.nearest(snap.view_addr)
        shown = name.lstrip("_") if name in VARIABLES else ""     # only at the start of a variable
        if current != shown:
            self.goto.set(shown)

    def _show_hex(self, snap):
        lines = []
        for row in range(0, VIEW_BYTES, HEX_ROW):
            base = (snap.view_addr + row) & 0xFFFF
            parts = [(f"{base:04X}  ", ("addr",))]
            chunk = snap.view[row:row + HEX_ROW]
            for k, b in enumerate(chunk):
                kind = self._accessed(base + k)
                parts.append((f"{b:02X}", (kind,) if kind else ()))
                parts.append((" ", ()))
            parts.append((" " + "".join(chr(b) if 32 <= b < 127 else "." for b in chunk), ()))
            lines.append(parts)
        self._set_text(self.mem, lines)

    def _show_image(self, snap):
        """28x28 pixels from view_addr, each with a thin ring in its access colour.

        0/1 data is shown black and white. Other bytes are signed (weights):
        orange positive, blue negative, brighter for larger values.
        """
        cell = self.cell
        bits = self.pixel.get() == "bit"
        value, offset = image_rows(snap.view, bits)
        self.img_rows = (snap.view_addr, value, offset, bits)
        addrs = (snap.view_addr + offset) & 0xFFFF
        kinds = np.where(self.age[addrs] > SHOWN, self.kind[addrs], 0)
        dark = np.array(IMG_DARK, dtype=np.float32)
        if bits or ((value == 0) | (value == 1)).all():
            colour = np.where(value[..., None] != 0, IMG_LIT, IMG_DARK).astype(np.float32)
        else:
            mag = (np.abs(value) / max(1, np.abs(value).max()))[..., None]
            target = np.where(value[..., None] > 0, IMG_POS, IMG_NEG)
            colour = dark + (target - dark) * mag
        ones = np.ones((cell, cell, 1), dtype=np.float32)
        rgb = np.kron(colour, ones).astype(np.uint8)
        big_kind = np.kron(kinds, np.ones((cell, cell), dtype=np.uint8))
        yy, xx = np.indices(big_kind.shape) % cell
        grid = (yy == cell - 1) | (xx == cell - 1)
        ring = ~grid & ((yy == 0) | (xx == 0) | (yy == cell - 2) | (xx == cell - 2))
        rgb[grid] = IMG_GRID
        hit = ring & (big_kind != 0)
        rgb[hit] = PALETTE[big_kind[hit]].astype(np.uint8)
        self.img_photo = ImageTk.PhotoImage(Image.fromarray(rgb, "RGB"), master=self)
        c = self.img
        c.delete("all")
        c.create_image(IMG_LABEL_W, 0, image=self.img_photo, anchor="nw")
        for row in range(0, value.shape[0], 4):        # every 4 rows, a whole byte in both modes
            a = (snap.view_addr + offset[row, 0]) & 0xFFFF
            c.create_text(IMG_LABEL_W - 6, row * cell, text=f"{a:04X}", anchor="ne",
                          font=self.small, fill="#808080")

    # --- navigation -----------------------------------------------------------

    def view_len(self):
        """Bytes shown: 256 for hex, 784 for a byte image, 98 for a bit image."""
        if self.view_mode.get() == "image":
            return image_bytes(self.pixel.get() == "bit")
        return VIEW_BYTES

    def row_bytes(self):
        """Bytes per scroll step: one hex row, one image row, or 2 bit-image rows (7 bytes)."""
        if self.view_mode.get() == "image":
            return 7 if self.pixel.get() == "bit" else IMG_W
        return HEX_ROW

    def set_view(self, addr):
        """Show memory starting at addr. In snapshot mode this reads it right away."""
        self.view_addr = addr & 0xFFFF
        if self.live is None:
            self.refresh()

    def scroll_rows(self, n):
        self.set_view(self.view_addr + n * self.row_bytes())
        return "break"

    def _scroll_pages(self, n):
        return self.scroll_rows(n * (self.view_len() // self.row_bytes()))

    def _on_mem_wheel(self, event):
        """One row per wheel step. macOS sends small deltas, Windows 120 per notch."""
        if event.num in (4, 5):
            delta = 1 if event.num == 4 else -1
        else:
            delta = event.delta
        if delta == 0:
            return "break"
        steps = -(delta // 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        return self.scroll_rows(steps)

    def _on_mem_scroll(self, action, value, unit=None):
        """Scrollbar over the whole 64 KB. The thumb is the visible window."""
        if action == "moveto":
            addr = int(float(value) * 65536)
            self.set_view(max(0, min(addr, 0x10000 - self.view_len())) & ~(HEX_ROW - 1))
        elif unit == "pages":
            self._scroll_pages(int(value))
        else:
            self.scroll_rows(int(value))

    def show_weights_of(self, j):
        """Show hidden neuron j's 784 weights (its W1q row) as a 28x28 image."""
        self.view_addr = (self.symbols["_weights"] + N_IN * j) & 0xFFFF
        self.pixel.set("byte")
        self.view_mode.set("image")
        self._on_view_mode()

    def _on_view_mode(self):
        self._apply_view_mode()
        self._after_view_change()

    def _apply_view_mode(self):
        """Switch between hex and image, with the matching buttons and hint."""
        image = self.view_mode.get() == "image"
        for b in self.pixel_buttons:
            b.config(state="normal" if image else "disabled")
        if image:
            self._auto_pixel()
            self.mem.pack_forget()
            self.img.pack(side="left")
        else:
            self.img.pack_forget()
            self.mem.pack(side="left")
        self._show_hint()

    def _on_pixel(self):
        self._show_hint()
        self._after_view_change()

    def _auto_pixel(self):
        """Pick the pixel size for x (one byte per pixel) or buf (one bit per pixel)."""
        for name, pixel in (("_x", "byte"), ("_buf", "bit")):
            if self.symbols.addr.get(name) == self.view_addr:
                self.pixel.set(pixel)

    def _after_view_change(self):
        if self.live is None:
            self.refresh()
        elif self.snap is not None:
            self.show(self.capture(self.view_addr, self.view_len()))

    def _show_hint(self):
        if self.view_mode.get() == "image":
            text = ("1 bit per pixel, 28 per row (buf is the image as sent)" if self.pixel.get() == "bit" else
                    "1 byte per pixel, 28 per row. 0/1 in white, other bytes signed: + orange, - blue")
            text += " · PgDn for the next 28×28"
        else:
            text = "Scroll with the wheel, PgUp/PgDn or the bar · Hover the image view for addresses"
        self.mem_hint.config(text=text)

    def _on_img_motion(self, event):
        if self.img_rows is None:
            return
        base, value, offset, bits = self.img_rows
        col, row = (event.x - IMG_LABEL_W) // self.cell, event.y // self.cell
        if not (0 <= col < IMG_W and 0 <= row < value.shape[0]):
            self._show_hint()
            return
        addr = (base + offset[row, col]) & 0xFFFF
        name = self.symbols.nearest(addr)
        where = f"{addr:04X}" + (f" bit {7 - (row * IMG_W + col) % 8}" if bits else "")
        text = f"row {row}, col {col} · {where} = {value[row, col]}"
        w = self.symbols.addr.get("_weights")
        if not bits and w is not None and 0 <= addr - w < N_HID * N_IN:
            j, p = divmod(addr - w, N_IN)
            text += f" · weight of neuron #{j} for pixel ({p // IMG_W}, {p % IMG_W})"
        elif name:
            text += f" · {name.lstrip('_')}"
        self.mem_hint.config(text=text)

    def _on_goto(self, _event):
        choice = self.goto.get()
        if choice in ("SP", "PC"):
            if self.snap is not None:
                self.set_view(self.snap.regs[choice.lower()])
        else:
            self.set_view(self.symbols["_" + choice])
            if self.view_mode.get() == "image":
                self._auto_pixel()
                self._show_hint()
        self._after_view_change()

    def _on_addr_entry(self, _event):
        try:
            self.set_view(int(self.addr_entry.get(), 16))
        except ValueError:
            self.addr_entry.delete(0, "end")

    def _bar_addr(self, event):
        """(block, first address, end address) under the mouse, or None."""
        hit = x_to_addr(self.layout, event.x)
        if hit is None:
            return None
        block, addr = hit
        nxt = x_to_addr(self.layout, event.x + 1)
        end = nxt[1] if nxt is not None and nxt[0] is block else block[2]
        return block, addr, max(addr + 1, end)

    def _on_bar_motion(self, event):
        hit = self._bar_addr(event)
        if hit is None:
            self.hover.config(text="")
            return
        (title, *_), addr, end = hit
        name = self.symbols.nearest(addr)
        parts = [title, f"{addr:04X}" if end - addr == 1 else f"{addr:04X}-{end - 1:04X}"]
        if name and title not in ("Free", "Stack"):
            parts.append(name.lstrip("_"))
        recent = self.kind[addr:end][self.age[addr:end] > SHOWN]
        used = [n for k, n in ((READ, "read"), (WRITE, "written"), (PROG, "running")) if (recent == k).any()]
        if used:
            parts.append(", ".join(used))
        self.hover.config(text="  ·  ".join(parts))

    def _on_bar_click(self, event):
        hit = self._bar_addr(event)
        if hit is not None:
            self.set_view(hit[1] & 0xFFF0)
