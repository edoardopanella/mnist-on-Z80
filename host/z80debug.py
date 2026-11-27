"""
z80debug.py - Data for the debug panel, wherever the Z80 runs.

    MapSymbols       symbols and memory-map blocks from a z88dk .map file
    Snapshot         one capture: registers, stack, call stack, memory window, accesses
    SnapshotSource   captures with the G and M commands (emulator or board)
    LiveSource       reads the shared file of the patched z88dk-ticks (make ticks-viz)

Both sources have capture(view_addr, view_len) -> Snapshot. SnapshotSource
sends commands, so it must run in the worker thread with the other commands.
LiveSource only reads shared memory, so it can run in Tk's thread at any time.
"""

import mmap
import os
import re
import struct
import time
from dataclasses import dataclass

import numpy as np

READ, WRITE, PROG = 1, 2, 4     # bits of Snapshot.activity, one byte per address
VIEW_BYTES = 256                # hex view: 16 rows of 16 bytes
IMG_W = 28                      # image view: pixels per row
STACK_BYTES = 256               # bytes read from SP up (STACK_COPY in z80_serial_main.c)
LIVE_STACK_BYTES = 1024         # live reads are free, so read deep enough for main's frame
MAX_FRAMES = 8                  # call stack depth

# Variables of firmware/z80_serial_main.c listed in the memory view
VARIABLES = ("_weights", "_x", "_trace", "_line", "_buf", "_regs")
# Memory-map block title for each variable
BLOCK_TITLES = {"_weights": "Weights", "_x": "Image", "_trace": "Results",
                "_buf": "Serial", "_line": "Serial", "_stack_copy": "Debug", "_regs": "Debug"}
FREE_MIN = 1024                 # smaller gaps are merged into a neighbouring block

N_IN, N_HID, N_OUT = 784, 32, 10      # firmware/nn.h
TRACE_SIZE = 4 * N_HID + N_HID + 4 * N_OUT     # struct nn_trace: acc1, h_q, acc2
# struct nn_weights (no padding, docs/spec.md §7.2): W1q, b1q, M0, shift, W2q, b2q
W2Q_OFS = N_HID * N_IN + 4 * N_HID + 4 + 1     # 25,221
WEIGHTS_SIZE = W2Q_OFS + N_OUT * N_HID + 4 * N_OUT     # 25,581
# Functions of firmware/nn.c. Any of them on the call stack means we are inside nn_infer.
NN_FUNCS = ("_nn_infer", "_layer1_neuron", "_requant", "_layer2_neuron")

_MAP_LINE = re.compile(r"^(\w+)\s+= \$([0-9A-F]{4}) ; (\w+), ")
# Compiler-generated labels, not useful as names in the stack view
_LOCAL_LABEL = re.compile(r"^(i_\d+|l_\w+_\d{5}|___str_\d+|loop_?\d*)$")


class MapSymbols:
    """Symbols of one build, from the .map file z88dk writes with -m.

    addr    name -> address of code and variables (C names start with '_')
    const   name -> value of linker constants (__head, __BSS_END_tail, ...)
    """

    def __init__(self, path):
        self.addr, self.const = {}, {}
        for line in open(path):
            m = _MAP_LINE.match(line)
            if m:
                name, value, kind = m.group(1), int(m.group(2), 16), m.group(3)
                (self.addr if kind == "addr" else self.const)[name] = value
        self._named = sorted((a, n) for n, a in self.addr.items()
                             if not n.startswith("__") and not _LOCAL_LABEL.match(n))
        # Section boundaries are addresses too. Other constants are plain numbers.
        self._bounds = sorted(v for n, v in self.const.items()
                              if n.startswith("__") and n.endswith(("_head", "_tail")))
        self.code = (self.const["__head"], self.const["__CODE_END_tail"])
        self.data = (self.const["__data_crt_head"], self.const["__BSS_END_tail"])
        self.stack_top = self.const["__register_sp"] or 0x10000    # SP = 0 wraps to 0xFFFF

    def __getitem__(self, name):
        return self.addr[name]

    def nearest(self, addr):
        """'name+0xNN' for the closest named symbol at or below addr.

        None outside code and data, where the closest symbol would be
        meaningless.
        """
        if not (self.in_code(addr) or self.data[0] <= addr < self.data[1]):
            return None
        best = None
        for a, n in self._named:
            if a > addr:
                break
            best = (a, n)
        if best is None:
            return None
        return best[1] if addr == best[0] else f"{best[1]}+0x{addr - best[0]:X}"

    def in_code(self, addr):
        return self.code[0] <= addr < self.code[1]

    def size_of(self, name):
        """Bytes from name to the next symbol or section boundary (includes padding)."""
        start = self.addr[name]
        later = [a for a, _ in self._named if a > start] + [b for b in self._bounds if b > start]
        return min(later) - start if later else 0

    def blocks(self, sp=None):
        """The memory map as [(title, start, end)] covering 0x0000-0xFFFF with no gaps.

        Blocks are Program, the variables in BLOCK_TITLES (neighbours with the
        same title are merged) and Stack. Stack starts on a 256-byte boundary at
        least STACK_BYTES below sp, so it has room for pushes and rarely moves.
        Gaps of FREE_MIN bytes or more become Free. Smaller gaps join the
        block before them, or the one after if the one before is Free.
        """
        if sp is None or not self.data[1] <= sp <= self.stack_top:
            sp = self.stack_top
        sp = max(self.data[1], min(self.stack_top - STACK_BYTES, (sp - STACK_BYTES) & ~0xFF))
        named = [("Program", *self.code)]
        for name, title in BLOCK_TITLES.items():
            if name in self.addr:
                start = self.addr[name]
                named.append((title, start, start + self.size_of(name)))
        named.append(("Stack", sp, self.stack_top))
        named = [b for b in sorted(named, key=lambda b: b[1]) if b[1] < b[2]]

        out = []
        pos, carry = 0, None            # carry = start of a small gap after a Free block
        for title, start, end in named + [(None, 0x10000, 0x10000)]:
            if start - pos >= FREE_MIN:
                out.append(["Free", pos, start])
            elif start > pos:
                if out and out[-1][0] != "Free":
                    out[-1][2] = start
                else:
                    carry = pos
            if title is None:
                break
            if carry is not None:
                start, carry = carry, None
            if out and out[-1][0] == title and out[-1][2] == start:
                out[-1][2] = end
            else:
                out.append([title, start, end])
            pos = end
        if out[-1][2] < 0x10000:        # extend the last block to the end
            out[-1][2] = 0x10000
        return [tuple(b) for b in out]

    def block_of(self, addr, blocks):
        """The title of the block holding addr."""
        for title, start, end in blocks:
            if start <= addr < end:
                return title
        return None


def image_bytes(bits, rows=IMG_W):
    """Bytes needed for rows of IMG_W pixels, at one bit or one byte per pixel."""
    return (rows * IMG_W + 7) // 8 if bits else rows * IMG_W


def image_rows(data, bits, rows=IMG_W):
    """Memory as a picture. Returns (value, offset), both shaped (rows, IMG_W).

    value is the pixel, offset the index of the byte it comes from.
    bits=False: one int8 per pixel. Good for x (0/1) and for W1q rows.
    bits=True: one bit per pixel, MSB first, as in buf (the packed image).
    Pixels past the end of data are 0.
    """
    n = rows * IMG_W
    raw = np.frombuffer(bytes(data), dtype=np.uint8)
    if bits:
        value = np.unpackbits(raw)[:n].astype(np.int16)
        offset = np.arange(n) // 8
    else:
        value = raw[:n].view(np.int8).astype(np.int16)
        offset = np.arange(n)
    value = np.pad(value, (0, n - len(value)))
    return value.reshape(rows, IMG_W), offset.reshape(rows, IMG_W)


@dataclass
class Snapshot:
    regs: dict              # z80nn.REG_NAMES -> int
    stack: list             # (address, word) from SP upwards
    frames: list            # call stack as (code address, symbol), PC first
    view_addr: int
    view: bytes             # view_len bytes from view_addr
    activity: np.ndarray    # (65536,) uint8, READ | WRITE | PROG bits
    acc2: list              # the 10 output scores, decoded from _trace
    live: bool              # True: real accesses. False: bytes changed since the last capture
    tstates: int = None     # live only


class SnapshotSource:
    """Captures with G and M. Works on the emulator and on the board.

    It cannot see the bus, so activity only has WRITE bits, for bytes that
    changed since the last capture. Watched areas are the variables (not the
    weights), the stack and the memory window.

    The stack is read from stack_copy, because G and M overwrite the real one.
    """

    live = False

    def __init__(self, z80, symbols):
        self.z80 = z80
        self.symbols = symbols
        self.previous = {}      # (start, length) -> bytes read at the last capture

    def capture(self, view_addr, view_len=VIEW_BYTES):
        sym = self.symbols
        view_addr &= 0xFFFF
        regs = self.z80.registers()
        sp = regs["sp"]
        stack_len = max(0, min(STACK_BYTES, sym.stack_top - sp)) & ~1

        areas = [(sym[n], sym.size_of(n)) for n in VARIABLES if n in sym.addr and n != "_weights"]
        areas += [(view_addr, view_len)]
        current = {}
        activity = np.zeros(65536, dtype=np.uint8)
        for start, n in areas:
            if n == 0 or (start, n) in current:
                continue
            data = self.z80.read_memory(start, n)
            current[(start, n)] = data
            old = self.previous.get((start, n))
            if old is not None:
                changed = np.nonzero(np.frombuffer(data, np.uint8) != np.frombuffer(old, np.uint8))[0]
                activity[(start + changed) & 0xFFFF] |= WRITE
        stack_bytes = self.z80.read_memory(sym["_stack_copy"], stack_len) if stack_len else b""
        old = self.previous.get("stack")
        if old is not None and len(old) == len(stack_bytes):
            changed = np.nonzero(np.frombuffer(stack_bytes, np.uint8) != np.frombuffer(old, np.uint8))[0]
            activity[(sp + changed) & 0xFFFF] |= WRITE
        current["stack"] = stack_bytes
        self.previous = current

        stack = [(sp + k, stack_bytes[k] | stack_bytes[k + 1] << 8)
                 for k in range(0, len(stack_bytes), 2)]
        trace = current.get((sym["_trace"], sym.size_of("_trace")), b"") if "_trace" in sym.addr else b""
        return Snapshot(regs, stack, call_stack(regs, stack_bytes, sym), view_addr,
                        current[(view_addr, view_len)], activity,
                        decode_acc2(trace) if len(trace) >= TRACE_SIZE else [], live=False)


def call_stack(regs, stack_bytes, sym):
    """Follow SDCC's frame chain. IX points at the caller's saved IX, with the return address above it.

    stack_bytes start at SP. SP and IX can be odd, so words are read at any
    offset. Functions that do not set up IX (small leaves, library helpers)
    are missing. Stops when the chain leaves stack_bytes.
    """
    sp = regs["sp"]

    def word(addr):
        k = addr - sp
        return stack_bytes[k] | stack_bytes[k + 1] << 8 if 0 <= k < len(stack_bytes) - 1 else None

    frames = [(regs["pc"], sym.nearest(regs["pc"]))]
    ix = regs["ix"]
    while len(frames) < MAX_FRAMES:
        saved, ret = word(ix), word(ix + 2)
        if saved is None or ret is None or not sym.in_code(ret):
            break
        frames.append((ret, sym.nearest(ret)))
        if saved <= ix:         # a valid saved IX is higher up the stack
            break
        ix = saved
    return frames


def infer_step(snap, sym):
    """Where nn_infer is, based on which weight rows were read since the last capture.

    layer1_neuron(j) only reads row j of W1q and layer2_neuron(k) only row k
    of W2q. Returns
        None            not inside nn_infer
        ("layer1", j)   computing hidden neuron j
        ("layer2", k)   computing the score of digit k
        ("trace", None) copying the results into trace
        ("", None)      inside nn_infer, but no weights read
    Only live sources have read bits, so snapshots only give None or ("", None).
    """
    if not any(name and name.split("+")[0] in NN_FUNCS for _, name in snap.frames):
        return None
    if "_weights" not in sym.addr:
        return "", None
    w = sym["_weights"]
    read = np.nonzero(snap.activity[w:w + WEIGHTS_SIZE] & READ)[0]
    w2 = read[(read >= W2Q_OFS) & (read < W2Q_OFS + N_OUT * N_HID)]
    if w2.size:
        return "layer2", int(w2.max() - W2Q_OFS) // N_HID
    w1 = read[read < N_HID * N_IN]
    if w1.size:
        return "layer1", int(w1.max()) // N_IN
    if "_trace" in sym.addr:
        t = sym["_trace"]
        if (snap.activity[t:t + TRACE_SIZE] & WRITE).any():
            return "trace", None
    return "", None


def decode_acc2(trace):
    """The 10 int32 scores of struct nn_trace (little-endian, after acc1 and h_q)."""
    off = 4 * N_HID + N_HID
    return np.frombuffer(trace[off:off + 4 * N_OUT], dtype="<i4").tolist()


def flags(af):
    """The F register as 'SZ5H3PNC', with '.' for clear bits."""
    f = af & 0xFF
    return "".join(c if f & (0x80 >> k) else "." for k, c in enumerate("SZ5H3PNC"))


# --- Live view: file shared with the patched z88dk-ticks (tools/ticks-viz.patch) ----

SHM_SIZE = 64 + 2 * 65536
SHM_RAM, SHM_ACT = 64, 64 + 65536
SPEED_OFS, PAUSED_OFS = 12, 16


class LiveSource:
    """Registers, memory and accesses of the running emulator, from its -shm file.

    Layout (viz.h in the patch): a header with seq, speed, pause, T-states
    and registers, then the emulated RAM, then one byte of access bits per
    address. ticks sets the bits and capture() clears them.
    """

    live = True

    def __init__(self, path, symbols, timeout=5.0):
        deadline = time.time() + timeout
        while not (os.path.exists(path) and os.path.getsize(path) == SHM_SIZE):
            if time.time() > deadline:
                raise RuntimeError(f"{path}: the emulator did not create it (is it the patched ticks?)")
            time.sleep(0.01)
        fd = os.open(path, os.O_RDWR)
        try:
            self.m = mmap.mmap(fd, SHM_SIZE)
        finally:
            os.close(fd)
        while self.m[0:4] != b"Z8VZ":
            if time.time() > deadline:
                raise RuntimeError(f"{path}: not a ticks-viz file")
            time.sleep(0.01)
        self.symbols = symbols
        self.ram = memoryview(self.m)[SHM_RAM:SHM_RAM + 65536]
        self.act = np.frombuffer(self.m, dtype=np.uint8, count=65536, offset=SHM_ACT)

    def _header(self):
        """(tstates, regs), retried until consistent. seq is odd while ticks writes."""
        for _ in range(1000):
            seq = struct.unpack_from("<I", self.m, 8)[0]
            tstates, = struct.unpack_from("<Q", self.m, 24)
            words = struct.unpack_from("<12H2B", self.m, 32)
            if seq % 2 == 0 and struct.unpack_from("<I", self.m, 8)[0] == seq:
                break
            time.sleep(0.0001)
        names = ["af", "bc", "de", "hl", "ix", "iy", "sp", "pc", "af2", "bc2", "de2", "hl2", "i", "r"]
        return tstates, dict(zip(names, words))

    def idle_tstates(self, timeout=0.2):
        """T-states since ticks started, once the count stops moving.

        ticks publishes just before it blocks waiting for input, and a reply
        can arrive before that. So wait until two reads agree.
        """
        last = self._header()[0]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.005)
            now = self._header()[0]
            if now == last:
                break
            last = now
        return last

    def read(self, addr, n):
        addr &= 0xFFFF
        if addr + n <= 65536:
            return bytes(self.ram[addr:addr + n])
        return bytes(self.ram[addr:]) + bytes(self.ram[:addr + n - 65536])

    def capture(self, view_addr, view_len=VIEW_BYTES):
        sym = self.symbols
        view_addr &= 0xFFFF
        tstates, regs = self._header()
        activity = self.act.copy()
        self.act[:] = 0         # bits set between copy and clear are lost, which is fine
        sp = regs["sp"]
        stack_len = max(0, min(LIVE_STACK_BYTES, sym.stack_top - sp)) & ~1
        stack_bytes = self.read(sp, stack_len)
        stack = [(sp + k, stack_bytes[k] | stack_bytes[k + 1] << 8) for k in range(0, stack_len, 2)]
        trace = self.read(sym["_trace"], TRACE_SIZE) if "_trace" in sym.addr else b""
        return Snapshot(regs, stack, call_stack(regs, stack_bytes, sym), view_addr,
                        self.read(view_addr, view_len), activity,
                        decode_acc2(trace) if trace else [], live=True, tstates=tstates)

    @property
    def speed(self):
        return struct.unpack_from("<I", self.m, SPEED_OFS)[0]

    @speed.setter
    def speed(self, hz):
        """Emulated clock in Hz. 0 means as fast as possible."""
        struct.pack_into("<I", self.m, SPEED_OFS, int(hz))

    @property
    def paused(self):
        return struct.unpack_from("<I", self.m, PAUSED_OFS)[0] != 0

    @paused.setter
    def paused(self, value):
        struct.pack_into("<I", self.m, PAUSED_OFS, 1 if value else 0)

    def close(self):
        self.ram.release()
        del self.act
        self.m.close()
