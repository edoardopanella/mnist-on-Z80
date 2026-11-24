"""
z80debug.py - What the debug panel shows, independent of where the Z80 runs.

    MapSymbols       symbols and memory regions from a z88dk .map file
    Snapshot         one capture: registers, stack, call stack, a memory window, access map
    SnapshotSource   captures through the G and M commands (emulator or board)
    LiveSource       reads the patched z88dk-ticks' shared file (make ticks-viz), any time

A source has capture(view_addr) -> Snapshot. SnapshotSource talks to the Z80,
so it must run in the same thread as the other Z80 commands: draw.py runs
both in its worker thread. LiveSource only reads memory shared with ticks, so
it can run in Tk's thread while the Z80 works.
"""

import mmap
import os
import re
import struct
import time
from dataclasses import dataclass

import numpy as np

READ, WRITE, PROG = 1, 2, 4     # bits of Snapshot.activity, one byte per address
VIEW_BYTES = 256                # memory window: 16 rows of 16 bytes
STACK_BYTES = 256               # bytes from SP upwards (= STACK_COPY in z80_serial_main.c)
LIVE_STACK_BYTES = 1024         # live: free to read, deep enough for main's frame
MAX_FRAMES = 8                  # call stack depth

# Variables of firmware/z80_serial_main.c shown under the memory bar
VARIABLES = ("_weights", "_x", "_trace", "_line", "_buf", "_regs")

N_HID, N_OUT = 32, 10           # firmware/nn.h
TRACE_SIZE = 4 * N_HID + N_HID + 4 * N_OUT     # struct nn_trace: acc1, h_q, acc2

_MAP_LINE = re.compile(r"^(\w+)\s+= \$([0-9A-F]{4}) ; (\w+), ")
# Compiler-made labels: no use as names in the stack view
_LOCAL_LABEL = re.compile(r"^(i_\d+|l_\w+_\d{5}|___str_\d+|loop_?\d*)$")


class MapSymbols:
    """Symbols of one build, read from the .map file z88dk writes with -m.

    addr:   name -> address of code and variables (C names start with '_')
    const:  name -> value of linker constants (__head, __BSS_END_tail, ...)
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
        # Section boundaries are addresses too; other constants are plain numbers.
        self._bounds = sorted(v for n, v in self.const.items()
                              if n.startswith("__") and n.endswith(("_head", "_tail")))
        self.code = (self.const["__head"], self.const["__CODE_END_tail"])
        self.data = (self.const["__data_crt_head"], self.const["__BSS_END_tail"])
        self.stack_top = self.const["__register_sp"] or 0x10000    # 0: SP wraps to 0xFFFF

    def __getitem__(self, name):
        return self.addr[name]

    def nearest(self, addr):
        """'name+0xNN' for the closest named symbol at or below addr.

        None outside code and data (free memory, stack): the closest symbol
        there would be the last variable, which says nothing.
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

    def regions(self, sp=None):
        """(label, start, end, level) to draw under the memory bar.

        Level 0: code, data, stack. Level 1: the variables inside data.
        """
        out = [("code", *self.code, 0), ("data", *self.data, 0)]
        if sp is not None and sp < self.stack_top:
            out.append(("stack", sp, self.stack_top, 0))
        for name in VARIABLES:
            if name in self.addr:
                start = self.addr[name]
                out.append((name.lstrip("_"), start, start + self.size_of(name), 1))
        return out


@dataclass
class Snapshot:
    regs: dict              # z80nn.REG_NAMES -> int
    stack: list             # (address, word) from SP upwards
    frames: list            # call stack, innermost first: (code address, symbol), PC first
    view_addr: int
    view: bytes             # VIEW_BYTES from view_addr
    activity: np.ndarray    # (65536,) uint8, READ | WRITE | PROG bits
    acc2: list              # the 10 output scores, decoded from _trace
    live: bool              # True: real accesses; False: bytes that changed between captures
    tstates: int = None     # live sources only


class SnapshotSource:
    """Captures through G and M. Works on the emulator and on the board.

    It cannot see the bus, so activity has only WRITE bits: the bytes of the
    watched areas (variables but not the weights, stack, memory window) that
    changed since the previous capture.

    The stack comes from the firmware's stack_copy, taken with the registers:
    by the time M runs, G and M have reused the real stack area.
    """

    live = False

    def __init__(self, z80, symbols):
        self.z80 = z80
        self.symbols = symbols
        self.previous = {}      # (start, length) -> bytes read at the last capture

    def capture(self, view_addr):
        sym = self.symbols
        view_addr &= 0xFFFF
        regs = self.z80.registers()
        sp = regs["sp"]
        stack_len = max(0, min(STACK_BYTES, sym.stack_top - sp)) & ~1

        areas = [(sym[n], sym.size_of(n)) for n in VARIABLES if n in sym.addr and n != "_weights"]
        areas += [(view_addr, VIEW_BYTES)]
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
                        current[(view_addr, VIEW_BYTES)], activity,
                        decode_acc2(trace) if len(trace) >= TRACE_SIZE else [], live=False)


def call_stack(regs, stack_bytes, sym):
    """Follow SDCC's frame chain: IX points at the caller's saved IX, the return address is above it.

    stack_bytes are the bytes from SP. SP and IX can be odd (SDCC pushes single
    bytes), so words are read at any offset. Functions that do not set up IX
    (small leaf functions, library helpers) do not appear. Stops when the
    chain leaves the bytes read.
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
        if saved <= ix:         # the saved IX must be higher up the stack
            break
        ix = saved
    return frames


def decode_acc2(trace):
    """The 10 int32 scores of struct nn_trace (little-endian, after acc1 and h_q)."""
    off = 4 * N_HID + N_HID
    return np.frombuffer(trace[off:off + 4 * N_OUT], dtype="<i4").tolist()


def flags(af):
    """The F register as 'SZ5H3PNC', with '.' for the bits that are clear."""
    f = af & 0xFF
    return "".join(c if f & (0x80 >> k) else "." for k, c in enumerate("SZ5H3PNC"))


# --- Live view: the file shared with the patched z88dk-ticks (tools/ticks-viz.patch) ----

SHM_SIZE = 64 + 2 * 65536
SHM_RAM, SHM_ACT = 64, 64 + 65536
SPEED_OFS, PAUSED_OFS = 12, 16


class LiveSource:
    """Registers, memory and accesses of the running emulator, read from its -shm file.

    Layout (viz.h in the patch): header with seq, speed, pause, T-states and
    registers, then the emulated RAM itself, then one byte of access bits per
    address. ticks sets the bits, capture() takes them and clears them.
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
        """(tstates, regs) from a consistent publish: seq is odd while ticks writes."""
        for _ in range(1000):
            seq = struct.unpack_from("<I", self.m, 8)[0]
            tstates, = struct.unpack_from("<Q", self.m, 24)
            words = struct.unpack_from("<12H2B", self.m, 32)
            if seq % 2 == 0 and struct.unpack_from("<I", self.m, 8)[0] == seq:
                break
            time.sleep(0.0001)
        names = ["af", "bc", "de", "hl", "ix", "iy", "sp", "pc", "af2", "bc2", "de2", "hl2", "i", "r"]
        return tstates, dict(zip(names, words))

    def read(self, addr, n):
        addr &= 0xFFFF
        if addr + n <= 65536:
            return bytes(self.ram[addr:addr + n])
        return bytes(self.ram[addr:]) + bytes(self.ram[:addr + n - 65536])

    def capture(self, view_addr):
        sym = self.symbols
        view_addr &= 0xFFFF
        tstates, regs = self._header()
        activity = self.act.copy()
        self.act[:] = 0         # a bit set between copy and clear is lost: one frame
        sp = regs["sp"]
        stack_len = max(0, min(LIVE_STACK_BYTES, sym.stack_top - sp)) & ~1
        stack_bytes = self.read(sp, stack_len)
        stack = [(sp + k, stack_bytes[k] | stack_bytes[k + 1] << 8) for k in range(0, stack_len, 2)]
        trace = self.read(sym["_trace"], TRACE_SIZE) if "_trace" in sym.addr else b""
        return Snapshot(regs, stack, call_stack(regs, stack_bytes, sym), view_addr,
                        self.read(view_addr, VIEW_BYTES), activity,
                        decode_acc2(trace) if trace else [], live=True, tstates=tstates)

    @property
    def speed(self):
        return struct.unpack_from("<I", self.m, SPEED_OFS)[0]

    @speed.setter
    def speed(self, hz):
        """Emulated clock in Hz, 0 = as fast as the laptop can."""
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
