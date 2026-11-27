"""End-to-end tests: host/z80nn.py talking to firmware/z80_serial_main.c in z88dk-ticks.

Every case in artifacts/testvec.bin must match exactly (acc1, h_q, acc2, digit).
Runs on both builds, and skips the ones that are missing:
    build/emu.bin     make emu, console I/O
    build/board.rom   make rom, the EEPROM image (ticks emulates the ACIA at 0x80/0x81)
"""

import os
import re
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "host"))
sys.path.insert(0, os.path.join(ROOT, "training"))
from z80nn import Z80NN  # noqa: E402

EMU_BIN = os.path.join(ROOT, "build", "emu.bin")
BOARD_ROM = os.path.join(ROOT, "build", "board.rom")
WEIGHTS = os.path.join(ROOT, "artifacts", "weights.bin")
TESTVEC = os.path.join(ROOT, "artifacts", "testvec.bin")

pytestmark = pytest.mark.skipif(
    shutil.which("z88dk-ticks") is None
    or not all(os.path.exists(p) for p in (WEIGHTS, TESTVEC)),
    reason="needs z88dk-ticks and artifacts/",
)


def read_map(path):
    """C symbols of a z88dk .map file: ({name: address}, {function name: address})."""
    syms, code = {}, {}
    for line in open(path):
        m = re.match(r"(_\w+)\s+= \$([0-9A-F]{4}) ; (\w+),.*?, (\w+), [^,]*$", line.rstrip())
        if m:
            name, addr, kind, section = m.group(1), int(m.group(2), 16), m.group(3), m.group(4)
            syms[name] = addr
            if kind == "addr" and section == "code_compiler":
                code[name] = addr
    m = re.search(r"^__register_sp\s+= \$([0-9A-F]{4})", open(path).read(), re.M)
    syms["__register_sp"] = int(m.group(1), 16)
    return syms, code


def function_at(code, addr):
    """The C function containing addr (the closest symbol below it)."""
    return max((a, name) for name, a in code.items() if a <= addr)[1]


@pytest.fixture(scope="module", params=[EMU_BIN, BOARD_ROM], ids=["emu", "board-rom"])
def z80(request):
    if not os.path.exists(request.param):
        pytest.skip(f"{os.path.relpath(request.param, ROOT)} not built")
    z = Z80NN("emu", request.param)
    z.load_weights(WEIGHTS)          # raises on a checksum mismatch
    z.map = read_map(os.path.splitext(request.param)[0] + ".map")
    yield z
    z.close()                        # kills board.rom, which waits for input forever


def test_version(z80):
    assert z80.version() == "Z80NN 2"


def test_all_test_vectors(z80):
    from export import read_test_vectors

    records = read_test_vectors(TESTVEC)
    assert len(records) == 105
    for i, rec in enumerate(records):
        digit, acc1, h_q, acc2 = z80.classify_debug(rec["x"])
        assert acc1 == rec["acc1"].tolist(), f"case {i}: acc1"
        assert h_q == rec["h_q"].tolist(), f"case {i}: h_q"
        assert acc2 == rec["acc2"].tolist(), f"case {i}: acc2"
        assert digit == rec["digit"], f"case {i}: digit"
        assert z80.classify(rec["x"]) == digit, f"case {i}: I and D disagree"


def test_errors_keep_it_running(z80):
    for bad, answer in [("X", "ERR CMD"), ("I 00", "ERR LEN"), ("W 0000 GG", "ERR HEX"),
                        ("W 63ED 00", "ERR OFS"), ("I " + "0" * 300, "ERR LONG"),
                        ("M 8000", "ERR LEN"), ("M 8000 00", "ERR LEN"), ("M 8000 41", "ERR LEN"),
                        ("M 80G0 01", "ERR HEX"), ("M 8000 0G", "ERR HEX")]:
        z80._write_line(bad)
        assert z80._read_line() == answer, bad
    assert z80.version() == "Z80NN 2"


def test_registers_after_inference(z80):
    from export import read_test_vectors

    syms, code = z80.map
    z80.classify(read_test_vectors(TESTVEC)[0]["x"])
    regs = z80.registers()
    assert set(regs) == {"af", "bc", "de", "hl", "ix", "iy", "sp", "pc",
                         "af2", "bc2", "de2", "hl2", "i", "r"}
    assert function_at(code, regs["pc"]) == "_cmd_infer"       # saved right after nn_infer
    top = syms["__register_sp"] or 0x10000                     # board: SP = 0 wraps to 0xFFFF
    highest_var = max(syms[n] for n in ("_weights", "_x", "_trace", "_line", "_buf", "_regs",
                                        "_stack_copy"))
    assert highest_var < regs["sp"] < top

    # stack_copy is the stack at the snapshot. In cmd_infer's frame, IX points at the
    # saved IX, followed by the return address into main.
    stack = z80.read_memory(syms["_stack_copy"], 256)
    k = regs["ix"] - regs["sp"]
    assert 0 <= k < 254
    assert function_at(code, stack[k + 2] | stack[k + 3] << 8) == "_main"


def test_memory_reads(z80):
    from export import read_test_vectors

    syms, _ = z80.map
    payload = open(WEIGHTS, "rb").read()[5:]
    assert z80.read_memory(syms["_weights"], len(payload)) == payload

    x = read_test_vectors(TESTVEC)[3]["x"]
    z80.classify(x)
    assert z80.read_memory(syms["_x"], 784) == bytes(x.tolist())  # one byte per pixel

    assert z80.read_memory(syms["_weights"], 1) == payload[:1]     # smallest read
    wrap = z80.read_memory(0xFFF0, 32)                             # wraps at 0xFFFF
    assert wrap[16:] == z80.read_memory(0x0000, 16)
