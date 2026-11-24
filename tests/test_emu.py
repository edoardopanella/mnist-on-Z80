"""End-to-end: host/z80nn.py <-> firmware/z80_serial_main.c in z88dk-ticks.

Every case of artifacts/testvec.bin must match bit for bit (acc1, h_q, acc2,
digit). Runs twice:
    build/emu.bin     make emu, console I/O
    build/board.rom   make rom, the EEPROM image; ticks emulates the ACIA at 0x80/0x81
A build that is missing is skipped. Run with: pytest tests/test_emu.py
"""

import os
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


@pytest.fixture(scope="module", params=[EMU_BIN, BOARD_ROM], ids=["emu", "board-rom"])
def z80(request):
    if not os.path.exists(request.param):
        pytest.skip(f"{os.path.relpath(request.param, ROOT)} not built")
    z = Z80NN("emu", request.param)
    z.load_weights(WEIGHTS)          # raises on a checksum mismatch
    yield z
    if request.param == BOARD_ROM:
        z.proc.kill()                # the ROM waits for input forever, like the board
        z.proc.wait()
    else:
        z.close()


def test_version(z80):
    assert z80.version() == "Z80NN 1"


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
                        ("W 63ED 00", "ERR OFS"), ("I " + "0" * 300, "ERR LONG")]:
        z80._write_line(bad)
        assert z80._read_line() == answer, bad
    assert z80.version() == "Z80NN 1"
