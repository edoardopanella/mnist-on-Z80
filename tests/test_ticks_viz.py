"""The patched z88dk-ticks (make ticks-viz) and host/z80debug.py LiveSource.

    - with -shm, all 105 test vectors still match on both builds
    - the shared file shows the RAM, the registers and real accesses
    - the speed setting holds the emulated clock, Pause stops it
    - without -shm the patched ticks works as before

Skipped when build/ticks-viz is not built.
"""

import os
import shutil
import sys
import tempfile
import threading
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "host"))
sys.path.insert(0, os.path.join(ROOT, "training"))
from z80debug import PROG, READ, WRITE, LiveSource, MapSymbols  # noqa: E402
from z80nn import Z80NN  # noqa: E402

VIZ_TICKS = os.path.join(ROOT, "build", "ticks-viz", "src", "ticks", "z88dk-ticks")
WEIGHTS = os.path.join(ROOT, "artifacts", "weights.bin")
TESTVEC = os.path.join(ROOT, "artifacts", "testvec.bin")
BUILDS = {"emu": ("build/emu.bin", "build/emu.map"), "board-rom": ("build/board.rom", "build/board.map")}

pytestmark = pytest.mark.skipif(
    not os.path.exists(VIZ_TICKS) or not all(os.path.exists(p) for p in (WEIGHTS, TESTVEC)),
    reason="needs make ticks-viz and artifacts/",
)


def records():
    from export import read_test_vectors
    return read_test_vectors(TESTVEC)


def in_function(sym, name, addrs):
    """How many of addrs fall inside the named function (up to the next named symbol)."""
    start = sym[name]
    return sum(1 for a in addrs if start <= a < start + sym.size_of(name))


@pytest.fixture(scope="module", params=list(BUILDS))
def live(request):
    prog, map_path = (os.path.join(ROOT, p) for p in BUILDS[request.param])
    if not os.path.exists(prog):
        pytest.skip(f"{BUILDS[request.param][0]} not built")
    tmp = tempfile.mkdtemp(prefix="z80viz-test-")
    shm = os.path.join(tmp, "ticks.shm")
    z80 = Z80NN("emu", prog, ticks=VIZ_TICKS, shm=shm)
    assert z80.version() == "Z80NN 2"
    source = LiveSource(shm, MapSymbols(map_path))
    source.capture(0)                   # drop the start-up accesses (BSS clearing)
    z80.load_weights(WEIGHTS)
    yield z80, source
    source.paused = False
    source.speed = 0
    z80.close()
    source.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_upload_is_visible(live):
    z80, src = live
    sym = src.symbols
    payload = open(WEIGHTS, "rb").read()[5:]
    assert src.read(sym["_weights"], len(payload)) == payload        # the RAM is live
    snap = src.capture(sym["_weights"])
    w = np.arange(sym["_weights"], sym["_weights"] + len(payload))
    assert (snap.activity[w] & WRITE).all()                          # every weight byte written
    prog = np.nonzero(snap.activity & PROG)[0]
    assert in_function(sym, "_cmd_write", prog) > 0                  # by cmd_write's code
    assert not (snap.activity[w] & PROG).any()


def test_all_test_vectors(live):
    z80, _ = live
    for i, rec in enumerate(records()):
        digit, acc1, h_q, acc2 = z80.classify_debug(rec["x"])
        assert (acc1, h_q, acc2, digit) == (rec["acc1"].tolist(), rec["h_q"].tolist(),
                                            rec["acc2"].tolist(), rec["digit"]), f"case {i}"


def test_inference_accesses(live):
    z80, src = live
    sym = src.symbols
    rec = records()[0]
    src.capture(0)
    z80.classify_debug(rec["x"])
    snap = src.capture(sym["_trace"])
    act = snap.activity
    assert snap.acc2 == rec["acc2"].tolist()                         # decoded from the live RAM

    # layer 1 reads only the weight rows of lit pixels: W1q[j][i] for x[i] = 1
    lit = np.nonzero(rec["x"])[0]
    w1 = sym["_weights"]
    assert all(act[w1 + j * 784 + i] & READ for j in (0, 31) for i in lit)
    dark = np.nonzero(rec["x"] == 0)[0]
    assert not any(act[w1 + i] & READ for i in dark)

    trace = np.arange(sym["_trace"], sym["_trace"] + 200)
    assert (act[trace] & WRITE).any()
    prog = np.nonzero(act & PROG)[0]
    assert in_function(sym, "_nn_infer", prog) > 0
    assert not (act[np.arange(sym.data[0], sym.data[1])] & PROG).any()  # no code runs in data

    # idle again: waiting for input, SP back in main's frame, call stack down to main
    assert sym.data[1] <= snap.regs["sp"] < sym.stack_top


def test_call_stack_mid_inference(live):
    z80, src = live
    sym = src.symbols
    src.speed = 1_000_000
    t = threading.Thread(target=z80.classify_debug, args=(records()[2]["x"],))
    t.start()
    time.sleep(1.0)                     # 5 M T-states at 1 MHz: well inside nn_infer
    src.paused = True
    time.sleep(0.1)
    names = [n.split("+")[0] for _, n in src.capture(0).frames]
    src.paused = False
    src.speed = 0
    t.join()
    assert names[-3:] == ["_nn_infer", "_cmd_infer", "_main"], names


def test_speed_holds_the_clock(live):
    z80, src = live
    src.speed = 2_000_000
    t0, w0 = src.capture(0).tstates, time.monotonic()
    z80.classify_debug(records()[1]["x"])
    t1, w1 = src.capture(0).tstates, time.monotonic()
    src.speed = 0
    mhz = (t1 - t0) / (w1 - w0) / 1e6
    # The upper bound proves the throttle (unthrottled ticks runs at well over 100 MHz).
    # The lower one is loose: a busy laptop can only make the emulator slower.
    assert 1.2 < mhz < 2.3, mhz


def test_pause_stops_between_instructions(live):
    z80, src = live
    rec = records()[3]
    src.speed = 1_000_000
    result = {}
    t = threading.Thread(target=lambda: result.update(digit=z80.classify(rec["x"])))
    t.start()
    time.sleep(0.5)
    src.paused = True
    time.sleep(0.2)
    first = src.capture(0)
    time.sleep(0.5)
    second = src.capture(0)
    assert second.tstates == first.tstates and second.regs == first.regs
    assert not second.activity.any()                                 # no accesses either
    src.paused = False
    src.speed = 0
    t.join(timeout=30)
    assert result["digit"] == rec["digit"]


def test_without_shm_it_is_plain_ticks():
    prog = os.path.join(ROOT, BUILDS["emu"][0])
    if not os.path.exists(prog):
        pytest.skip("build/emu.bin not built")
    z80 = Z80NN("emu", prog, ticks=VIZ_TICKS)
    assert z80.version() == "Z80NN 2"
    z80.load_weights(WEIGHTS)
    rec = records()[0]
    assert z80.classify(rec["x"]) == rec["digit"]
    z80.close()
