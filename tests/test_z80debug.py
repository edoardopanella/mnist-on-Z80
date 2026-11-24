"""host/z80debug.py: map parsing, and snapshots through G/M on the emulated Z80.

The map tests need build/emu.map and build/board.map (make emu rom); the
snapshot tests also need z88dk-ticks. Missing pieces are skipped.
"""

import os
import shutil
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "host"))
sys.path.insert(0, os.path.join(ROOT, "training"))
from z80debug import WRITE, MapSymbols, SnapshotSource, flags  # noqa: E402

WEIGHTS = os.path.join(ROOT, "artifacts", "weights.bin")
TESTVEC = os.path.join(ROOT, "artifacts", "testvec.bin")
BUILDS = {"emu": ("build/emu.bin", "build/emu.map"), "board-rom": ("build/board.rom", "build/board.map")}
PAYLOAD_SIZE = 25581


def symbols_for(build):
    path = os.path.join(ROOT, BUILDS[build][1])
    if not os.path.exists(path):
        pytest.skip(f"{BUILDS[build][1]} not built")
    return MapSymbols(path)


@pytest.mark.parametrize("build", BUILDS)
def test_map_symbols(build):
    sym = symbols_for(build)
    assert sym.size_of("_weights") == PAYLOAD_SIZE
    assert sym.size_of("_x") == 784
    assert sym.nearest(sym["_cmd_infer"]) == "_cmd_infer"
    assert sym.nearest(sym["_cmd_infer"] + 0x10) == "_cmd_infer+0x10"
    assert sym.in_code(sym["_main"]) and not sym.in_code(sym["_x"])
    assert sym.code[0] < sym.code[1] <= sym.data[0] < sym.data[1] <= sym.stack_top
    labels = {r[0]: r for r in sym.regions(sp=sym.stack_top - 0x100)}
    assert {"code", "data", "stack", "weights", "x", "trace", "line", "buf", "regs"} <= set(labels)
    for name in ("weights", "x", "trace", "line", "buf", "regs"):      # variables inside data
        _, start, end, level = labels[name]
        assert level == 1 and sym.data[0] <= start < end <= sym.data[1]


def test_map_board_layout():
    sym = symbols_for("board-rom")
    assert sym.code[0] == 0x0000 and sym.code[1] <= 0x8000     # ROM
    assert sym.data[0] >= 0x8000                               # RAM
    assert sym.stack_top == 0x10000                            # SP starts at 0: wraps to 0xFFFF


def test_flags():
    assert flags(0x00FF) == "SZ5H3PNC"
    assert flags(0x0041) == ".Z.....C"
    assert flags(0xFF00) == "........"


@pytest.fixture(scope="module", params=list(BUILDS))
def source(request):
    from z80nn import Z80NN

    prog = os.path.join(ROOT, BUILDS[request.param][0])
    if shutil.which("z88dk-ticks") is None or not os.path.exists(WEIGHTS):
        pytest.skip("needs z88dk-ticks and artifacts/")
    if not os.path.exists(prog):
        pytest.skip(f"{BUILDS[request.param][0]} not built")
    z = Z80NN("emu", prog)
    z.load_weights(WEIGHTS)
    yield SnapshotSource(z, symbols_for(request.param))
    z.close()


def test_snapshot_after_inference(source):
    from export import read_test_vectors

    z80, sym = source.z80, source.symbols
    recs = read_test_vectors(TESTVEC)
    z80.classify_debug(recs[0]["x"])
    first = source.capture(sym["_x"])
    assert not first.live and first.activity.sum() == 0        # nothing to compare with yet
    z80.classify_debug(recs[1]["x"])
    snap = source.capture(sym["_x"])

    assert snap.acc2 == recs[1]["acc2"].tolist()
    assert snap.view == bytes(recs[1]["x"][:256].tolist())
    assert [name.split("+")[0] for _, name in snap.frames] == ["_cmd_infer", "_main"]
    assert snap.stack[0][0] == snap.regs["sp"]

    changed = {a for a in range(65536) if snap.activity[a]}
    assert all(snap.activity[a] == WRITE for a in changed)     # a snapshot sees writes only
    diff = {sym["_x"] + k for k in range(784) if recs[0]["x"][k] != recs[1]["x"][k]}
    assert diff <= changed                                     # every pixel that changed
    weights = range(sym["_weights"], sym["_weights"] + PAYLOAD_SIZE)
    assert not any(a in weights for a in changed)              # weights are not watched


def test_worker_runs_jobs_in_order_off_the_tk_thread():
    tk = pytest.importorskip("tkinter")
    from draw import Worker

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    worker, seen = Worker(root), []
    main = threading.current_thread()

    def job(k):
        time.sleep(0.01)
        return k, threading.current_thread() is not main

    for k in range(3):
        worker.submit(lambda k=k: job(k), lambda result, error: seen.append((result, error)))
    worker.submit(lambda: 1 / 0, lambda result, error: seen.append((result, type(error))))
    deadline = time.time() + 5
    while len(seen) < 4 and time.time() < deadline:
        root.update()
        time.sleep(0.01)
    root.destroy()
    assert seen == [((0, True), None), ((1, True), None), ((2, True), None), (None, ZeroDivisionError)]
