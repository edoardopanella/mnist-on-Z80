"""host/z80debug.py: map parsing, and snapshots through G/M on the emulated Z80.

The map tests need build/emu.map and build/board.map (make emu rom); the
snapshot tests also need z88dk-ticks. Missing pieces are skipped.
"""

import os
import shutil
import sys
import threading
import time

import numpy as np
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
    sp = sym.stack_top - 0x100
    blocks = sym.blocks(sp)
    assert blocks[0][1] == 0 and blocks[-1][2] == 0x10000                    # all of memory
    assert all(a[2] == b[1] for a, b in zip(blocks, blocks[1:]))             # no gaps
    assert all(start < end for _, start, end in blocks)
    titles = [t for t, _, _ in blocks]
    assert {"Program", "Weights", "Image", "Results", "Serial", "Debug", "Free", "Stack"} == set(titles)
    assert len(titles) == len(set(titles) - {"Free"}) + titles.count("Free")  # one block per title
    for name, title in (("_main", "Program"), ("_weights", "Weights"), ("_x", "Image"),
                        ("_trace", "Results"), ("_buf", "Serial"), ("_line", "Serial"),
                        ("_stack_copy", "Debug"), ("_regs", "Debug")):
        assert sym.block_of(sym[name], blocks) == title
        assert sym.block_of(sym[name] + sym.size_of(name) - 1, blocks) == title
    assert sym.block_of(sp - 0x100, blocks) == "Stack"                       # room for pushes
    assert sym.block_of(sp - 0x200, blocks) == "Free"
    deep = sym.stack_top - 0x1C9
    assert sym.block_of(deep - 0x100, sym.blocks(deep)) == "Stack"
    assert sym.blocks() == sym.blocks(sym.stack_top)                         # default: empty stack


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


def test_drawing_app_stop():
    """Stop drops the running answer; Recognize while busy stops, then runs the new drawing."""
    tk = pytest.importorskip("tkinter")
    from draw import DrawingApp

    go = threading.Event()          # the fake Z80 answers when this is set
    submitted, events = [], []

    def classify(x):
        submitted.append(int(x.sum()))
        assert go.wait(5)
        go.clear()
        return str(int(x.sum()))

    def on_stop():
        events.append("stop")
        go.set()

    try:
        app = DrawingApp(on_submit=classify, on_stop=on_stop, on_stopped=lambda: events.append("stopped"))
    except tk.TclError:
        pytest.skip("no display")
    app.withdraw()

    def wait(cond):
        deadline = time.time() + 5
        while not cond() and time.time() < deadline:
            app.update()
            time.sleep(0.01)
        assert cond()

    def draw(width):
        app.pen.rectangle([0, 0, 280, 280], fill=0)
        app.pen.rectangle([100, 40, 100 + width, 240], fill=255)

    try:
        draw(30)
        app.recognize()
        wait(lambda: len(submitted) == 1)
        assert app.busy and str(app.stop_button["state"]) == "normal"
        app.stop()
        wait(lambda: not app.busy)
        assert events == ["stop", "stopped"]
        assert app.result["text"] == "" and app.status["text"].startswith("Stopped")
        assert str(app.stop_button["state"]) == "disabled"

        app.recognize()                     # same drawing again
        wait(lambda: len(submitted) == 2)
        draw(80)
        app.recognize()                     # while busy: stop, then this one
        wait(lambda: len(submitted) == 3)
        assert events == ["stop", "stopped", "stop", "stopped"]
        assert submitted[2] != submitted[1] and app.busy
        go.set()
        wait(lambda: not app.busy)
        assert app.result["text"] == str(submitted[2])
        assert events == ["stop", "stopped", "stop", "stopped"]
    finally:
        go.set()
        app.destroy()


def test_image_rows():
    from z80debug import image_bytes, image_rows

    img = np.zeros((28, 28), dtype=np.uint8)
    img[3, 5] = img[27, 27] = img[10, 0] = 1
    lit, offset = image_rows(bytes(img.ravel()) + b"\xff" * 10, bits=False)
    assert (lit == img).all() and offset[3, 5] == 3 * 28 + 5
    packed = np.packbits(img.ravel()).tobytes()                 # as pack_image() sends it
    assert len(packed) == image_bytes(True) == 98 and image_bytes(False) == 784
    lit, offset = image_rows(packed, bits=True)
    assert (lit == img).all() and offset[1, 0] == 3 and offset[3, 5] == (3 * 28 + 5) // 8
    lit, _ = image_rows(packed[:50], bits=True)                 # short read: the rest is dark
    assert lit[27].sum() == 0
    value, _ = image_rows(bytes([0x7F, 0x80, 0xFF, 0x05]), bits=False)    # weights: int8
    assert list(value[0, :5]) == [127, -128, -1, 5, 0]


def test_snapshot_image_views(source):
    """After a real inference, x (bytes) and buf (bits) both show the digit that was sent."""
    from export import read_test_vectors
    from z80debug import image_bytes, image_rows

    z80, sym = source.z80, source.symbols
    case = read_test_vectors(TESTVEC)[3]
    z80.classify_debug(case["x"])
    expected = np.asarray(case["x"]).reshape(28, 28)
    x = source.capture(sym["_x"], image_bytes(False))
    buf = source.capture(sym["_buf"], image_bytes(True))
    assert len(x.view) == 784 and len(buf.view) == 98
    assert (image_rows(x.view, bits=False)[0] == expected).all()
    # buf[0..2] hold the arguments of the capture's own last M command (cmd_memory
    # parses into buf): 24 bits of row 0. The rest is the image that was sent.
    lit = image_rows(buf.view, bits=True)[0]
    assert (lit[1:] == expected[1:]).all() and (lit[0, 24:] == expected[0, 24:]).all()


def fake_panel(tk, memory, stack_words=100):
    """A DebugPanel on a fake Z80: memory is a 64 KB bytearray, captures are synchronous."""
    from debug_panel import DebugPanel
    from z80debug import Snapshot

    sym = symbols_for("emu")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    sp = sym.stack_top - 2 * stack_words
    regs = dict(af=0, bc=0, de=0, hl=0, ix=0, iy=0, sp=sp, pc=sym["_main"], af2=0, bc2=0, de2=0, hl2=0, i=0, r=0)

    def capture(addr, n=256):
        view = bytes(memory[(addr + k) & 0xFFFF] for k in range(n))
        stack = [(sp + 2 * k, memory[(sp + 2 * k) & 0xFFFF]) for k in range(stack_words)]
        return Snapshot(dict(regs), stack, [], addr, view, np.zeros(65536, np.uint8), [], live=False)

    panel = DebugPanel(root, sym, capture, lambda fn, done: done(fn(), None))
    panel.pack()
    panel.refresh()
    root.update()
    return root, panel, sym


def test_panel_memory_scrolling():
    tk = pytest.importorskip("tkinter")
    from types import SimpleNamespace

    root, panel, _ = fake_panel(tk, bytearray(65536))
    try:
        panel.set_view(0x1000)
        assert panel.snap.view_addr == 0x1000 and len(panel.snap.view) == 256
        panel._on_mem_wheel(SimpleNamespace(num=0, delta=-1))            # macOS: one step down
        assert panel.view_addr == 0x1010
        panel._on_mem_wheel(SimpleNamespace(num=0, delta=240))           # Windows: two notches up
        assert panel.view_addr == 0x0FF0
        panel._scroll_pages(1)
        assert panel.view_addr == 0x10F0 and panel.snap.view_addr == 0x10F0
        panel._on_mem_scroll("moveto", "1.0")                            # bar at the bottom
        assert panel.view_addr == 0xFF00
        panel.scroll_rows(16)                                            # wraps past FFFF
        assert panel.view_addr == 0x0000
        panel._on_mem_scroll("scroll", "-1", "units")
        assert panel.view_addr == 0xFFF0
        first, last = panel.mem_scroll.get()
        assert abs(first - 0xFFF0 / 65536) < 1e-6
        panel.addr_entry.delete(0, "end")
        panel.addr_entry.insert(0, "8a00")
        panel._on_addr_entry(None)
        assert panel.view_addr == 0x8A00
    finally:
        root.destroy()


def test_panel_image_view():
    tk = pytest.importorskip("tkinter")
    memory = bytearray(65536)
    img = np.zeros((28, 28), dtype=np.uint8)
    img[4:24, 12:16] = 1                                                 # a "1"
    sym = symbols_for("emu")
    memory[sym["_x"]:sym["_x"] + 784] = bytes(img.ravel())
    memory[sym["_buf"]:sym["_buf"] + 98] = np.packbits(img.ravel()).tobytes()
    root, panel, _ = fake_panel(tk, memory)
    try:
        panel.view_mode.set("image")
        panel.goto.set("x")
        panel._on_goto(None)
        panel._on_view_mode()
        assert panel.pixel.get() == "byte" and len(panel.snap.view) == 784
        assert (panel.img_rows[1] == img).all()
        panel.scroll_rows(1)                                             # one image row
        assert panel.view_addr == sym["_x"] + 28
        panel.goto.set("buf")
        panel._on_goto(None)
        assert panel.pixel.get() == "bit" and len(panel.snap.view) == 98
        assert (panel.img_rows[1] == img).all()
        panel.view_mode.set("hex")
        panel._on_view_mode()
        assert len(panel.snap.view) == 256
    finally:
        root.destroy()


def test_panel_stack_keeps_its_place():
    tk = pytest.importorskip("tkinter")
    memory = bytearray(65536)
    root, panel, sym = fake_panel(tk, memory)
    try:
        panel.stack.yview(40)
        root.update()
        top = panel.stack.index("@0,0")
        panel.refresh()                                                  # same content
        root.update()
        assert panel.stack.index("@0,0") == top
        sp = panel.snap.regs["sp"]
        memory[sp + 2] = 0x55                                            # one word changes
        panel.refresh()
        root.update()
        assert panel.stack.index("@0,0") == top
        assert "0055" in panel.stack.get("1.0", "end")
    finally:
        root.destroy()
