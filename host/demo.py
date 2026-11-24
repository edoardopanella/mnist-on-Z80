"""
demo.py - Draw a digit and let the Z80 classify it.

    python host/demo.py                          emulator: build/emu.bin in z88dk-ticks (make emu)
    python host/demo.py board /dev/tty.usbserial-XXXX    the board, over serial
    python host/demo.py board /dev/tty.usbserial-XXXX --debug    the board, with the debug panel

Starts the Z80 through z80nn.Z80NN, uploads artifacts/weights.bin (checked
with the checksum), then opens draw.DrawingApp: every "Recognize" sends the
preprocessed image and shows the digit the Z80 returns.

With the debug panel (default on the emulator, --debug on the board) the image
goes with the D command and host/debug_panel.py shows registers, stack and
memory. Symbols come from the program's .map file, which must match the
program that is running. Two modes:
    live      emulator with the patched ticks (make ticks-viz): read from the
              memory shared with ticks 20 times a second, also during an
              inference, with every read, write and instruction fetch
    snapshot  board, or emulator without the patch (or --no-live): after each
              result, read with the G and M commands
"""

import argparse
import os
import shutil
import tempfile

from draw import DrawingApp
from z80nn import Z80NN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "artifacts", "weights.bin")
EMU_BIN = os.path.join(ROOT, "build", "emu.bin")
BOARD_MAP = os.path.join(ROOT, "build", "board.map")
VIZ_TICKS = os.path.join(ROOT, "build", "ticks-viz", "src", "ticks", "z88dk-ticks")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("target", nargs="?", default="emu", choices=["emu", "board", "local"])
    parser.add_argument("where", nargs="?", default=EMU_BIN,
                        help="emu/local: program to run (default build/emu.bin); board: serial port")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None,
                        help="show the debug panel (default: on for emu, off for board and local)")
    parser.add_argument("--map", help="the program's .map file (default: next to the emu program; "
                                      "build/board.map for the board)")
    parser.add_argument("--live", action=argparse.BooleanOptionalAction, default=None,
                        help="emu: live debug view (default: on when build/ticks-viz exists)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="live view: emulated clock in MHz, 0 = as fast as possible (default 1)")
    args = parser.parse_args()
    debug = args.debug if args.debug is not None else args.target == "emu"
    if debug and args.target == "local":
        parser.error("--debug needs the Z80 firmware (G and M commands)")
    map_path = args.map or (os.path.splitext(args.where)[0] + ".map" if args.target == "emu" else BOARD_MAP)
    live = debug and args.target == "emu" and (args.live if args.live is not None else os.path.exists(VIZ_TICKS))
    if live and not os.path.exists(VIZ_TICKS):
        parser.error("--live needs the patched emulator: make ticks-viz")

    shm_dir = shm = None
    if live:
        shm_dir = tempfile.mkdtemp(prefix="z80viz-")
        shm = os.path.join(shm_dir, "ticks.shm")
        z80 = Z80NN("emu", args.where, ticks=VIZ_TICKS, shm=shm)
    else:
        z80 = Z80NN(args.target, args.where)
    source = None
    try:
        version = z80.version()
        print("Connected:", version)
        if debug and version != "Z80NN 2":
            raise RuntimeError("this firmware has no debug commands: rebuild it (make emu / make rom)")
        z80.load_weights(WEIGHTS)
        print("Weights loaded, checksum OK")

        if not debug:
            DrawingApp(on_submit=lambda x: str(z80.classify(x))).mainloop()
            return

        from debug_panel import DebugPanel
        from z80debug import LiveSource, MapSymbols, SnapshotSource

        symbols = MapSymbols(map_path)
        if live:
            source = LiveSource(shm, symbols)
            source.speed = int(args.speed * 1e6)    # the weights went up at full speed
            print(f"Live view, emulated clock {args.speed:g} MHz" if args.speed else "Live view, full speed")
            note = ("Live (patched ticks): every read, write and instruction fetch, "
                    "fading in about half a second.\nClock: how fast the emulated Z80 runs; "
                    "Pause stops it between two instructions.")
        else:
            source = SnapshotSource(z80, symbols)
            note = ("Snapshot after each inference: registers and stack as nn_infer returned, "
                    "memory as it is now.\nRed = changed since the previous snapshot "
                    "(no live reads or prog: the firmware cannot see the bus).")
            if args.target == "emu":
                print("Snapshot view (make ticks-viz for the live one)")

        def panel(app):
            p = DebugPanel(app, symbols, source.capture, app.run_in_background, note=note,
                           live=source if live else None)
            p.refresh()                     # memory map and registers before the first digit
            panels.append(p)
            return p

        # Stop (live only): the inference finishes at full speed and its answer is dropped
        panels = []
        stop = dict(on_stop=lambda: panels[0].fast_forward(),
                    on_stopped=lambda: panels[0].restore_clock()) if live else {}
        def classify(x):
            if live:
                panels[0].start_digit()     # the T-states count restarts for this digit
            return str(z80.classify_debug(x)[0])

        DrawingApp(on_submit=classify, side_panel=panel, **stop).mainloop()
    finally:
        if live and source is not None:
            source.paused = False           # a paused ticks would not see the end of input
        z80.close()
        if source is not None and live:
            source.close()
        if shm_dir:
            shutil.rmtree(shm_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
