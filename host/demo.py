"""
demo.py - Draw a digit and let the Z80 classify it.

    python host/demo.py                          emulator: build/emu.bin in z88dk-ticks (make emu)
    python host/demo.py board /dev/tty.usbserial-XXXX    the board, over serial

Starts the Z80 through z80nn.Z80NN, uploads artifacts/weights.bin (checked
with the checksum), then opens draw.DrawingApp: every "Recognize" sends the
preprocessed image with the I command and shows the digit the Z80 returns.
"""

import argparse
import os

from draw import DrawingApp
from z80nn import Z80NN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "artifacts", "weights.bin")
EMU_BIN = os.path.join(ROOT, "build", "emu.bin")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("target", nargs="?", default="emu", choices=["emu", "board", "local"])
    parser.add_argument("where", nargs="?", default=EMU_BIN,
                        help="emu/local: program to run (default build/emu.bin); board: serial port")
    args = parser.parse_args()

    z80 = Z80NN(args.target, args.where)
    try:
        print("Connected:", z80.version())
        z80.load_weights(WEIGHTS)
        print("Weights loaded, checksum OK")
        DrawingApp(on_submit=lambda x: str(z80.classify(x))).mainloop()
    finally:
        z80.close()


if __name__ == "__main__":
    main()
