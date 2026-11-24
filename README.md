# MNIST on Z80

Handwritten digit recognition on a Z80. A 784 → 32 → 10 MLP is trained in floating point with PyTorch, converted to integer-only arithmetic, and run in C on the Z80. You draw a digit on the laptop, and the Z80 classifies it, either in the `z88dk-ticks` emulator or on Grant Searle's breadboard Z80. The C code reproduces the Python integer reference bit for bit.

## Layout

```
training/   Python pipeline: train → quantize → check → export
firmware/   C inference library and Z80 programs
host/       laptop tools: drawing app, Z80 link library, demo, DeZog UI
tests/      pytest tests and DeZog input generator
docs/       spec.md (formats and algorithm), plan.md, toolchain guide
artifacts/  generated files (MNIST data, weights, test vectors), not committed
```

## Quick start

```
make          # trains if artifacts/ is missing, then builds every Z80 program
make test     # all tests, including the 105 test vectors on the emulated Z80
make demo     # draw a digit, the emulated Z80 classifies it
```

Requirements: Python 3 with `torch`, `torchvision`, `numpy`, `pillow`, `pytest` (and `pyserial` for the board). Also z88dk (`zcc`, `z88dk-ticks`) on `PATH`, and VS Code with DeZog for step-by-step debugging. Installation: `docs/z80-toolchain-macos-guide.md`.

## 1. Training pipeline

Run from the repository root, in this order (`make train` runs them all). Each step reads what the previous one wrote to `artifacts/`.

```
python training/train.py       # -> artifacts/mlp.pt
python training/quantize.py    # -> artifacts/quantized.npz
python training/reference.py   # prints float vs integer accuracy (also: make check)
python training/export.py      # -> artifacts/weights.bin, artifacts/testvec.bin
```

| File | What it does |
| --- | --- |
| `train.py` | Downloads MNIST and binarizes it (pixel = 1 if > 0.5). Trains the MLP (Adam, lr 1e-3, batch 64, 10 epochs, seed 0) and saves the float weights. Also defines `MLP`, `get_data()` and the paths that the other scripts import. |
| `quantize.py` | Quantizes weights to `int8` (symmetric, one scale per layer) and biases to `int32`. Calibrates the hidden-layer scale on the training set. Replaces the float rescale between the layers with an integer multiply `M0` and a right `shift`, choosing the largest `shift` that cannot overflow 32 bits. |
| `reference.py` | `infer_int()` is the integer-only network, computed step by step exactly as the C code does: layer 1, ReLU, requantize, layer 2, argmax. `evaluate()` compares it with the float model. |
| `export.py` | Writes little-endian binaries for the C code: `weights.bin` with the weights, and `testvec.bin` with 105 test cases (100 MNIST images plus 5 edge cases). Each case stores the image and every intermediate value. The script reads both files back to check them. Formats: `docs/spec.md` §5. |

Test-set results: float 96.01%, integer 95.96%, float/integer agreement 99.67%.

## 2. Inference on the Z80

| File | What it does |
| --- | --- |
| `firmware/nn.h`, `nn.c` | `nn_infer()` is the C version of `infer_int()`. It uses only fixed-width types, so it is correct with the Z80's 16-bit `int`. Because the inputs are binary, layer 1 needs only additions. `struct nn_weights` has the same layout as the `weights.bin` payload. |
| `firmware/z80_serial_main.c` | The demo firmware: receives commands from the laptop, runs `nn_infer`, replies. See section 3. |
| `firmware/z80_nn_main.c` | DeZog test: runs `nn_infer` on one test case loaded into memory. Writes to the mailbox whether `acc1`, `h_q`, `acc2` and the digit all match `testvec.bin`. |
| `firmware/z80_draw_main.c` | DeZog drawing demo: reads a packed image from simulated I/O ports, runs `nn_infer`, and sends back the digit, the activations and a register snapshot. |
| `firmware/z80_test_main.c` | Toolchain sanity check: writes a fixed value to a mailbox at `0xF000`. |

## 3. How the laptop talks to the Z80

The drawing window runs on the laptop; the network runs on the Z80. They exchange text lines. The firmware, the bridge and the protocol are the same whether the Z80 is emulated or real; only the way characters travel changes.

```
host/draw.py ─► host/demo.py ─► host/z80nn.py ─┬─ emu ──► z88dk-ticks running build/emu.bin    (console)
  drawing         glue           Z80NN bridge  │          z88dk-ticks running build/board.rom  (emulated ACIA)
  window                                       └─ board ─► USB serial ─► ACIA ─► build/board.rom on the EEPROM
```

### The protocol (`docs/spec.md` §7.5)

One command per line, one reply per command. Bytes are sent as hexadecimal text.

| Command | Reply | Meaning |
| --- | --- | --- |
| `V` | `Z80NN 1` | Version: checks that the Z80 is there |
| `W oooo <hex>` | `OK` | Writes up to 64 bytes of the weights at offset `oooo` (400 lines in total) |
| `C` | `C xxxx` | 16-bit checksum of the weights received, compared by the laptop |
| `I <196 hex>` | `R d` | Classifies a 28×28 image packed in 98 bytes; `d` is the digit |
| `D <196 hex>` | `A1 …`, `H …`, `A2 …`, `R d` | Same as `I`, plus every intermediate value |

Errors (`ERR CMD`, `ERR HEX`, `ERR LEN`, `ERR OFS`, `ERR LONG`) leave the firmware running. The weights live in the Z80's RAM, so they are uploaded again every time the program starts.

### One firmware, two builds

`firmware/z80_serial_main.c` is a single source file. Only its three character functions (`ser_init`, `ser_getc`, `ser_putc`) differ between builds:

| Build | Command | Output | Characters travel through |
| --- | --- | --- | --- |
| Emulator | `make emu` | `build/emu.bin` | The console of `z88dk-ticks` (its stdin/stdout). Uses `fgetc_cons()`, which does not echo the input back. |
| Board | `make rom` | `build/board.rom` | The board's MC6850 ACIA at ports `0x80`/`0x81`, polled with interrupts off. 115,200 baud with the 7.3728 MHz crystal. |

Board memory map: code in ROM from `0x0000` (about 3.6 KB); weights and buffers in RAM from `0x8000` to `0xE97D`; stack below `0xFFFF`.

### The bridge: `host/z80nn.py`

`Z80NN(target, where)` hides where the Z80 is. After it connects, `version()`, `load_weights()`, `classify()` and `classify_debug()` work the same way everywhere.

| `target` | `where` | What it does |
| --- | --- | --- |
| `emu` | a Z80 program (`build/emu.bin` or `build/board.rom`) | Starts `z88dk-ticks` on it. Two options matter: `-counter`, because ticks otherwise stops after about 10 classifications, and `-lua host/ticks_acia.lua` (see below). |
| `board` | a serial port, e.g. `/dev/tty.usbserial-XXXX` | Opens it at 115,200 baud, 8N1, RTS/CTS flow control |
| `local` | a program on the laptop | Runs it and talks to it through stdin/stdout (none is built by this repo) |

### Testing the EEPROM image before writing it

`z88dk-ticks` already emulates an ACIA on the same ports as the board, but it never flushes its output, so replies sent through a pipe never arrive. `host/ticks_acia.lua` hooks the output port and flushes after every byte. With it, the exact `build/board.rom` that goes on the EEPROM runs in the emulator. `tests/test_emu.py` runs all 105 test vectors on both builds and checks every intermediate value.

## 4. Running the demo

Run everything from the repository root. You need `artifacts/weights.bin`; `make` creates it if it is missing.

### On the emulator

```
make demo
```

This is the same as:

```
make emu                  # -> build/emu.bin
python host/demo.py       # target emu, program build/emu.bin
```

### On the board

```
make rom                  # -> build/board.rom
```

1. Write `build/board.rom` to the EEPROM with your programmer, for example `minipro -p <chip> -w build/board.rom -s`. `-s` accepts a file smaller than the chip.
2. Put the EEPROM in the board, connect the USB-serial adapter, and find its port with `ls /dev/tty.usb*`.
3. Run:

   ```
   make demo PORT=/dev/tty.usbserial-XXXX
   ```

   This is the same as:

   ```
   python host/demo.py board /dev/tty.usbserial-XXXX
   ```

### Arguments of `host/demo.py`

```
python host/demo.py [TARGET] [WHERE]
```

| `TARGET` | `WHERE` | Example |
| --- | --- | --- |
| `emu` (default) | Z80 program for the emulator, default `build/emu.bin` | `python host/demo.py` |
| `board` | Serial port of the board (required) | `python host/demo.py board /dev/tty.usbserial-XXXX` |
| `local` | Program on the laptop that speaks the protocol | `python host/demo.py local ./my_program` |

### What you see

The script prints `Connected: Z80NN 1`, uploads the weights, prints `Weights loaded, checksum OK`, and opens the window. Draw a digit with the mouse, then press **Recognize** or Enter; the digit the Z80 computed appears below. Press **Clear** or Esc to start again. The preview on the right shows the 28×28 image the Z80 receives.

One classification costs about 5 million Z80 clock cycles: instant on the emulator, about 0.7 s on the board at 7.3728 MHz. On the board, uploading the weights takes a few seconds.

## 5. Debugging with DeZog

Memory map: code at `0x0000`, weights at `0x1000`, test case at `0x7800`, variables from `0x8000`, mailbox at `0xF000`. `.vscode/launch.json` has one launch config per program:

```
make inputs CASE=0   # build/weights_payload.bin (-> 0x1000), build/case.bin (-> 0x7800)
make firmware        # z80_nn_main.c   -> launch config "Z80 zsim", result in the mailbox
make draw            # z80_draw_main.c -> launch config "Z80 draw UI"
```

The "Z80 draw UI" config loads the custom UI in `host/dezog/`: `draw_ui.html` is the drawing panel, and `draw_logic.js` simulates the I/O ports. `make firmware FW=firmware/z80_test_main.c` builds the toolchain sanity check instead.

## 6. Host tools

| File | What it does |
| --- | --- |
| `host/draw.py` | Tkinter drawing window. `preprocess()` reproduces MNIST's framing: crop, scale to 20 px, center of mass at (14, 14) in 28×28, binarize. `python host/draw.py` runs it without a Z80 and prints the image as ASCII. |
| `host/z80nn.py` | The `Z80NN` bridge (section 3). |
| `host/demo.py` | Connects `DrawingApp` to `Z80NN` (section 4). |
| `host/ticks_acia.lua` | Flushes the emulated ACIA's output so `build/board.rom` runs in `z88dk-ticks` (section 3). |
| `host/dezog/` | Custom UI for DeZog (section 5). |

## Tests

```
make test     # pytest
make check    # integer reference vs float model
```

| Test | What it checks |
| --- | --- |
| `tests/test_preprocess.py` | The drawing preprocessing: empty input, bounding box, 20 px scaling, centering. |
| `tests/test_emu.py` | All 105 test vectors through `Z80NN` on the emulated Z80, for both `build/emu.bin` and `build/board.rom`: every `acc1`, `h_q`, `acc2` and digit must match Python. Also checks that errors leave the firmware running. A build that is missing is skipped (`make emu rom`). |
