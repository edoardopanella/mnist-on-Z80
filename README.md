# MNIST on Z80

Handwritten digit recognition on a Z80. A 784 → 32 → 10 MLP is trained in floating point with PyTorch, converted to integer-only arithmetic, and run in C on the Z80. The emulator is DeZog's zsim; the hardware target is Grant Searle's breadboard Z80. The C code must reproduce the Python integer reference bit for bit.

## Layout

```
training/   Python pipeline: train → quantize → check → export
firmware/   C inference library and Z80 programs
host/       laptop tools: drawing app, Z80 link library, DeZog UI
tests/      pytest tests and DeZog input generator
docs/       spec.md (formats and algorithm), plan.md, toolchain guide
artifacts/  generated files (MNIST data, weights, test vectors), not committed
```

## 1. Training pipeline

Run from the repository root, in this order (`make train` runs them all). Each step reads what the previous one wrote to `artifacts/`.

```
python training/train.py       # -> artifacts/mlp.pt
python training/quantize.py    # -> artifacts/quantized.npz
python training/reference.py   # prints float vs integer accuracy
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
| `firmware/z80_test_main.c` | Toolchain sanity check: writes a fixed value to a mailbox at `0xF000`. |
| `firmware/z80_nn_main.c` | Runs `nn_infer` on one test case loaded into memory. Writes to the mailbox whether `acc1`, `h_q`, `acc2` and the digit all match `testvec.bin`. |
| `firmware/z80_draw_main.c` | Interactive firmware for DeZog. It reads a packed image from simulated I/O ports, runs `nn_infer`, and sends back the digit, the activations and a register snapshot. |
| `firmware/z80_serial_main.c` | Line-protocol firmware (`docs/spec.md` §7.5), the Z80 side of `host/z80nn.py`. Commands: `V` version, `W` weight chunk, `C` checksum, `I` classify, `D` classify with intermediate values. Two builds differ only in how characters move. `make emu` (`build/emu.bin`) uses the console of the `z88dk-ticks` emulator. `make rom` (`build/board.rom`, the EEPROM image) polls the board's MC6850 ACIA at ports `0x80`/`0x81` (115,200 baud with the 7.3728 MHz crystal). |

### Demo: draw on the laptop, classify on the Z80

```
make demo                                  # emulator: builds build/emu.bin, runs host/demo.py
make rom                                   # board: write build/board.rom to the EEPROM, then
make demo PORT=/dev/tty.usbserial-XXXX     #        same window, classified by the board
```

`host/demo.py` connects through `Z80NN`, uploads the weights, checks the checksum, and opens the drawing window. Each "Recognize" sends the image to the Z80, which returns the digit. One classification costs about 5M T-states, roughly 0.7 s at 7.3728 MHz.

Board memory: code in ROM from `0x0000` (about 3.6 KB); weights and buffers in RAM from `0x8000` to `0xE97D`; stack below `0xFFFF`. Interrupts stay off.

### DeZog

Memory map: code at `0x0000`, weights at `0x1000`, test case at `0x7800`, variables from `0x8000`, mailbox at `0xF000`.

Running in the DeZog emulator (`.vscode/launch.json` has one launch config per program):

```
make inputs CASE=0   # build/weights_payload.bin (-> 0x1000), build/case.bin (-> 0x7800)
make firmware        # z80_nn_main.c  -> launch config "Z80 zsim", result in the mailbox
make draw            # z80_draw_main.c -> launch config "Z80 draw UI"
```

The "Z80 draw UI" config loads the custom UI in `host/dezog/`: `draw_ui.html` is the drawing panel, and `draw_logic.js` simulates the I/O ports. `make firmware FW=firmware/z80_test_main.c` builds the toolchain sanity check instead. `make` with no target builds everything that is missing.

## 3. Host tools

| File | What it does |
| --- | --- |
| `host/draw.py` | Tkinter drawing canvas. `preprocess()` reproduces MNIST's framing: crop, scale to 20 px, center of mass at (14, 14) in 28×28, binarize. `python host/draw.py` prints the resulting image as ASCII. |
| `host/z80nn.py` | `Z80NN` talks to a local executable, the `z88dk-ticks` emulator, or the board over serial (`pyserial`). It speaks the line protocol of `docs/spec.md` §7.5: version, weight upload with checksum, classify, classify with debug output. |
| `host/demo.py` | Connects `DrawingApp` to `Z80NN`. `python host/demo.py` uses the emulator; `python host/demo.py board <port>` uses the board. |
| `host/ticks_acia.lua` | Port hook that `Z80NN` passes to `z88dk-ticks`. It flushes the emulated ACIA's output, so `build/board.rom` can also run in the emulator. |

## Tests

```
make test     # pytest: preprocessing tests, and all 105 test vectors through z80nn.py + emulator
make check    # integer reference vs float model
```

`tests/test_emu.py` runs the 105 vectors against both builds, `build/emu.bin` and `build/board.rom`, so the EEPROM image is tested before it is written. It skips any build that is missing (`make emu rom`).

## Requirements

Python 3 with `torch`, `torchvision`, `numpy`, `pillow`, `pytest` (and `pyserial` for the board). z88dk and VS Code with DeZog for the Z80 part; see `docs/z80-toolchain-macos-guide.md`.
