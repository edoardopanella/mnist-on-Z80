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

Run from the repository root, in this order. Each step reads what the previous one wrote to `artifacts/`.

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
| `firmware/z80_draw_main.c` | Interactive firmware. It reads a packed image from simulated I/O ports, runs `nn_infer`, and sends back the digit, the activations and a register snapshot. |

Build with z88dk (the command is in the header of `z80_draw_main.c`; use the same flags for the other programs):

```
zcc +z80 -compiler=sdcc -m --list --c-code-in-asm -pragma-define:CRT_MODEL=0 \
    -o build/draw.bin firmware/z80_draw_main.c firmware/nn.c
```

Memory map: code at `0x0000`, weights at `0x1000`, test case at `0x7800`, variables from `0x8000`, mailbox at `0xF000`.

Running in the DeZog emulator:

1. `python tests/make_dezog_inputs.py [case]` writes `build/weights_payload.bin` (loaded at `0x1000`) and `build/case.bin` (loaded at `0x7800`).
2. Run `z80_nn_main` and read the result in the mailbox, or run `z80_draw_main` with the custom UI in `host/dezog/`. `draw_ui.html` is the drawing panel; `draw_logic.js` simulates the I/O ports.

## 3. Host tools

| File | What it does |
| --- | --- |
| `host/draw.py` | Tkinter drawing canvas. `preprocess()` reproduces MNIST's framing: crop, scale to 20 px, center of mass at (14, 14) in 28×28, binarize. `python host/draw.py` prints the resulting image as ASCII. |
| `host/z80nn.py` | `Z80NN` talks to a local executable, the `z88dk-ticks` emulator, or the board over serial (`pyserial`). It speaks the line protocol of `docs/spec.md` §7.5: version, weight upload with checksum, classify, classify with debug output. |

## Tests

```
python -m pytest tests/      # preprocessing tests for host/draw.py
```

## Requirements

Python 3 with `torch`, `torchvision`, `numpy`, `pillow`, `pytest` (and `pyserial` for the board). z88dk and VS Code with DeZog for the Z80 part; see `docs/z80-toolchain-macos-guide.md`.
