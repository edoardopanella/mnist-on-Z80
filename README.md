# MNIST on Z80

Handwritten digit recognition on an 8-bit Z80 processor. You draw a digit on the laptop; a Z80 classifies it with a small neural network and sends the answer back.

- A 784 → 32 → 10 neural network (MLP) is trained in floating point with PyTorch, on binarized MNIST.
- It is converted to integer-only arithmetic: `int8` weights, `int32` sums, no floating point on the Z80.
- The C inference code runs on the Z80 and reproduces the Python integer reference **bit for bit**.
- The Z80 runs in the `z88dk-ticks` emulator. The same firmware also builds as an EEPROM image intended for Grant Searle's breadboard Z80, and the emulator runs that exact image through all the tests.

| Model | Accuracy on the 10,000 MNIST test images |
| --- | --- |
| Float (PyTorch) | 96.01% |
| Integer (Python reference, and the Z80) | 95.96% |
| Float and integer give the same digit | 99.67% |

One classification takes about 5 million Z80 clock cycles: about 0.7 s at 7.3728 MHz, the clock of the board the firmware is meant for.

Federico Scaffidi Muta, Edoardo Panella. Software Engineering (20875) project, Bocconi University, Fall 2026.

## Contents

1. [Repository structure](#repository-structure)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Reproducing the results](#reproducing-the-results)
5. [Running on the board](#running-on-the-board)
6. [How it works](#how-it-works)
7. [Make targets](#make-targets)
8. [Further documentation](#further-documentation)
9. [License](#license)

## Repository structure

```text
mnist-on-Z80/
├── src/
│   ├── training/     Python: train the network, quantize it, check it, export it for the Z80
│   ├── firmware/     C code for the Z80: the inference library and the Z80 programs
│   ├── host/         Python on the laptop: drawing app, link to the Z80, network view, debug panel
│   └── tools/        ticks-viz.patch: adds a live view to the z88dk-ticks emulator
├── tests/            pytest tests: the emulated Z80 against Python, bit for bit
├── artifacts/        weights.bin and testvec.bin: the trained network and test vectors, for the Z80
├── docs/             guide.md, spec.md, Z80 toolchain installation guide
├── Makefile          every build, test and demo command (see "Make targets")
├── pyproject.toml    Python dependencies (uv.lock and requirements.txt pin the exact versions)
└── LICENSE
```

Generated and not committed: `build/` (the Z80 programs), `artifacts/data/` (MNIST, downloaded on first training), `artifacts/mlp.pt` and `artifacts/quantized.npz` (intermediate training outputs).

## Requirements

Developed and tested on macOS (Apple silicon).

| What | Needed for | How to get it |
| --- | --- | --- |
| Python 3.12 or newer, with Tkinter | All Python code (the drawing window uses Tk) | [uv](https://docs.astral.sh/uv/) installs a Python that includes Tk. With Homebrew's Python, also `brew install python-tk@3.12` |
| [uv](https://docs.astral.sh/uv/getting-started/installation/), or pip | The virtual environment | `curl -LsSf https://astral.sh/uv/install.sh \| sh`, or use pip (below) |
| [z88dk](https://github.com/z88dk/z88dk): `zcc` and `z88dk-ticks` on `PATH` | Compiling C for the Z80, and emulating it | [docs/z80-toolchain-macos-guide.md](docs/z80-toolchain-macos-guide.md), Step 1 (Step 2, sjasmplus, is not needed here) |
| make, a C compiler, patch | The build, and the live debug view | Xcode command-line tools: `xcode-select --install` |
| *Optional:* z88dk's source tree (`src/` and `ext/` next to its `bin/`) | The live debug view (`make ticks-viz`) | There when z88dk is built from source (toolchain guide, Option C). Check: `ls "$(dirname "$(command -v zcc)")/../src/ticks"` |
| *Optional:* VS Code with DeZog | Stepping through the C code on a simulated Z80 | Toolchain guide, Step 3 |
| *Optional:* Grant Searle's Z80 board, an EEPROM programmer, a USB-serial adapter | Running on the hardware the firmware is intended for | [Running on the board](#running-on-the-board) |

## Setup

```bash
git clone https://github.com/edoardopanella/mnist-on-Z80.git
cd mnist-on-Z80

# Python environment in .venv, with the exact versions of uv.lock
uv sync
# or, without uv:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Check the setup
.venv/bin/python -c "import tkinter, torch; print('Python OK')"
command -v zcc z88dk-ticks          # both must print a path
```

The Makefile uses `.venv/bin/python` by itself, so you do not need to activate the environment for `make`. To use another interpreter: `make PYTHON=/path/to/python ...`.

## Reproducing the results

Run every command from the repository root. Steps 1 to 3 use the committed network in `artifacts/`; step 4 trains it again from scratch.

### 1. Build the Z80 programs

```bash
make
```

This compiles the firmware with z88dk into `build/`: `emu.bin` (for the emulator's console), `board.rom` (the EEPROM image, which the emulator can also run) and `firmware_*.bin` (for DeZog).

### 2. Run the tests

```bash
make test
```

This builds what the tests need, including the patched emulator when z88dk's sources are available, and then runs pytest. Expected result: **`53 passed`**. The central test runs all 105 test vectors on the emulated Z80, with both builds, and checks that every intermediate value matches Python exactly. If something is missing (for example z88dk's sources), the affected tests are skipped and listed with the reason.

### 3. Run the demo

```bash
make demo
```

A window opens. Draw a digit with the mouse and press **Recognize** (or Enter): the emulated Z80 classifies it and the digit appears below the drawing. **Clear** (or Esc) starts again. Under the drawing you see the network's hidden neurons and the 10 scores; on the right, a debug panel shows the Z80's memory, registers and stack.

With the patched emulator (built by `make test`, or once with `make ticks-viz`) the panel is live: the Z80 runs at 1 MHz and you can watch it compute neuron by neuron. Without it, the panel updates after each digit. All demo options are in [docs/guide.md §4](docs/guide.md#4-running-the-demo).

### 4. Train the network from scratch (optional)

```bash
make train    # downloads MNIST, trains, quantizes, exports (under a minute on a laptop CPU, plus the download)
make check    # prints the accuracies
```

Expected output of `make check`:

```text
Float accuracy:       96.01%
Integer accuracy:     95.96%
Float/int agreement:  99.67%
```

`make train` overwrites `artifacts/weights.bin` and `artifacts/testvec.bin`. Training is seeded, and on our machine it reproduces the committed files byte for byte; `git status artifacts/` shows whether yours differ. After retraining, run `make test` again: it checks the Z80 against the new test vectors.

## Running on the board

The board build is intended for Grant Searle's breadboard Z80: a Z80 at 7.3728 MHz, code in an EEPROM from `0x0000`, RAM from `0x8000`, and an MC6850 ACIA serial port at 115,200 baud.

1. Build the EEPROM image: `make rom` writes `build/board.rom`.
2. Write it to the EEPROM, for example `minipro -p <chip> -w build/board.rom -s` (`-s` accepts a file smaller than the chip).
3. Put the EEPROM in the board, connect the USB-serial adapter and find its port: `ls /dev/tty.usb*`.
4. Run the demo on it: `make demo PORT=/dev/tty.usbserial-XXXX`. Add `DEBUG=1` for the debug panel, read over the serial line.

## How it works

```text
 laptop (Python)                                              Z80 (C, compiled with z88dk)
 ───────────────                                              ────────────────────────────
 src/training/   train → quantize → export
                                      │
                                      ▼
                         artifacts/weights.bin ─── uploaded at start ───► weights in RAM
                                                                                │
 src/host/       drawing window ─── 28×28 image, as text lines ────────► nn_infer()  (src/firmware/)
                 network view   ◄── digit and intermediate values ──────────────┘
                 debug panel
```

1. **Training** (`src/training/`). PyTorch trains the MLP on MNIST with every pixel set to 0 or 1. The weights are then quantized to `int8`, with one scale per layer. The float rescale between the two layers becomes an integer multiply and shift. `reference.py` runs this integer network in Python; it is the reference the Z80 must match exactly. `export.py` writes the weights, and 105 test cases with every intermediate value, as little-endian binaries.
2. **Inference** (`src/firmware/`). `nn.c` implements the same integer network in portable C, correct with the Z80's 16-bit `int`. The inputs are binary, so the first layer needs only additions.
3. **Link** (`src/firmware/z80_serial_main.c`, `src/host/z80nn.py`). The laptop and the Z80 exchange text lines: upload the weights, send an image, read back the digit, the registers or any memory. The same firmware is built for the emulator and for the board. Only the way characters travel differs: the emulator's console, or the board's serial chip (ACIA), which `z88dk-ticks` also emulates.
4. **Demo and debug view** (`src/host/`). The drawing window crops, scales and centres the digit the way MNIST did, then sends it. A patched `z88dk-ticks` shares the Z80's memory with the laptop, so the debug panel shows every read, write and instruction fetch while the network runs.

Details of each part: [docs/guide.md](docs/guide.md).

## Make targets

| Command | What it does |
| --- | --- |
| `make` | Builds every Z80 program into `build/` (and the DeZog inputs). Trains only if `artifacts/weights.bin` is missing |
| `make test` | Builds what the tests need, then runs all tests |
| `make demo` | Drawing demo on the emulator, running the EEPROM image `build/board.rom` |
| `make demo PORT=/dev/tty.usbserial-XXXX` | Drawing demo on the board, over its serial port (`DEBUG=1`: with the debug panel) |
| `make train` | Trains, quantizes and exports again from scratch |
| `make check` | Prints float vs integer accuracy |
| `make emu` / `make rom` | Builds the emulator program `build/emu.bin` / the EEPROM image `build/board.rom` |
| `make ticks-viz` | Builds the patched emulator for the live debug view in `build/ticks-viz` |
| `make firmware`, `make inputs CASE=n` | DeZog program and inputs for one test case ([guide §5](docs/guide.md#5-debugging-with-dezog)) |
| `make clean` | Removes `build/`, the patched emulator included; `artifacts/` are kept |

## Further documentation

| Document | Contents |
| --- | --- |
| [docs/guide.md](docs/guide.md) | Each part in detail: the training scripts, the firmware, the serial protocol, every demo option, the debug panel, DeZog, the tests |
| [docs/spec.md](docs/spec.md) | The contract between the parts: the quantization scheme, the integer algorithm, the binary file formats, the protocol, the live-view file |
| [docs/z80-toolchain-macos-guide.md](docs/z80-toolchain-macos-guide.md) | Installing z88dk and VS Code with DeZog on macOS |

## License

[MIT](LICENSE), © 2026 Federico Scaffidi Muta, Edoardo Panella.

`src/tools/ticks-viz.patch` modifies the source code of [z88dk](https://github.com/z88dk/z88dk), which is distributed under its own license, the Clarified Artistic License. MNIST is not included in this repository; it is downloaded by torchvision when the network is trained.
