# MNIST on Z80

Handwritten digit recognition on a Z80. A 784 → 32 → 10 MLP is trained in floating point with PyTorch, converted to integer-only arithmetic, and run in C on the Z80. You draw a digit on the laptop, and the Z80 classifies it, either in the `z88dk-ticks` emulator or on Grant Searle's breadboard Z80. The C code reproduces the Python integer reference bit for bit.

## Layout

```
training/   Python pipeline: train → quantize → check → export
firmware/   C inference library and Z80 programs
host/       laptop tools: drawing app, Z80 link library, demo, debug panel
tests/      pytest tests and DeZog input generator
tools/      ticks-viz.patch: z88dk-ticks with a live view for the debug panel
docs/       spec.md (formats and algorithm), plan.md, toolchain guide
artifacts/  generated files (MNIST data, weights, test vectors), not committed
```

## Quick start

```
make             # trains if artifacts/ is missing, then builds every Z80 program
make ticks-viz   # once: the patched emulator for the live debug view (section 4)
make test        # all tests, including the 105 test vectors on the emulated Z80
make demo        # draw a digit, the emulated Z80 classifies it
```

`make clean` removes `build/`, `build/ticks-viz` included: run `make ticks-viz` again after it. The Makefile uses `.venv/bin/python` when there is a `.venv`; set `PYTHON=` to use another interpreter.

Requirements: Python 3 with `torch`, `torchvision`, `numpy`, `pillow`, `pytest` (and `pyserial` for the board). Also z88dk (`zcc`, `z88dk-ticks`) on `PATH`, and VS Code with DeZog for step-by-step debugging. The live debug view (`make ticks-viz`) also needs z88dk's sources (`src/` next to its `bin/`), a C compiler and `patch` (Xcode command-line tools). Installation: `docs/z80-toolchain-macos-guide.md`.

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
| `firmware/z80_test_main.c` | Toolchain sanity check: writes a fixed value to a mailbox at `0xF000`. |
| `firmware/snapregs.h`, `snapregs.c` | `snap_regs()` copies every register into `struct regs`, for the `G` command of `z80_serial_main.c`. |

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
| `V` | `Z80NN 2` | Version: checks that the Z80 is there |
| `W oooo <hex>` | `OK` | Writes up to 64 bytes of the weights at offset `oooo` (400 lines in total) |
| `C` | `C xxxx` | 16-bit checksum of the weights received, compared by the laptop |
| `I <196 hex>` | `R d` | Classifies a 28×28 image packed in 98 bytes; `d` is the digit |
| `D <196 hex>` | `A1 …`, `H …`, `A2 …`, `R d` | Same as `I`, plus every intermediate value |
| `G` | `G af bc de hl ix iy sp pc af' bc' de' hl' i r` | Registers in hex, snapped right after the last `I`/`D` inference (all `0` before the first one). The same inference copies 256 bytes of stack from SP into `stack_copy`, for `M` |
| `M aaaa nn` | `M <hex>` | Reads `nn` bytes (`01`..`40` hex) from address `aaaa`; wraps at `FFFF` |

Errors (`ERR CMD`, `ERR HEX`, `ERR LEN`, `ERR OFS`, `ERR LONG`) leave the firmware running. The weights live in the Z80's RAM, so they are uploaded again every time the program starts.

### One firmware, two builds

`firmware/z80_serial_main.c` is a single source file. Only its three character functions (`ser_init`, `ser_getc`, `ser_putc`) differ between builds:

| Build | Command | Output | Characters travel through |
| --- | --- | --- | --- |
| Emulator | `make emu` | `build/emu.bin` | The console of `z88dk-ticks` (its stdin/stdout). Uses `fgetc_cons()`, which does not echo the input back. |
| Board | `make rom` | `build/board.rom` | The board's MC6850 ACIA at ports `0x80`/`0x81`, polled with interrupts off. 115,200 baud with the 7.3728 MHz crystal. |

Board memory map: code in ROM from `0x0000` (about 3.9 KB); weights, buffers and `stack_copy` in RAM from `0x8000` to about `0xEA97`; stack below `0xFFFF`. The `.map` files in `build/` have the exact addresses (`docs/spec.md` §7.4).

### The bridge: `host/z80nn.py`

`Z80NN(target, where)` hides where the Z80 is. After it connects, `version()`, `load_weights()`, `classify()`, `classify_debug()`, `registers()` and `read_memory()` work the same way everywhere.

| `target` | `where` | What it does |
| --- | --- | --- |
| `emu` | a Z80 program (`build/emu.bin` or `build/board.rom`) | Starts `z88dk-ticks` on it with `-counter`, because ticks otherwise stops after about 10 classifications. Its stdout is a pseudo-terminal (see below). |
| `board` | a serial port, e.g. `/dev/tty.usbserial-XXXX` | Opens it at 115,200 baud, 8N1, RTS/CTS flow control |
| `local` | a program on the laptop | Runs it and talks to it through stdin/stdout (none is built by this repo) |

### Testing the EEPROM image before writing it

`z88dk-ticks` already emulates an ACIA on the same ports as the board, but it never flushes its output, so replies sent through a pipe never arrive. `Z80NN` therefore gives ticks a pseudo-terminal as stdout: on a terminal the C library flushes at every `\n`. With it, the exact `build/board.rom` that goes on the EEPROM runs in the emulator. `tests/test_emu.py` runs all 105 test vectors on both builds and checks every intermediate value.

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

   Add `DEBUG=1` (or `--debug`) for the debug panel (below).

### Arguments of `host/demo.py`

```
python host/demo.py [TARGET] [WHERE] [--debug | --no-debug] [--map FILE] [--live | --no-live] [--speed MHZ]
```

| `TARGET` | `WHERE` | Example |
| --- | --- | --- |
| `emu` (default) | Z80 program for the emulator, default `build/emu.bin` | `python host/demo.py` |
| `board` | Serial port of the board (required) | `python host/demo.py board /dev/tty.usbserial-XXXX` |
| `local` | Program on the laptop that speaks the protocol | `python host/demo.py local ./my_program` |

| Option | Meaning |
| --- | --- |
| `--debug` / `--no-debug` | Show or hide the debug panel. Default: shown on `emu`, hidden on `board` and `local` (`local` cannot have it). |
| `--map FILE` | The `.map` file of the program that runs, for the symbols. Default: next to the emulator program (`build/emu.map`), or `build/board.map` for the board. It must come from the same build as the EEPROM contents. |
| `--live` / `--no-live` | Emulator: live debug view with the patched ticks. Default: on when `make ticks-viz` has been run. |
| `--speed MHZ` | Live view: the emulated clock, default `1` (one inference takes about 5 s to watch); `0` = as fast as possible. Changeable in the panel. |

### What you see

The script prints `Connected: Z80NN 2`, uploads the weights, prints `Weights loaded, checksum OK`, and opens the window. Draw a digit with the mouse, then press **Recognize** or Enter; the digit the Z80 computed appears below. Press **Clear** or Esc to start again. The preview on the right shows the 28×28 image the Z80 receives.

In the live view (emulator, debug panel) a slow clock makes one digit take seconds. **Stop** ends it: the Z80 finishes that inference at full speed, its answer is dropped, and the clock goes back to the one you chose. Pressing **Recognize** (or Enter) while the Z80 works does the same, then runs the new drawing; **Clear** (or Esc) also stops it.

One classification costs about 5 million Z80 clock cycles: instant on the emulator, about 0.7 s on the board at 7.3728 MHz. On the board, uploading the weights takes a few seconds. The Z80 works in a background thread, so the window stays responsive meanwhile.

### The debug panel

With the panel, the window has a DeZog-style view on the right. Images go with `D` instead of `I` (so `_trace` is filled). It has two modes:

| Mode | When | How often |
| --- | --- | --- |
| **Live** | Emulator, after `make ticks-viz` | 20 times a second, also during an inference. Every read, write and instruction fetch. |
| **Snapshot** | The board (`--debug`), or the emulator without the patch (or `--no-live`) | After each result, read with `G` and `M`. |

| Part | What it shows |
| --- | --- |
| Memory map | All of memory as named blocks in address order: Program, Weights, Image (`x`), Results (`trace`), Serial (`buf`, `line`), Debug (`stack_copy`, `regs`), Free, Stack. Each block has its own strip of accesses in DeZog's colours (read yellow, written red, code running blue), so small variables are as visible as the weights. White line and arrow: PC; green box: the memory window. Under it, one sentence says what the Z80 is doing (from the call stack) and which blocks it reads and writes. Hover for the address and symbol, click to show that address. |
| Registers | All registers, red when they changed since the previous snapshot; F and F' decoded (`SZ5H3PNC`). |
| Call stack | PC, then the return addresses found through SDCC's IX frame chain (`_cmd_infer+0xB4`, `_main+0xE3`). |
| Stack | Words from SP, with code addresses named; frame return addresses in blue. It scrolls, and keeps its place while the view refreshes. |
| Memory | A window anywhere in the 64 KB. **Go to** a variable, SP or PC, or type an **Address** (hex, Enter); scroll with the wheel, PgUp/PgDn or the scroll bar (the bar is the whole 64 KB). Only the rows shown are read, so it works on the board too. **View**: *Hex*, 16 × 16 bytes and ASCII, each byte coloured by its access; or *Image 28×28*, 28 pixels per row like the MNIST frame, with a thin ring in the access colour. **Pixel** *1 byte* shows `x` (0/1: white = 1); other bytes are drawn as signed, + orange and − blue, so at `weights` each 28×28 is one hidden neuron's `W1q` row (PgDn: the next neuron). *1 bit* shows `buf`, the packed image the laptop sent. Go to picks the right one for `x` and `buf`. Hover the image for the address. In snapshot mode the first 3 bytes of `buf` hold the arguments of the panel's own last `M` command. **Refresh** reads it again. |
| Scores | The 10 values of `acc2`, the winner in green. |
| Clock (live only) | The emulated clock: 100 kHz, 1 MHz, 3.5 MHz, 7.37 MHz (the board's), or max; **Pause** stops the Z80 between two instructions. Under it, the T-states of the current digit (from 0 at each Recognize), what they would take on the board, and the measured MHz (`idle` while the Z80 waits). A digit in the panel costs more than 5 million T-states: `D` also decodes the image and sends back every intermediate value. |

**Live mode.** `make ticks-viz` copies z88dk's `src/ticks` into `build/ticks-viz`, applies `tools/ticks-viz.patch` and builds it there; the installed z88dk is not changed. The patch adds one option, `-shm <file>`: ticks keeps the emulated RAM in that file and marks there every access (bits read, write, prog, set in `get_memory()`/`put_memory()`), and publishes the registers and T-states about 100 times per emulated second. It also sleeps to match the clock the panel asks for, and waits while paused. `LiveSource` (`host/z80debug.py`) maps the same file; the panel shows the accesses fading out in about half a second, like DeZog. Without `-shm` the patched ticks behaves like the original. The weights always go up at full speed; the clock applies from the first digit. The patch is made for the z88dk sources installed here: if a newer z88dk no longer accepts it, `make ticks-viz` fails and the demo falls back to snapshots.

**The ticks patch.** `tools/ticks-viz.patch` adds `viz.h` and `viz.c` to `src/ticks`, and short hooks to `memory.c` (access bits), `ticks.c` (publishing, clock and pause in the CPU loop), `ticks_main.c` (the `-shm` option), `hook_console.c` (publishing before a blocking read) and its `Makefile`. The shared file is 131,136 bytes: a 64-byte header (magic, a sequence counter, clock, pause, T-states, registers), the 64 KB of RAM, then 64 KB of access bits. Its exact layout and rules are in `docs/spec.md` §7.7.

**Snapshot mode.** Snapshots are: registers and stack as they were when `nn_infer` returned (the firmware copies the stack into `stack_copy`, because `G` and `M` reuse that area), memory as it is when read. The firmware cannot see its own bus, so the bar shows only writes, as the bytes that changed since the previous snapshot (variables, stack and memory window; the weights are not compared). On the board one refresh moves about 2 KB over the serial line, a fraction of a second.

## 5. Debugging with DeZog

The demo runs in z88dk-ticks (section 4). DeZog is for debugging the C code step by step in VS Code: breakpoints, single steps, C source lines, registers and memory, in DeZog's own simulator (zsim). It runs `firmware/z80_nn_main.c` on one test case.

Memory map: code at `0x0000`, weights at `0x1000`, test case at `0x7800`, variables from `0x8000`, mailbox at `0xF000`.

```
make inputs CASE=0   # build/weights_payload.bin (-> 0x1000), build/case.bin (-> 0x7800)
make firmware        # z80_nn_main.c -> launch config "Z80 zsim"
```

Start the "Z80 zsim" config in VS Code. It stops at the first instruction; set breakpoints in `nn.c` or `z80_nn_main.c` and continue. When the program finishes, the mailbox at `0xF000` says whether `acc1`, `h_q`, `acc2` and the digit all match `testvec.bin` (byte layout at the top of `z80_nn_main.c`). `make firmware FW=firmware/z80_test_main.c` builds the toolchain sanity check instead.

## 6. Host tools

| File | What it does |
| --- | --- |
| `host/draw.py` | Tkinter drawing window. `preprocess()` reproduces MNIST's framing: crop, scale to 20 px, center of mass at (14, 14) in 28×28, binarize. `python host/draw.py` runs it without a Z80 and prints the image as ASCII. |
| `host/z80nn.py` | The `Z80NN` bridge (section 3). |
| `host/demo.py` | Connects `DrawingApp` to `Z80NN` (section 4), with the debug panel if asked. |
| `host/z80debug.py` | `MapSymbols` (symbols and memory-map blocks from a z88dk `.map`), `SnapshotSource` (a `Snapshot` through `G`/`M`), `LiveSource` (a `Snapshot` from the patched ticks' shared file, plus clock and pause), the call-stack walk. No Tk. |
| `host/debug_panel.py` | `DebugPanel`, the Tk view of a `Snapshot` (section 4). |
| `tools/ticks-viz.patch` | The live-view patch for `z88dk-ticks`, built by `make ticks-viz` (section 4). |

## Tests

```
make test     # builds emu, rom (and ticks-viz when z88dk's sources are there), then pytest
make check    # integer reference vs float model
```

`make test` lists every skipped test with its reason: a test is skipped only when what it needs is missing (a build, `artifacts/`, `z88dk-ticks`, a display for the `Worker` test).

| Test | What it checks |
| --- | --- |
| `tests/test_preprocess.py` | The drawing preprocessing: empty input, bounding box, 20 px scaling, centering. |
| `tests/test_z80debug.py` | The map parser on both builds (sizes, symbols, memory-map blocks, board layout), flag decoding, snapshots on both builds (scores, call stack, changed bytes), the order and errors of `draw.Worker`, the image decoding of `x` (bytes) and `buf` (bits) after a real inference, the panel's memory scrolling (wheel, pages, bar, wrap at `FFFF`), image view and the stack keeping its place, and Stop in `DrawingApp` (answer dropped, the next drawing run after it). |
| `tests/test_ticks_viz.py` | The patched ticks, on both builds: all 105 test vectors with `-shm`; the upload seen as writes by `cmd_write`; layer 1 reading exactly the weight rows of lit pixels; no instruction fetched from data; the call stack down to `main` in the middle of an inference; `x` and `buf` as images; the per-digit T-states; the clock (2 MHz asked, at most 2.3 measured, so it is throttled) and Pause. Skipped without `make ticks-viz`. |
| `tests/test_emu.py` | All 105 test vectors through `Z80NN` on the emulated Z80, for both `build/emu.bin` and `build/board.rom`: every `acc1`, `h_q`, `acc2` and digit must match Python. Also checks the register snapshot (`G`: PC inside `cmd_infer`, SP in the stack) and memory reads (`M`: the weights and the unpacked image), and that errors leave the firmware running. A build that is missing is skipped (`make emu rom`). |
