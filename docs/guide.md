# MNIST on Z80 — Guide

How each part of the project works, and how to use it in detail. For installation and the steps to reproduce the results, start with the [README](../README.md). The exact file formats and the integer algorithm are in [spec.md](spec.md).

Commands run from the repository root. `python` means the project's virtual environment: activate it with `source .venv/bin/activate`, or write `uv run python` instead.

## Contents

1. [Training pipeline](#1-training-pipeline)
2. [Inference on the Z80](#2-inference-on-the-z80)
3. [How the laptop talks to the Z80](#3-how-the-laptop-talks-to-the-z80)
4. [Running the demo](#4-running-the-demo): options, network view, focus on hover, debug panel
5. [Debugging with DeZog](#5-debugging-with-dezog)
6. [Host tools](#6-host-tools)
7. [Tests](#7-tests)

## 1. Training pipeline

Run in this order (`make train` runs them all). Each step reads what the previous one wrote to `artifacts/`.

```bash
python src/training/train.py       # -> artifacts/mlp.pt
python src/training/quantize.py    # -> artifacts/quantized.npz
python src/training/reference.py   # prints float vs integer accuracy (also: make check)
python src/training/export.py      # -> artifacts/weights.bin, artifacts/testvec.bin
```

| File | What it does |
| --- | --- |
| `train.py` | Downloads MNIST and binarizes it (pixel = 1 if > 0.5). Trains the MLP (Adam, lr 1e-3, batch 64, 10 epochs, seed 0) and saves the float weights. Also defines `MLP`, `get_data()` and the paths that the other scripts import. |
| `quantize.py` | Quantizes weights to `int8` (symmetric, one scale per layer) and biases to `int32`. Calibrates the hidden-layer scale on the training set. Replaces the float rescale between the layers with an integer multiply `M0` and a right `shift`, choosing the largest `shift` that cannot overflow 32 bits. |
| `reference.py` | `infer_int()` is the integer-only network, computed step by step exactly as the C code does: layer 1, ReLU, requantize, layer 2, argmax. `evaluate()` compares it with the float model. |
| `export.py` | Writes little-endian binaries for the C code: `weights.bin` with the weights, and `testvec.bin` with 105 test cases (100 MNIST images plus 5 edge cases). Each case stores the image and every intermediate value. The script reads both files back to check them. Formats: [spec.md §4](spec.md#4-binary-file-formats-normative). |

Test-set results: float 96.01%, integer 95.96%, float/integer agreement 99.67%.

**How the quantization works.** The spec explains it in full:

- [spec.md §2, Quantization scheme](spec.md#2-quantization-scheme): how each layer's scale is chosen, why the biases use the accumulator's scale, how the hidden layer is calibrated, how `M0` and `shift` are chosen so that nothing overflows, and the current values.
- [spec.md §3, Integer inference algorithm](spec.md#3-integer-inference-algorithm-normative): the exact integer computation the Z80 performs, and the guaranteed range of every value.

## 2. Inference on the Z80

| File | What it does |
| --- | --- |
| `src/firmware/nn.h`, `nn.c` | `nn_infer()` is the C version of `infer_int()`. It uses only fixed-width types, so it is correct with the Z80's 16-bit `int`. Because the inputs are binary, layer 1 needs only additions. `struct nn_weights` has the same layout as the `weights.bin` payload. |
| `src/firmware/z80_serial_main.c` | The demo firmware: receives commands from the laptop, runs `nn_infer`, replies. See section 3. |
| `src/firmware/z80_nn_main.c` | DeZog test: runs `nn_infer` on one test case loaded into memory. Writes to the mailbox whether `acc1`, `h_q`, `acc2` and the digit all match `testvec.bin`. |
| `src/firmware/z80_test_main.c` | Toolchain sanity check: writes a fixed value to a mailbox at `0xF000`. |
| `src/firmware/snapregs.h`, `snapregs.c` | `snap_regs()` copies every register into `struct regs`, for the `G` command of `z80_serial_main.c`. |

## 3. How the laptop talks to the Z80

The drawing window runs on the laptop; the network runs on the Z80. They exchange text lines. The firmware, the bridge and the protocol are the same in the emulator and on the board they are intended for; only the way characters travel changes.

```text
src/host/draw.py ─► src/host/demo.py ─► src/host/z80nn.py ─┬─ emu ──► z88dk-ticks running build/board.rom  (emulated ACIA, the demo's default)
  drawing             glue               Z80NN bridge      │          z88dk-ticks running build/emu.bin    (console)
  window                                                   └─ board ─► USB serial ─► ACIA ─► build/board.rom on the EEPROM
```

### The protocol (`docs/spec.md` §5.5)

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

`src/firmware/z80_serial_main.c` is a single source file, built for the emulator and for the board it is intended for, Grant Searle's breadboard Z80. Only its three character functions (`ser_init`, `ser_getc`, `ser_putc`) differ between builds:

| Build | Command | Output | Characters travel through |
| --- | --- | --- | --- |
| Emulator | `make emu` | `build/emu.bin` | The console of `z88dk-ticks` (its stdin/stdout). Uses `fgetc_cons()`, which does not echo the input back. |
| Board | `make rom` | `build/board.rom` | The board's MC6850 ACIA at ports `0x80`/`0x81`, polled with interrupts off. 115,200 baud with the 7.3728 MHz crystal. |

Board memory map: code in ROM from `0x0000` (about 3.9 KB); weights, buffers and `stack_copy` in RAM from `0x8000` to about `0xEA97`; stack below `0xFFFF`. The `.map` files in `build/` have the exact addresses (`docs/spec.md` §5.4).

### The bridge: `src/host/z80nn.py`

`Z80NN(target, where)` hides where the Z80 is. After it connects, `version()`, `load_weights()`, `classify()`, `classify_debug()`, `registers()` and `read_memory()` work the same way everywhere.

| `target` | `where` | What it does |
| --- | --- | --- |
| `emu` | a Z80 program (`build/emu.bin` or `build/board.rom`) | Starts `z88dk-ticks` on it with `-counter`, because ticks otherwise stops after about 10 classifications. Its stdout is a pseudo-terminal (see below). |
| `board` | a serial port, e.g. `/dev/tty.usbserial-XXXX` | Opens it at 115,200 baud, 8N1, RTS/CTS flow control |
| `local` | a program on the laptop | Runs it and talks to it through stdin/stdout (none is built by this repo) |

### Testing the EEPROM image before writing it

`z88dk-ticks` already emulates an ACIA on the same ports as the board, but it never flushes its output, so replies sent through a pipe never arrive. `Z80NN` therefore gives ticks a pseudo-terminal as stdout: on a terminal the C library flushes at every `\n`. With it, the exact `build/board.rom` that goes on the EEPROM runs in the emulator, and the demo runs that image by default. `tests/test_emu.py` runs all 105 test vectors on both builds and checks every intermediate value.

While the firmware waits for a command, ticks waits too, using no CPU, although `ser_getc()` polls the ACIA's status register. This relies on a quirk of the installed z88dk: its status read reports a received byte even when none is waiting (`kbhit()` returns −1, not 0, and the emulation takes that as true), so the firmware goes straight to reading the data register, and that read blocks until a byte arrives. A z88dk that reports "empty" correctly would make the firmware spin, using a full core; `tests/test_ticks_viz.py` fails if that happens (`docs/spec.md` §5.6).

At end of input (the laptop closed ticks' stdin) the firmware cannot tell: plain ticks gives it `0xFF` bytes forever, so `Z80NN.close()` kills it after 2 s. The patched ticks of the live view (`-shm`, section 4) exits by itself instead.

## 4. Running the demo

The demo uses the committed `artifacts/weights.bin`; `make` trains again only if it is missing.

### On the emulator

```bash
make demo
```

This is the same as:

```bash
make rom                      # -> build/board.rom
python src/host/demo.py       # target emu, program build/board.rom
```

The emulator runs the same `build/board.rom` that goes on the EEPROM. To run the console build instead: `make emu`, then `python src/host/demo.py emu build/emu.bin`.

### On the board

`build/board.rom` is meant for the EEPROM of Grant Searle's breadboard Z80 (7.3728 MHz, MC6850 ACIA at 115,200 baud).

```bash
make rom                  # -> build/board.rom
```

1. Write `build/board.rom` to the EEPROM with your programmer, for example `minipro -p <chip> -w build/board.rom -s`. `-s` accepts a file smaller than the chip.
2. Put the EEPROM in the board, connect the USB-serial adapter, and find its port with `ls /dev/tty.usb*`.
3. Run:

   ```bash
   make demo PORT=/dev/tty.usbserial-XXXX
   ```

   This is the same as:

   ```bash
   python src/host/demo.py board /dev/tty.usbserial-XXXX
   ```

   Add `DEBUG=1` (or `--debug`) for the debug panel (below).

### Arguments of `src/host/demo.py`

```text
python src/host/demo.py [TARGET] [WHERE] [--debug | --no-debug] [--map FILE] [--live | --no-live] [--speed MHZ]
```

| `TARGET` | `WHERE` | Example |
| --- | --- | --- |
| `emu` (default) | Z80 program for the emulator, default `build/board.rom` | `python src/host/demo.py`, `python src/host/demo.py emu build/emu.bin` |
| `board` | Serial port of the board (required) | `python src/host/demo.py board /dev/tty.usbserial-XXXX` |
| `local` | Program on the laptop that speaks the protocol | `python src/host/demo.py local ./my_program` |

| Option | Meaning |
| --- | --- |
| `--debug` / `--no-debug` | Show or hide the debug panel. Default: shown on `emu`, hidden on `board` and `local` (`local` cannot have it). |
| `--map FILE` | The `.map` file of the program that runs, for the symbols. Default: next to the emulator program (`build/board.map` for `build/board.rom`, `build/emu.map` for `build/emu.bin`), or `build/board.map` for the board. It must come from the same build as the EEPROM contents. |
| `--live` / `--no-live` | Emulator: live debug view with the patched ticks. Default: on when `make ticks-viz` has been run. |
| `--speed MHZ` | Live view: the emulated clock, default `1` (one inference takes about 5 s to watch); `0` = as fast as possible. Changeable in the panel. |

### What you see

The script prints `Connected: Z80NN 2`, uploads the weights, prints `Weights loaded, checksum OK`, and opens the window. Draw a digit with the mouse, then press **Recognize** or Enter; the digit the Z80 computed appears below. Press **Clear** or Esc to start again. The preview on the right shows the 28×28 image the Z80 receives.

The UI is meant to explain what happens inside the machine: every part has a short caption in plain words.

In the live view (emulator, debug panel) a slow clock makes one digit take seconds. **Stop** ends it: the Z80 finishes that inference at full speed, its answer is dropped, and the clock goes back to the one you chose. Pressing **Recognize** (or Enter) while the Z80 works does the same, then runs the new drawing; **Clear** (or Esc) also stops it.

One classification costs about 5 million Z80 clock cycles: instant on the emulator, about 0.7 s at the 7.3728 MHz of the board it is intended for. Over the board's serial line the weight upload is about 54 KB of hex text, about 5 s at 115,200 baud. The Z80 works in a background thread, so the window stays responsive meanwhile.

### The network view

With the debug panel, the space under the drawing shows the network the Z80 computes (`src/host/network_view.py`), in order:

| Part | What it shows |
| --- | --- |
| Headline | What the Z80 is doing now (live: *Unpacking the image it received*, *Layer 1, neuron #12: adding its weights at the 74 lit pixels*, *Layer 2, digit 4*, then *Network done: writing the 74 results as text for the laptop*), then the answer with its score and the runner-up. With `D` that last phase is about a third of a digit's time: every `acc1`, `h_q` and `acc2` value goes out as decimal text, and each digit of a 32-bit number costs a 32-bit division. |
| Layer 1 | 32 cells, one per hidden neuron: `h_q`, black = 0 (cut by ReLU) to white = the strongest of this digit. Live: the neuron being computed is outlined in yellow, the ones done turn grey; the values come with the `D` reply. Hover: the sum (`acc1`), ReLU and the rescaled value. Click: Memory shows that neuron's 784 weights as a 28×28 image. |
| Layer 2 | The 10 scores (`acc2`), the highest in green: the answer. Live: the score being computed is outlined. |

The live step comes from `infer_step()` (`src/host/z80debug.py`): `layer1_neuron(j)` reads only row *j* of `W1q`, `layer2_neuron(k)` only row *k* of `W2q`, so the weight rows read since the previous capture say which neuron the Z80 is on. At 1 MHz each hidden neuron takes about 0.15 s, slow enough to follow.

### Focus on hover

With the network view and the debug panel, the part under the mouse stays sharp and every other part fades, to keep the attention on one thing: the drawing, the preview, layer 1, layer 2, and each frame of the debug panel. The faded parts keep updating. The **Focus on hover** checkbox next to Clear turns it on; it is off by default.

Tk cannot blur or fade a widget, so `src/host/focus.py` lays a borderless window, 75% opaque and in the window's background colour, over each faded part; four thin blue windows draw the focused part's border. It reads the mouse position 12 times a second rather than Tk's enter and leave events, which also fire on child widgets and on the veils. In a gap between parts the focus stays where it was; outside the window, or with another app in front, nothing is faded. While the window is dragged the veils hide (Tk learns the window's new position only now and then, so veils moved along would trail behind it) and come back 0.25 s after it stops. A veil would take a click meant for the part under it, but the part comes into focus as soon as the mouse moves over it.

### The debug panel

With the panel, the window has a DeZog-style view on the right. Images go with `D` instead of `I` (so `_trace` is filled). It has two modes:

| Mode | When | How often |
| --- | --- | --- |
| **Live** | Emulator, after `make ticks-viz` | 20 times a second, also during an inference. Every read, write and instruction fetch. |
| **Snapshot** | The board (`--debug`), or the emulator without the patch (or `--no-live`) | After each result, read with `G` and `M`. |

| Part | What it shows |
| --- | --- |
| Memory map | All of memory as named blocks in address order, each with its size: Program, Weights, Image (`x`), Results (`trace`), Serial (`buf`, `line`), Debug (`stack_copy`, `regs`), Free, Stack. Each block has its own strip of accesses in DeZog's colours (read yellow, written red, code running blue), so small variables are as visible as the weights. White line and arrow: PC; green box: the memory window. Under it, one sentence says what the Z80 is doing (from the call stack) and which blocks it reads and writes. Hover for the address and symbol, click to show that address. |
| Registers | All registers, red when they changed since the previous snapshot; F and F' decoded (`SZ5H3PNC`). |
| Call stack | PC, then the return addresses found through SDCC's IX frame chain (`_cmd_infer+0xB4`, `_main+0xE3`). |
| Stack | Words from SP, with code addresses named; frame return addresses in blue. It scrolls, and keeps its place while the view refreshes. |
| Memory | A window anywhere in the 64 KB. **Go to** a variable, SP or PC, or type an **Address** (hex, Enter); scroll with the wheel, PgUp/PgDn or the scroll bar (the bar is the whole 64 KB). Only the rows shown are read, so it is also meant for the board's serial line. **View**: *Hex*, 16 × 16 bytes and ASCII, each byte coloured by its access; or *Image 28×28*, 28 pixels per row like the MNIST frame, with a thin ring in the access colour. **Pixel** *1 byte* shows `x` (0/1: white = 1); other bytes are drawn as signed, + orange and − blue, so at `weights` each 28×28 is one hidden neuron's `W1q` row (PgDn: the next neuron). *1 bit* shows `buf`, the packed image the laptop sent. Go to picks the right one for `x` and `buf`. It opens on `x` as an image. Hover the image for the address (at `weights`: which neuron and pixel the weight belongs to). In snapshot mode the first 3 bytes of `buf` hold the arguments of the panel's own last `M` command. **Refresh** reads it again. |
| Clock (live only) | The emulated clock: 100 kHz, 1 MHz, 3.5 MHz, 7.37 MHz (the board's), or max; **Pause** stops the Z80 between two instructions. Under it, the T-states of the current digit (from 0 at each Recognize), what they would take on the board, and the measured MHz (`idle` while the Z80 waits). A digit in the panel costs more than 5 million T-states: `D` also decodes the image and sends back every intermediate value. |

**Live mode.** `make ticks-viz` copies z88dk's `src/ticks` into `build/ticks-viz`, applies `src/tools/ticks-viz.patch` and builds it there; the installed z88dk is not changed. The patch adds one option, `-shm <file>`: ticks keeps the emulated RAM in that file and marks there every access (bits read, write, prog, set in `get_memory()`/`put_memory()`), and publishes the registers and T-states about 100 times per emulated second. It also sleeps to match the clock the panel asks for, waits while paused, and exits at end of input when it runs `build/board.rom` (section 3). `LiveSource` (`src/host/z80debug.py`) maps the same file; the panel shows the accesses fading out in about half a second, like DeZog. Without `-shm` the patched ticks behaves like the original. The weights always go up at full speed; the clock applies from the first digit. The patch is made for the z88dk sources installed here: if a newer z88dk no longer accepts it, `make ticks-viz` fails and the demo falls back to snapshots.

**The ticks patch.** `src/tools/ticks-viz.patch` adds `viz.h` and `viz.c` to z88dk's `src/ticks`, and short hooks to `memory.c` (access bits), `ticks.c` (publishing, clock and pause in the CPU loop), `ticks_main.c` (the `-shm` option), `hook_console.c` (publishing before a blocking read, the console's and the ACIA's), `acia.c` (exit when the ACIA's data read finds the end of input) and its `Makefile`. The shared file is 131,136 bytes: a 64-byte header (magic, a sequence counter, clock, pause, T-states, registers), the 64 KB of RAM, then 64 KB of access bits. Its exact layout and rules are in `docs/spec.md` §5.6.

**Snapshot mode.** Snapshots are: registers and stack as they were when `nn_infer` returned (the firmware copies the stack into `stack_copy`, because `G` and `M` reuse that area), memory as it is when read. The firmware cannot see its own bus, so the bar shows only writes, as the bytes that changed since the previous snapshot (variables, stack and memory window; the weights are not compared). Over the board's serial line one refresh is about 2 KB, a fraction of a second at 115,200 baud.

## 5. Debugging with DeZog

The demo runs in z88dk-ticks (section 4). DeZog is for debugging the C code step by step in VS Code: breakpoints, single steps, C source lines, registers and memory, in DeZog's own simulator (zsim). It runs `src/firmware/z80_nn_main.c` on one test case.

Memory map: code at `0x0000`, weights at `0x1000`, test case at `0x7800`, variables from `0x8000`, mailbox at `0xF000`.

```bash
make inputs CASE=0   # build/weights_payload.bin (-> 0x1000), build/case.bin (-> 0x7800)
make firmware        # z80_nn_main.c -> launch config "Z80 zsim"
```

Start the "Z80 zsim" config (`.vscode/launch.json`) in VS Code. It stops at the first instruction; set breakpoints in `nn.c` or `z80_nn_main.c` and continue. When the program finishes, the mailbox at `0xF000` says whether `acc1`, `h_q`, `acc2` and the digit all match `testvec.bin` (byte layout at the top of `z80_nn_main.c`). `make firmware FW=src/firmware/z80_test_main.c` builds the toolchain sanity check instead.

## 6. Host tools

| File | What it does |
| --- | --- |
| `src/host/draw.py` | Tkinter drawing window. `preprocess()` reproduces MNIST's framing: crop, scale to 20 px, center of mass at (14, 14) in 28×28, binarize. `python src/host/draw.py` runs it without a Z80 and prints the image as ASCII. |
| `src/host/z80nn.py` | The `Z80NN` bridge (section 3). |
| `src/host/demo.py` | Connects `DrawingApp` to `Z80NN` (section 4), with the debug panel if asked. |
| `src/host/z80debug.py` | `MapSymbols` (symbols and memory-map blocks from a z88dk `.map`), `SnapshotSource` (a `Snapshot` through `G`/`M`), `LiveSource` (a `Snapshot` from the patched ticks' shared file, plus clock and pause), the call-stack walk. No Tk. |
| `src/host/focus.py` | `HoverFocus`: the part under the mouse in focus, the others faded (section 4). |
| `src/host/network_view.py` | `NetworkView`, the network under the drawing: hidden neurons and scores, step by step (section 4). |
| `src/host/debug_panel.py` | `DebugPanel`, the Tk view of a `Snapshot` (section 4). |
| `src/tools/ticks-viz.patch` | The live-view patch for `z88dk-ticks`, built by `make ticks-viz` (section 4). |

## 7. Tests

```bash
make test     # builds emu, rom (and ticks-viz when z88dk's sources are there), then pytest
make check    # integer reference vs float model
```

`make test` lists every skipped test with its reason: a test is skipped only when what it needs is missing (a build, `artifacts/`, `z88dk-ticks`, a display for the `Worker` test).

| Test | What it checks |
| --- | --- |
| `tests/test_preprocess.py` | The drawing preprocessing: empty input, bounding box, 20 px scaling, centering. |
| `tests/test_z80debug.py` | The map parser on both builds (sizes, symbols, memory-map blocks, board layout), flag decoding, snapshots on both builds (scores, call stack, changed bytes), the order and errors of `draw.Worker`, the image decoding of `x` (bytes) and `buf` (bits) after a real inference, `infer_step()` on both maps, the panel's memory scrolling (wheel, pages, bar, wrap at `FFFF`), image view (and a neuron's weights), focus on hover (veils, gaps, outside the window, off) and the stack keeping its place, and Stop in `DrawingApp` (answer dropped, the next drawing run after it). |
| `tests/test_ticks_viz.py` | The patched ticks, on both builds: all 105 test vectors with `-shm`; the upload seen as writes by `cmd_write`; layer 1 reading exactly the weight rows of lit pixels; no instruction fetched from data; the call stack down to `main` in the middle of an inference; `infer_step()` following layer 1 neuron by neuron, then layer 2; `x` and `buf` as images; the per-digit T-states; the clock (2 MHz asked, at most 2.3 measured, so it is throttled) and Pause; `build/board.rom` waiting for a command with no T-states and no accesses (the guard for the ACIA quirk, section 3), the panel's clock saying `idle` then, and ticks exiting with status 0 within 1 s of the end of input. Skipped without `make ticks-viz`. |
| `tests/test_emu.py` | All 105 test vectors through `Z80NN` on the emulated Z80, for both `build/emu.bin` and `build/board.rom`: every `acc1`, `h_q`, `acc2` and digit must match Python. Also checks the register snapshot (`G`: PC inside `cmd_infer`, SP in the stack) and memory reads (`M`: the weights and the unpacked image), and that errors leave the firmware running. A build that is missing is skipped (`make emu rom`). |
| `tests/make_dezog_inputs.py` | Not a test: cuts one test case and the weights into the raw files DeZog loads (`make inputs`, section 5). |
