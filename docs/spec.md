# MNIST on Z80 — Specification

This document is the contract between the parts of the project. The Python pipeline defines what the correct answer is; the C code on the Z80 (emulated or real) must reproduce it **bit for bit**. When this document and the code disagree, fix one of them so they agree again.

Sections 2–5 are **normative**: they define exactly what every implementation must compute and how files are laid out. So are the serial protocol (§7.5) and the live-view file (§7.7), which two programs must read the same way. The rest are implementation guides.

## Contents

1. Repository layout
2. The network
3. Quantization scheme
4. Integer inference algorithm (normative)
5. Binary file formats (normative)
6. Python pipeline
7. C implementation
8. Assembly (optional)
9. Testing strategy
10. Open points

---

## 1. Repository layout

```
mnist-on-Z80/
├── training/      train.py, quantize.py, reference.py, export.py
├── firmware/      nn.h, nn.c                  inference library
│                  z80_serial_main.c           the firmware: serial protocol (§7.5), emulator and board
│                  snapregs.h, snapregs.c      register snapshot for the G command
│                  z80_nn_main.c               one test case in DeZog (§9)
│                  z80_test_main.c             toolchain sanity check
├── host/          z80nn.py                    bridge to the Z80 (§7.6)
│                  draw.py, demo.py            drawing window and demo
│                  z80debug.py, debug_panel.py debug panel (§7.7)
├── tests/         test_emu.py, test_z80debug.py, test_ticks_viz.py, test_preprocess.py,
│                  make_dezog_inputs.py
├── tools/         ticks-viz.patch             z88dk-ticks with a live view (§7.7)
├── artifacts/     generated files, not committed (see .gitignore)
│   ├── data/          MNIST, downloaded by torchvision
│   ├── mlp.pt         float weights
│   ├── quantized.npz  integer weights and constants
│   ├── weights.bin    integer weights for the C code
│   └── testvec.bin    test vectors for the C code
├── build/         Z80 programs, DeZog inputs, build/ticks-viz; not committed
├── docs/          this file, plan.md (historical), toolchain guide
├── hardware/      photos, wiring notes
├── Makefile
└── README.md
```

| Folder | Runs on | Who uses it |
| --- | --- | --- |
| `training/` | Laptop, Python | Run once to produce `artifacts/` |
| `firmware/` | Z80: `z88dk-ticks`, DeZog's zsim, or the board | The product |
| `host/` | Laptop, Python | Talks to the Z80, emulated or real |
| `tests/` | Laptop, pytest | Verifies `firmware/` and `host/` against `artifacts/testvec.bin` |
| `tools/` | Laptop | Builds the patched emulator for the live debug view |

---

## 2. The network

| Item | Value |
| --- | --- |
| Architecture | MLP 784 → 32 → 10, ReLU after the first layer, no activation after the second |
| Input | 28 × 28 MNIST image, **binarized**: pixel = 1 if its value after `ToTensor()` is > 0.5, else 0 |
| Pixel order | Row by row from the top, left to right in each row (same as `nn.Flatten`). Pixel index `i = 28 · row + col` |
| Output | The digit with the largest output (argmax) |
| Training | PyTorch, Adam (lr 1e-3), batch 64, 10 epochs, cross-entropy, seed 0 |

There is no normalization of the input: the network sees exactly the 0/1 values that the Z80 will receive.

---

## 3. Quantization scheme

All weights become integers; every scale factor is used only on the laptop.

### 3.1 Notation

| Symbol | Meaning |
| --- | --- |
| `W1`, `b1` | Float weights (32 × 784) and biases (32) of layer 1 |
| `W2`, `b2` | Float weights (10 × 32) and biases (10) of layer 2 |
| `s` | A **scale**: the real value of one integer step |
| `q` suffix | Quantized (integer) version |

The relation between a real value `v`, its scale `s` and its integer version `vq` is always `v ≈ s · vq`.

### 3.2 Weights: symmetric per-layer quantization

For each weight matrix `W`:

```
s  = max(|W|) / 127
Wq = clip(round(W / s), -127, 127)       stored as int8
```

The value -128 is never used, so the range is symmetric and 0 maps exactly to 0.

### 3.3 Biases: the scale of the accumulator

A bias is added to an accumulator, so it must be expressed in the accumulator's units. It is **not** quantized with its own scale.

| Bias | Formula | Why |
| --- | --- | --- |
| `b1q` | `round(b1 / s1)` | The input is 0/1 (scale 1), so `acc1` is in units of `s1` |
| `b2q` | `round(b2 / (s_h · s2))` | Layer 2 multiplies values in units of `s_h` by weights in units of `s2` |

Biases are stored as int32 and are **not** clipped.

### 3.4 Activations of layer 1

The scale of the hidden activations is calibrated on the **training set** (never on the test set):

```
h_max = max over all training images of ReLU(W1 · x + b1)     (float)
s_h   = h_max / 127
```

### 3.5 Requantization constants `M0` and `shift`

Between the layers, `acc1` (units of `s1`) must be converted to units of `s_h`, i.e. multiplied by `M = s1 / s_h`. The Z80 has no floats, so `M` is approximated as:

```
M ≈ M0 / 2^shift          M0, shift integers
```

`shift` is chosen as large as possible (best precision) subject to the **no-overflow condition**:

```
bound · M0 + 2^(shift-1) < 2^31
```

where `bound` is the largest value `acc1` can reach for **any** binary image:

```
bound = max over neurons j of ( b1q[j] + sum of the positive entries of W1q[j] )
```

This is a proof, not a measurement: the worst image turns on exactly the pixels with positive weight.

### 3.6 Current values

These change every time the network is retrained. They are printed by `quantize.py` and `reference.py`.

| Quantity | Value |
| --- | --- |
| `s1` | 0.0101027 |
| `s2` | 0.0111489 |
| `h_max` | 26.297 |
| `s_h` | 0.207063 |
| `M` | 0.0487906 |
| `M0` | 204,643 |
| `shift` | 22 |
| Relative error of `M0 / 2^shift` | 2.41e-6 |
| `bound` | 6,965 |
| Largest `acc1` observed on the test set | 2,316 |
| Accuracy, float model (test set) | 96.01% |
| Accuracy, integer model (test set) | 95.96% |
| Agreement float vs integer | 99.67% |
| `h_q` values saturated at 127 (test set) | 0% |

---

## 4. Integer inference algorithm (normative)

Every implementation (`reference.py`, `nn.c`, and any assembly) must compute exactly these values.

**Input**: `x[784]`, each 0 or 1.

1. **Layer 1**, for each neuron `j` in 0..31:
   ```
   acc1[j] = b1q[j] + sum of W1q[j][i] for every i where x[i] == 1        (int32)
   ```
2. **ReLU**: `r = max(acc1[j], 0)`.
3. **Requantization**:
   ```
   h_q[j] = min(127, (r · M0 + 2^(shift-1)) >> shift)                     (0..127)
   ```
   `r · M0` is computed in 32-bit signed arithmetic; the condition of §3.5 guarantees no overflow. `r ≥ 0`, so the right shift is a division rounded down. Adding `2^(shift-1)` first turns it into rounding to nearest.
4. **Layer 2**, for each digit `k` in 0..9:
   ```
   acc2[k] = b2q[k] + sum over j of W2q[k][j] · h_q[j]                     (int32)
   ```
   Each product fits in 16 bits: |W2q · h_q| ≤ 127 · 127 = 16,129.
5. **Argmax**: the result is the smallest `k` such that `acc2[k]` is maximal (**ties go to the lowest index**, like `numpy.argmax`). In C this means comparing with a strict `>`.

### 4.1 Value ranges (guaranteed)

| Value | Range | Type |
| --- | --- | --- |
| `x[i]` | 0 or 1 | uint8 |
| `acc1[j]` | ≥ -(784 · 127 + max \|b1q\|), ≤ `bound` | int32 |
| `r · M0 + 2^(shift-1)` | < 2^31 | int32 |
| `h_q[j]` | 0..127 | uint8 |
| `W2q[k][j] · h_q[j]` | -16,129..16,129 | int16 |
| `acc2[k]` | \|acc2\| ≤ 32 · 127 · 127 + max \|b2q\| = 516,128 + max \|b2q\| | int32 |
| result | 0..9 | uint8 |

---

## 5. Binary file formats (normative)

Produced by `training/export.py`. All multi-byte integers are **little-endian** (least significant byte first), the byte order of the Z80. Two-dimensional arrays are stored **row by row**, which is also how C lays out `int8_t a[R][C]` in memory.

### 5.1 `weights.bin` (25,586 bytes)

| Offset | Size (bytes) | Field | Type | Content |
| --- | --- | --- | --- | --- |
| 0 | 4 | magic | bytes | `Z8NN` |
| 4 | 1 | version | uint8 | 1 |
| 5 | 25,088 | `W1q` | int8 [32][784] | Neuron 0's 784 weights first |
| 25,093 | 128 | `b1q` | int32 [32] | |
| 25,221 | 4 | `M0` | int32 | |
| 25,225 | 1 | `shift` | uint8 | 1..31 |
| 25,226 | 320 | `W2q` | int8 [10][32] | Digit 0's 32 weights first |
| 25,546 | 40 | `b2q` | int32 [10] | |

The **payload** is everything after the 5-byte header (offsets 5..25,585, 25,581 bytes). It is what the serial protocol transfers (§7.5).

### 5.2 `testvec.bin` (7 + n · 985 bytes)

Header:

| Offset | Size | Field | Type |
| --- | --- | --- | --- |
| 0 | 4 | magic `Z8TV` | bytes |
| 4 | 1 | version (1) | uint8 |
| 5 | 2 | number of cases `n` | uint16 |

Then `n` records of 985 bytes each:

| Size | Field | Type | Content |
| --- | --- | --- | --- |
| 784 | `x` | uint8 [784] | Image, 0 or 1 |
| 128 | `acc1` | int32 [32] | **Before** the ReLU |
| 32 | `h_q` | uint8 [32] | 0..127 |
| 40 | `acc2` | int32 [10] | |
| 1 | `digit` | uint8 | Result of the integer network, **not** the MNIST label |

Current content: 105 cases — the first 100 test-set images, then `all_black`, `all_white`, `only_pixel_0`, `only_pixel_783`, and `worst_neuron_N` (the image that drives neuron N to exactly `bound`).

---

## 6. Python pipeline

Run from the repository root, in this order:

```
python training/train.py       # -> artifacts/mlp.pt
python training/quantize.py    # -> artifacts/quantized.npz
python training/reference.py   # prints accuracies
python training/export.py      # -> artifacts/weights.bin, artifacts/testvec.bin
```

### 6.1 `train.py`

Defines `MLP` (an `nn.Sequential` of `Flatten`, `Linear(784, 32)`, `ReLU`, `Linear(32, 10)`, so the linear layers are `layers[1]` and `layers[3]`) and `get_data()`, which returns binarized train and test datasets and loaders. Trains for 10 epochs and saves the `state_dict` to `artifacts/mlp.pt`. Also defines the paths used by every other script (`DATA_DIR`, `WEIGHTS_PATH`).

### 6.2 `quantize.py`

Implements §3: quantizes the weights (`quantize_symmetric`), calibrates `h_max` on the training set, computes `bound`, and chooses `M0` and `shift` (`choose_m0_shift`). Asserts the no-overflow condition and the integer types, then saves `W1q`, `b1q`, `W2q`, `b2q`, `M0`, `shift` (plus `s1`, `s2`, `s_h`, `h_max`, `bound` for diagnostics) to `artifacts/quantized.npz`.

### 6.3 `reference.py`

The executable form of §4.

| Function | Purpose |
| --- | --- |
| `load_quantized(path)` | Loads `quantized.npz`; arrays as int64, scalars as Python numbers |
| `load_test_images()` | Test set as an int64 array (10000, 784) of 0/1, plus labels |
| `infer_int(x, q)` | One image → `(digit, acc1, h_q, acc2)`, integers only |
| `load_float_model()` | The float PyTorch model, for comparison |
| `evaluate(images, labels, q)` | Accuracies, agreement, saturation rate, largest `acc1` |

`infer_int` processes **one** image on purpose: it mirrors the C code step by step.

### 6.4 `export.py`

Implements §5. Writes both files, refuses to convert any value outside its type's range (numpy's `astype` would wrap silently), reads both files back and compares every value. Also checks that the `worst_neuron_N` image reaches exactly `bound`.

---

## 7. C implementation

### 7.1 Portability rules

`nn.c` is compiled by z88dk (sdcc) for the Z80, where `int` is **16 bits**. It is written in plain C99 so that it would also compile unchanged on a laptop, where `int` is 32 bits; code that works there can silently fail on the Z80. These rules apply to every C file in `firmware/`:

1. **Use only fixed-width types** from `<stdint.h>` for data: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`. Use plain `int` only where the value is guaranteed to fit in 16 bits.
2. **Never shift an untyped constant into the upper bits.** Write `(int32_t)1 << (shift - 1)`, never `1 << (shift - 1)`: on the Z80 the second form is undefined behavior for `shift > 15`.
3. **No signed overflow.** Every accumulator is `int32_t`, and §4.1 proves the ranges.
4. **Loop indices must fit their range.** A loop to 784 needs `uint16_t`; `uint8_t` wraps at 256 and the loop never ends.
5. **Right shift only non-negative values.** Shifting a negative number right is implementation-defined. The requantization only shifts after the ReLU.
6. **Casts in products**: write `(int16_t)a * (int16_t)b` in layer 2 so the compiler uses a 16-bit multiplication.
7. **No dynamic memory** (`malloc`/`free`).
8. **Large data is never on the stack.** The weights (≈25.6 KB) are a global or `static` variable, or sit at a fixed address. The Z80 stack is a few hundred bytes.
9. **No `printf`.** The firmware prints numbers with its own `put_hex4`/`put_i32`: smaller, and no `%ld` support to configure.
10. **`nn.c` does no I/O**: it does not include `<stdio.h>`. All input/output lives in the `z80_*_main.c` files.

### 7.2 `firmware/nn.h` — interface

```c
#ifndef NN_H
#define NN_H
#include <stdint.h>

#define N_IN  784
#define N_HID 32
#define N_OUT 10

struct nn_weights {                 /* same fields and order as weights.bin */
    int8_t  W1q[N_HID][N_IN];
    int32_t b1q[N_HID];
    int32_t M0;
    uint8_t shift;
    int8_t  W2q[N_OUT][N_HID];
    int32_t b2q[N_OUT];
};

struct nn_trace {                   /* intermediate values, for tests */
    int32_t acc1[N_HID];            /* before ReLU */
    uint8_t h_q[N_HID];
    int32_t acc2[N_OUT];
};

uint8_t nn_infer(const struct nn_weights *w,
                 const uint8_t x[N_IN],
                 struct nn_trace *trace);

#endif
```

**Contract of `nn_infer`**:

| Parameter | Meaning |
| --- | --- |
| `w` | Pointer to the weights. Not modified |
| `x` | 784 pixels, each 0 or 1. Not modified |
| `trace` | Where to store the intermediate values, or `NULL` if they are not needed |
| return value | The digit, 0..9, as defined in §4 |

With sdcc the struct has **no padding** (no field needs alignment on the Z80), so it has exactly the layout of the `weights.bin` payload (§5.1). The firmware relies on this and copies the payload into it byte for byte; both `z80_serial_main.c` and `z80_nn_main.c` check it at compile time:

```c
_Static_assert(sizeof(struct nn_weights) == 25581, "nn_weights layout != weights.bin payload");
```

A compiler that pads the struct fails to build instead of computing wrong results.

### 7.3 `firmware/nn.c` — inference

Four functions; the first three are `static` (private to the file).

| Function | Computes | Notes |
| --- | --- | --- |
| `layer1_neuron(row, bias, x)` | `acc1[j]` (§4 step 1) | Adds `row[i]` when `x[i] != 0`. Index is `uint16_t` |
| `requant(acc, M0, shift)` | `h_q[j]` (§4 steps 2–3) | ReLU first; `(int32_t)1 << (shift - 1)`; clamp to 127 |
| `layer2_neuron(row, bias, h_q)` | `acc2[k]` (§4 step 4) | `(int16_t)` casts on both factors |
| `nn_infer(w, x, trace)` | Everything | Strict `>` in the argmax; fills `trace` only if not `NULL` |

Pseudocode of `nn_infer`:

```
acc1[32], h_q[32], acc2[10] as local arrays (≈200 bytes, fine on the stack)
for j in 0..31:
    acc1[j] = layer1_neuron(w->W1q[j], w->b1q[j], x)
    h_q[j]  = requant(acc1[j], w->M0, w->shift)
best = 0
for k in 0..9:
    acc2[k] = layer2_neuron(w->W2q[k], w->b2q[k], h_q)
    if acc2[k] > acc2[best]: best = k
if trace != NULL: copy acc1, h_q, acc2 into trace
return best
```

### 7.4 The Z80 programs

| Program | Built by | Runs in | Input and output |
| --- | --- | --- | --- |
| `z80_serial_main.c` | `make emu` → `build/emu.bin` | `z88dk-ticks` | The line protocol (§7.5) on the console: `fgetc_cons()`/`fputc_cons()` |
| `z80_serial_main.c` | `make rom` → `build/board.rom` | The board, or `z88dk-ticks` | The same protocol through the MC6850 ACIA at ports `0x80`/`0x81` |
| `z80_nn_main.c` | `make inputs firmware` → `build/firmware_*.bin` | DeZog (zsim) | One test case loaded at `0x7800`, result in the mailbox at `0xF000` (§9) |
| `z80_test_main.c` | `make firmware FW=firmware/z80_test_main.c` | DeZog (zsim) | Writes a fixed value to the mailbox: checks the toolchain |

Build flags (see the `Makefile`; every build also writes a `.map` file with `-m`):

| Build | Flags | Memory map (from the `.map`) |
| --- | --- | --- |
| `emu` | `+test -compiler=sdcc` | 64 KB of RAM. Code from `0x0000` (to about `0x10E8`), variables to about `0x7BE3`, stack below `0xFF00` |
| `rom` | `+z80 -compiler=sdcc -pragma-define:CRT_ENABLE_EIDI=0x01 -create-app` | Code in ROM from `0x0000` (about 3.9 KB). Variables in RAM from `0x8000` (weights, buffers and `stack_copy`, to about `0xEA97`), stack below `0xFFFF`. `-create-app` writes `build/board.rom`, the file for the EEPROM. Interrupts stay off: the ACIA is polled |
| `firmware` | `+z80 -compiler=sdcc -m --list --c-code-in-asm -pragma-define:CRT_MODEL=0` | Code from `0x0000` (below `0x1000`), weights loaded at `0x1000`, test case at `0x7800`, variables from `0x8000`, mailbox `0xF000`. `--list` writes the `firmware/*.lis` listings DeZog uses for source-level debugging |

Check the `.map` file after a change that adds data: the variables and the stack must not meet. The addresses above move with every change; the `.map` is the reference, and the debug panel reads it.

### 7.5 `firmware/z80_serial_main.c` — firmware and serial protocol

The firmware is an endless loop: read one line, execute the command, reply with one or more lines. The same source is built for the emulator and for the board (§7.4); only its three character functions `ser_init`, `ser_getc`, `ser_putc` differ. The weights live in a global `struct nn_weights` in RAM, so they are uploaded again every time the program starts.

**Line format**:

- Commands are ASCII lines terminated by `\n`. A `\r` before it is ignored.
- Bytes are encoded as **two hexadecimal digits** (`00`..`FF`); upper and lower case are both accepted on input, the board replies in upper case.
- The receive buffer holds 198 characters, the longest valid line (`I ` + 196 hex digits). A longer line is discarded and answered with `ERR LONG`.
- At end of input (the emulator's stdin closed) `build/emu.bin` exits. The board keeps polling, and so does `build/board.rom` in plain `z88dk-ticks`, which then gives it `0xFF` bytes forever; with `-shm` the patched ticks exits instead (§7.7).

**Commands**:

| Command | Arguments | Reply | Effect |
| --- | --- | --- | --- |
| `V` | — | `Z80NN 2` | Name and protocol version (2: adds `G` and `M`) |
| `W` | `oooo` + space + up to 128 hex digits | `OK`, or `ERR <reason>` | Writes up to 64 bytes of the weights payload at offset `oooo` (4 hex digits) |
| `C` | — | `C xxxx` | 16-bit checksum of the whole payload received so far |
| `I` | 196 hex digits | `R d` | Classifies one image, `d` = digit |
| `D` | 196 hex digits | `A1 …`, `H …`, `A2 …`, then `R d` | Like `I`, plus the intermediate values in decimal, separated by spaces |
| `G` | — | `G af bc de hl ix iy sp pc af' bc' de' hl' i r` | Registers, 4 hex digits each (`i`, `r`: 2), snapped by `snap_regs()` right after `nn_infer` returns in the last `I`/`D`; all zero before the first one. PC is the return address of that call, inside `cmd_infer`. The same `I`/`D` copies the 256 bytes from that SP into `stack_copy` (read it with `M`): `G` and `M` reuse that stack area for their own frames |
| `M` | `aaaa` + space + `nn` | `M` + space + `2·nn` hex digits, or `ERR <reason>` | Reads `nn` bytes (`01`..`40` hex) from address `aaaa`; the address wraps at `FFFF`. `nn` = `00` or above `40` is `ERR LEN` |

**Weights transfer**: the payload of `weights.bin` (§5.1, without the 5-byte header) is sent in 64-byte chunks: 399 full lines plus one line of 45 bytes, offsets `0000`, `0040`, `0080`, … The firmware copies each chunk to `(uint8_t *)&weights + offset`: the struct has the payload's layout (§7.2), and the Z80 is little-endian like the file, so no field needs decoding. A chunk that would end past byte 25,581 is `ERR OFS`.

**Checksum**: the sum of all 25,581 payload bytes, as unsigned bytes, modulo 65,536, printed as 4 upper-case hex digits. The host computes the same value and compares.

**Image encoding**: 784 bits packed into 98 bytes. Pixel `i` is bit `7 - (i mod 8)` of byte `i / 8`: the first pixel is the **most significant** bit of the first byte.

**Errors**: `ERR CMD` (unknown command), `ERR HEX` (invalid hex digit), `ERR LEN` (wrong number of digits), `ERR OFS` (offset out of range), `ERR LONG` (line too long). After an error the firmware keeps running.

**Serial line (board)**: MC6850 ACIA, control register `0x16`: clock ÷ 64, 8N1, RTS low, no interrupts. With Searle's 7.3728 MHz crystal that is 115,200 baud. The host opens the port with RTS/CTS flow control. In the emulator, `z88dk-ticks` emulates the same ACIA on its stdin/stdout; it never flushes that output, so the host gives ticks a pseudo-terminal as stdout (on a terminal the C library flushes at every `\n`).

**Input without echo**: the emulator build reads with `fgetc_cons()`, which does not echo (unlike `getchar()`); the board build reads the ACIA data register directly.

**Debug commands** `G` and `M` serve the debug panel (§7.7) in snapshot mode. They cost about 460 bytes of ROM (`board.rom` grew from 3,511 to 3,970 bytes) and 282 bytes of RAM (`regs`, `stack_copy`).

### 7.6 `host/z80nn.py` — the bridge, and the `Makefile`

`Z80NN(target, where)` hides where the Z80 is; `version()`, `load_weights()`, `classify()`, `classify_debug()`, `registers()` and `read_memory()` then work the same everywhere.

| `target` | `where` | Connection |
| --- | --- | --- |
| `emu` | `build/emu.bin` or `build/board.rom` | Starts `z88dk-ticks -counter 100000000000` on it (ticks otherwise stops after 10^8 T-states, about 10 classifications), stdin a pipe, stdout a pseudo-terminal. Optional `ticks=` (another emulator binary) and `shm=` (the live-view file, §7.7) |
| `board` | serial port | 115,200 baud, 8N1, RTS/CTS, 5 s timeout (`pyserial`) |
| `local` | a program on the laptop | Its stdin/stdout |

`close()` closes the input; a program that does not exit within 2 s is killed. That happens to `board.rom` in plain ticks, which never sees the end of input (§7.5); with `-shm` ticks exits at once.

**Makefile targets**:

| Target | Effect |
| --- | --- |
| `make` (`all`) | Missing artifacts, DeZog inputs, and the `firmware`, `emu` and `rom` builds |
| `make train` | Retrain from scratch, then quantize and export again |
| `make check` | `reference.py`: float vs integer accuracy |
| `make test` | Builds `emu`, `rom` and, when z88dk's sources are there, `ticks-viz`; then pytest, listing the reason of every skipped test |
| `make emu` / `make rom` | `build/emu.bin` / `build/board.rom` (§7.4) |
| `make firmware` | `build/firmware_*` for DeZog; `FW=` picks another source |
| `make inputs CASE=n` | `build/weights_payload.bin` and `build/case.bin` for DeZog |
| `make ticks-viz` | The patched emulator in `build/ticks-viz` (§7.7) |
| `make demo` | Builds `rom` and runs the drawing demo on the emulator with `build/board.rom` (`host/demo.py`'s default, map `build/board.map`); `PORT=…` on the board, `DEBUG=1` with the debug panel there |
| `make clean` | Removes `build/` (including `build/ticks-viz`) and the listings; `artifacts/` stay |

`PYTHON` defaults to `.venv/bin/python` when that exists, else `python`. Recipe lines start with a **TAB**.

### 7.7 Debug panel and live-view file

`host/debug_panel.py` shows registers, call stack, stack, memory and memory accesses of the running Z80. It gets them from one of two sources in `host/z80debug.py`, which have the same `capture(view_addr)`:

| Source | Used for | How |
| --- | --- | --- |
| `SnapshotSource` | The board, or the emulator without the patch | After each result: `G`, then `M` for the variables, `stack_copy` and the memory window. Bytes that changed since the previous capture count as writes; there are no reads or instruction fetches (the firmware cannot see its own bus) |
| `LiveSource` | The emulator with the patched ticks | Reads the live-view file below, any time, also during an inference |

Symbols and memory-map blocks come from the program's `.map` file (`MapSymbols`). The call stack follows SDCC's frame chain: a function that uses a frame does `push ix; ld ix,0; add ix,sp`, so `(IX)` is the caller's IX and `(IX+2)` the return address. SP and IX can be odd (SDCC sometimes pushes single bytes), so words are read at any offset.

**Live-view file** (normative between `tools/ticks-viz.patch` and `LiveSource`). `z88dk-ticks -shm <file>` creates `<file>` (131,136 bytes) and maps it shared; the host maps it too. Little-endian, the host's byte order:

| Offset | Size | Field | Written by |
| --- | --- | --- | --- |
| 0 | 4 | magic `Z8VZ`, written last when the file is ready | ticks |
| 4 | 4 | version (1) | ticks |
| 8 | 4 | `seq`: odd while ticks writes the next three fields, +2 per publish | ticks |
| 12 | 4 | `speed_hz`: emulated clock, 0 = as fast as possible | host |
| 16 | 4 | `paused`: non-zero = stop between two instructions | host |
| 24 | 8 | T-states | ticks |
| 32 | 24 | `af bc de hl ix iy sp pc af' bc' de' hl'`, uint16 each (the order of `struct regs`) | ticks |
| 56 | 2 | `i`, `r` | ticks |
| 64 | 65,536 | The emulated memory itself | ticks (it is the CPU's RAM) |
| 65,600 | 65,536 | One byte per address: bit 0 read, bit 1 write, bit 2 prog (instruction fetch) | ticks sets bits, the host clears them after reading |

Rules:

- ticks publishes the registers about 100 times per emulated second (every 200,000 T-states at full speed), only between instructions, and once more before it blocks waiting for input. A reader takes a value only when `seq` is even and the same before and after reading.
- Blocking reads: `emu.bin` reads the console (`cmd_readkey()`, which calls `getchar()`); `board.rom` reads the ACIA data register (`acia_in()`, port `0x81`, which calls `getch()`). Both publish first (`viz_idle()`) and then block in the C library until a byte arrives, so the T-states stop and the panel shows `idle`. Pause and clock changes made while blocked take effect at the next publish after the read returns.
- **Dependency on z88dk's ACIA emulation.** `board.rom` waits in the data read only because the status read (port `0x80`) never reports "empty": it tests `kbhit()` for truth, and `kbhit()` returns −1, not 0, when no byte is waiting, whatever stdin is. So RX full is always set and `ser_getc()`'s polling loop never repeats. This is undocumented behaviour of the installed z88dk. If a later z88dk reports "empty" correctly, the firmware spins on the status register, using a full core and counting T-states while idle, and never reaches the data read, so the end-of-input exit below stops working too. `test_board_rom_waits_without_running` in `tests/test_ticks_viz.py` is the guard: it fails when the T-states move while `board.rom` waits for a command, so a z88dk upgrade fails loudly instead of quietly burning a core.
- End of input: when the ACIA data read gets `EOF` (the host closed stdin), the patched ticks with `-shm` exits with status 0. The firmware cannot tell end of input from data (plain ticks returns `0xFF`), so without this the program would never end. `emu.bin` needs no help: its console read returns −1 and the firmware returns from `main`.
- At `speed_hz` > 0 ticks sleeps to keep the emulated clock; time spent paused or waiting for input is not caught up.
- Access bits are set in `get_memory()` (read or prog, by the access type) and `put_memory()` (write); loading the program sets none. A write to write-protected memory still counts as a write.
- Only the standard 64 KB memory model maps its RAM into the file.
- Without `-shm`, the patched ticks behaves exactly like the original.

The patch (`tools/ticks-viz.patch`) adds `viz.h`/`viz.c` and hooks in `memory.c`, `ticks.c` (the CPU loop), `ticks_main.c` (the option), `hook_console.c` (before blocking reads), `acia.c` (exit at end of input) and the `Makefile`. `make ticks-viz` applies it to a copy of the installed z88dk's `src/ticks` in `build/ticks-viz`; the installed z88dk is not changed.

---

## 8. Assembly (optional)

Assembly is **not** required. It belongs to the optimization phase, after the C version passes all tests on the board.

**Rules**:

1. **Keep the C version.** The assembly replaces one function behind a compile-time switch, so both can be built and compared.
2. **Same contract, same tests.** The assembly function must make `make test` pass with 105/105 exactly like the C one.
3. **Measure before and after** with `z88dk-ticks` (it prints the T-states at exit; `-start`/`-end` limit the count to a PC range) and record both, for example in a `docs/measurements.md`.
4. **Start from the compiler's output** (`--list`) for the same function and improve it, instead of writing from scratch.
5. **Calling convention**: how arguments reach an assembly function depends on the compiler and its options. Check the z88dk documentation for sdcc's convention (`-compiler=sdcc`, the classic library used here) before writing any code.

**Candidates**, in order of expected benefit:

| Candidate | Idea |
| --- | --- |
| Layer 1 inner loop | It is ~99% of the work. Best combined with a **transposed** `W1q` (784 rows × 32): for each lit pixel, add its 32 weights to 32 accumulators |
| 16-bit accumulators | Accumulate at most 257 lit pixels in `int16_t` (127 · 257 = 32,639 fits), then add to `int32_t` |
| Layer-2 multiplication | Quarter-square table: `a · b = ⌊(a+b)²/4⌋ − ⌊(a−b)²/4⌋`, exact for integers |

Changing the layout of `W1q` changes §5: bump the file format **version** to 2, update `export.py`, `nn.h`/`nn.c`, `z80nn.py` (it checks the version) and `tests/make_dezog_inputs.py` together, and update this document.

---

## 9. Testing strategy

The same 105 cases of `testvec.bin` are the reference at every level.

| Level | Command | What it proves |
| --- | --- | --- |
| 1. Python | `make check`, and `export.py`'s read-back | The integer reference agrees with the float model; the files hold exactly what `infer_int` computes |
| 2. Emulator | `make test` | `nn.c` is right with 16-bit `int`, in both builds (`emu.bin` and the EEPROM image `board.rom`); the protocol, the debug commands, the debug panel's sources and the patched ticks work |
| 3. Step by step | DeZog, launch config "Z80 zsim" | One case (`make inputs CASE=n firmware`) with breakpoints; the mailbox reports the first mismatch |
| 4. Board | `make demo PORT=…` | The hardware, the ROM and the serial link: connection, weight upload with checksum, classifications |

**Acceptance**: 105/105 at level 2, every intermediate value identical.

**Tests** (`tests/`):

| File | Checks |
| --- | --- |
| `test_emu.py` | All 105 cases with `D` on both builds, `I` agreeing with `D`, every error reply, `G` (PC inside `cmd_infer`, SP in the stack, `stack_copy` holding the return address into `main`) and `M` (the weights, the unpacked image, wrap-around) |
| `test_z80debug.py` | The `.map` parser on both builds, flag decoding, snapshots on both builds, `draw.Worker` |
| `test_ticks_viz.py` | The patched ticks with `-shm` on both builds: the 105 cases, accesses (upload, layer 1 reading exactly the rows of lit pixels, no instruction fetched from data), the call stack mid-inference, the clock and Pause. Skipped without `make ticks-viz` |
| `test_preprocess.py` | The drawing preprocessing of `host/draw.py` |

**Mutation check**: temporarily introduce a known bug (for example `>=` instead of `>` in the argmax, or remove the `(int32_t)` cast in `requant`), confirm that `make test` fails, then undo it.

**Host side** (`host/`): `z80nn.py` (§7.6); `draw.py`, the drawing window and MNIST-style preprocessing (crop to the bounding box, longest side to 20 px, center of mass at (14, 14) in 28 × 28, the training threshold); `demo.py`, which joins them and sends images with `I`, or `D` with the debug panel (§7.7).

---

## 10. Open points

Resolved:

- [x] Output file of `-create-app`: `build/board.rom`, written to the EEPROM (§7.4).
- [x] Echo and line buffering: `fgetc_cons()` on the emulator, the ACIA polled directly on the board (§7.5).
- [x] Cycle count: `z88dk-ticks` prints the T-states at exit; `-start`/`-end` limit the count (§8).
- [x] Memory: code, weights and stack fit in both builds (§7.4).

Open:

- [ ] Run the 105 cases on the real board (level 4 only classifies drawn digits). `Z80NN("board", port)` already supports everything `test_emu.py` uses.
- [ ] sdcc's calling convention between C and assembly, before writing any assembly (§8).
- [ ] Searle's schematic: exact 74HC chips, RTS/CTS wiring, ACIA interrupt line to `/INT`.
- [ ] `tools/ticks-viz.patch` is made for the z88dk sources installed now; a newer z88dk may need it updated.
