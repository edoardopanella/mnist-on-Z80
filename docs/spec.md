# MNIST on Z80 — Specification

This document is the contract between the parts of the project. The Python pipeline defines what the correct answer is; the C code on the Z80, in the emulator or on the board it is intended for, must reproduce it **bit for bit**. When this document and the code disagree, fix one of them so they agree again.

Sections 1–4 are **normative**: they define exactly what every implementation must compute and how files are laid out. So are the serial protocol (§5.5) and the live-view file (§5.6), which two programs must read the same way. The rest are implementation guides. How to install, build and run everything is in the [README](../README.md) and the [guide](guide.md).

## Contents

1. [The network](#1-the-network)
2. [Quantization scheme](#2-quantization-scheme)
   1. [Notation](#21-notation)
   2. [Weights: symmetric per-layer quantization](#22-weights-symmetric-per-layer-quantization)
   3. [Biases: the scale of the accumulator](#23-biases-the-scale-of-the-accumulator)
   4. [Activations of layer 1](#24-activations-of-layer-1)
   5. [Requantization constants `M0` and `shift`](#25-requantization-constants-m0-and-shift)
   6. [Current values](#26-current-values)
3. [Integer inference algorithm (normative)](#3-integer-inference-algorithm-normative)
   1. [Value ranges (guaranteed)](#31-value-ranges-guaranteed)
4. [Binary file formats (normative)](#4-binary-file-formats-normative)
   1. [`weights.bin`](#41-weightsbin-25586-bytes)
   2. [`testvec.bin`](#42-testvecbin-7--n--985-bytes)
5. [C implementation](#5-c-implementation)
   1. [Portability rules](#51-portability-rules)
   2. [`nn.h`: interface](#52-srcfirmwarennh--interface)
   3. [`nn.c`: inference](#53-srcfirmwarennc--inference)
   4. [The Z80 programs](#54-the-z80-programs)
   5. [`z80_serial_main.c`: firmware and serial protocol](#55-srcfirmwarez80_serial_mainc--firmware-and-serial-protocol)
   6. [Debug panel and live-view file](#56-debug-panel-and-live-view-file)
6. [Testing strategy](#6-testing-strategy)

---

## 1. The network

| Item | Value |
| --- | --- |
| Architecture | MLP 784 → 32 → 10, ReLU after the first layer, no activation after the second |
| Input | 28 × 28 MNIST image, **binarized**: pixel = 1 if its value after `ToTensor()` is > 0.5, else 0 |
| Pixel order | Row by row from the top, left to right in each row (same as `nn.Flatten`). Pixel index `i = 28 · row + col` |
| Output | The digit with the largest output (argmax) |
| Training | PyTorch, Adam (lr 1e-3), batch 64, 10 epochs, cross-entropy, seed 0 |

There is no normalization of the input: the network sees exactly the 0/1 values that the Z80 will receive.

---

## 2. Quantization scheme

All weights become integers; every scale factor is used only on the laptop.

`src/training/quantize.py` implements this section (`quantize_symmetric`, `choose_m0_shift`). It asserts the no-overflow condition and the integer types, then saves `W1q`, `b1q`, `W2q`, `b2q`, `M0` and `shift` (plus `s1`, `s2`, `s_h`, `h_max` and `bound` for diagnostics) to `artifacts/quantized.npz`.

### 2.1 Notation

| Symbol | Meaning |
| --- | --- |
| `W1`, `b1` | Float weights (32 × 784) and biases (32) of layer 1 |
| `W2`, `b2` | Float weights (10 × 32) and biases (10) of layer 2 |
| `s` | A **scale**: the real value of one integer step |
| `q` suffix | Quantized (integer) version |

The relation between a real value `v`, its scale `s` and its integer version `vq` is always `v ≈ s · vq`.

### 2.2 Weights: symmetric per-layer quantization

For each weight matrix `W`:

```
s  = max(|W|) / 127
Wq = clip(round(W / s), -127, 127)       stored as int8
```

The value -128 is never used, so the range is symmetric and 0 maps exactly to 0.

### 2.3 Biases: the scale of the accumulator

A bias is added to an accumulator, so it must be expressed in the accumulator's units. It is **not** quantized with its own scale.

| Bias | Formula | Why |
| --- | --- | --- |
| `b1q` | `round(b1 / s1)` | The input is 0/1 (scale 1), so `acc1` is in units of `s1` |
| `b2q` | `round(b2 / (s_h · s2))` | Layer 2 multiplies values in units of `s_h` by weights in units of `s2` |

Biases are stored as int32 and are **not** clipped.

### 2.4 Activations of layer 1

The scale of the hidden activations is calibrated on the **training set** (never on the test set):

```
h_max = max over all training images of ReLU(W1 · x + b1)     (float)
s_h   = h_max / 127
```

### 2.5 Requantization constants `M0` and `shift`

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

### 2.6 Current values

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

## 3. Integer inference algorithm (normative)

Every implementation (`reference.py` and `nn.c`) must compute exactly these values. `infer_int()` in `src/training/reference.py` is the executable form of this section; it processes **one** image on purpose, to mirror the C code step by step.

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
   `r · M0` is computed in 32-bit signed arithmetic; the condition of §2.5 guarantees no overflow. `r ≥ 0`, so the right shift is a division rounded down. Adding `2^(shift-1)` first turns it into rounding to nearest.
4. **Layer 2**, for each digit `k` in 0..9:
   ```
   acc2[k] = b2q[k] + sum over j of W2q[k][j] · h_q[j]                     (int32)
   ```
   Each product fits in 16 bits: |W2q · h_q| ≤ 127 · 127 = 16,129.
5. **Argmax**: the result is the smallest `k` such that `acc2[k]` is maximal (**ties go to the lowest index**, like `numpy.argmax`). In C this means comparing with a strict `>`.

### 3.1 Value ranges (guaranteed)

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

## 4. Binary file formats (normative)

Produced by `src/training/export.py`, which refuses to convert any value outside its type's range (numpy's `astype` would wrap silently), then reads both files back and compares every value. All multi-byte integers are **little-endian** (least significant byte first), the byte order of the Z80. Two-dimensional arrays are stored **row by row**, which is also how C lays out `int8_t a[R][C]` in memory.

### 4.1 `weights.bin` (25,586 bytes)

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

The **payload** is everything after the 5-byte header (offsets 5..25,585, 25,581 bytes). It is what the serial protocol transfers (§5.5).

### 4.2 `testvec.bin` (7 + n · 985 bytes)

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

Current content: 105 cases — the first 100 test-set images, then `all_black`, `all_white`, `only_pixel_0`, `only_pixel_783`, and `worst_neuron_N` (the image that drives neuron N to exactly `bound`; `export.py` checks that it does).

---

## 5. C implementation

### 5.1 Portability rules

`nn.c` is compiled by z88dk (sdcc) for the Z80, where `int` is **16 bits**. It is written in plain C99 so that it would also compile unchanged on a laptop, where `int` is 32 bits; code that works there can silently fail on the Z80. These rules apply to every C file in `src/firmware/`:

1. **Use only fixed-width types** from `<stdint.h>` for data: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`. Use plain `int` only where the value is guaranteed to fit in 16 bits.
2. **Never shift an untyped constant into the upper bits.** Write `(int32_t)1 << (shift - 1)`, never `1 << (shift - 1)`: on the Z80 the second form is undefined behavior for `shift > 15`.
3. **No signed overflow.** Every accumulator is `int32_t`, and §3.1 proves the ranges.
4. **Loop indices must fit their range.** A loop to 784 needs `uint16_t`; `uint8_t` wraps at 256 and the loop never ends.
5. **Right shift only non-negative values.** Shifting a negative number right is implementation-defined. The requantization only shifts after the ReLU.
6. **Casts in products**: write `(int16_t)a * (int16_t)b` in layer 2 so the compiler uses a 16-bit multiplication.
7. **No dynamic memory** (`malloc`/`free`).
8. **Large data is never on the stack.** The weights (≈25.6 KB) are a global or `static` variable, or sit at a fixed address. The Z80 stack is a few hundred bytes.
9. **No `printf`.** The firmware prints numbers with its own `put_hex4`/`put_i32`: smaller, and no `%ld` support to configure.
10. **`nn.c` does no I/O**: it does not include `<stdio.h>`. All input/output lives in the `z80_*_main.c` files.

### 5.2 `src/firmware/nn.h` — interface

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
| return value | The digit, 0..9, as defined in §3 |

With sdcc the struct has **no padding** (no field needs alignment on the Z80), so it has exactly the layout of the `weights.bin` payload (§4.1). The firmware relies on this and copies the payload into it byte for byte; both `z80_serial_main.c` and `z80_nn_main.c` check it at compile time:

```c
_Static_assert(sizeof(struct nn_weights) == 25581, "nn_weights layout != weights.bin payload");
```

A compiler that pads the struct fails to build instead of computing wrong results.

### 5.3 `src/firmware/nn.c` — inference

Four functions; the first three are `static` (private to the file).

| Function | Computes | Notes |
| --- | --- | --- |
| `layer1_neuron(row, bias, x)` | `acc1[j]` (§3 step 1) | Adds `row[i]` when `x[i] != 0`. Index is `uint16_t` |
| `requant(acc, M0, shift)` | `h_q[j]` (§3 steps 2–3) | ReLU first; `(int32_t)1 << (shift - 1)`; clamp to 127 |
| `layer2_neuron(row, bias, h_q)` | `acc2[k]` (§3 step 4) | `(int16_t)` casts on both factors |
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

### 5.4 The Z80 programs

| Program | Built by | Runs in | Input and output |
| --- | --- | --- | --- |
| `z80_serial_main.c` | `make emu` → `build/emu.bin` | `z88dk-ticks` | The line protocol (§5.5) on the console: `fgetc_cons()`/`fputc_cons()` |
| `z80_serial_main.c` | `make rom` → `build/board.rom` | The board, or `z88dk-ticks` | The same protocol through the MC6850 ACIA at ports `0x80`/`0x81` |
| `z80_nn_main.c` | `make inputs firmware` → `build/firmware_*.bin` | DeZog (zsim) | One test case loaded at `0x7800`, result in the mailbox at `0xF000` (§6) |
| `z80_test_main.c` | `make firmware FW=src/firmware/z80_test_main.c` | DeZog (zsim) | Writes a fixed value to the mailbox: checks the toolchain |

Build flags (see the `Makefile`; every build also writes a `.map` file with `-m`):

| Build | Flags | Memory map (from the `.map`) |
| --- | --- | --- |
| `emu` | `+test -compiler=sdcc` | 64 KB of RAM. Code from `0x0000` (to about `0x10E8`), variables to about `0x7BE3`, stack below `0xFF00` |
| `rom` | `+z80 -compiler=sdcc -pragma-define:CRT_ENABLE_EIDI=0x01 -create-app` | Code in ROM from `0x0000` (about 3.9 KB). Variables in RAM from `0x8000` (weights, buffers and `stack_copy`, to about `0xEA97`), stack below `0xFFFF`. `-create-app` writes `build/board.rom`, the file for the EEPROM. Interrupts stay off: the ACIA is polled |
| `firmware` | `+z80 -compiler=sdcc -m --list --c-code-in-asm -pragma-define:CRT_MODEL=0` | Code from `0x0000` (below `0x1000`), weights loaded at `0x1000`, test case at `0x7800`, variables from `0x8000`, mailbox `0xF000`. `--list` writes the `src/firmware/*.lis` listings DeZog uses for source-level debugging |

Check the `.map` file after a change that adds data: the variables and the stack must not meet. The addresses above move with every change; the `.map` is the reference, and the debug panel reads it.

### 5.5 `src/firmware/z80_serial_main.c` — firmware and serial protocol

The firmware is an endless loop: read one line, execute the command, reply with one or more lines. The same source is built for the emulator and for the board (§5.4); only its three character functions `ser_init`, `ser_getc`, `ser_putc` differ. The weights live in a global `struct nn_weights` in RAM, so they are uploaded again every time the program starts.

**Line format**:

- Commands are ASCII lines terminated by `\n`. A `\r` before it is ignored.
- Bytes are encoded as **two hexadecimal digits** (`00`..`FF`); upper and lower case are both accepted on input, the firmware replies in upper case.
- The receive buffer holds 198 characters, the longest valid line (`I ` + 196 hex digits). A longer line is discarded and answered with `ERR LONG`.
- At end of input (the emulator's stdin closed) `build/emu.bin` exits. The board keeps polling, and so does `build/board.rom` in plain `z88dk-ticks`, which then gives it `0xFF` bytes forever; with `-shm` the patched ticks exits instead (§5.6).

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

**Weights transfer**: the payload of `weights.bin` (§4.1, without the 5-byte header) is sent in 64-byte chunks: 399 full lines plus one line of 45 bytes, offsets `0000`, `0040`, `0080`, … The firmware copies each chunk to `(uint8_t *)&weights + offset`: the struct has the payload's layout (§5.2), and the Z80 is little-endian like the file, so no field needs decoding. A chunk that would end past byte 25,581 is `ERR OFS`.

**Checksum**: the sum of all 25,581 payload bytes, as unsigned bytes, modulo 65,536, printed as 4 upper-case hex digits. The host computes the same value and compares.

**Image encoding**: 784 bits packed into 98 bytes. Pixel `i` is bit `7 - (i mod 8)` of byte `i / 8`: the first pixel is the **most significant** bit of the first byte.

**Errors**: `ERR CMD` (unknown command), `ERR HEX` (invalid hex digit), `ERR LEN` (wrong number of digits), `ERR OFS` (offset out of range), `ERR LONG` (line too long). After an error the firmware keeps running.

**Serial line (board)**: MC6850 ACIA, control register `0x16`: clock ÷ 64, 8N1, RTS low, no interrupts. With Searle's 7.3728 MHz crystal that is 115,200 baud. The host opens the port with RTS/CTS flow control. In the emulator, `z88dk-ticks` emulates the same ACIA on its stdin/stdout; it never flushes that output, so the host gives ticks a pseudo-terminal as stdout (on a terminal the C library flushes at every `\n`).

**Input without echo**: the emulator build reads with `fgetc_cons()`, which does not echo (unlike `getchar()`); the board build reads the ACIA data register directly.

**Debug commands** `G` and `M` serve the debug panel (§5.6) in snapshot mode. They cost about 460 bytes of ROM (`board.rom` grew from 3,511 to 3,970 bytes) and 282 bytes of RAM (`regs`, `stack_copy`).

### 5.6 Debug panel and live-view file

`src/host/debug_panel.py` shows registers, call stack, stack, memory and memory accesses of the running Z80. It gets them from one of two sources in `src/host/z80debug.py`, which have the same `capture(view_addr)`:

| Source | Used for | How |
| --- | --- | --- |
| `SnapshotSource` | The board, or the emulator without the patch | After each result: `G`, then `M` for the variables, `stack_copy` and the memory window. Bytes that changed since the previous capture count as writes; there are no reads or instruction fetches (the firmware cannot see its own bus) |
| `LiveSource` | The emulator with the patched ticks | Reads the live-view file below, any time, also during an inference |

Symbols and memory-map blocks come from the program's `.map` file (`MapSymbols`). The call stack follows SDCC's frame chain: a function that uses a frame does `push ix; ld ix,0; add ix,sp`, so `(IX)` is the caller's IX and `(IX+2)` the return address. SP and IX can be odd (SDCC sometimes pushes single bytes), so words are read at any offset.

**Live-view file** (normative between `src/tools/ticks-viz.patch` and `LiveSource`). `z88dk-ticks -shm <file>` creates `<file>` (131,136 bytes) and maps it shared; the host maps it too. Little-endian, the host's byte order:

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

The patch (`src/tools/ticks-viz.patch`) adds `viz.h`/`viz.c` and hooks in `memory.c`, `ticks.c` (the CPU loop), `ticks_main.c` (the option), `hook_console.c` (before blocking reads), `acia.c` (exit at end of input) and the `Makefile`. `make ticks-viz` applies it to a copy of the installed z88dk's `src/ticks` in `build/ticks-viz`; the installed z88dk is not changed.

---

## 6. Testing strategy

The same 105 cases of `testvec.bin` are the reference at every level.

| Level | Command | What it proves |
| --- | --- | --- |
| 1. Python | `make check`, and `export.py`'s read-back | The integer reference agrees with the float model; the files hold exactly what `infer_int` computes |
| 2. Emulator | `make test` | `nn.c` is right with 16-bit `int`, in both builds (`emu.bin` and the EEPROM image `board.rom`); the protocol, the debug commands, the debug panel's sources and the patched ticks work |
| 3. Step by step | DeZog, launch config "Z80 zsim" | One case (`make inputs CASE=n firmware`) with breakpoints; the mailbox reports the first mismatch |
| 4. Board | `make demo PORT=…` | Intended for the board: the ROM and the serial link, through the connection, the weight upload with its checksum, and classifications |

**Acceptance**: 105/105 at level 2, every intermediate value identical. What each test file checks is listed in the [guide, section 7](guide.md#7-tests).

**Mutation check**: temporarily introduce a known bug (for example `>=` instead of `>` in the argmax, or remove the `(int32_t)` cast in `requant`), confirm that `make test` fails, then undo it.
