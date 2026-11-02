# MNIST on Z80 — Specification

This document is the contract between the parts of the project. The Python pipeline defines what the correct answer is; the C code (on the laptop and on the Z80) must reproduce it **bit for bit**. When this document and the code disagree, fix one of them so they agree again.

Sections 2–5 are **normative**: they define exactly what every implementation must compute and how files and messages are laid out. Sections 6–9 are implementation guides.

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
├── firmware/      nn.h, nn.c, z80_main.c   (+ optional .asm)
├── tests/         host_main.c, test_e2e.py
├── host/          z80nn.py, demo.py
├── artifacts/     generated files, not committed (see .gitignore)
│   ├── data/          MNIST, downloaded by torchvision
│   ├── mlp.pt         float weights
│   ├── quantized.npz  integer weights and constants
│   ├── weights.bin    integer weights for the C code
│   └── testvec.bin    test vectors for the C code
├── docs/          this file, measurements, report
├── hardware/      photos, wiring notes
├── Makefile
└── README.md
```

| Folder | Runs on | Who uses it |
| --- | --- | --- |
| `training/` | Laptop, Python | Run once to produce `artifacts/` |
| `firmware/` | Z80 (and laptop, for testing) | The product |
| `tests/` | Laptop | Verifies `firmware/` against `artifacts/testvec.bin` |
| `host/` | Laptop, Python | Talks to the board (or the emulator) |

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

The same `nn.c` is compiled by `gcc`/`clang` on the laptop (where `int` is 32 bits) and by z88dk on the Z80 (where `int` is **16 bits**). Code that works on the laptop can silently fail on the Z80. These rules apply to every C file in `firmware/`:

1. **Use only fixed-width types** from `<stdint.h>` for data: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`. Use plain `int` only where the value is guaranteed to fit in 16 bits.
2. **Never shift an untyped constant into the upper bits.** Write `(int32_t)1 << (shift - 1)`, never `1 << (shift - 1)`: on the Z80 the second form is undefined behavior for `shift > 15`.
3. **No signed overflow.** Every accumulator is `int32_t`, and §4.1 proves the ranges.
4. **Loop indices must fit their range.** A loop to 784 needs `uint16_t`; `uint8_t` wraps at 256 and the loop never ends.
5. **Right shift only non-negative values.** Shifting a negative number right is implementation-defined. The requantization only shifts after the ReLU.
6. **Casts in products**: write `(int16_t)a * (int16_t)b` in layer 2 so the compiler uses a 16-bit multiplication.
7. **No dynamic memory** (`malloc`/`free`).
8. **Large data is never on the stack.** The weights (≈25.6 KB) are a global or `static` variable. The Z80 stack is a few hundred bytes to a few KB.
9. **`printf` with explicit casts**: `printf("%ld", (long)v)` for an `int32_t`, `printf("%u", (unsigned)v)` for small unsigned values.
10. **`nn.c` does no I/O**: it does not include `<stdio.h>`. All input/output lives in `host_main.c` or `z80_main.c`.

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

The struct is a convenient container, **not** a description of a file layout: the compiler may insert padding between fields. Always fill it field by field, never with a single read of the whole struct.

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

### 7.4 `tests/host_main.c` — laptop test runner

**Usage**: `test_host <weights.bin> <testvec.bin>`. Exit code 0 if every case matches, 1 otherwise.

**Behavior**:

1. Read `weights.bin` into a `static struct nn_weights`, field by field, checking magic, version, and that the file ends exactly after `b2q`.
2. Read the header of `testvec.bin`.
3. For each case: read the record, call `nn_infer` with a trace, and compare in this order: `acc1[0..31]`, `h_q[0..31]`, `acc2[0..9]`, digit. Report the **first** mismatch (case index, field, index, expected, actual) and exit with 1.
4. Print `n/n cases OK` and exit with 0.

**Helper functions** (recommended):

| Helper | Behavior |
| --- | --- |
| `die(msg)` | Print to `stderr`, `exit(1)` |
| `read_exact(f, dest, n)` | `fread` exactly `n` bytes or `die` |
| `read_u8(f)` | One byte |
| `read_u16_le(f)` | `b[0] \| b[1] << 8` |
| `read_i32_le(f)` | `b[0] \| b[1] << 8 \| b[2] << 16 \| b[3] << 24`, each byte converted to `uint32_t` before shifting, the result converted to `int32_t` |

`int8` arrays (`W1q`, `W2q`) are read directly with `read_exact` into the struct field: a single byte has no byte order.

**Compiler flags** (see `Makefile`):

```
-std=c99 -Wall -Wextra -Werror -O1 -g
-fsanitize=undefined,address -fno-sanitize-recover=all
```

UBSan stops at the first undefined behavior, ASan at the first out-of-bounds access. Passing on the laptop is **necessary but not sufficient**: `int` is 32 bits there, so 16-bit overflows only show up on the Z80 (emulator or board).

### 7.5 `firmware/z80_main.c` — firmware and serial protocol

The firmware is an endless loop: read one line from the serial port, execute the command, reply with one or more lines. It uses the same `nn.c`. The weights live in a global `struct nn_weights` in RAM.

**Line format**:

- Commands are ASCII lines terminated by `\n`. A `\r` before it is ignored.
- Bytes are encoded as **two hexadecimal digits** (`00`..`FF`); upper and lower case are both accepted on input, the board replies in upper case.
- A line longer than the receive buffer is discarded and answered with `ERR LONG`.

**Commands**:

| Command | Arguments | Reply | Effect |
| --- | --- | --- | --- |
| `V` | — | `Z80NN 1` | Name and protocol version |
| `W` | `oooo` + space + up to 128 hex digits | `OK`, or `ERR <reason>` | Writes up to 64 bytes of the weights payload at offset `oooo` (4 hex digits) |
| `C` | — | `C xxxx` | 16-bit checksum of the whole payload received so far |
| `I` | 196 hex digits | `R d` | Classifies one image, `d` = digit |
| `D` | 196 hex digits | `A1 …`, `H …`, `A2 …`, then `R d` | Like `I`, plus the intermediate values in decimal, separated by spaces |

**Weights transfer**: the payload of `weights.bin` (§5.1, without the 5-byte header) is sent in 64-byte chunks: 399 full lines plus one line of 45 bytes, offsets `0000`, `0040`, `0080`, … The firmware decodes each chunk into the correct fields of `struct nn_weights` according to the offsets of §5.1 minus 5. Little-endian int32 fields are assembled exactly as in `read_i32_le`.

**Checksum**: the sum of all 25,581 payload bytes, as unsigned bytes, modulo 65,536, printed as 4 upper-case hex digits. The host computes the same value and compares.

**Image encoding**: 784 bits packed into 98 bytes. Pixel `i` is bit `7 - (i mod 8)` of byte `i / 8`: the first pixel is the **most significant** bit of the first byte.

**Errors**: `ERR CMD` (unknown command), `ERR HEX` (invalid hex digit), `ERR LEN` (wrong number of digits), `ERR OFS` (offset out of range), `ERR LONG` (line too long). After an error the firmware keeps running.

**Build** (z88dk, target `+rc2014`, which matches Searle's board: ACIA at ports `0x80`/`0x81`, ROM from `0x0000`, RAM from `0x8000`):

| Purpose | Subtype | Code location |
| --- | --- | --- |
| Board (EEPROM) | `-subtype=acia` | ROM from `0x0000`, data in RAM from `0x8000` |
| Emulator (`z88dk-ticks`) | `-subtype=basic` | RAM from `0x9000` |

Common flags: `+rc2014 -clib=sdcc_iy -SO3 -m`. Check the `.map` file after every build: the weights struct and the stack must fit in RAM.

**Input driver**: z88dk's serial input may echo characters and buffer whole lines by default; the protocol needs echo disabled. See the open points (§10).

### 7.6 `Makefile`

| Target | Effect |
| --- | --- |
| `make test-host` | Builds `build/test_host` and runs it on `artifacts/weights.bin` and `artifacts/testvec.bin` |
| `make sim` | Builds the firmware for `z88dk-ticks` |
| `make rom` | Builds the firmware for the EEPROM |
| `make burn` | Writes the ROM image with `minipro` |
| `make test` | `test-host`, then the end-to-end test on the emulator |
| `make clean` | Removes `build/` |

Recipe lines start with a **TAB**.

---

## 8. Assembly (optional)

Assembly is **not** required. It belongs to the optimization phase, after the C version passes all tests on the board.

**Rules**:

1. **Keep the C version.** The assembly replaces one function behind a compile-time switch, so both can be built and compared.
2. **Same contract, same tests.** The assembly function must make `make test` pass with 105/105 exactly like the C one.
3. **Measure before and after** with `z88dk-ticks` and record both in `docs/measurements.md`.
4. **Start from the compiler's output** (`--list`) for the same function and improve it, instead of writing from scratch.
5. **Calling convention**: how arguments reach an assembly function depends on the compiler and its options. Check the z88dk documentation for the convention used with `sdcc_iy` before writing any code.

**Candidates**, in order of expected benefit:

| Candidate | Idea |
| --- | --- |
| Layer 1 inner loop | It is ~99% of the work. Best combined with a **transposed** `W1q` (784 rows × 32): for each lit pixel, add its 32 weights to 32 accumulators |
| 16-bit accumulators | Accumulate at most 257 lit pixels in `int16_t` (127 · 257 = 32,639 fits), then add to `int32_t` |
| Layer-2 multiplication | Quarter-square table: `a · b = ⌊(a+b)²/4⌋ − ⌊(a−b)²/4⌋`, exact for integers |

Changing the layout of `W1q` changes §5: bump the file format **version** to 2, update `export.py`, `host_main.c` and the firmware together, and update this document.

---

## 9. Testing strategy

The same 105 cases of `testvec.bin` are checked at three levels. A level is only attempted after the previous one passes.

| Level | Command | What it proves |
| --- | --- | --- |
| 1. Laptop | `make test-host` | The algorithm in `nn.c` is right; no UB detected by the sanitizers |
| 2. Emulator | `make test` (with `z88dk-ticks`) | `nn.c` is right with 16-bit `int`; the protocol works |
| 3. Board | `python tests/test_e2e.py --port /dev/...` | The hardware, the ROM and the serial link work |

**Acceptance**: 105/105 at each level, every intermediate value identical.

**Mutation check**: after level 1 passes, temporarily introduce a known bug (for example `>=` instead of `>` in the argmax, or remove the `(int32_t)` cast in `requant`) and confirm that a test fails. Then undo it.

**Host tools** (`host/`):

| File | Role |
| --- | --- |
| `z80nn.py` | Library: connect to the board or to `z88dk-ticks`, send commands, load weights and verify the checksum, classify images |
| `tests/test_e2e.py` | Sends the 105 cases with `D` and compares every value with `testvec.bin` |
| `demo.py` | Drawing canvas, MNIST-style preprocessing (crop to bounding box, scale longest side to 20 px, center of mass at (14, 14) in 28 × 28, same 0.5 threshold), sends the image with `I`, shows the digit |

---

## 10. Open points

These details are not verified yet and must be confirmed against the official documentation before being relied on:

- [ ] z88dk: exact output file names of `-create-app` for the `acia` and `basic` subtypes, and which one to write to the EEPROM.
- [ ] z88dk: how to disable echo and line buffering in the serial input driver, and which line terminator it expects.
- [ ] z88dk: options of `z88dk-ticks` to print the cycle count.
- [ ] z88dk: whether code + weights fit in RAM in the `basic` subtype, and the right `CRT_ORG_CODE` if not.
- [ ] z88dk: calling convention between C (`sdcc_iy`) and assembly.
- [ ] Searle's schematic: exact 74HC chips, RTS/CTS wiring, ACIA interrupt line to `/INT`.