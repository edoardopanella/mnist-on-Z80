# MNIST on Z80 — Implementation Plan

Status as of 2026-11-02. Companion to `docs/spec.md`, which remains the normative contract.

## 1. Status of the blocks

| Block | File(s) | Status |
| --- | --- | --- |
| Training | `training/train.py` | Done |
| Quantization | `training/quantize.py` | Done, one bug (§3.1) |
| Golden integer model | `training/reference.py` | Done |
| Export | `training/export.py` | Done (read-back checks, edge-case vectors) |
| Specification | `docs/spec.md` | Done |
| Inference library | `firmware/nn.h`, `firmware/nn.c` | Missing |
| Laptop test runner | `tests/host_main.c` | Missing |
| Build | `Makefile` | Missing |
| Firmware (serial protocol, weight loading) | `firmware/z80_main.c` | Missing |
| Host tools | `host/z80nn.py`, `tests/test_e2e.py`, `host/demo.py` | Missing |
| Assembly optimization | optional `.asm` | Not started (optional, spec §8) |
| Measurements | `docs/measurements.md` | Missing |

## 2. Where the cost is: multiplication is not the bottleneck

The input is binarized, so layer 1 contains no multiplications:

    acc1[j] = b1q[j] + sum of W1q[j][i] over the lit pixels i

That is 32 × 784 ≈ 25k conditional 32-bit additions, about 99% of the work.

All multiplications in the network:

- Layer 2: 320 products, 8 bit × 8 bit, result in 16 bits.
- Requantization: 32 products, 32 bit × `M0`.

352 multiplications per image. The z88dk library routines handle them correctly and their cost is negligible next to layer 1. A hand-written assembly routine should therefore target the **layer-1 accumulation loop**, not multiplication.

Intuition: a binary input turns the matrix-vector product into "select columns and add them". Multiplication only appears in layer 2, whose input `h_q` (0..127) is not binary.

### Layer-1 improvements (bit-identical results)

1. **Visit only lit pixels.** A binarized MNIST digit has roughly 100–200 lit pixels out of 784. Build the list of lit indices once per image; each neuron's loop then runs about 5× fewer iterations.
2. **Transpose `W1q`** (spec §8): for each lit pixel, add its 32 weights to 32 accumulators. Memory is read sequentially, which an assembly loop exploits well. This changes the file format (version 2: update `export.py`, `host_main.c`, firmware and spec together), so it comes only after the C version passes all tests.

## 3. Known bugs and inconsistencies

1. **`quantize.py` crashes on a clean run.** It calls `get_data(DATA_DIR=DATA_DIR)`, but `train.py` defines `get_data(data_dir=...)`, so the call raises `TypeError: unexpected keyword argument`. Probably introduced by the repo reorganization. Existing generated files are fine; a clean rerun is not. Fix: `get_data()` or `get_data(data_dir=DATA_DIR)`.
2. **README out of date.** It says to run the scripts from `training/` and write to `model/`; the code and spec use `artifacts/` and run from the repo root.
3. **`.gitignore` does not list `artifacts/`.** Generated files are currently excluded only by the `*.pt` / `*.npz` / `*.bin` / `data/` patterns. Add an explicit `artifacts/` line; the `model/` line can go.
4. **Emulator memory is tight** (already an open point in spec §10). With subtype `basic`, RAM starts at `0x9000`: about 28 KB for 25.6 KB of weights plus code, C runtime library, buffers and stack. Linking `printf` can exceed it. Settle this first with the `.map` file of a minimal build.
5. **Minor:** comments and printed messages in `reference.py` are in Italian; the rest of the code is in English.

## 4. Next steps, in dependency order

1. Fix bug §3.1 and rerun the whole Python pipeline once from the repo root.
2. Write `nn.h` and `nn.c` exactly as in spec §7.2–7.3. Core block: plain C with no I/O, so the same file compiles on the laptop and on the Z80.
3. Write `tests/host_main.c` and the `test-host` Makefile target. Reach 105/105 on the laptop with sanitizers on, then run the mutation check of spec §9 (plant a known bug, confirm a test fails, undo it).
4. Build a minimal z88dk program for `z88dk-ticks` (print one character) to settle the open points of spec §10: output file names, memory fit, cycle counting.
5. Write `z80_main.c` (serial protocol) and `host/z80nn.py`, then run the end-to-end test on the emulator. First point where problems from the Z80's 16-bit `int` can appear.
6. Optional: layer-1 loop in assembly, following spec §8 (C version kept behind a compile-time switch, cycle counts before and after in `docs/measurements.md`).

The C/assembly calling convention (spec §10) matters only at step 6. Steps 2–5 are the deliverable; step 6 is extra.

## 5. Resources

- z88dk wiki: "Classic vs New library" and the `sdcc_iy` pages (for spec §10).
- SDCC manual: Z80 calling-convention sections (`__sdcccall`), for step 6.
- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018, §2.2: the `M0` / `shift` requantization.
