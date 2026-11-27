"""
export.py - Write the quantized weights and the test vectors as binary files for the C code.

Multi-byte numbers are little-endian, as on the Z80. The format must match docs/spec.md.

weights.bin  (5 + 25,581 = 25,586 bytes)
    4         bytes  magic "Z8NN"
    1         uint8  version (1)
    32 * 784  int8   W1q, row by row (all weights of neuron 0 first)
    32        int32  b1q
    1         int32  M0
    1         uint8  shift
    10 * 32   int8   W2q, row by row
    10        int32  b2q

testvec.bin  (7 + n_cases * 985 bytes)
    4         bytes  magic "Z8TV"
    1         uint8  version (1)
    1         uint16 number of cases
    for each case (985 bytes):
        784   uint8  image, 0 or 1, row by row (same order as nn.Flatten)
        32    int32  acc1, BEFORE the ReLU
        32    uint8  h_q, between 0 and 127
        10    int32  acc2
        1     uint8  expected digit: the one from infer_int, NOT the MNIST label
"""

import os
import numpy as np

from reference import QUANT_PATH, load_quantized, load_test_images, infer_int

# --- Paths ----------------------------------------------------------------
OUT_DIR = os.path.dirname(QUANT_PATH)
WEIGHTS_BIN = os.path.join(OUT_DIR, "weights.bin")
TESTVEC_BIN = os.path.join(OUT_DIR, "testvec.bin")

# --- Format ---------------------------------------------------------------
WEIGHTS_MAGIC = b"Z8NN"
TESTVEC_MAGIC = b"Z8TV"
VERSION = 1

N_IN, N_HID, N_OUT = 784, 32, 10
N_REAL = 100                      # real test-set images to export

INT32_MIN, INT32_MAX = -2**31, 2**31 - 1

WEIGHTS_SIZE = (4 + 1                      # magic + version
                + N_HID * N_IN             # W1q
                + N_HID * 4                # b1q
                + 4 + 1                    # M0 + shift
                + N_OUT * N_HID            # W2q
                + N_OUT * 4)               # b2q
TESTVEC_HEADER = 4 + 1 + 2                 # magic + version + number of cases
RECORD_SIZE = N_IN + N_HID * 4 + N_HID + N_OUT * 4 + 1   # 985


# --- Range-checked conversions ----------------------------------------------
# astype() does not raise on out-of-range values, it silently wraps them.
# So the range is checked before converting.

def _check_range(name, arr, lo, hi):
    arr = np.asarray(arr)
    if arr.min() < lo or arr.max() > hi:
        raise ValueError(f"{name} out of range [{lo}, {hi}]: "
                         f"min {arr.min()}, max {arr.max()}")
    return arr


def int8_bytes(name, arr):
    return _check_range(name, arr, -127, 127).astype(np.int8).tobytes()


def int32_bytes(name, arr):
    return _check_range(name, arr, INT32_MIN, INT32_MAX).astype("<i4").tobytes()


def uint8_bytes(name, arr, lo=0, hi=255):
    return _check_range(name, arr, lo, hi).astype(np.uint8).tobytes()


# --- Weights ----------------------------------------------------------------

def write_weights(q, path):
    assert q["W1q"].shape == (N_HID, N_IN), q["W1q"].shape
    assert q["W2q"].shape == (N_OUT, N_HID), q["W2q"].shape

    parts = [
        WEIGHTS_MAGIC,
        bytes([VERSION]),
        int8_bytes("W1q", q["W1q"]),
        int32_bytes("b1q", q["b1q"]),
        int32_bytes("M0", [q["M0"]]),
        uint8_bytes("shift", [q["shift"]], 1, 31),
        int8_bytes("W2q", q["W2q"]),
        int32_bytes("b2q", q["b2q"]),
    ]
    with open(path, "wb") as f:
        for part in parts:
            f.write(part)


def read_weights(path):
    """Read weights.bin back, as the C code sees it. Only used for checks."""
    with open(path, "rb") as f:
        data = f.read()

    if len(data) != WEIGHTS_SIZE:
        raise ValueError(f"weights.bin: expected {WEIGHTS_SIZE} bytes, found {len(data)}")
    if data[:4] != WEIGHTS_MAGIC or data[4] != VERSION:
        raise ValueError("weights.bin: wrong magic or version")

    pos = 5

    def take(dtype, count):
        nonlocal pos
        arr = np.frombuffer(data, dtype=dtype, count=count, offset=pos)
        pos += arr.nbytes
        return arr.astype(np.int64)

    w = {
        "W1q": take(np.int8, N_HID * N_IN).reshape(N_HID, N_IN),
        "b1q": take("<i4", N_HID),
        "M0": int(take("<i4", 1)[0]),
        "shift": int(take(np.uint8, 1)[0]),
        "W2q": take(np.int8, N_OUT * N_HID).reshape(N_OUT, N_HID),
        "b2q": take("<i4", N_OUT),
    }
    assert pos == len(data), "weights.bin: leftover bytes at the end of the file"
    return w


# --- Test cases -------------------------------------------------------------

def build_test_images(images, q):
    """N_REAL test images plus a few edge cases, as (name, image) pairs."""
    cases = [(f"mnist_{i}", images[i]) for i in range(N_REAL)]

    zeros = np.zeros(N_IN, dtype=np.int64)
    first_pixel = zeros.copy()
    first_pixel[0] = 1
    last_pixel = zeros.copy()
    last_pixel[-1] = 1

    # "Worst" image: lights the pixels with a positive weight for the neuron
    # with the largest possible acc1, so that neuron reaches bound.
    max_acc_per_neuron = np.maximum(q["W1q"], 0).sum(axis=1) + q["b1q"]
    worst_neuron = int(np.argmax(max_acc_per_neuron))
    worst = (q["W1q"][worst_neuron] > 0).astype(np.int64)

    cases += [
        ("all_black", zeros),
        ("all_white", np.ones(N_IN, dtype=np.int64)),
        ("only_pixel_0", first_pixel),
        ("only_pixel_783", last_pixel),
        (f"worst_neuron_{worst_neuron}", worst),
    ]

    # Check that it really does
    _, acc1, _, _ = infer_int(worst, q)
    assert acc1[worst_neuron] == q["bound"], (acc1[worst_neuron], q["bound"])

    return cases


def write_test_vectors(cases, q, path):
    if len(cases) > 65535:
        raise ValueError("Too many cases for a uint16")

    with open(path, "wb") as f:
        f.write(TESTVEC_MAGIC)
        f.write(bytes([VERSION]))
        f.write(np.array([len(cases)], dtype="<u2").tobytes())

        for name, x in cases:
            digit, acc1, h_q, acc2 = infer_int(x, q)
            f.write(uint8_bytes(f"{name}: image", x, 0, 1))
            f.write(int32_bytes(f"{name}: acc1", acc1))
            f.write(uint8_bytes(f"{name}: h_q", h_q, 0, 127))
            f.write(int32_bytes(f"{name}: acc2", acc2))
            f.write(uint8_bytes(f"{name}: digit", [digit], 0, 9))


def read_test_vectors(path):
    """Read testvec.bin back as a list of dicts. Also used by the tests."""
    with open(path, "rb") as f:
        data = f.read()

    if data[:4] != TESTVEC_MAGIC or data[4] != VERSION:
        raise ValueError("testvec.bin: wrong magic or version")
    n = int(np.frombuffer(data, dtype="<u2", count=1, offset=5)[0])
    if len(data) != TESTVEC_HEADER + n * RECORD_SIZE:
        raise ValueError(f"testvec.bin: size {len(data)} does not match {n} cases")

    records = []
    pos = TESTVEC_HEADER
    for _ in range(n):
        def take(dtype, count):
            nonlocal pos
            arr = np.frombuffer(data, dtype=dtype, count=count, offset=pos)
            pos += arr.nbytes
            return arr.astype(np.int64)

        records.append({
            "x": take(np.uint8, N_IN),
            "acc1": take("<i4", N_HID),
            "h_q": take(np.uint8, N_HID),
            "acc2": take("<i4", N_OUT),
            "digit": int(take(np.uint8, 1)[0]),
        })
    return records


# --- Main -------------------------------------------------------------------

def main():
    q = load_quantized(QUANT_PATH)
    images, _ = load_test_images()

    # 1. Weights
    write_weights(q, WEIGHTS_BIN)
    assert os.path.getsize(WEIGHTS_BIN) == WEIGHTS_SIZE
    w = read_weights(WEIGHTS_BIN)
    for name in ("W1q", "b1q", "W2q", "b2q"):
        assert np.array_equal(w[name], q[name]), f"{name} differs after reading it back"
    assert w["M0"] == q["M0"] and w["shift"] == q["shift"]

    # 2. Test vectors
    cases = build_test_images(images, q)
    write_test_vectors(cases, q, TESTVEC_BIN)
    records = read_test_vectors(TESTVEC_BIN)
    assert len(records) == len(cases)
    for (name, x), rec in zip(cases, records):
        digit, acc1, h_q, acc2 = infer_int(x, q)
        assert np.array_equal(rec["x"], x), name
        assert np.array_equal(rec["acc1"], acc1), name
        assert np.array_equal(rec["h_q"], h_q), name
        assert np.array_equal(rec["acc2"], acc2), name
        assert rec["digit"] == digit, name

    # 3. Summary
    print(f"Weights: {WEIGHTS_BIN}  ({os.path.getsize(WEIGHTS_BIN)} bytes)")
    print(f"Tests:   {TESTVEC_BIN}  ({os.path.getsize(TESTVEC_BIN)} bytes, {len(cases)} cases)")
    print("Special cases:", ", ".join(name for name, _ in cases[N_REAL:]))
    print("Both files read back correctly: OK")


if __name__ == "__main__":
    main()