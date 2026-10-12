"""
export.py - Esporta pesi quantizzati e vettori di test in file binari per il C.

Tutti i numeri a piu' byte sono little-endian (il byte meno significativo per
primo), come sullo Z80. Questo formato va tenuto identico in docs/spec.md.

weights.bin  (5 + 25.581 = 25.586 byte)
    4         byte   firma "Z8NN"
    1         uint8  versione (1)
    32 * 784  int8   W1q, riga per riga (prima tutti i pesi del neurone 0)
    32        int32  b1q
    1         int32  M0
    1         uint8  shift
    10 * 32   int8   W2q, riga per riga
    10        int32  b2q

testvec.bin  (7 + numero_casi * 985 byte)
    4         byte   firma "Z8TV"
    1         uint8  versione (1)
    1         uint16 numero di casi
    per ogni caso (985 byte):
        784   uint8  immagine, 0 o 1, riga per riga (come nn.Flatten)
        32    int32  acc1, PRIMA della ReLU
        32    uint8  h_q, tra 0 e 127
        10    int32  acc2
        1     uint8  cifra attesa: quella di infer_int, NON l'etichetta MNIST
"""

import os
import numpy as np

from reference import QUANT_PATH, load_quantized, load_test_images, infer_int

# --- Percorsi -------------------------------------------------------------
OUT_DIR = os.path.dirname(QUANT_PATH)
WEIGHTS_BIN = os.path.join(OUT_DIR, "weights.bin")
TESTVEC_BIN = os.path.join(OUT_DIR, "testvec.bin")

# --- Formato --------------------------------------------------------------
WEIGHTS_MAGIC = b"Z8NN"
TESTVEC_MAGIC = b"Z8TV"
VERSION = 1

N_IN, N_HID, N_OUT = 784, 32, 10
N_REAL = 100                      # immagini vere del test set da esportare

INT32_MIN, INT32_MAX = -2**31, 2**31 - 1

WEIGHTS_SIZE = (4 + 1                      # firma + versione
                + N_HID * N_IN             # W1q
                + N_HID * 4                # b1q
                + 4 + 1                    # M0 + shift
                + N_OUT * N_HID            # W2q
                + N_OUT * 4)               # b2q
TESTVEC_HEADER = 4 + 1 + 2                 # firma + versione + numero casi
RECORD_SIZE = N_IN + N_HID * 4 + N_HID + N_OUT * 4 + 1   # 985


# --- Conversioni con controllo dell'intervallo -------------------------------
# astype() NON segnala errori: un valore fuori intervallo viene "avvolto" e
# diventa un numero sbagliato in silenzio. Per questo si controlla PRIMA.

def _check_range(name, arr, lo, hi):
    arr = np.asarray(arr)
    if arr.min() < lo or arr.max() > hi:
        raise ValueError(f"{name} fuori intervallo [{lo}, {hi}]: "
                         f"min {arr.min()}, max {arr.max()}")
    return arr


def int8_bytes(name, arr):
    return _check_range(name, arr, -127, 127).astype(np.int8).tobytes()


def int32_bytes(name, arr):
    return _check_range(name, arr, INT32_MIN, INT32_MAX).astype("<i4").tobytes()


def uint8_bytes(name, arr, lo=0, hi=255):
    return _check_range(name, arr, lo, hi).astype(np.uint8).tobytes()


# --- Pesi -----------------------------------------------------------------

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
    """Rilegge weights.bin come farebbe il C. Serve solo per i controlli."""
    with open(path, "rb") as f:
        data = f.read()

    if len(data) != WEIGHTS_SIZE:
        raise ValueError(f"weights.bin: attesi {WEIGHTS_SIZE} byte, trovati {len(data)}")
    if data[:4] != WEIGHTS_MAGIC or data[4] != VERSION:
        raise ValueError("weights.bin: firma o versione sbagliate")

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
    assert pos == len(data), "weights.bin: byte avanzati in fondo al file"
    return w


# --- Casi di test ---------------------------------------------------------

def build_test_images(images, q):
    """Restituisce una lista di (nome, immagine)."""
    cases = [(f"mnist_{i}", images[i]) for i in range(N_REAL)]

    zeros = np.zeros(N_IN, dtype=np.int64)
    first_pixel = zeros.copy()
    first_pixel[0] = 1
    last_pixel = zeros.copy()
    last_pixel[-1] = 1

    # Immagine "peggiore": accende esattamente i pixel con peso positivo del
    # neurone che ha il massimo acc1 possibile. Porta quel neurone a bound.
    max_acc_per_neuron = np.maximum(q["W1q"], 0).sum(axis=1) + q["b1q"]
    worst_neuron = int(np.argmax(max_acc_per_neuron))
    worst = (q["W1q"][worst_neuron] > 0).astype(np.int64)

    cases += [
        ("tutta_nera", zeros),
        ("tutta_bianca", np.ones(N_IN, dtype=np.int64)),
        ("solo_pixel_0", first_pixel),
        ("solo_pixel_783", last_pixel),
        (f"peggiore_neurone_{worst_neuron}", worst),
    ]

    # L'immagine peggiore deve davvero raggiungere bound
    _, acc1, _, _ = infer_int(worst, q)
    assert acc1[worst_neuron] == q["bound"], (acc1[worst_neuron], q["bound"])

    return cases


def write_test_vectors(cases, q, path):
    if len(cases) > 65535:
        raise ValueError("Troppi casi per un uint16")

    with open(path, "wb") as f:
        f.write(TESTVEC_MAGIC)
        f.write(bytes([VERSION]))
        f.write(np.array([len(cases)], dtype="<u2").tobytes())

        for name, x in cases:
            digit, acc1, h_q, acc2 = infer_int(x, q)
            f.write(uint8_bytes(f"{name}: immagine", x, 0, 1))
            f.write(int32_bytes(f"{name}: acc1", acc1))
            f.write(uint8_bytes(f"{name}: h_q", h_q, 0, 127))
            f.write(int32_bytes(f"{name}: acc2", acc2))
            f.write(uint8_bytes(f"{name}: cifra", [digit], 0, 9))


def read_test_vectors(path):
    """Rilegge testvec.bin come farebbe il C. Serve solo per i controlli."""
    with open(path, "rb") as f:
        data = f.read()

    if data[:4] != TESTVEC_MAGIC or data[4] != VERSION:
        raise ValueError("testvec.bin: firma o versione sbagliate")
    n = int(np.frombuffer(data, dtype="<u2", count=1, offset=5)[0])
    if len(data) != TESTVEC_HEADER + n * RECORD_SIZE:
        raise ValueError(f"testvec.bin: dimensione {len(data)} non coerente con {n} casi")

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


# --- Main -----------------------------------------------------------------

def main():
    q = load_quantized(QUANT_PATH)
    images, _ = load_test_images()

    # 1. Pesi
    write_weights(q, WEIGHTS_BIN)
    assert os.path.getsize(WEIGHTS_BIN) == WEIGHTS_SIZE
    w = read_weights(WEIGHTS_BIN)
    for name in ("W1q", "b1q", "W2q", "b2q"):
        assert np.array_equal(w[name], q[name]), f"{name} riletto diverso"
    assert w["M0"] == q["M0"] and w["shift"] == q["shift"]

    # 2. Vettori di test
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

    # 3. Riepilogo
    print(f"Pesi:  {WEIGHTS_BIN}  ({os.path.getsize(WEIGHTS_BIN)} byte)")
    print(f"Test:  {TESTVEC_BIN}  ({os.path.getsize(TESTVEC_BIN)} byte, {len(cases)} casi)")
    print("Casi speciali:", ", ".join(name for name, _ in cases[N_REAL:]))
    print("Rilettura di entrambi i file: OK")


if __name__ == "__main__":
    main()