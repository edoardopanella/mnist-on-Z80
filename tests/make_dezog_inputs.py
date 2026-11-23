"""Cut artifacts/weights.bin and artifacts/testvec.bin into raw files DeZog loads.

    python tests/make_dezog_inputs.py [case_index]     (default 0)

Writes
    build/weights_payload.bin   weights.bin without the 5-byte header -> loaded at 0x1000
    build/case.bin              one 985-byte testvec record          -> loaded at 0x7800
"""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
OUT = ROOT / "build"
REC = 985  # x (784) + acc1 (128) + h_q (32) + acc2 (40) + digit (1), spec §5.2

w = (ART / "weights.bin").read_bytes()
assert w[:4] == b"Z8NN" and w[4] == 1 and len(w) == 25586, "unexpected weights.bin"

t = (ART / "testvec.bin").read_bytes()
assert t[:4] == b"Z8TV" and t[4] == 1, "unexpected testvec.bin"
(n,) = struct.unpack_from("<H", t, 5)
assert len(t) == 7 + n * REC, "testvec.bin size does not match its header"

k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
assert 0 <= k < n, f"case index must be 0..{n - 1}"
rec = t[7 + k * REC : 7 + (k + 1) * REC]

OUT.mkdir(exist_ok=True)
(OUT / "weights_payload.bin").write_bytes(w[5:])
(OUT / "case.bin").write_bytes(rec)
print(f"case {k}/{n - 1}: expected digit {rec[-1]}, lit pixels {sum(rec[:784])}")
