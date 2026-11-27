'''
Host-side client for the Z80 inference firmware: sends weights, images and
debug commands over the line protocol to a native build, the emulator or the board.
'''


import os
import pty
import subprocess
import tty

import numpy as np

# Order of the G reply = struct regs in firmware/snapregs.h
REG_NAMES = ["af", "bc", "de", "hl", "ix", "iy", "sp", "pc", "af2", "bc2", "de2", "hl2", "i", "r"]
MAX_CHUNK = 64      # bytes per W or M line

class Z80NN:
    def __init__(self, target, where, ticks="z88dk-ticks", shm=None):
        # target: "local" (native build on the host), "emu" (emulator), "board" (the board)
        # emu only: ticks = the emulator to run; shm = file for the live view, which
        # needs the patched ticks (make ticks-viz, see host/z80debug.py LiveSource)
        self.target = target

        if target == "local":
            self.proc = subprocess.Popen([where], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

        elif target == "emu":
            # where = build/emu.bin (make emu) or build/board.rom (make rom, ACIA emulated by ticks).
            # -counter: ticks quits after 1e8 T-states by default, about 10 inferences
            # stdout is a pseudo-terminal, not a pipe: ticks' emulated ACIA never flushes,
            # and only on a terminal does the C library flush at every '\n'.
            master, slave = pty.openpty()
            tty.setraw(slave)       # no '\n' -> '\r\n' translation, no echo
            live = ["-shm", shm] if shm else []
            self.proc = subprocess.Popen([ticks, "-counter", "100000000000", *live, where],
                                         stdin=subprocess.PIPE, stdout=slave, text=True)
            os.close(slave)
            self.out = os.fdopen(master, "r", encoding="ascii", newline="")

        elif target == "board":
            import serial
            self.port = serial.Serial(where, 115200,rtscts=True, timeout=5)


        else:
            raise ValueError("target must be 'local', 'emu' or 'board'")

    # Helper functions: write and read line
    def _write_line(self, text):
        if self.target == "board":
            self.port.write((text + "\n").encode("ascii"))
        else:
            self.proc.stdin.write(text + "\n")
            self.proc.stdin.flush()

    def _read_line(self):
        if self.target == "board":
            line = self.port.readline().decode("ascii")
        elif self.target == "emu":
            try:
                line = self.out.readline()
            except OSError:         # the pty reports EIO once ticks has exited
                line = ""
        else:
            line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("no answer from the Z80")

        return line.strip()


    # Send a command
    def send(self, text):
        self._write_line(text)
        answer = self._read_line()

        if answer.startswith("ERR"):
            raise RuntimeError(f"the Z80 answered {answer!r}")
        return answer

    def version(self):
        return self.send("V")

    # Sending the weights as chunks of 64 bytes each

    def load_weights(self, path):
        with open(path, "rb") as f:
            data = f.read()
        if data[:4] != b"Z8NN" or data[4] != 1:
            raise ValueError("this is not a weights.bin file")

        payload = data[5:] # skip the 5-byte header

        for offset in range(0, len(payload), 64): # 0, 64, 128, ...
            chunk = payload[offset: offset + 64] 
            self.send(f"W {offset:04X} {chunk.hex().upper()}")


        # checksum
        expected = sum(payload) % 65536
        answer = self.send("C") # e.g. "C 3A7F": 16-bit sum of the weight bytes received
        got = int(answer.split()[1], 16)

        if got != expected:
            raise RuntimeError(f"checksum mismatch: expected {expected:04X}, got {got:04X}")


    def classify(self, x):
        answer = self.send("I " + pack_image(x))
        return int(answer.split()[1])

    def classify_debug(self, x):
        self._write_line("D " + pack_image(x))
        acc1 = self._read_numbers("A1", 32)
        h_q = self._read_numbers("H", 32)
        acc2 = self._read_numbers("A2", 10)
        digit = self._read_numbers("R", 1)[0]
        return digit, acc1, h_q, acc2

    # Debug: register snapshot and memory reads (G and M commands)

    def registers(self):
        """Registers snapped right after the last I/D, as {name: int} (REG_NAMES)."""
        parts = self.send("G").split()
        if parts[0] != "G" or len(parts) != len(REG_NAMES) + 1:
            raise RuntimeError(f"unexpected answer: {' '.join(parts)!r}")
        return {name: int(v, 16) for name, v in zip(REG_NAMES, parts[1:])}

    def read_memory(self, addr, n):
        """n bytes of Z80 memory from addr (wraps at 0xFFFF), in 64-byte M commands."""
        data = bytearray()
        while len(data) < n:
            k = min(MAX_CHUNK, n - len(data))
            a = (addr + len(data)) & 0xFFFF
            parts = self.send(f"M {a:04X} {k:02X}").split()
            if parts[0] != "M" or len(parts) != 2 or len(parts[1]) != 2 * k:
                raise RuntimeError(f"unexpected answer: {' '.join(parts)!r}")
            data += bytes.fromhex(parts[1])
        return bytes(data)

    def _read_numbers(self, label, count):
        # reads a line like "A1 323 420 -193 ..." and returns the numbers
        line = self._read_line()
        parts = line.split()
        if parts[0] != label or len(parts) != count + 1:
            raise RuntimeError(f"unexpected answer: {line!r}")
        return [int(p) for p in parts[1:]]


    def close(self):
        if self.target == "board":
            self.port.close()

        else:
            self.proc.stdin.close()     # emu.bin exits at end of input...
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()        # ...board.rom polls the ACIA forever, like the board
                self.proc.wait()
            if self.target == "emu":
                self.out.close()
   

# 784 binary pixels -> 98 bytes (MSB first) -> uppercase hex

def pack_image(x):
    bits = np.asarray(x, dtype=np.uint8).reshape(784)
    return np.packbits(bits).tobytes().hex().upper()




    



