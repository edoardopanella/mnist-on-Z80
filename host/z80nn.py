'''
Libreria di collegamento tra input mac e z80. Serve a fare da tramite tra l'input che diamo noi dal mac 
e la scheda z80
'''


import subprocess # È un modulo già incluso in Python (non va installato) che serve a lanciare altri programmi dall’interno di uno script Python. È come se Python aprisse un terminale, scrivesse il nome di un programma e premesse Invio al posto vostro.
import numpy as np

class Z80NN:
    def __init__(self, target, where):
        # target: "local" (programma sul mac), "emu" (emulatore), "board" (la scheda)
        self.target = target

        if target == "local":
            # lancia il programma, come se lo scrivessimo sul terminale
            self.proc = subprocess.Popen([where], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) # avvia un programma e lo lascia acceso, POPEN sta per PROCESS OPEN

        elif target == "emu":
            # uguale a prima ma il programma che lancia è l'emulator
            # where = build/emu.bin (make emu, +test target, starts at 0x0000).
            # -counter: ticks quits after 1e8 T-states by default, about 10 inferences
            self.proc = subprocess.Popen(["z88dk-ticks", "-counter", "100000000000", where], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

        elif target == "board":
            import serial # usa porte seriali, ovvero USB
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
        return self.send("V") # manda al target il commando V e ritorna la risposta

    # Sending the weights as chunks of 64 bytes each

    def load_weights(self, path):
        with open(path, "rb") as f:
            data = f.read()
        if data[:4] != b"Z8NN" or data[4] != 1:
            raise ValueError("this is not a weights.bin file")

        payload = data[5:] # everything but the first 5 bytes of intestazione

        for offset in range(0, len(payload), 64): # 0, 64, 128, ...
            chunk = payload[offset: offset + 64] 
            self.send(f"W {offset:04X} {chunk.hex().upper()}")


        # checksum
        expected = sum(payload) % 65536
        answer = self.send("C") # risposta tipo "C 3A7F", è l'istruzione checksum, ovvero somma tutti i byte dei pesi che hai ricevuto
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

    def _read_numbers(self, label, count):
        # legge una riga tipo "A1 323 420 -193 ..." e restituisce i numeri
        line = self._read_line()
        parts = line.split()
        if parts[0] != label or len(parts) != count + 1:
            raise RuntimeError(f"unexpected answer: {line!r}")
        return [int(p) for p in parts[1:]]


    def close(self):
        if self.target == "board":
            self.port.close()

        else:
            self.proc.stdin.close()
            self.proc.wait()
   

# utility per trasformare un immagine in bytes invece che bits

def pack_image(x):
    bits = np.asarray(x, dtype=np.uint8).reshape(784)
    return np.packbits(bits).tobytes().hex().upper() # in esadecimale




    



