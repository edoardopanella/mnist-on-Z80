# MNIST on Z80. Run from the repository root.
#
#   make                  missing artifacts, DeZog inputs, both Z80 programs
#   make train            retrain from scratch, then quantize and export again
#   make check            integer reference vs float model (accuracies)
#   make test             pytest
#   make firmware         build/firmware_*  (z80_nn_main.c, launch config "Z80 zsim")
#   make firmware FW=firmware/z80_test_main.c    toolchain sanity check instead
#   make draw             build/draw_*      (z80_draw_main.c, launch config "Z80 draw UI")
#   make emu              build/emu.bin     (z80_serial_main.c for z88dk-ticks)
#   make demo             drawing app classifying on the emulator
#   make inputs CASE=n    build/weights_payload.bin + build/case.bin for DeZog (default 0)
#   make clean            remove build output (artifacts/ are kept)

PYTHON ?= python
ZCC    ?= zcc
CASE   ?= 0
FW     ?= firmware/z80_nn_main.c
ZFLAGS  = +z80 -compiler=sdcc -m --list --c-code-in-asm -pragma-define:CRT_MODEL=0

ART = artifacts

.PHONY: all train check test firmware draw emu demo inputs clean

all: inputs firmware draw emu

# --- Python pipeline ----------------------------------------------------------
# Artifacts depend on each other only, not on the scripts, so editing a script
# never starts a 10-epoch retrain by surprise. "make train" forces it.

$(ART)/mlp.pt:
	$(PYTHON) training/train.py

$(ART)/quantized.npz: $(ART)/mlp.pt
	$(PYTHON) training/quantize.py

# Pattern rule with two targets: one run of export.py writes both files.
%/weights.bin %/testvec.bin: %/quantized.npz
	$(PYTHON) training/export.py

train:
	$(PYTHON) training/train.py
	$(PYTHON) training/quantize.py
	$(PYTHON) training/export.py

check: $(ART)/quantized.npz
	$(PYTHON) training/reference.py

test:
	$(PYTHON) -m pytest tests/

# --- DeZog --------------------------------------------------------------------

inputs: $(ART)/weights.bin $(ART)/testvec.bin
	$(PYTHON) tests/make_dezog_inputs.py $(CASE)

# --- Z80 (z88dk) --------------------------------------------------------------
# Always rebuilt: FW can switch the source file, and a build takes seconds.
# Output names (NAME_CODE.bin, NAME_DATA.bin, NAME.map) match .vscode/launch.json.

build:
	mkdir -p build

firmware: | build
	$(ZCC) $(ZFLAGS) -o build/firmware.bin $(FW) firmware/nn.c

draw: | build
	$(ZCC) $(ZFLAGS) -o build/draw.bin firmware/z80_draw_main.c firmware/nn.c

# +test: console I/O is z88dk-ticks' stdin/stdout. No --list, so the DeZog
# listings (firmware/*.lis) are left alone.
emu: | build
	$(ZCC) +test -compiler=sdcc -m -o build/emu.bin firmware/z80_serial_main.c firmware/nn.c

demo: emu $(ART)/weights.bin
	$(PYTHON) host/demo.py

clean:
	rm -rf build firmware/*.lis
