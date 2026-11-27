# MNIST on Z80. Run from the repository root.
#
#   make                  missing artifacts, DeZog inputs, the Z80 programs (firmware, emu, rom)
#   make train            retrain from scratch, then quantize and export again
#   make check            integer reference vs float model (accuracies)
#   make test             builds emu, rom (and ticks-viz if z88dk's sources are there), then pytest
#   make firmware         build/firmware_*  (z80_nn_main.c, launch config "Z80 zsim")
#   make firmware FW=firmware/z80_test_main.c    toolchain sanity check instead
#   make emu              build/emu.bin     (z80_serial_main.c for z88dk-ticks)
#   make rom              build/board.rom   (z80_serial_main.c for the board's EEPROM)
#   make demo             drawing app classifying on the emulator
#   make demo PORT=/dev/tty.usbserial-XXXX      same, on the board (DEBUG=1: debug panel)
#   make inputs CASE=n    build/weights_payload.bin + build/case.bin for DeZog (default 0)
#   make ticks-viz        patched z88dk-ticks for the live debug view (build/ticks-viz)
#   make clean            remove build output, build/ticks-viz included (artifacts/ are kept)

# The project's virtual environment when there is one
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
ZCC    ?= zcc
CASE   ?= 0
FW     ?= firmware/z80_nn_main.c
ZFLAGS  = +z80 -compiler=sdcc -m --list --c-code-in-asm -pragma-define:CRT_MODEL=0

ART = artifacts

# Patched z88dk-ticks for the live debug view (target ticks-viz below)
TICKS_SRC ?= $(abspath $(dir $(shell command -v z88dk-ticks))../src)
VIZ       ?= build/ticks-viz
VIZ_TICKS  = $(VIZ)/src/ticks/z88dk-ticks

.PHONY: all train check test firmware emu rom demo inputs ticks-viz clean

all: inputs firmware emu rom

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

# Builds what the tests run, so a green run does not hide skipped tests.
# -rs lists every skipped test with its reason.
test: emu rom $(ART)/weights.bin $(ART)/testvec.bin $(if $(wildcard $(TICKS_SRC)/ticks),ticks-viz)
	$(PYTHON) -m pytest -rs tests/

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

# +test: console I/O is z88dk-ticks' stdin/stdout. No --list, so the DeZog
# listings (firmware/*.lis) are left alone.
emu: | build
	$(ZCC) +test -compiler=sdcc -m -o build/emu.bin firmware/z80_serial_main.c firmware/nn.c firmware/snapregs.c

# Board: +z80's default ROM layout (code from 0x0000, RAM from 0x8000).
# CRT_ENABLE_EIDI=0x01: interrupts stay off, the firmware polls the ACIA.
# Write build/board.rom to the EEPROM.
rom: | build
	$(ZCC) +z80 -compiler=sdcc -m -pragma-define:CRT_ENABLE_EIDI=0x01 -create-app \
	    -o build/board.bin firmware/z80_serial_main.c firmware/nn.c firmware/snapregs.c

# --- Patched z88dk-ticks for the live debug view -----------------------------
# Copies z88dk's ticks source into build/, applies tools/ticks-viz.patch
# (option -shm: memory, registers and accesses shared with host/z80debug.py)
# and builds it there. The installed z88dk is not touched.

ticks-viz: $(VIZ_TICKS)

$(VIZ_TICKS): tools/ticks-viz.patch
	rm -rf $(VIZ)
	mkdir -p $(VIZ)/src $(VIZ)/ext
	cp -R $(TICKS_SRC)/ticks $(TICKS_SRC)/common $(TICKS_SRC)/Make.common $(VIZ)/src/
	cp -R $(TICKS_SRC)/../ext/uthash $(VIZ)/ext/
	find $(VIZ) \( -name '*.o' -o -name '*.d' \) -delete
	patch -d $(VIZ) -p1 < tools/ticks-viz.patch
	$(MAKE) -C $(VIZ)/src/ticks z88dk-ticks

# --- Demo ---------------------------------------------------------------------

ifdef PORT
demo: $(ART)/weights.bin
	$(PYTHON) host/demo.py board $(PORT) $(if $(DEBUG),--debug)
else
demo: emu $(ART)/weights.bin
	$(PYTHON) host/demo.py
endif

clean:
	rm -rf build firmware/*.lis
