# Z80 Development Toolchain on macOS: Installation Guide

This guide installs a complete Z80 development environment on macOS (Apple silicon or Intel): z88dk for C and assembly, sjasmplus as a standalone assembler, and DeZog in VS Code as a debugger with a built-in Z80 simulator. Version-specific details (option names, configuration keys, download file names) change over time; the linked project documentation takes precedence over this guide.

## What each tool does

| Tool | Role | Input | Output |
| --- | --- | --- | --- |
| [z88dk](https://github.com/z88dk/z88dk) | C compiler, assembler, linker, Z80 libraries | `.c`, `.asm` | Z80 binaries |
| [sjasmplus](https://github.com/z00m128/sjasmplus) | Assembler with macros, listings, debug maps | `.asm` | binary, listing, SLD map |
| [VS Code](https://code.visualstudio.com) + [DeZog](https://github.com/maziac/DeZog) | Debugger front end with the zsim simulator | binary + map | stepping, registers, memory |

Each tool is independent: z88dk and sjasmplus both produce Z80 machine code, and DeZog only consumes the files they write.

## Before you start

- **Use paths without spaces.** Several build scripts and tools mishandle spaces. Install into `~/z88dk` and `~/tools/sjasmplus` rather than a folder such as `Documents/My Projects/`.
- **Know your shell.** Current macOS uses zsh, configured in `~/.zshrc`. If your shell is bash, use `~/.bash_profile` instead. Check with `echo $SHELL`.
- **Know your architecture.** `uname -m` prints `arm64` (Apple silicon) or `x86_64` (Intel). Apple silicon can also run x86_64 binaries through Rosetta 2, but native builds are preferable.
- **Install the command-line tools** if you plan to build anything from source: `xcode-select --install` (provides the compiler, make and git).
- **[Homebrew](https://brew.sh)** is optional but simplifies several steps. On Apple silicon it installs to `/opt/homebrew`, which compilers do not search by default (see the source-build section).

## Concepts used throughout

- **PATH** is the list of directories the shell searches when you type a command name. "Command not found" means no PATH directory contains that file; the file may still exist elsewhere on disk.
- **Environment variables** such as `ZCCCFG` are named strings inherited by every program the shell launches. They are read when a shell starts, so after editing `~/.zshrc` run `source ~/.zshrc` or open a new terminal.
- **Quarantine.** macOS Gatekeeper tags downloaded files and may block unsigned executables on first use. Clear the tag only for software you trust, from the official project page.
- **Host and target.** The tools run on your Mac (host) and emit code for the Z80 (target). A tool built for the wrong host architecture will not run natively.

## Step 1: Install z88dk

### Option A: prebuilt archive (recommended)

1. Download the macOS archive from the z88dk project ([releases page](https://github.com/z88dk/z88dk/releases) or the [nightly builds](http://nightly.z88dk.org) linked from the [README](https://github.com/z88dk/z88dk#readme)) and unpack it.
2. Move the folder to `~/z88dk`. Check that both `~/z88dk/bin` and `~/z88dk/lib/config` exist. If they sit one level deeper, move the inner folder up.
3. Add these lines to your shell configuration file:

   ```
   export ZCCCFG="$HOME/z88dk/lib/config"
   export PATH="$PATH:$HOME/z88dk/bin"
   ```

4. Clear the quarantine flag:

   ```
   xattr -dr com.apple.quarantine ~/z88dk
   ```

5. Reload the shell configuration (`source ~/.zshrc`) and verify:
   - `which zcc` prints a path inside `~/z88dk/bin`
   - `zcc` with no arguments prints usage
   - `file ~/z88dk/bin/zcc` shows the architecture (arm64 or x86_64)

Common mistakes: pointing `ZCCCFG` at `lib` instead of `lib/config`, and pointing `PATH` at the root folder instead of `bin`.

### Option B: Homebrew

If a z88dk formula is available (`brew search z88dk`), `brew install z88dk` places the binaries on PATH. The formula's post-install message states whether `ZCCCFG` still needs to be set.

### Option C: build from source

Needed only for a newer version than the releases provide. The build compiles z88dk's own host programs first, then uses them to compile the Z80 libraries, so `bin` and `lib/config` exist only after it finishes.

1. Install prerequisites with Homebrew. The z88dk README lists the current set; at least `gmp`, `libxml2` and `pkg-config`, and typically bison, flex, m4, re2c, ragel, texinfo and dos2unix:

   ```
   brew install gmp libxml2 pkg-config
   ```

2. **Apple silicon only:** Homebrew packages are not on the compiler's default search path. Run `brew shellenv` once as the README directs, and add to your shell configuration:

   ```
   export CPATH="$HOMEBREW_PREFIX/include:$CPATH"
   export LIBRARY_PATH="$HOMEBREW_PREFIX/lib:$LIBRARY_PATH"
   ```

3. Clone with submodules (`--recursive` is required) or download the nightly tarball. If `wget` is missing, use `git clone` or `curl -LO <url>`; macOS includes `curl` and `tar`.

   ```
   git clone --recursive https://github.com/z88dk/z88dk.git
   ```

4. Follow the README's build steps from inside the repository, setting `PATH` and `ZCCCFG` to the repository's `bin` and `lib/config` for the build shell, then run `./build.sh`. Setting `BUILD_SDCC=1` also builds the `zsdcc` compiler, which generally produces better Z80 code than the older built-in one.
5. A failed build almost always means a missing prerequisite; install it with `brew install` and rerun.

## Step 2: Install sjasmplus

### Option A: prebuilt release

Download the macOS archive from the [Releases page](https://github.com/z00m128/sjasmplus/releases) of the sjasmplus repository, unpack it into `~/tools/sjasmplus`, then continue at "Put it on PATH".

### Option B: build from source

The repository contains source only; it holds no runnable program until built.

1. Clone or download the repository into a path without spaces.
2. Run `make` inside the folder. This produces an executable named `sjasmplus`.
3. Test it in place with `./sjasmplus --help`. The `./` is required because the shell does not search the current directory.

### Put it on PATH

1. Copy only the executable into a stable folder:

   ```
   mkdir -p ~/tools/sjasmplus
   cp sjasmplus ~/tools/sjasmplus/
   ```

2. Add the folder to your shell configuration and reload:

   ```
   export PATH="$PATH:$HOME/tools/sjasmplus"
   ```

3. Verify with `which sjasmplus` and `sjasmplus --help`. If a downloaded binary is blocked, run `chmod +x` on it and clear the quarantine flag as in Step 1.

PATH must name the directory that directly contains the executable, not a parent folder.

## Step 3: Install VS Code and DeZog

1. Install [VS Code](https://code.visualstudio.com).
2. In the Extensions panel, search for **DeZog** (author maziac) and install it, or use its [Marketplace page](https://marketplace.visualstudio.com/items?itemName=maziac.dezog).
3. Open your project folder with File > Open Folder (again, no spaces in the path).
4. Write a small assembly file, including a memory model directive (`DEVICE`), an origin (`ORG`), a final loop or `HALT`, and a snapshot save directive (`SAVESNA`). Assemble it with sjasmplus, requesting the raw output, the listing and the SLD (source-level-debug) file. Option names are in the sjasmplus documentation.
5. In Run and Debug, create a `launch.json` with a DeZog entry. The settings to supply, in words:
   - remote type: `zsim`, the built-in simulator
   - memory model: a preset such as ZX48K, used only as a simple 64 KB RAM layout (a custom ROM/RAM model can replace it later)
   - source mapping: the `.sld` file under the `sjasmplus` key (z88dk output uses the `.map` file under its own key)
   - the program to load, and the start address
   - `topOfStack`: a numeric address such as `0xFFF0`, or a real label present in the map
6. Set a breakpoint by clicking the gutter beside a line, press F5 to run to it, and press F11 to execute one instruction at a time.

Re-assemble before every debug session; a stale `.sld` causes breakpoints to be ignored. Automating the assemble step as a VS Code pre-launch task avoids this.

## How the pieces connect

1. Source (`.asm`) goes to sjasmplus (or z88dk).
2. The assembler writes a memory image, a listing (bytes beside source) and a map file (address to source line).
3. DeZog loads the image and the map into zsim.
4. zsim executes Z80 instructions; DeZog shows registers, flags, memory and the current source line.

## Verification checklist

| Check | Expected result |
| --- | --- |
| `which zcc` | path inside the z88dk `bin` folder |
| `echo $ZCCCFG` and `ls $ZCCCFG` | the config path, listing target files |
| `which sjasmplus` | path inside the sjasmplus folder |
| `sjasmplus --help` | usage text |
| DeZog on a test program | stops at a breakpoint; registers change when stepping |

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| command not found | executable not in a PATH directory, or shell configuration not reloaded |
| "cannot be opened because the developer cannot be verified" | quarantine flag; clear with `xattr -dr com.apple.quarantine <folder>` |
| `zcc` cannot find its configuration | `ZCCCFG` unset, or pointing to `lib` instead of `lib/config` |
| source build: header or library not found | Apple silicon Homebrew paths missing; set `CPATH` and `LIBRARY_PATH` |
| source build: tool not found or version error | missing prerequisite; install with `brew install` |
| build or scripts fail oddly | spaces in the path; move the folder |
| "cannot evaluate top of stack" in DeZog | `topOfStack` unresolved; use a numeric address or a label that exists in the map |
| breakpoints ignored | stale or wrong `.sld` path, or load address differs from `ORG` |
| program runs into empty memory | missing final jump-to-self or `HALT`; zero bytes execute as `NOP` |

## Caveats

- Configuration key names (DeZog) and command-line options (sjasmplus) vary by version; verify against the current documentation.
- zsim is a development simulator. Do not rely on its T-state counts for hardware timing claims without checking its documentation; use a cycle-accurate, pin-level model (for example the [floooh/chips](https://github.com/floooh/chips) Z80 core) for that.
- Prebuilt archives may target only one architecture; check with `file` and use Rosetta 2 if needed.

## Resources

- [z88dk wiki](https://github.com/z88dk/z88dk/wiki) and [README](https://github.com/z88dk/z88dk#readme): installation, compiler targets, new-target creation
- [sjasmplus README and documentation](https://github.com/z00m128/sjasmplus), with the [documentation site](https://z00m128.github.io/sjasmplus/documentation.html)
- [DeZog documentation](https://github.com/maziac/DeZog/tree/main/documentation): setup, zsim, sjasmplus and z88dk pages
- [Zilog UM0080, *Z80 CPU User Manual*](https://www.zilog.com/docs/z80/um0080.pdf)
- [clrhome.org Z80 opcode table](https://clrhome.org/table/), for encodings and T-states
- [floooh/chips](https://github.com/floooh/chips): cycle-accurate Z80 core for timing validation
