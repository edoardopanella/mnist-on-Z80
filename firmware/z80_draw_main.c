/* Draw-a-digit firmware for DeZog (zsim) with a Custom UI.
 *
 * The image comes from the "Z80 Simulator" panel through I/O ports simulated
 * by host/dezog/draw_logic.js; the UI is host/dezog/draw_ui.html.
 *
 * Ports
 *   IN  0x10  next byte queued by the UI, 0 = nothing queued
 *   OUT 0x11  one byte of a reply
 *   OUT 0x12  end of reply, value = reply type
 *
 * Commands (first byte of each UI message)
 *   'I' + 98 bytes   packed 28x28 image, MSB first (like z80nn.pack_image)
 *                    -> 'R': digit (1), h_q (32), acc2 (10 x int32 LE)
 *                    -> 'G': register snapshot (struct regs, 26 bytes)
 *   'M' lo hi len    read len bytes (0 = 256) from address hi:lo
 *                    -> 'M': the bytes
 * At start-up       -> 'A': addresses of x, trace, regs (3 x uint16 LE)
 *
 * Memory map: code 0x0000-0x0FFF, weights 0x1000-0x73ED (loaded by DeZog),
 * variables from 0x8000, stack below 0xFFFF.
 *
 * Build:
 *   zcc +z80 -compiler=sdcc -m --list --c-code-in-asm -pragma-define:CRT_MODEL=0 \
 *       -o build/draw.bin firmware/z80_draw_main.c firmware/nn.c
 */
#include <stdint.h>
#include "nn.h"

#define WEIGHTS ((const struct nn_weights *)0x1000)

__sfr __at 0x10 PORT_IN;
__sfr __at 0x11 PORT_OUT;
__sfr __at 0x12 PORT_END;

#define IMG_BYTES (N_IN / 8)        /* 98 */

/* Register snapshot. Offsets are used by snap_regs() below, keep the order. */
struct regs {
    uint16_t af, bc, de, hl;        /*  0  2  4  6 */
    uint16_t ix, iy, sp, pc;        /*  8 10 12 14 */
    uint16_t af2, bc2, de2, hl2;    /* 16 18 20 22 */
    uint8_t  i, r;                  /* 24 25 */
};

static uint8_t x[N_IN];             /* unpacked image, one 0/1 byte per pixel */
static struct nn_trace trace;
struct regs regs;                   /* not static: snap_regs refers to _regs */

/* Copy all registers into regs. SP and PC are the caller's: SP as it was
 * before the call, PC = the return address (the instruction after the call).
 * HL and AF are saved first because the code below uses them. */
static void snap_regs(void) __naked
{
    __asm
    ld  (_regs+6), hl
    push af
    pop hl
    ld  (_regs+0), hl
    ld  (_regs+2), bc
    ld  (_regs+4), de
    ld  (_regs+8), ix
    ld  (_regs+10), iy
    ld  hl, 2
    add hl, sp
    ld  (_regs+12), hl
    pop hl
    push hl
    ld  (_regs+14), hl
    defb 0x08                       ; ex af,af (written as a byte: the quote confuses the C preprocessor)
    push af
    pop hl
    ld  (_regs+16), hl
    defb 0x08
    exx
    ld  (_regs+18), bc
    ld  (_regs+20), de
    ld  (_regs+22), hl
    exx
    ld  a, i
    ld  (_regs+24), a
    ld  a, r
    ld  (_regs+25), a
    ret
    __endasm;
}

static void send(const void *p, uint16_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    while (n--)
        PORT_OUT = *b++;
}

static void send_u16(uint16_t v)
{
    PORT_OUT = (uint8_t)v;
    PORT_OUT = (uint8_t)(v >> 8);
}

/* Read the 98 packed bytes into x[], 8 pixels per byte, MSB = leftmost. */
static void read_image(void)
{
    uint8_t *p = x;
    uint8_t i, b, bit;

    for (i = 0; i < IMG_BYTES; i++) {
        b = PORT_IN;
        for (bit = 0x80; bit != 0; bit >>= 1)
            *p++ = (b & bit) ? 1 : 0;
    }
}

static void cmd_infer(void)
{
    uint8_t digit;

    read_image();
    digit = nn_infer(WEIGHTS, x, &trace);
    snap_regs();

    PORT_OUT = digit;
    send(trace.h_q, sizeof trace.h_q);
    send(trace.acc2, sizeof trace.acc2);
    PORT_END = 'R';

    send(&regs, sizeof regs);
    PORT_END = 'G';
}

static void cmd_memory(void)
{
    uint16_t addr = PORT_IN;
    uint16_t len;

    addr |= (uint16_t)PORT_IN << 8;
    len = PORT_IN;
    if (len == 0)
        len = 256;
    send((const uint8_t *)addr, len);
    PORT_END = 'M';
}

int main(void)
{
    uint8_t cmd;

    send_u16((uint16_t)x);
    send_u16((uint16_t)&trace);
    send_u16((uint16_t)&regs);
    PORT_END = 'A';

    for (;;) {
        /* idle: wait for the UI. Not "while ((cmd = PORT_IN) == 0)": sdcc
         * reads the port again for the test and loses a byte. */
        do {
            cmd = PORT_IN;
        } while (cmd == 0);
        if (cmd == 'I')
            cmd_infer();
        else if (cmd == 'M')
            cmd_memory();
    }
}
