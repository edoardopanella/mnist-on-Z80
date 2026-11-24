/* Register snapshot for z80_serial_main.c (the G command).
 *
 * snap_regs() copies every register into the global regs. SP and PC are the
 * caller's: SP as it was before the call, PC = the return address (the
 * instruction after the call). Before the first call regs is all zero.
 */
#ifndef SNAPREGS_H
#define SNAPREGS_H

#include <stdint.h>

/* Offsets are hard-coded in snapregs.c, keep the order. */
struct regs {
    uint16_t af, bc, de, hl;        /*  0  2  4  6 */
    uint16_t ix, iy, sp, pc;        /*  8 10 12 14 */
    uint16_t af2, bc2, de2, hl2;    /* 16 18 20 22 */
    uint8_t  i, r;                  /* 24 25 */
};

extern struct regs regs;

void snap_regs(void) __naked;

#endif
