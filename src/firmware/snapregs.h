/* Register snapshot for the G command in z80_serial_main.c.
 *
 * snap_regs() copies every register into the global regs. SP and PC are the
 * caller's: SP is its value before the call, PC is the return address.
 * regs is all zero until the first call.
 */
#ifndef SNAPREGS_H
#define SNAPREGS_H

#include <stdint.h>

/* snapregs.c hard-codes these offsets */
struct regs {
    uint16_t af, bc, de, hl;        /*  0  2  4  6 */
    uint16_t ix, iy, sp, pc;        /*  8 10 12 14 */
    uint16_t af2, bc2, de2, hl2;    /* 16 18 20 22 */
    uint8_t  i, r;                  /* 24 25 */
};

extern struct regs regs;

void snap_regs(void) __naked;

#endif
