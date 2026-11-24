/* Register snapshot: see snapregs.h. */
#include "snapregs.h"

struct regs regs;                   /* not static: the asm below refers to _regs */

/* HL and AF are saved first because the code below uses them. */
void snap_regs(void) __naked
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
