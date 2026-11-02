        DEVICE ZXSPECTRUM48     ; memory model: 64 KB address space, RAM from 0x4000

result  EQU 0x9000
stack   EQU 0xFFF0

        ORG 0x8000
start:
        LD   SP, stack
        XOR  A                  ; A = 0 (running sum)
        LD   B, 10              ; loop counter
loop:
        ADD  A, B               ; sum += B
        DJNZ loop               ; B -= 1; jump if B != 0
        LD   (result), A        ; store the sum in memory
done:
        JR   done               ; stop here (a jump to itself)

        SAVESNA "main.sna", start