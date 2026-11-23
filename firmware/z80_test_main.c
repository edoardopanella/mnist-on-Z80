#include <stdint.h>


#define MAILBOX ((volatile uint8_t *)0xF000)

static uint16_t sum_to(uint8_t n)
{
    uint16_t s = 0;
    uint8_t i;
    for (i=1; i<=n; i++)
        s += i;
    return s;
}

int main(void)
{
    uint16_t s = sum_to(10);

    MAILBOX[0] = 0x42;
    MAILBOX[1] = (uint8_t)(s & 0xFF);
    MAILBOX[2] = (uint8_t)(s >> 8);

    for (;;) { }
}