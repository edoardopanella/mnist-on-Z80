/* Line-protocol firmware (docs/spec.md §7.5), the Z80 side of host/z80nn.py.
 *
 * One command per line, one reply per command (D replies with four lines):
 *   V                  -> "Z80NN 1"
 *   W oooo <hex>       -> "OK"      up to 64 bytes of the weights payload at offset oooo
 *   C                  -> "C xxxx"  16-bit sum of the whole payload
 *   I <196 hex>        -> "R d"     classify a packed image (98 bytes, MSB = first pixel)
 *   D <196 hex>        -> "A1 ...", "H ...", "A2 ...", "R d"
 * Errors: ERR CMD, ERR HEX, ERR LEN, ERR OFS, ERR LONG. The program keeps running.
 *
 * I/O goes through the console: fgetc_cons() reads without echo (getchar()
 * would echo every byte back into the replies). Under z88dk-ticks (make emu)
 * the console is the emulator's stdin/stdout.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "nn.h"

#define PAYLOAD_SIZE 25581          /* weights.bin without its 5-byte header */
#define MAX_CHUNK    64
#define IMG_BYTES    (N_IN / 8)     /* 98 */
#define LINE_MAX     (2 + 2 * IMG_BYTES)   /* longest valid line: "I " + 196 hex */

_Static_assert(sizeof(struct nn_weights) == PAYLOAD_SIZE, "nn_weights layout != weights.bin payload");

static struct nn_weights weights;   /* filled by W, byte for byte (Z80 is little-endian) */
static struct nn_trace trace;
static uint8_t x[N_IN];
static uint8_t buf[MAX_CHUNK > IMG_BYTES ? MAX_CHUNK : IMG_BYTES];
static char line[LINE_MAX + 1];

/* --- Output ---------------------------------------------------------------- */

static void put_str(const char *s)
{
    while (*s)
        fputc_cons(*s++);
}

static void put_hex4(uint16_t v)
{
    static const char digits[] = "0123456789ABCDEF";
    int8_t sh;
    for (sh = 12; sh >= 0; sh -= 4)
        fputc_cons(digits[(v >> sh) & 0x0F]);
}

/* Decimal int32 without printf (smaller, and no %ld support to configure). */
static void put_i32(int32_t v)
{
    char d[10];
    uint8_t n = 0;
    uint32_t u = (uint32_t)v;

    if (v < 0) {
        fputc_cons('-');
        u = 0 - u;
    }
    do {
        d[n++] = (char)('0' + u % 10);
        u /= 10;
    } while (u != 0);
    while (n)
        fputc_cons(d[--n]);
}

static void put_list(const char *label, const int32_t *v, uint8_t n)
{
    uint8_t i;
    put_str(label);
    for (i = 0; i < n; i++) {
        fputc_cons(' ');
        put_i32(v[i]);
    }
    fputc_cons('\n');
}

static void reply(const char *s)
{
    put_str(s);
    fputc_cons('\n');
}

/* --- Input ----------------------------------------------------------------- */

/* Read one line into line[], without '\r' and '\n'.
 * Returns its length, -1 if it was longer than LINE_MAX, -2 at end of input. */
static int16_t read_line(void)
{
    uint8_t n = 0, too_long = 0;
    int c;

    for (;;) {
        c = fgetc_cons();
        if (c == EOF)
            return -2;
        if (c == '\n')
            break;
        if (c == '\r')
            continue;
        if (n < LINE_MAX)
            line[n++] = (char)c;
        else
            too_long = 1;
    }
    line[n] = '\0';
    return too_long ? -1 : n;
}

static int8_t hex_digit(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

/* Decode n bytes from 2n hex digits into dst. Returns 0, or -1 on a bad digit. */
static int8_t hex_bytes(const char *s, uint8_t *dst, uint8_t n)
{
    int8_t hi, lo;
    while (n--) {
        hi = hex_digit(*s++);
        lo = hex_digit(*s++);
        if (hi < 0 || lo < 0)
            return -1;
        *dst++ = (uint8_t)(hi << 4 | lo);
    }
    return 0;
}

/* --- Commands -------------------------------------------------------------- */

/* "W oooo <hex>": arg points at oooo, len = strlen(arg). */
static void cmd_write(const char *arg, uint8_t len)
{
    uint8_t n;
    uint16_t offset;

    if (len < 7 || arg[4] != ' ' || (len - 5) % 2 != 0 || (len - 5) / 2 > MAX_CHUNK) {
        reply("ERR LEN");
        return;
    }
    n = (uint8_t)((len - 5) / 2);
    if (hex_bytes(arg, buf, 2) != 0 || hex_bytes(arg + 5, buf + 2, n) != 0) {
        reply("ERR HEX");
        return;
    }
    offset = (uint16_t)buf[0] << 8 | buf[1];
    if (offset > PAYLOAD_SIZE - n) {
        reply("ERR OFS");
        return;
    }
    memcpy((uint8_t *)&weights + offset, buf + 2, n);
    reply("OK");
}

static void cmd_checksum(void)
{
    const uint8_t *p = (const uint8_t *)&weights;
    uint16_t sum = 0, i;

    for (i = 0; i < PAYLOAD_SIZE; i++)
        sum += p[i];
    put_str("C ");
    put_hex4(sum);
    fputc_cons('\n');
}

/* "I <hex>" / "D <hex>": arg points at the hex digits. */
static void cmd_infer(const char *arg, uint8_t len, uint8_t debug)
{
    uint8_t *p = x;
    uint8_t i, bit, digit;

    if (len != 2 * IMG_BYTES) {
        reply("ERR LEN");
        return;
    }
    if (hex_bytes(arg, buf, IMG_BYTES) != 0) {
        reply("ERR HEX");
        return;
    }
    for (i = 0; i < IMG_BYTES; i++)
        for (bit = 0x80; bit != 0; bit >>= 1)
            *p++ = (buf[i] & bit) ? 1 : 0;

    digit = nn_infer(&weights, x, debug ? &trace : NULL);

    if (debug) {
        int32_t h[N_HID];               /* h_q widened so put_list prints it too */
        for (i = 0; i < N_HID; i++)
            h[i] = trace.h_q[i];
        put_list("A1", trace.acc1, N_HID);
        put_list("H", h, N_HID);
        put_list("A2", trace.acc2, N_OUT);
    }
    put_str("R ");
    fputc_cons((char)('0' + digit));
    fputc_cons('\n');
}

int main(void)
{
    int16_t len;

    for (;;) {
        len = read_line();
        if (len == -2)
            return 0;                   /* end of input: the host closed the pipe */
        if (len == -1) {
            reply("ERR LONG");
            continue;
        }

        if (len == 1 && line[0] == 'V')
            reply("Z80NN 1");
        else if (len == 1 && line[0] == 'C')
            cmd_checksum();
        else if (len > 2 && line[1] == ' ' && line[0] == 'W')
            cmd_write(line + 2, (uint8_t)(len - 2));
        else if (len > 2 && line[1] == ' ' && (line[0] == 'I' || line[0] == 'D'))
            cmd_infer(line + 2, (uint8_t)(len - 2), line[0] == 'D');
        else
            reply("ERR CMD");
    }
}
