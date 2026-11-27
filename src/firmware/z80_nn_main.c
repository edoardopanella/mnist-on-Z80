/* Runs nn_infer on one test case inside DeZog (zsim), no serial I/O.
 *
 * DeZog loads the data straight into memory (see .vscode/launch.json):
 *   0x1000  weights payload  (weights.bin without its 5-byte header, 25,581 bytes)
 *   0x7800  one testvec record (985 bytes: x, acc1, h_q, acc2, digit)
 * Both files are made by tests/make_dezog_inputs.py.
 *
 * Result in the mailbox at 0xF000:
 *   [0] 0x42           program finished
 *   [1] digit          computed by nn_infer on the Z80
 *   [2] expected digit from testvec.bin
 *   [3] 0x00 if acc1, h_q, acc2 and digit all match, else the first mismatch:
 *       0x01 acc1, 0x02 h_q, 0x03 acc2, 0x04 digit
 *   [4] index of the first mismatching element (0 when [3] is 0x00 or 0x04)
 */
#include <stdint.h>
#include <string.h>
#include "nn.h"

/* Memory map: code 0x0000-0x0FFF, weights 0x1000-0x73ED, case 0x7800-0x7BD8,
 * variables from 0x8000, mailbox 0xF000, stack below 0xFFFF. */
#define WEIGHTS ((const struct nn_weights *)0x1000)
#define CASE    ((const uint8_t *)0x7800)
#define MAILBOX ((volatile uint8_t *)0xF000)

/* Offsets inside one testvec record (spec §4.2). */
#define REC_X     0
#define REC_ACC1  784
#define REC_HQ    (REC_ACC1 + 4 * N_HID)
#define REC_ACC2  (REC_HQ + N_HID)
#define REC_DIGIT (REC_ACC2 + 4 * N_OUT)

/* The struct must have exactly the payload layout (no padding). */
_Static_assert(sizeof(struct nn_weights) == 25581, "nn_weights layout != weights.bin payload");

static struct nn_trace trace;

/* Little-endian int32 at p. The Z80 is little-endian too, so memcpy is enough. */
static int32_t rd_i32(const uint8_t *p)
{
    int32_t v;
    memcpy(&v, p, 4);
    return v;
}

static void report(uint8_t digit, uint8_t what, uint8_t index)
{
    MAILBOX[1] = digit;
    MAILBOX[2] = CASE[REC_DIGIT];
    MAILBOX[3] = what;
    MAILBOX[4] = index;
    MAILBOX[0] = 0x42;               /* written last, so it means "results ready" */
}

int main(void)
{
    uint8_t digit = nn_infer(WEIGHTS, CASE + REC_X, &trace);
    uint8_t i;

    for (i = 0; i < N_HID; i++)
        if (trace.acc1[i] != rd_i32(CASE + REC_ACC1 + 4 * i)) { report(digit, 0x01, i); goto done; }
    for (i = 0; i < N_HID; i++)
        if (trace.h_q[i] != CASE[REC_HQ + i])                  { report(digit, 0x02, i); goto done; }
    for (i = 0; i < N_OUT; i++)
        if (trace.acc2[i] != rd_i32(CASE + REC_ACC2 + 4 * i)) { report(digit, 0x03, i); goto done; }
    if (digit != CASE[REC_DIGIT])                              { report(digit, 0x04, 0); goto done; }
    report(digit, 0x00, 0);

done:
    for (;;) { }
}
