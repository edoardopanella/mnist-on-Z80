#ifndef NN_H
#define NN_H
#include <stdint.h>

#define N_IN  784
#define N_HID 32
#define N_OUT 10

struct nn_weights {                 /* same fields and order as weights.bin */
    int8_t  W1q[N_HID][N_IN];
    int32_t b1q[N_HID];
    int32_t M0;
    uint8_t shift;
    int8_t  W2q[N_OUT][N_HID];
    int32_t b2q[N_OUT];
};

struct nn_trace {                   /* intermediate values, for tests and the D command */
    int32_t acc1[N_HID];            /* before ReLU */
    uint8_t h_q[N_HID];
    int32_t acc2[N_OUT];
};

uint8_t nn_infer(const struct nn_weights *w,
                 const uint8_t x[N_IN],
                 struct nn_trace *trace);

#endif
