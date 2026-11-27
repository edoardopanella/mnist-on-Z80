#include <stddef.h>   /* for NULL */
#include "nn.h"


/* One hidden neuron before ReLU. Pixels are 0 or 1, so each lit pixel just adds its weight. */
static int32_t layer1_neuron(const int8_t *row, int32_t bias, const uint8_t *x)
{
    int32_t acc = bias;

    for (uint16_t i = 0; i < N_IN; i++){
        if (x[i] != 0){
            acc += row[i];
        }
    }

    return acc;
}

/* ReLU, then scale to 0..127 with a rounded (acc * M0) >> shift. */
static uint8_t requant(int32_t acc, int32_t M0, uint8_t shift)
{
    int32_t t;

    if (acc < 0){ /* ReLU */
        acc = 0;
    }

    /* No overflow: 0 <= acc <= bound, and quantize.py picks M0 and shift so that
       bound * M0 + 2^(shift-1) < 2^31. */
    t = (acc * M0 + ((int32_t)1 << (shift - 1))) >> shift;

    if (t > 127){
        return 127;
    }

    return (uint8_t)t;
}

/* One output score (logit). */
static int32_t layer2_neuron(const int8_t *row, int32_t bias, const uint8_t *h_q)
{
    int32_t acc = bias;

    for (uint8_t j = 0; j < N_HID; j++){
        acc += (int16_t)row[j] * (int16_t)h_q[j];
    }

    return acc;
}

uint8_t nn_infer(const struct nn_weights *w, const uint8_t x[N_IN], struct nn_trace *trace)
{
    int32_t acc1[N_HID];
    uint8_t h_q[N_HID];
    int32_t acc2[N_OUT];
    uint8_t best = 0;

    /* Layer 1 + requantization */
    for (uint8_t j = 0; j < N_HID; j++){
        acc1[j] = layer1_neuron(w->W1q[j], w->b1q[j], x);
        h_q[j] = requant(acc1[j], w->M0, w->shift);
    }

    /* Layer 2 + argmax */
    for (uint8_t k = 0; k < N_OUT; k++){
        acc2[k] = layer2_neuron(w->W2q[k], w->b2q[k], h_q);
        if (acc2[k] > acc2[best]){
            best = k;
        }
    }

    if (trace != NULL){
        for (uint8_t j = 0; j < N_HID; j++){
            trace->acc1[j] = acc1[j];
            trace->h_q[j] = h_q[j];
        }
        for (uint8_t k = 0; k < N_OUT; k++){
            trace->acc2[k] = acc2[k];
        }
    }

    return best;
}

