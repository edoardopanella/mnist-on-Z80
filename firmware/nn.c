#include <stddef.h>   /* for NULL */
#include "nn.h"


static int32_t layer1_neuron(const int8_t *row, int32_t bias, const uint8_t *x)
{
    /* value of one layer-1 neuron; called once per hidden neuron (32 times) */
    int32_t acc = bias;

    for (uint16_t i = 0; i < N_IN; i++){
        if (x[i] != 0){ /* binary input: each lit pixel adds its weight */
            acc += row[i];
        }
    }

    return acc;
}

static uint8_t requant(int32_t acc, int32_t M0, uint8_t shift)
{
    /* ReLU, then rescale acc1 to h_q with M0 and shift (rounded), then clamp to 127 */
    int32_t t;

    if (acc < 0){ /* ReLU */
        acc = 0;
    }

    t = (acc * M0 + ((int32_t)1 << (shift - 1))) >> shift; /* acc >= 0 e acc <= bound: quantize.py garantisce acc * M0 + 2^(shift-1) < 2^31 */

    if (t > 127){
        return 127;
    }

    return (uint8_t)t;
}

static int32_t layer2_neuron(const int8_t *row, int32_t bias, const uint8_t *h_q)
{
    /* value of one layer-2 neuron (logit) */
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

