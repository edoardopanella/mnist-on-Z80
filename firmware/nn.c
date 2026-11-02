#include <stddef.h>   /* per NULL */
#include "nn.h"


static int32_t layer1_neuron(const int8_t *row, int32_t bias, const uint8_t *x)
{
    /* calcola il valore di un neurone nel primo layer, verrà poi chiamata 32 volte*/
    int32_t acc = bias; /* aggiungi il bias del neurone*/

    for (uint16_t i = 0; i < N_IN; i++){
        if (x[i] != 0){ /* controlla che il pixel i sia acceso*/
            acc += row[i]; /* se il pixel è acceso, aggiungi all'acc il valore dei pesi che si collegano ad esso*/
        }
    }

    return acc; /* ritorna il valore del neurone*/
}

static uint8_t requant(int32_t acc, int32_t M0, uint8_t shift)
{
    /* per un neurone, calcola ReLU, poi conversione da acc1 a h_q con M0 e shift, poi limite a 127*/
    int32_t t;

    if (acc < 0){ /* RELU*/
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
    /* calcola attivazione di un neurone nel secondo layer*/
    int32_t acc = bias; /* come prima, aggiungi il bias all'attivazione del neurone nel secondo layer*/

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

    /* Layer 1 + requantizzazione */
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

