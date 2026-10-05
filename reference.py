from train import WEIGHTS_PATH, get_data

import torch
import numpy as np
import os

QUANT_PATH = os.path.join(os.path.dirname(WEIGHTS_PATH), "quantized.npz")

ARRAYS = ["W1q", "b1q", "W2q", "b2q"]                          
SCALARS = ["M0", "shift", "bound", "s1", "s2", "s_h", "h_max"] 


def load_quantized(path):
    q = {}
    with np.load(path) as f:
        for name in ARRAYS:
            q[name] = f[name].astype(np.int64)
        for name in SCALARS:
            q[name] = f[name].item()
    return q




