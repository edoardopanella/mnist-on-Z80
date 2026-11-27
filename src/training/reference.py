"""
reference.py - Integer inference in numpy, computed exactly as the Z80 does it.

Run it to compare the integer model with the float one on the MNIST test set.
"""

from train import WEIGHTS_PATH, get_data, MLP

import torch
import numpy as np
import os

QUANT_PATH = os.path.join(os.path.dirname(WEIGHTS_PATH), "quantized.npz")

ARRAYS = ["W1q", "b1q", "W2q", "b2q"]
SCALARS = ["M0", "shift", "bound", "s1", "s2", "s_h", "h_max"]


def load_quantized(path):
    """quantized.npz as a dict: arrays as int64, scalars as Python numbers."""
    q = {}
    with np.load(path) as f:
        for name in ARRAYS:
            q[name] = f[name].astype(np.int64)
        for name in SCALARS:
            q[name] = f[name].item()
    return q


def load_test_images():
    """The 10,000 binarized test images as (10000, 784) int64, and their labels."""
    test_loader = get_data()["test_loader"]

    images, labels = [], []
    for batch, target in test_loader:
        # (B, 1, 28, 28) -> (B, 784), row by row, like nn.Flatten
        images.append(batch.reshape(batch.shape[0], 784).numpy().astype(np.int64))
        labels.append(target.numpy())

    images = np.concatenate(images)
    labels = np.concatenate(labels)

    assert images.shape == (10000, 784)
    assert set(np.unique(images)) <= {0, 1}, "images are not binarized"
    return images, labels


def infer_int(x, q):
    """Returns (digit, acc1, h_q, acc2). acc1 is before the ReLU."""
    assert x.shape == (784,), f"expected shape (784,), got {x.shape}"
    assert np.all((x == 0) | (x == 1)), "image contains values other than 0 and 1"

    acc1 = (q['W1q'] @ x) +q['b1q']
    acc1_relu = np.maximum(acc1, 0)

    assert np.max(acc1_relu) * q['M0'] + 2**(q['shift']-1) < 2**31

    h_q = (acc1_relu * q['M0'] + 2**(q['shift']-1)) >> q['shift']
    h_q = np.minimum(h_q, 127)

    acc2 = (q['W2q'] @ h_q) + q['b2q']

    digit = int(np.argmax(acc2))

    return digit, acc1, h_q, acc2

def load_float_model():
    model = MLP()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model

@torch.no_grad()
def evaluate(images, labels, q):
    """Accuracy of both models, how often they agree, and how often h_q saturates."""
    N = images.shape[0]

    model = load_float_model()
    float_logits = model(torch.from_numpy(images).float())   # (N, 10)
    pred_float = float_logits.argmax(dim=1).numpy()          # (N,)

    correct_int = 0
    agreed = 0
    count_127 = 0
    max_acc1 = float('-inf')

    for i in range(N):
        digit, acc1, h_q, acc2 = infer_int(images[i], q)
        if digit == labels[i]:
            correct_int += 1
        if digit == pred_float[i]:
            agreed += 1
        count_127 += np.count_nonzero(h_q == 127)
        max_acc1 = max(max_acc1, int(np.max(acc1)))

    return {
        "acc_float": float(np.mean(pred_float == labels)),
        "acc_int": correct_int / N,
        "agreement": agreed / N,
        "perc_127": count_127 / (N * 32),
        "max_acc1": max_acc1,
    }


def main():
    q = load_quantized(QUANT_PATH)
    images, labels = load_test_images()
    r = evaluate(images, labels, q)

    print(f"Float accuracy:       {r['acc_float']:.2%}")
    print(f"Integer accuracy:     {r['acc_int']:.2%}")
    print(f"Float/int agreement:  {r['agreement']:.2%}")
    print(f"h_q saturated at 127: {r['perc_127']:.4%}")
    print(f"Max acc1 observed:    {r['max_acc1']}  (bound: {q['bound']})")


if __name__ == "__main__":
    main()
