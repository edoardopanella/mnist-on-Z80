from train import WEIGHTS_PATH, DATA_DIR, get_data, MLP

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



def load_test_images():
    test_loader = get_data()["test_loader"]

    images, labels = [], []
    for batch, target in test_loader:
        # (B, 1, 28, 28) -> (B, 784), riga per riga come nn.Flatten
        images.append(batch.reshape(batch.shape[0], 784).numpy().astype(np.int64))
        labels.append(target.numpy())

    images = np.concatenate(images)   # forma (10000, 784)
    labels = np.concatenate(labels)   # forma (10000,)

    assert images.shape == (10000, 784)
    assert set(np.unique(images)) <= {0, 1}, "Le immagini non sono binarizzate"
    return images, labels


def infer_int(x, q):
    assert x.shape == (784,), f"Attese 784 posizioni, trovate {x.shape}"
    assert np.all((x == 0) | (x == 1)), "L'immagine contiene valori diversi da 0 e 1"


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
    N = images.shape[0]

    # Step 1: predizioni del modello float
    model = load_float_model()
    float_logits = model(torch.from_numpy(images).float())   # forma (N, 10)
    pred_float = float_logits.argmax(dim=1).numpy()          # forma (N,)

    # Step 2: contatori
    correct_int = 0
    agreed = 0
    count_127 = 0
    max_acc1 = float('-inf')

    # Step 3: riferimento a interi, un'immagine alla volta
    for i in range(N):
        digit, acc1, h_q, acc2 = infer_int(images[i], q)
        if digit == labels[i]:
            correct_int += 1
        if digit == pred_float[i]:
            agreed += 1
        count_127 += np.count_nonzero(h_q == 127)
        max_acc1 = max(max_acc1, int(np.max(acc1)))

    # Step 4: risultati
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

    print(f"Accuratezza float:   {r['acc_float']:.2%}")
    print(f"Accuratezza interi:  {r['acc_int']:.2%}")
    print(f"Accordo float/int:   {r['agreement']:.2%}")
    print(f"h_q saturati a 127:  {r['perc_127']:.4%}")
    print(f"Max acc1 osservato:  {r['max_acc1']}  (bound: {q['bound']})")


if __name__ == "__main__":
    main()