import torch
import numpy as np
import os
from train import MLP, get_data, WEIGHTS_PATH, DATA_DIR

QUANT_PATH = os.path.join(os.path.dirname(WEIGHTS_PATH), "quantized.npz")


def quantize_symmetric(w):
    s = (torch.max(torch.abs(w)) / 127).item()
    wq = torch.round(w / s).clamp(-127, 127).to(torch.int8)

    return wq, s


def choose_m0_shift(M, bound):
    M0_best, shift_best = None, None

    for shift in range(1, 31):
        M0 = round(M*2**shift)

        if bound * M0 + 2**(shift - 1) < 2**31:
            M0_best, shift_best = M0, shift
        else:
            break
    if M0_best is None and shift_best is None:
        raise ValueError('Quantization not supported')

    return M0_best, shift_best


def main():
    model = MLP()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location = "cpu"))
    model.eval()

    fc1 = model.layers[1]
    fc2 = model.layers[3]

    W1 = fc1.weight.detach()
    b1 = fc1.bias.detach()

    W2 = fc2.weight.detach()
    b2 = fc2.bias.detach()


    #print(f"W1: {W1.shape}\nb1: {b1.shape}\n\nW2: {W2.shape}\nb2: {b2.shape}")

    # Quantize layer 1
    W1q, s1 = quantize_symmetric(W1)
    b1q = torch.round(b1 / s1).to(torch.int32)

    # Quantize the neurons after layer 1

    data = get_data(DATA_DIR=DATA_DIR)
    train_loader = data['train_loader']

    h_max = 0
    with torch.no_grad():
        first_layer = model.layers[:3]

        for image, _ in train_loader:
            h = first_layer(image)
            local_max = torch.max(torch.abs(h))
            if local_max > h_max:
                h_max = local_max.item()

    s_h = h_max / 127

    # Quantize layer 2
    W2q, s2 = quantize_symmetric(W2)
    b2q = torch.round(b2 / (s_h * s2)).to(torch.int32)


    M = s1 / s_h


    W1q_pos = W1q.to(torch.int64).clamp(min=0)
    b1q_i64 = b1q.to(torch.int64)

    max_acc_per_neuron = torch.sum(W1q_pos, dim = 1) + b1q_i64

    bound = max_acc_per_neuron.max().item()

    M0, shift = choose_m0_shift(M=M, bound=bound)

    # Checks and assertions and types
    assert bound * M0 + 2**(shift - 1) < 2**31, "Overflow possibile in acc1 * M0"
    assert W1q.dtype == torch.int8 and W2q.dtype == torch.int8
    assert b1q.dtype == torch.int32 and b2q.dtype == torch.int32

    print(f"s1: {s1:.6g}  s2: {s2:.6g}  h_max: {h_max:.6g}  s_h: {s_h:.6g}")
    print(f"M: {M:.6g}  M0: {M0}  shift: {shift}  bound: {bound}")
    print(f"Relative error for M: {abs(M0 / 2**shift - M) / M:.2e}")

    # Saving
    np.savez(
        QUANT_PATH,
        W1q=W1q.numpy(), b1q=b1q.numpy(),        # weights and biases of layer 1
        W2q=W2q.numpy(), b2q=b2q.numpy(),        # weights and biases of layer 2
        M0=M0, shift=shift,                      # requantization
        s1=s1, s2=s2, s_h=s_h, h_max=h_max, bound=bound,   # diagnostics
    )

    # Final check
    saved = np.load(QUANT_PATH)
    assert np.array_equal(saved["W1q"], W1q.numpy())
    assert np.array_equal(saved["W2q"], W2q.numpy())
    print(f"Saved in {QUANT_PATH}")


if __name__ == "__main__":
    main()
