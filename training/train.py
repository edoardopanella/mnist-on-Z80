"""
train.py - Train a 784 -> 32 -> 10 MLP on binarized MNIST.

Output: artifacts/mlp.pt (float weights, state_dict).
Also defines MLP, get_data() and the paths the other scripts use.
Importing it has no side effects.
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, ToTensor

# --- Paths ------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
DATA_DIR = os.path.join(ARTIFACTS_DIR, "data")
MODEL_DIR = ARTIFACTS_DIR
WEIGHTS_PATH = os.path.join(MODEL_DIR, "mlp.pt")

# --- Hyperparameters ----------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Model --------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # quantize.py reads the linear layers as layers[1] and layers[3]
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.layers(x)


def build_model():
    model = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, criterion, optimizer


# --- Data -----------------------------------------------------------------------

def get_data(data_dir=DATA_DIR):
    """Binarized MNIST: every pixel is 0 or 1, as the Z80 receives it."""
    os.makedirs(data_dir, exist_ok=True)
    transform = Compose([ToTensor(), Lambda(lambda x: (x > 0.5).float())])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
    }


# --- Training and evaluation --------------------------------------------------

def correct(output, target):
    """Number of correctly classified images in a batch."""
    predicted_digits = output.argmax(1)
    return (predicted_digits == target).sum().item()


def train_one_epoch(data_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0
    total_correct = 0

    for data, target in data_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += correct(output, target)

    train_loss = total_loss / len(data_loader)
    accuracy = total_correct / len(data_loader.dataset)
    print(f"  train loss: {train_loss:.6f}, accuracy: {accuracy:.2%}")
    return train_loss, accuracy


@torch.no_grad()
def evaluate(data_loader, model, criterion):
    """Loss and accuracy on data_loader, without training."""
    model.eval()
    total_loss = 0
    total_correct = 0

    for data, target in data_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        output = model(data)
        total_loss += criterion(output, target).item()
        total_correct += correct(output, target)

    eval_loss = total_loss / len(data_loader)
    accuracy = total_correct / len(data_loader.dataset)
    print(f"  test  loss: {eval_loss:.6f}, accuracy: {accuracy:.2%}")
    return eval_loss, accuracy


# --- Main ---------------------------------------------------------------------

def main():
    print("PyTorch version:", torch.__version__)
    print("Device:", DEVICE)
    torch.manual_seed(SEED)

    data = get_data()
    model, criterion, optimizer = build_model()

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(data["train_loader"], model, criterion, optimizer)
        test_loss, test_acc = evaluate(data["test_loader"], model, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")
    return history


if __name__ == "__main__":
    main()