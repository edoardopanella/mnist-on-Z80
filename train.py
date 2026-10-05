# imports

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Lambda

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Device check
print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.') 
    device = torch.device('cpu')


# Knobs
BATCH_SIZE = 64
EPOCHS = 10
LOSS = nn.CrossEntropyLoss()
SEED = torch.manual_seed(0)

# Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "mlp.pt")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

'''
returned in get_data()

train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)'''


'''
TEST FOR DATA, to be deleted
for (data, target) in train_loader:
    print('data:', data.size(), 'type:', data.type())
    print('target:', target.size(), 'type:', target.type())
    break


pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    data, target = next(iter(train_loader))
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(data[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
    plt.title('Class: '+str(target[i].item()))

plt.show()
'''

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    def forward(self, x):
        return self.layers(x)


# Test model

'''model = MLP().to(device)
print(model)'''


def correct(output, target):
        predicted_digits = output.argmax(1)    
        correct_ones = (predicted_digits == target).type(torch.float)
        return correct_ones.sum().item()

def build_model():

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    return  model, criterion, optimizer
    

def train_one_epoch(data_loader, model, criterion, optimizer):

    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_loss = 0
    total_correct = 0

    for data, target in data_loader:
        # NOTE: optional, move data to GPU, will remove perhaps late
        data = data.to(device)
        target = target.to(device)

        # Do a forward pass
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)
        total_loss += loss.item()

        # Count number of correct digits
        total_correct += correct(output=output, target=target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss / num_batches
    accuracy = total_correct / num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")

    return train_loss, accuracy




def get_data(DATA_DIR):
    transform = Compose([ToTensor(), Lambda(lambda x: (x > 0.5).float())])
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return {
        'train_dataset' : train_dataset,
        'test_dataset' : test_dataset,
        'train_loader' : train_loader,
        'test_loader': test_loader
    }

@torch.no_grad()
def evaluate(data_loader, model, criterion):
    '''
    one epoch evaluator on the given loader (val or test), no weight updates
    '''
    model.eval()
    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_loss = 0
    total_correct = 0

    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        # Do a forward pass
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)
        total_loss += loss.item()

        # Count number of correct digits
        total_correct += correct(output=output, target=target)

    eval_loss = total_loss / num_batches
    accuracy = total_correct / num_items
    print(f"Eval loss: {eval_loss:7f}, accuracy: {accuracy:.2%}")

    return eval_loss, accuracy



def main():

    # Step 1: get data
    data = get_data(DATA_DIR=DATA_DIR)
    train_dataset = data['train_dataset']
    test_dataset = data['test_dataset']

    train_loader = data['train_loader']
    test_loader = data['test_loader']

    # Step 2: build model
    model, criterion, optimizer = build_model()

    # Step 3: training loop
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(data_loader=train_loader,
                                                model=model,
                                                criterion=criterion,
                                                optimizer=optimizer)
        test_loss, test_acc = evaluate(data_loader=test_loader,
                                       model=model,
                                       criterion=criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

    return history, model


if __name__ == "__main__":
    _, model = main()

    torch.save(model.state_dict(), WEIGHTS_PATH)



