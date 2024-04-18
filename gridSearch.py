import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
import os
import time

import model as Model
import data as Data

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--batchsize', default=100, type=int, help='batch size')
parser.add_argument('--epochs', default=30, type=int, help='number of epochs')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum of optimizer')
parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay of optimizer')
parser.add_argument('--Tmax', default=200, type=int, help='scheduler T max')

args = parser.parse_args()

checkStep = 30

learningRate = [0.01, 0.001, 0.0001]
batchSize = [32, 64, 128]
epochs = [10, 20, 30, 40]
momentum = [0.8, 0.9, 0.99]
weightDecay = [5e-4, 1e-4, 5e-6]


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            with open(f"logs/loss.txt", "a", encoding="utf8") as f:
                f.write(f"{loss:>7f}\n")
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    # poisonedRate = 1
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    with open(f"logs/grid.txt", "a", encoding="utf8") as f:
        f.write(f"{(100*correct):>0.3f} {test_loss:>8f}\n")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu")

model = Model.ResNet18().to(device)
data = Data.Data()

loss_fn = torch.nn.CrossEntropyLoss()
valid_loader = DataLoader(dataset=data.val_set, batch_size=100, shuffle=True)


for a in epochs:
    for b in learningRate:
        for c in batchSize:
            for d in momentum:
                for e in weightDecay:
                    train_loader = DataLoader(dataset=data.train_data, batch_size=c, shuffle=True)
                    test_loader = DataLoader(dataset=data.test_dataset, batch_size=c, shuffle=True)
                    optimizer = torch.optim.SGD(model.parameters(), lr=b, momentum=d, weight_decay=e)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Tmax)
                    T1 = time.time()
                    for t in range(a):
                        print(f"Epoch {t+1}\n-----------------------------------------")
                        train(train_loader, model, loss_fn, optimizer)
                        # test(test_loader, model, loss_fn)
                        scheduler.step()
                        # if (t+1)%checkStep == 0:
                            # Enter = input("Do you want to save this model?[y/n]: ")
                            # if Enter == "y":
                            #     name = input("Name it as [your-enter].pth: ")
                            #     torch.save(model.state_dict(), f"model/{name}.pth")
                            #     print(f"Saved PyTorch Model State to {name}.pth")
                            #     exit()
                    T2 = time.time()
                    print(f"Epochs: {a}, LearningRate: {b}, BatchSize: {c}, Momentum: {d}, WeightDecay: {e}, time: {(T1-T2)*1000}")
                    with open(f"logs/grid.txt", "a", encoding="utf8") as f:
                        f.write(f"Epochs: {a}, LearningRate: {b}, BatchSize: {c}, Momentum: {d}, WeightDecay: {e}, time: {(T1-T2)*1000}\n")
                    test(valid_loader, model, loss_fn)
                    with open(f"logs/loss.txt", "a", encoding="utf8") as f:
                        f.write(f"Epochs: {a}, LearningRate: {b}, BatchSize: {c}, Momentum: {d}, WeightDecay: {e}, time: {(T1-T2)*1000}\n")
                        f.write(f"\n")


print("Done!")