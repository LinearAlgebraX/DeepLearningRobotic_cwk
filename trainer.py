import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
import os

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
    with open(f"logs/log.txt", "a", encoding="utf8") as f:
        f.write(f"{(100*correct):>0.3f} {test_loss:>8f}\n")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu")

model = Model.ResNet18().to(device)
data = Data.Data()
train_loader = DataLoader(dataset=data.train_data, batch_size=args.batchsize, shuffle=True)
test_loader = DataLoader(dataset=data.test_dataset, batch_size=args.batchsize, shuffle=True)
valid_loader = DataLoader(dataset=data.val_set, batch_size=args.batchsize, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weightdecay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Tmax)

for t in range(args.epochs):
    print(f"Epoch {t+1}\n-----------------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
    scheduler.step()
    if (t+1)%checkStep == 0:
        Enter = input("Do you want to save this model?[y/n]: ")
        if Enter == "y":
            name = input("Name it as [your-enter].pth: ")
            torch.save(model.state_dict(), f"model/{name}.pth")
            print(f"Saved PyTorch Model State to {name}.pth")
            exit()

print("Done!")