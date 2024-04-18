import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
import os
import time
import random

import model as Model
import data as Data


times = 10
epochs = 10

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
            # with open(f"logs/loss.txt", "a", encoding="utf8") as f:
            #     f.write(f"{loss:>7f}\n")
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
    return correct, test_loss

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu")

data = Data.Data()
valid_loader = DataLoader(dataset=data.val_set, batch_size=100, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()

best_score = 0
best_para = None
for a in range(times):
    model = Model.ResNet18().to(device)

    learningRate = random.uniform(0.001, 0.01)
    batchSize = random.randint(32, 128)
    momentum = random.uniform(0.7, 0.99)
    weightDecay = random.uniform(1e-4, 5e-6)

    train_loader = DataLoader(dataset=data.train_data, batch_size=batchSize, shuffle=True)
    test_loader = DataLoader(dataset=data.test_dataset, batch_size=batchSize, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum, weight_decay=weightDecay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    T1 = time.time()
    for t in range(epochs):
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
    print(f"LearningRate: {learningRate}, BatchSize: {batchSize}, Momentum: {momentum}, WeightDecay: {weightDecay}, time: {(T1-T2)*1000}")
    with open(f"logs/grid.txt", "a", encoding="utf8") as f:
        f.write(f"LearningRate: {learningRate}, BatchSize: {batchSize}, Momentum: {momentum}, WeightDecay: {weightDecay}, time: {(T1-T2)*1000}\n")
    current_score, current_loss = test(valid_loader, model, loss_fn)
    # with open(f"logs/loss.txt", "a", encoding="utf8") as f:
    #     f.write(f"LearningRate: {learningRate}, BatchSize: {batchSize}, Momentum: {momentum}, WeightDecay: {weightDecay}, time: {(T1-T2)*1000}\n")
        # f.write(f"\n")
    
    if best_score < current_score:
        best_score = current_score
        best_para = (learningRate, batchSize, momentum, weightDecay)

print(f"Best Score: {best_score}, Best Parameters: {best_para}")
print("Done!")