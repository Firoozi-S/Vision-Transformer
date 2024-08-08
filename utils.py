import torch
import torch.nn as nn

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        break

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(input = outputs, dim = 1)
            correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)