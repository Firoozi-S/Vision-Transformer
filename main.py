import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

import globals
import model
import utils

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load the training data
trainset = CIFAR100(root = './data', train = True, download = True, transform = transform)
trainloader = DataLoader(trainset, batch_size = globals.BATCH_SIZE, shuffle = True, num_workers = 0)

# Download and load the test data
testset = CIFAR100(root = './data', train = False, download = True, transform = transform)
testloader = DataLoader(testset, batch_size = globals.BATCH_SIZE, shuffle = False, num_workers = 0)

vit = model.VisionTransformer()
optimizer = torch.optim.Adam(vit.parameters(), lr = globals.LEARNING_RATE)

for epoch in range(globals.EPOCHS):
    train_loss = utils.train(vit, trainloader, optimizer)
    test_loss, test_acc = utils.evaluate(vit, testloader)

    print(f'Epoch {epoch + 1}/{globals.EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
