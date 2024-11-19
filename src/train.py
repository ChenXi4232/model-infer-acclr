import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from network.ResNet import *
from module.transforms import Cutout
import numpy as np
import logging
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from module.metrics import acc
from copy import deepcopy
from torch.utils.data import random_split, DataLoader
import json
import matplotlib.pyplot as plt
from torchsummary import summary

output_dir = os.path.join("/home/v-chenwang6/model-infer-acclr/outputs", "CIFAR10", "ResNet9")
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(output_dir, "ResNet11.log"), mode="w"
        ),
    ],
)
SW = SummaryWriter(output_dir, flush_secs=30)

num_epochs = 64
batch_size = 256
learning_rate = 0.01
seed = 42
grad_clip = 0.12
weight_decay = 1e-4
have_test = False

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

train_val_dataset = torchvision.datasets.CIFAR10(root='/home/v-chenwang6/model-infer-acclr/data/CIFAR10', train=True, download=True)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
# train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
data = train_val_dataset.data/255.0
mean, std = data.mean(axis=(0, 1, 2)), data.std(axis=(0, 1, 2))

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    # Cutout(8, 8),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# train_dataset.dataset.transform = train_transform
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# val_dataset.dataset.transform = val_test_transform
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# test_dataset = torchvision.datasets.CIFAR10(root='/home/v-chenwang6/model-infer-acclr/data/CIFAR10', train=False, download=True, transform=val_test_transform)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_dataset = torchvision.datasets.CIFAR10(root='/home/v-chenwang6/model-infer-acclr/data/CIFAR10', train=True, download=True, transform=train_transform)
val_dataset = torchvision.datasets.CIFAR10(root='/home/v-chenwang6/model-infer-acclr/data/CIFAR10', train=False, download=True, transform=val_test_transform) 

loader_kwargs = {"num_workers": 4, "pin_memory": True}
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet11(3, 10)
model = model.to(device)
logging.info(summary(model, (3, 32, 32)))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_sheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, learning_rate, epochs=num_epochs, steps_per_epoch=len(train_loader)
)

def train(model, data_loader, criterion, device, grad_clip=None, lr_sheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        correct += (labels == outputs.argmax(dim=1)).sum().item()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()
        if lr_sheduler:
            lr_sheduler.step()

        running_loss += loss.item()

    return running_loss / len(data_loader), correct / len(data_loader.dataset)

def validate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            correct += (labels == outputs.argmax(dim=1)).sum().item()
            loss = criterion(outputs, labels)

            running_loss += loss.item()

    return running_loss / len(data_loader), correct / len(data_loader.dataset)

def test(model, data_loader, criterion, device):
    return validate(model, data_loader, criterion, device)

best_acc = 0
best_model = None
best_acc_epoch = 0
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    train_loss, train_acc = train(model, train_loader, criterion, device, grad_clip, lr_sheduler)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    epoch_elapsed_time = time.time() - epoch_start_time  # Time taken for the current epoch
    remaining_time = (epoch_elapsed_time * (num_epochs - (epoch + 1)))  # Estimate remaining time
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
        SW.add_scalar("Learning Rate", cur_lr, epoch)
    logging.info(f"Epoch: {epoch + 1}/{num_epochs}, " +
                 f"Train Loss: {train_loss:.4f}, " +
                 f"Train Acc: {train_acc * 100:.2f}%, " +
                 f"Val Loss: {val_loss:.4f}, " +
                 f"Val Acc: {val_acc * 100:.2f}%, " +
                 f"LR: {cur_lr:.4f}, " +
                 f"Elapsed Time: {epoch_elapsed_time:.2f}s, " +
                 f"Remaining Time: {remaining_time:.2f}s")
    SW.add_scalars("Loss", {"train": train_loss, "valid": val_loss}, epoch)
    SW.add_scalars("Accuracy", {"train": train_acc, "valid": val_acc}, epoch)

    # lr_sheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        del best_model
        best_model = deepcopy(model)
        best_acc_epoch = epoch

elapsed_time = time.time() - start_time
logging.info(f"Training Time: {elapsed_time:.2f}s")
logging.info(f"Best validation accuracy: {best_acc * 100:.2f}% at epoch: {best_acc_epoch}")
torch.save(best_model.state_dict(), os.path.join(output_dir, "best_model.pth"))

SW.close()


if have_test:
    model.load_state_dict(best_model.state_dict())
    infer_start_time = time.time()
    test_loss, test_acc = test(model, test_loader, criterion, device)
    infer_elapsed_time = time.time() - infer_start_time
    infer_elapsed_time_per_sample = infer_elapsed_time / len(test_loader.dataset)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc * 100:.2f}%")
    logging.info(f"Total Inference Time: {infer_elapsed_time:.2f}s, Inference Time Per Sample: {infer_elapsed_time_per_sample:.6f}s")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({"test_loss": test_loss, "test_acc": test_acc}, f)
