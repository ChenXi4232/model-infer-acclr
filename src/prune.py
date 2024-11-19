import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch
import numpy as np
import os
import sys
from torchsummary import summary
from module.transforms import Cutout
from network.ResNet import ResNet11
import torch.nn.utils.prune as prune
import logging

from copy import deepcopy

output_dir = os.path.join("/home/v-chenwang6/model-infer-acclr/outputs", "CIFAR10", "ResNet11", "prune")
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

num_epochs = 64
batch_size = 256
learning_rate = 0.01
seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        Cutout(8, 8),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

val_test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_val_dataset = torchvision.datasets.CIFAR10(
    root="/home/v-chenwang6/model-infer-acclr/data/CIFAR10", train=True, download=True
)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset.dataset.transform = val_test_transform
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = torchvision.datasets.CIFAR10(
    root="/home/v-chenwang6/model-infer-acclr/data/CIFAR10",
    train=False,
    download=True,
    transform=val_test_transform,
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet11(3, 10)
model = model.to(device)
logging.info(summary(model, (3, 32, 32)))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.load_state_dict(
    torch.load(
        "/home/v-chenwang6/model-infer-acclr/outputs/CIFAR10/ResNet11/best_model.pth"
    )
)

print(model)

pruning_model = deepcopy(model)

prune.l1_unstructured(pruning_model.fc[0], name="weight", amount=0.3)
prune.remove(pruning_model.fc[0], "weight")

logging.info(summary(pruning_model, (3, 32, 32)))

pruning_model = pruning_model.to(device)

with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    record_shapes=True,
    with_stack=True,
) as prof:
    pruning_test_loss, pruning_test_acc = test(pruning_model, test_loader, criterion, device)

logging.info(f"Pruning Test Loss: {pruning_test_loss:.4f}, Pruning Test Acc: {pruning_test_acc:.4f}")
logging.info(prof.key_averages().table(sort_by="cpu_time_total"))

with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    record_shapes=True,
    with_stack=True,
) as prof:
    model_test_loss, model_test_acc = test(model, test_loader, criterion, device)
logging.info(f"Model Test Loss: {model_test_loss:.4f}, Model Test Acc: {model_test_acc:.4f}")
logging.info(prof.key_averages().table(sort_by="cpu_time_total"))
