import torch
import torch.nn as nn
import torch.optim as optim

from data import train_loader, val_loader, test_loader
from build_model import build_model
from trainer import train_model
from evaluate import evaluate_model
from evaluate import plot_training_curves


# ----------------------
# Device
# ----------------------
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

device = get_device()


# ----------------------
# Choose Model
# ----------------------
# Change this to "resnet" when training ResNet
model_name = "efficientnet"

model = build_model(device, model_name)


# ----------------------
# Loss & Optimizer
# ----------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)


# ----------------------
# Train
# ----------------------
model, train_losses, val_accuracies = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    model_name,
    epochs=20,
    patience=3
)


# ----------------------
# Evaluate (includes ROC)
# ----------------------
evaluate_model(model, test_loader, device, threshold=0.45)


# ----------------------
# Training Curves
# ----------------------
plot_training_curves(train_losses, val_accuracies)