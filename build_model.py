import torch.nn as nn
from torchvision import models

def build_model(device, model_name="resnet"):

    if model_name == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)

        # Freeze all
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in model.layer4.parameters():
            param.requires_grad = True

        # Unfreeze fc
        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 2)

        # Freeze all
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last 3 feature blocks
        for param in model.features[-1].parameters():
            param.requires_grad = True
        for param in model.features[-2].parameters():
            param.requires_grad = True
        for param in model.features[-3].parameters():
            param.requires_grad = True

        # Unfreeze classifier
        for param in model.classifier.parameters():
            param.requires_grad = True

    else:
        raise ValueError("Invalid model_name. Choose 'resnet' or 'efficientnet'.")

    model = model.to(device)
    return model