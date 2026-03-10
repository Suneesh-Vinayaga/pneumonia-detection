import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import os


def evaluate_model(model, test_loader, device, threshold=0.45):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            pneumonia_probs = probs[:, 1]
            predicted = (pneumonia_probs >= threshold).long()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(pneumonia_probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Accuracy
    test_acc = 100 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nTest Accuracy: {test_acc:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # AUC
    auc_score = roc_auc_score(all_labels, all_probs)
    print(f"AUC Score: {auc_score:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    os.makedirs("outputs", exist_ok=True)

    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=300)
    plt.show()
    plt.close()

    print("ROC curve saved to outputs/roc_curve.png")


def plot_training_curves(train_losses, val_accuracies):

    epochs = range(1, len(train_losses) + 1)

    plt.figure()

    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.title("Training vs Validation Performance")
    plt.legend()

    os.makedirs("outputs", exist_ok=True)

    plt.tight_layout()
    plt.savefig("outputs/training_curve.png", dpi=300)
    plt.show()
    plt.close()

    print("Training curve saved to outputs/training_curve.png")