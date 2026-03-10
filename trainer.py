import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(model, train_loader, val_loader, criterion, optimizer, device, model_name, epochs=20, patience=3):
    os.makedirs("models", exist_ok=True)

    best_val_acc = 0
    counter = 0

    train_losses = []
    val_accuracies = []

    # ✅ Create scheduler ONCE (epoch level, not batch level)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',      # because we monitor validation accuracy
        factor=0.5,      # reduce LR by half
        patience=1,      # wait 1 epoch before reducing
        verbose=True
    )

    print("Starting training...")

    for epoch in range(epochs):

        # --------------------
        # TRAINING
        # --------------------
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {running_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.2f}%")

        train_losses.append(running_loss)

        # --------------------
        # VALIDATION
        # --------------------
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        val_accuracies.append(val_acc)

        # ✅ Step scheduler AFTER validation
        scheduler.step(val_acc)

        # --------------------
        # EARLY STOPPING
        # --------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), f'models/{model_name}_best.pth')
            print("Best model saved.")
        else:
            counter += 1
            print(f"No improvement. Counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered.")
            break

    print("\nTraining Complete")

    model.load_state_dict(torch.load(f"models/{model_name}_best.pth"))
    print("Loaded best model.")

    return model, train_losses, val_accuracies