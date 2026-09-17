import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import CricketShotDataset
from gru_model import CricketShotGRU


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------
# Datasets
# -------------------------

train_dataset = CricketShotDataset(
    "features/train"
)

val_dataset = CricketShotDataset(
    "features/val"
)


# -------------------------
# DataLoaders
# -------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)


# -------------------------
# Model
# -------------------------

model = CricketShotGRU(
    input_size=1280,
    hidden_size=512,
    num_layers=2,
    num_classes=10,
    dropout=0.3
)

model = model.to(device)


# -------------------------
# Loss and optimizer
# -------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


# Reduce learning rate when validation loss stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# -------------------------
# Training settings
# -------------------------

num_epochs = 10

best_val_accuracy = 0.0
patience = 5
epochs_without_improvement = 0

os.makedirs("models", exist_ok=True)

best_model_path = "models/cricket_shot_gru_24frames.pth"

print("\nStarting improved training...\n")


# -------------------------
# Training loop
# -------------------------

for epoch in range(num_epochs):

    # =====================
    # Training
    # =====================

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = (
        100.0 * train_correct / train_total
    )

    average_train_loss = (
        train_loss / len(train_loader)
    )


    # =====================
    # Validation
    # =====================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for features, labels in val_loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = (
        100.0 * val_correct / val_total
    )

    average_val_loss = (
        val_loss / len(val_loader)
    )


    # Update learning rate
    scheduler.step(
        average_val_loss
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # =====================
    # Save best model
    # =====================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

        saved_text = " ? BEST MODEL"

    else:

        epochs_without_improvement += 1

        saved_text = ""


    # =====================
    # Print results
    # =====================

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"| Train Loss: {average_train_loss:.4f} "
        f"| Train Accuracy: {train_accuracy:.2f}% "
        f"| Val Loss: {average_val_loss:.4f} "
        f"| Val Accuracy: {val_accuracy:.2f}% "
        f"| LR: {current_lr:.6f}"
        f"{saved_text}"
    )


    # =====================
    # Early stopping
    # =====================

    if epochs_without_improvement >= patience:

        print(
            "\nEarly stopping triggered."
        )

        break


print("\nImproved training completed!")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Best model saved to:",
    best_model_path
)
