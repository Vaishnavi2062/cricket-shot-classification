import torch
from torch.utils.data import DataLoader

from dataset_loader import CricketShotDataset
from gru_model import CricketShotGRU


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Load test dataset
test_dataset = CricketShotDataset(
    "features/test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# Create model
model = CricketShotGRU(
    input_size=1280,
    hidden_size=512,
    num_layers=2,
    num_classes=10,
    dropout=0.3
)


# Load trained model
model.load_state_dict(
    torch.load(
        "models/cricket_shot_gru_24frames.pth",
        map_location=device
    )
)


# Move model to device
model = model.to(device)

# Set model to evaluation mode
model.eval()


# Store predictions and actual labels
all_predictions = []
all_labels = []


# Test the model
with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# Calculate accuracy
correct = sum(
    p == l
    for p, l in zip(
        all_predictions,
        all_labels
    )
)

total = len(all_labels)

accuracy = 100.0 * correct / total


# Display test results
print("\nTest Results")
print("------------")

print(
    "Total test samples:",
    total
)

print(
    "Correct predictions:",
    correct
)

print(
    "Wrong predictions:",
    total - correct
)

print(
    f"Test Accuracy: {accuracy:.2f}%"
)

