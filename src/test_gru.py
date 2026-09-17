import torch
from torch.utils.data import DataLoader

from dataset_loader import CricketShotDataset
from gru_model import CricketShotGRU


# Device
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
    hidden_size=256,
    num_layers=2,
    num_classes=10,
    dropout=0.3
)


# Load trained model
model.load_state_dict(
    torch.load(
     "models/cricket_shot_gru_best.pth",
    map_location=device
    )
)

model = model.to(device)
model.eval()


# Test
correct = 0
total = 0


with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = 100.0 * correct / total


print("\nTest Results")
print("------------")
print("Total test samples:", total)
print("Correct predictions:", correct)
print("Wrong predictions:", total - correct)
print(f"Test Accuracy: {accuracy:.2f}%")
