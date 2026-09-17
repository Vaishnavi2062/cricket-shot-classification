import torch
from torch.utils.data import DataLoader

from dataset_loader import CricketShotDataset


train_dataset = CricketShotDataset("features/train")
val_dataset = CricketShotDataset("features/val")
test_dataset = CricketShotDataset("features/test")


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

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

print("Number of train batches:", len(train_loader))
print("Number of validation batches:", len(val_loader))
print("Number of test batches:", len(test_loader))


features, labels = next(iter(train_loader))

print("Batch feature shape:", features.shape)
print("Batch label shape:", labels.shape)
print("DataLoader test successful!")
