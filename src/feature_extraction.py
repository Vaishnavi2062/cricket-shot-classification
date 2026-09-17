import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Import our preprocessing function
from preprocess import load_video_frames


# --------------------------------------------------
# 1. Select device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# --------------------------------------------------
# 2. Load pre-trained EfficientNet-B0
# --------------------------------------------------

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)


# --------------------------------------------------
# 3. Remove the final classification layer
# --------------------------------------------------

# EfficientNet-B0 classifier normally produces
# 1000 ImageNet class predictions.
#
# We remove it because we need feature vectors,
# not ImageNet predictions.

model.classifier = nn.Identity()

model = model.to(device)

model.eval()


# --------------------------------------------------
# 4. Preprocessing transform required by EfficientNet
# --------------------------------------------------

transform = weights.transforms()


# --------------------------------------------------
# 5. Load video frames
# --------------------------------------------------

video_path = "dataset/train/pull/pull_0001.avi"

frames = load_video_frames(video_path)

print("Preprocessed frames shape:", frames.shape)


# --------------------------------------------------
# 6. Convert NumPy frames to PyTorch tensor
# --------------------------------------------------

frames_tensor = torch.from_numpy(frames)


# Current shape:
# (16, 224, 224, 3)

# Change to:
# (16, 3, 224, 224)

frames_tensor = frames_tensor.permute(0, 3, 1, 2)


# --------------------------------------------------
# 7. Apply EfficientNet preprocessing
# --------------------------------------------------

frames_tensor = torch.stack(
    [transform(frame) for frame in frames_tensor]
)


frames_tensor = frames_tensor.to(device)


# --------------------------------------------------
# 8. Extract features
# --------------------------------------------------

with torch.no_grad():

    features = model(frames_tensor)


# --------------------------------------------------
# 9. Print results
# --------------------------------------------------

print("Feature extraction successful!")

print("Feature shape:", features.shape)

print("Data type:", features.dtype)

print("Minimum:", features.min().item())

print("Maximum:", features.max().item())
