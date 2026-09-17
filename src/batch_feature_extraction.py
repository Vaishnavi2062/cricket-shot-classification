import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from preprocess import load_video_frames


# --------------------------------------------------
# 1. Select CPU or GPU
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# 2. Load EfficientNet-B0
# --------------------------------------------------

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)

# Remove the final classification layer
model.classifier = nn.Identity()

model = model.to(device)
model.eval()

print("EfficientNet-B0 loaded successfully!")


# --------------------------------------------------
# 3. Preprocessing transformation
# --------------------------------------------------

transform = weights.transforms()


# --------------------------------------------------
# 4. Dataset folders
# --------------------------------------------------

dataset_root = "dataset"

splits = ["train", "val", "test"]


# --------------------------------------------------
# 5. Output folder for extracted features
# --------------------------------------------------

feature_root = "features"

os.makedirs(feature_root, exist_ok=True)


# --------------------------------------------------
# 6. Process each dataset split
# --------------------------------------------------

for split in splits:

    split_path = os.path.join(dataset_root, split)

    print("\n====================================")
    print("Processing:", split)
    print("====================================")

    # Get class folders
    classes = sorted(
        [
            folder
            for folder in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, folder))
        ]
    )

    print("Classes:", classes)

    # Process every class
    for class_name in classes:

        class_path = os.path.join(
            split_path,
            class_name
        )

        # Create corresponding feature folder
        output_class_path = os.path.join(
            feature_root,
            split,
            class_name
        )

        os.makedirs(
            output_class_path,
            exist_ok=True
        )

        # Find all AVI videos
        videos = sorted(
            [
                file
                for file in os.listdir(class_path)
                if file.lower().endswith(".avi")
            ]
        )

        print(
            f"\nClass: {class_name} | Videos: {len(videos)}"
        )

        # Process videos
        for count, video_name in enumerate(videos, start=1):

            video_path = os.path.join(
                class_path,
                video_name
            )

            try:

                # ----------------------------------
                # Extract 24 frames
                # ----------------------------------

                frames = load_video_frames(
                    video_path,
                    num_frames=24,
                    save_frames=False
                )

                # Shape:
                # (24, 224, 224, 3)

                frames_tensor = torch.from_numpy(
                    frames
                )

                # Convert:
                # (24, 224, 224, 3)
                #       ↓
                # (24, 3, 224, 224)

                frames_tensor = frames_tensor.permute(
                    0, 3, 1, 2
                )

                # Apply EfficientNet preprocessing
                frames_tensor = torch.stack(
                    [
                        transform(frame)
                        for frame in frames_tensor
                    ]
                )

                frames_tensor = frames_tensor.to(device)

                # ----------------------------------
                # EfficientNet-B0 feature extraction
                # ----------------------------------

                with torch.no_grad():

                    features = model(
                        frames_tensor
                    )

                # Shape:
                # (24, 1280)

                features = features.cpu().numpy()

                # ----------------------------------
                # Save features
                # ----------------------------------

                video_base_name = os.path.splitext(
                    video_name
                )[0]

                output_path = os.path.join(
                    output_class_path,
                    video_base_name + ".npy"
                )

                np.save(
                    output_path,
                    features
                )

                print(
                    f"[{count}/{len(videos)}] "
                    f"{video_name} → "
                    f"{features.shape}"
                )

            except Exception as e:

                print(
                    f"ERROR: {video_name} → {e}"
                )


print("\n====================================")
print("FEATURE EXTRACTION COMPLETED!")
print("====================================")

