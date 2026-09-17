import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CricketShotDataset(Dataset):

    def __init__(self, features_dir):

        self.samples = []

        self.class_names = sorted([
            name for name in os.listdir(features_dir)
            if os.path.isdir(
                os.path.join(features_dir, name)
            )
        ])

        self.class_to_index = {
            class_name: index
            for index, class_name in enumerate(self.class_names)
        }

        for class_name in self.class_names:

            class_folder = os.path.join(
                features_dir,
                class_name
            )

            for filename in os.listdir(class_folder):

                if filename.endswith(".npy"):

                    file_path = os.path.join(
                        class_folder,
                        filename
                    )

                    label = self.class_to_index[class_name]

                    self.samples.append(
                        (file_path, label)
                    )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        file_path, label = self.samples[index]

        features = np.load(file_path)

        features = torch.tensor(
            features,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return features, label


if __name__ == "__main__":

    dataset = CricketShotDataset(
        "features/train"
    )

    print("Number of samples:", len(dataset))
    print("Classes:", dataset.class_names)
    print("Class to index:", dataset.class_to_index)

    features, label = dataset[0]

    print("One sample shape:", features.shape)
    print("One label:", label.item())
