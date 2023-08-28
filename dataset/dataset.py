import pandas as pd
import numpy as np
from PIL import Image
import os

import torch
from torch.utils.data import Dataset


class SkinDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        assert self.mode in ["train", "valid", "test"]
        self.transform = transform
        csv_file = os.path.join(root, "ISIC_2019_Training_GroundTruth.csv")
        self.file = pd.read_csv(csv_file)

        self.images = self.file["image"].values
        self.labels = self.file.iloc[:, 1:].values.astype("int")
        self.targets = np.argmax(self.labels, axis=1)

        initial_len = len(self.images)

        # data split
        np.random.seed(0)
        idxs = np.random.permutation(initial_len)
        self.images = self.images[idxs]
        self.targets = self.targets[idxs]

        if self.mode == "train":
            self.images = self.images[:int(0.7*initial_len)]
            self.targets = self.targets[:int(0.7*initial_len)]
        elif self.mode == "valid":
            self.images = self.images[int(
                0.7*initial_len):int(0.8*initial_len)]
            self.targets = self.targets[int(
                0.7*initial_len):int(0.8*initial_len)]
        else:
            self.images = self.images[int(0.8*initial_len):]
            self.targets = self.targets[int(0.8*initial_len):]

        self.n_classes = len(np.unique(self.targets))
        assert self.n_classes == 8

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = os.path.join(
            self.root, "ISIC_2019_Training_Input", self.images[index] + ".jpg")
        img = Image.open(image_name).convert("RGB")
        label = self.targets[index]
        if self.transform is not None:
            if not isinstance(self.transform, list):
                img = self.transform(img)
            else:
                img0 = self.transform[0](img)
                img1 = self.transform[1](img)
                img = [img0, img1]
        return img, label

    def __len__(self):
        return len(self.images)
    

class ichDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        assert self.mode in ["train", "valid", "test"]
        self.transform = transform
        self.data = pd.read_csv(os.path.join(self.root, "data.csv"))
        self.image_id = self.data["id"].to_numpy()
        self.targets = self.data["class"].to_numpy()
        initial_len = len(self.targets)

        # split
        np.random.seed(0)
        idxs = np.random.permutation(len(self.targets))
        self.image_id = self.image_id[idxs]
        self.targets = self.targets[idxs]

        if self.mode == "train":
            self.image_id = self.image_id[:int(0.7*initial_len)]
            self.targets = self.targets[:int(0.7*initial_len)]
        elif self.mode == "valid":
            self.image_id = self.image_id[int(0.7*initial_len):int(0.8*initial_len)]
            self.targets = self.targets[int(0.7*initial_len):int(0.8*initial_len)]
        else:
            self.image_id = self.image_id[int(0.8*initial_len):]
            self.targets = self.targets[int(0.8*initial_len):]

        self.n_classes = len(np.unique(self.targets))
        assert self.n_classes == 5

    def __getitem__(self, index):
        id, target = self.image_id[index], self.targets[index]
        img = self.read_image(id)

        if self.transform is not None:
            if not isinstance(self.transform, list):
                img = self.transform(img)
            else:
                img0 = self.transform[0](img)
                img1 = self.transform[1](img)
                img = [img0, img1]
        return img, target
    
    def __len__(self):
        return len(self.targets)

    def read_image(self, id):
        image_path = os.path.join(self.root, "stage_1_train_images", id+".png")
        image = Image.open(image_path).convert("RGB")
        return image
