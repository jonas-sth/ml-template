import os

import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from src.s00_utils import constants


class CustomImageDataset(Dataset):
    """
    Custom Dataset to be used with torch Dataloader.
    Expects the label.csv to contain a file_name and label for each image,
    where the image is stored at image_dir/file_name.
    """
    def __init__(self, image_dir, label_path, image_transform=None, label_transform=None):
        self.label_path = label_path
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.label_transform = label_transform

        self.image_labels = pd.read_csv(label_path)

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_labels.iloc[idx, 0])
        image = cv2.imread(image_path)
        label = self.image_labels.iloc[idx, 1]
        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label


# Define custom datasets
MNIST_Dataset = CustomImageDataset(
    image_dir=os.path.join(constants.ROOT, r"data\d01_raw\mnist\train"),
    label_path=os.path.join(constants.ROOT, r"data\d01_raw\mnist\train\labels.csv"),
    image_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize(0.5, 0.2),
    ])
)
