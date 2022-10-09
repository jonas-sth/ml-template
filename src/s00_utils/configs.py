import os
import json
import dataclasses
import time

import torch

from src.s00_utils import constants
from src.s01_data import datasets


@dataclasses.dataclass
class Config:
    num_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    num_folds: int
    data: datasets.CustomImageDataset
    output_dir: str = os.path.join(constants.ROOT, rf"models\run_{time.strftime('%Y%m%d-%H%M%S')}")

    @classmethod
    def from_file(cls, file_path):
        """
        Loads config parameters from a file and returns a new config object.
        """
        with open(file_path, "r") as json_file:
            # Load the parameters from a json file as a dictionary
            config_dict = json.load(json_file)

            # Process the dataset information
            image_dir = config_dict.pop("image_dir")
            label_path = config_dict.pop("label_path")
            config_dict.pop("transform")
            transform_path = config_dict.pop("transform_path")
            transform = torch.load(transform_path)
            data = datasets.CustomImageDataset(image_dir, label_path, transform)
            config_dict["data"] = data

            # Create the config and return it
            return cls(**config_dict)

    def to_file(self):
        """
        Saves config parameters to a file.
        """
        # Create necessary directories
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Transform the config to a dictionary
        config_dict = dataclasses.asdict(self)

        # Process the dataset information
        config_dict.pop("data")
        config_dict["image_dir"] = self.data.image_dir
        config_dict["label_path"] = self.data.label_path
        if self.data.transform is not None:
            # Save transformation
            transform_path = os.path.join(self.output_dir, "transform.pt")
            torch.save(self.data.transform, transform_path)
            config_dict["transform"] = str(self.data.transform)
            config_dict["transform_path"] = transform_path

        # Save the dictionary to a json file
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w") as json_file:
            json.dump(config_dict, json_file)

    def as_dict(self):
        return dataclasses.asdict(self)
