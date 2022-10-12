import os
import dataclasses

import torch

from src.s01_data import datasets
from src.s03_modelling import models
from src.s04_training import runners


@dataclasses.dataclass
class Config:
    runner: runners.CustomKFoldRunner
    data: datasets.CustomImageDataset
    model: models.CustomConvNet
    optimizer: torch.optim.Optimizer
    loss_function: torch.nn.Module
    accuracy_function: torch.nn.Module

    @classmethod
    def from_file(cls, file_path):
        """
        Loads the config object at the given file path via torch and returns it.
        """
        config = torch.load(file_path)
        return config

    def to_file(self, output_dir):
        """
        Saves this config object in the given output directory via torch.
        """
        # Create necessary directories
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save this config object via torch
        output_path = os.path.join(output_dir, "config.pt")
        torch.save(self, output_path)

    def as_table(self):
        """
        Creates a Markdown table presenting the config parameters.
        Used to display in tensorboard.
        """
        # Transform the config to a dictionary
        config_dict = dataclasses.asdict(self)

        # Initialize Markdown table
        text = "| Parameter | Value |  \n" \
               "| --------- | ----- |  \n"

        # Parse all parameters
        for key, value in config_dict.items():
            if isinstance(value, datasets.CustomImageDataset):
                text += f"| image_dir | {value.image_dir} |  \n"
                text += f"| label_path | {value.label_path} |  \n"
                transform_string = str(value.transform).replace("\n", "<br />&nbsp;&nbsp;")
                text += f"| transform | {transform_string} |  \n"
            else:
                text += f"| {key} | {value} |  \n"

        return text
