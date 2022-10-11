import os
import json
import dataclasses
import time
from typing import Callable

import torch
import torchmetrics

from src.s00_utils import constants
from src.s01_data import datasets


@dataclasses.dataclass
class Config:
    num_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    num_folds: int
    device: torch.device
    data: datasets.CustomImageDataset
    loss_function: Callable
    accuracy_function: Callable
    optimizer: Callable
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
            transform_path = config_dict.pop("transform_path", None)
            transform = torch.load(transform_path) if transform_path is not None else None
            data = datasets.CustomImageDataset(image_dir, label_path, transform)
            config_dict["data"] = data

            # Process the device information
            device = config_dict.pop("device")
            config_dict["device"] = torch.device(device)

            # Process the callable methods
            loss_function_path = config_dict.pop("loss_function_path")
            config_dict["loss_function"] = torch.load(loss_function_path)

            accuracy_function_path = config_dict.pop("accuracy_function_path")
            config_dict["accuracy_function"] = torch.load(accuracy_function_path)

            optimizer_path = config_dict.pop("optimizer_path")
            config_dict["optimizer"] = torch.load(optimizer_path)

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

        # Initialize final dictionary
        parsed_dict = {}

        # Parse all parameters
        for key, value in config_dict.items():
            if isinstance(value, datasets.CustomImageDataset):
                # Process the dataset information and save the transform method via torch
                parsed_dict["image_dir"] = value.image_dir
                parsed_dict["label_path"] = value.label_path
                if value.transform is not None:
                    transform_path = os.path.join(self.output_dir, "transform.pt")
                    torch.save(value.transform, transform_path)
                    parsed_dict["transform_path"] = transform_path
            elif isinstance(value, torch.device):
                # Process the device information
                parsed_dict[key] = value.type
            elif callable(value):
                # Process all callable methods by saving them via torch
                output_path = os.path.join(self.output_dir, f"{key}.pt")
                torch.save(value, output_path)
                parsed_dict[f"{key}_path"] = output_path
            else:
                parsed_dict[key] = value

        # Save the dictionary to a json file
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w") as json_file:
            json.dump(parsed_dict, json_file, indent=2)

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


if __name__ == '__main__':
    test_config = Config(
        num_epochs=4,
        batch_size=32,
        learning_rate=0.01,
        momentum=0.5,
        num_folds=10,
        device=torch.device("cpu"),
        data=datasets.MNIST_Dataset,
        loss_function=torch.nn.CrossEntropyLoss,
        accuracy_function=torchmetrics.Accuracy,
        optimizer=torch.optim.SGD,
        output_dir="test",
    )

    print(test_config.loss_function)

    test_config.to_file()

    new_config = Config.from_file("test/config.json")

    print(new_config.loss_function)

    print(new_config)
    print(test_config)
