"""This module contains the config logic which combines all custom components that are used within the training."""

import inspect
import os
import dataclasses
from typing import Callable

import torch
import torchmetrics
from torch.utils.data import Dataset
from torch.utils import tensorboard
from torchvision.transforms import transforms

from src.s00_utils import constants
from src.s02_customizing import datasets, models, weight_inits, runners


@dataclasses.dataclass
class Config:
    """
    Combines all custom components that are used within the training.
    A config can be saved to a file and loaded again.
    A config can be logged to a tensorboard for better visualization.
    """
    runner: runners.CustomKFoldRunner
    data: torch.utils.data.Dataset
    model: torch.nn.Module
    weight_init: Callable
    optimizer: torch.optim.Optimizer
    loss_function: torch.nn.Module
    accuracy_function: torch.nn.Module

    def to_file(self, dir_path):
        """
        Saves this config object in the given directory via torch.
        """
        # Create necessary directories
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save this config object via torch
        output_path = os.path.join(dir_path, "config.pt")
        torch.save(self, output_path)

    def to_tensorboard(self, writer: tensorboard.writer.SummaryWriter) -> None:
        """
        Writes the config as formatted Markdown text to the given tensorboard writer.
        """
        # Add every component of this config with a unique tag to tensorboard as Markdown text.
        writer.add_text(tag="Runner", text_string=_markdown(str(self.runner)))
        writer.add_text(tag="Dataset", text_string=_markdown(str(self.data)))
        writer.add_text(tag="Model", text_string=_markdown(str(self.model)))
        writer.add_text(tag="Weight Initialization", text_string=_markdown(inspect.getsource(self.weight_init)))
        writer.add_text(tag="Optimizer", text_string=_markdown(str(self.optimizer)))
        writer.add_text(tag="Loss Function", text_string=_markdown(str(self.loss_function)))
        writer.add_text(tag="Accuracy Function", text_string=_markdown(str(self.accuracy_function)))

        # Get one sample of the data with the right dimensions and use it to log the graph to tensorboard
        data_sample = self.data[0][0][None, :]
        writer.add_graph(self.model, data_sample)


def _markdown(string: str):
    """
    Formats linebreaks and indentation to Markdown format.
    """
    formatted_string = string.replace(" ", "&nbsp;&nbsp;").replace("\n", "<br />")
    return formatted_string


def load_config(file_path):
    """
    Loads a config object at the given file path via torch and returns it.
    """
    config = torch.load(file_path)
    return config


def create_config():
    """
    Creates a config object with the given parameters.
    Use this method to create new configurations.
    """
    # Initialize runner
    runner = runners.CustomKFoldRunner(num_folds=5,
                                       num_epochs=20,
                                       batch_size=64,
                                       seed=777,
                                       device=torch.device("cpu")
                                       )

    # Initialize data
    data = datasets.CustomImageDataset(image_dir=os.path.join(constants.ROOT, r"data\d01_raw\mnist\train"),
                                       label_path=os.path.join(constants.ROOT, r"data\d01_raw\mnist\train\labels.csv"),
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Grayscale(),
                                                                     transforms.Normalize(0.5, 0.2)]
                                                                    )
                                       )

    # Initialize model
    model = models.CustomConvNetSequential()

    # Set weight initialization
    weight_init = weight_inits.xavier_weight_init

    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.001)

    # Initialize loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # Initialize accuracy function
    accuracy_function = torchmetrics.Accuracy()

    # Combine into one config
    config = Config(runner=runner,
                    data=data,
                    model=model,
                    weight_init=weight_init,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    accuracy_function=accuracy_function
                    )

    return config
