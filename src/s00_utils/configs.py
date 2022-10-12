import os
import dataclasses

import torch
from torch.utils import tensorboard

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
        writer.add_text(tag="Runner", text_string=_markdown(str(self.runner)))
        writer.add_text(tag="Dataset", text_string=_markdown(str(self.data)))
        writer.add_text(tag="Model", text_string=_markdown(str(self.model)))
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
