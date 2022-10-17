"""This module is used to conduct experiments."""

import os

import torch
import torchmetrics
from torchvision.transforms import transforms

from src.s00_utils import constants
from src.s02_customizing import datasets, models, weight_inits, trainers


def run(dir_path, clear_dir=True):
    """
    Runs an experiment with the specified parameters.
    Use this method to create new configurations.
    """
    # Initialize data
    data = datasets.CustomImageDataset(image_dir=os.path.join(constants.ROOT, r"data\d01_raw\mnist\train"),
                                       label_path=os.path.join(constants.ROOT, r"data\d01_raw\mnist\train\labels.csv"),
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Grayscale(),
                                                                     transforms.Normalize(0.5, 0.2)]
                                                                    )
                                       )

    # Initialize model
    model = models.CustomConvNet()

    # Set weight initialization
    weight_init = weight_inits.xavier_weight_init

    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.001)

    # Initialize loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # Initialize accuracy function
    accuracy_function = torchmetrics.Accuracy()

    # Initialize trainer
    trainer = trainers.CustomKFoldTrainer(
        num_folds=5,
        num_epochs=10,
        batch_size=64,
        device=torch.device("cpu"),
        data=data,
        model=model,
        weight_init=weight_init,
        optimizer=optimizer,
        loss_function=loss_function,
        accuracy_function=accuracy_function,
    )

    # Run experiment
    trainer.k_fold_cross_validation(dir_path, clear_dir)


def rerun(file_path, dir_path, clear_dir=True):
    """
    Reruns an experiment with the config from the given file path.
    """
    # Load config
    trainer = trainers.CustomKFoldTrainer.from_file(file_path)

    # Run experiment
    trainer.k_fold_cross_validation(dir_path, clear_dir)


if __name__ == "__main__":
    # Set output directory
    output_dir = os.path.join(constants.ROOT, r"models\exp")

    # Run experiment
    run(output_dir)

    # Test to rerun experiment
    # config_path = os.path.join(constants.ROOT, r"models\exp\config.pt")
    # new_output_dir = os.path.join(constants.ROOT, r"models\exp2")
    # rerun(config_path, new_output_dir)


