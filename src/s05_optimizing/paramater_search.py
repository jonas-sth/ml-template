"""This module contains methods to compare different configs used within the training."""

import os

import torch
import torchmetrics
from torchvision.transforms import transforms

from src.s00_utils import constants
from src.s02_customizing import runners, datasets, models, weight_inits
from src.s03_prototyping import configs
from src.s04_training import train


def complete_search():
    """
    An example where every part of the config is varied and all combinations are compared.
    """

    # Define variations for the runner (mainly batch size)
    possible_runners = []
    for num_folds in [2]:
        for num_epochs in [5]:
            for batch_size in [32, 64, 128]:
                for seed in [777]:
                    possible_runners.append(runners.CustomKFoldRunner(num_folds=num_folds,
                                                                      num_epochs=num_epochs,
                                                                      batch_size=batch_size,
                                                                      seed=seed,
                                                                      device=torch.device("cpu")
                                                                      )
                                            )

    # Define variations for the dataset (different data and label combinations or applied transforms)
    possible_datasets = [
        datasets.CustomImageDataset(image_dir=os.path.join(constants.ROOT, r"data\d01_raw\mnist\train"),
                                    label_path=os.path.join(constants.ROOT, r"data\d01_raw\mnist\train\labels.csv"),
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Grayscale(),
                                                                  transforms.Normalize(0.5, 0.2),
                                                                  transforms.GaussianBlur(3)]
                                                                 )
                                    ),
        datasets.CustomImageDataset(image_dir=os.path.join(constants.ROOT, r"data\d01_raw\mnist\train"),
                                    label_path=os.path.join(constants.ROOT, r"data\d01_raw\mnist\train\labels.csv"),
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Grayscale(),
                                                                  transforms.Normalize(0.5, 0.2)]
                                                                 )
                                    )
    ]

    # Define variations for the model (entirely different models)
    possible_models = [
        models.CustomConvNetSequential(),
        models.CustomConvNet()
    ]

    # Define variations for the weight initialization (different methods)
    possible_weight_inits = [
        weight_inits.xavier_weight_init
    ]

    # Define variations for the optimizer (different optimizer or different parameters)
    # Combine as tuple with model
    possible_optimizers = []
    for lr in [0.1, 0.01]:
        for momentum in [0.01, 0.001]:
            for model in possible_models:
                possible_optimizers.append((model, torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)))

    # Define variations for the loss function
    possible_loss_functions = [
        torch.nn.CrossEntropyLoss()
    ]

    # Define variations for the accuracy function
    possible_accuracy_function = [
        torchmetrics.Accuracy()
    ]

    # Train all combinations
    i = 0
    # all_scores = []
    # all_configs = []
    for runner in possible_runners:
        for data in possible_datasets:
            for weight_init in possible_weight_inits:
                for model, optimizer in possible_optimizers:
                    for loss_function in possible_loss_functions:
                        for accuracy_function in possible_accuracy_function:
                            config = configs.Config(runner=runner,
                                                    data=data,
                                                    model=model,
                                                    weight_init=weight_init,
                                                    optimizer=optimizer,
                                                    loss_function=loss_function,
                                                    accuracy_function=accuracy_function
                                                    )
                            output_dir = os.path.join(constants.ROOT, rf"models\hyper_search\run_{i}")
                            train.k_fold_cross_validation(config, output_dir)
                            # all_scores.append(train.k_fold_cross_validation(config, output_dir))
                            # all_configs.append(config)
                            i += 1

    # best_run = np.argmax(all_scores)
    # print(f"Best run was: {best_run}")
    # print(f"Used config: {all_configs}")


if __name__ == '__main__':
    complete_search()
