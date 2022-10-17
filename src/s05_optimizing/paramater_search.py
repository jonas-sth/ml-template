"""This module contains methods to compare different configs used within the training."""

import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter

from src.s00_utils import constants
from src.s02_customizing import datasets, models, weight_inits, trainers


def tune_sgd_optimizer(file_path, dir_path, clear_dir=True):
    # Clean up
    if clear_dir:
        shutil.rmtree(dir_path, ignore_errors=True)

    # Set parameters to test
    possible_learning_rates = [0.1, 0.01, 0.001]
    possible_moments = [0.1, 0.01, 0.001]
    possible_weight_decays = [0.1, 0.01, 0.001]

    total_possibilities = len(possible_learning_rates) * len(possible_moments) * len(possible_weight_decays)

    # Initialize logging
    comparison_writer = SummaryWriter(dir_path)

    # Iterate over possible hyper parameters
    i = 1
    for lr in possible_learning_rates:
        for momentum in possible_moments:
            for decay in possible_weight_decays:
                # Load trainer
                trainer = trainers.CustomKFoldTrainer.from_file(file_path)

                # Replace optimizer
                trainer.optimizer = torch.optim.SGD(params=trainer.model.parameters(),
                                                    lr=lr,
                                                    momentum=momentum,
                                                    weight_decay=decay)

                # Set output directory
                exp_dir = os.path.join(dir_path, f"exp_{i}")

                # Run training
                avg, std = trainer.k_fold_cross_validation(exp_dir, run_info=f"{i}/{total_possibilities}")

                # Log to tensorboard
                comparison_writer.add_hparams(
                    hparam_dict={
                        "learning rate": lr,
                        "momentum": momentum,
                        "weight decay": decay
                    },
                    metric_dict={
                        "accuracy": avg
                    },
                    hparam_domain_discrete={
                        "learning rate": possible_learning_rates,
                        "momentum": possible_moments,
                        "weight decay": possible_weight_decays
                    },
                    run_name=f"run_{i}"
                )

                # Increase counter
                i += 1

    comparison_writer.close()


if __name__ == "__main__":
    config_path = os.path.join(constants.ROOT, r"models\exp\config.pt")
    output_dir = os.path.join(constants.ROOT, r"models\hyper_search")
    tune_sgd_optimizer(config_path, output_dir)
