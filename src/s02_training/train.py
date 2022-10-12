import os
import random
from time import sleep

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.s00_utils import constants
from src.s02_training import configs


def train(writer: tensorboard.writer.SummaryWriter,
          train_loader: torch.utils.data.dataloader.DataLoader,
          epoch: int,
          config: configs.Config) -> None:
    """
    Trains one epoch.
    """
    config.model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        # Send data to GPU
        data = data.to(config.runner.device)
        target = target.to(config.runner.device)

        # Reset optimizer
        config.optimizer.zero_grad()

        # Calculate loss and update weights accordingly
        output = config.model(data)
        loss = config.loss_function(output, target)
        loss.backward()
        config.optimizer.step()

        # Calculate accuracy
        acc_function = config.accuracy_function.to(config.runner.device)
        acc = acc_function(output, target)

        # Log to tensorboard
        writer.add_scalar("Loss/Train", loss.item(), (batch_idx + epoch * len(train_loader)))
        writer.add_scalar("Accuracy/Train", acc.item(), (batch_idx + epoch * len(train_loader)))


def validate(writer: tensorboard.writer.SummaryWriter,
             val_loader: torch.utils.data.dataloader.DataLoader,
             epoch: int,
             config: configs.Config) -> (float, float):
    """
    Validates one epoch.
    """
    config.model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc="Validating")):
            # Send data to GPU
            data = data.to(config.runner.device)
            target = target.to(config.runner.device)

            # Calculate loss
            output = config.model(data)
            loss = config.loss_function(output, target)

            # Calculate accuracy
            acc_function = config.accuracy_function.to(config.runner.device)
            acc = acc_function(output, target)

            # Log to tensorboard
            writer.add_scalar("Loss/Validation", loss.item(), (batch_idx + epoch * len(val_loader)))
            writer.add_scalar("Accuracy/Validation", acc.item(), (batch_idx + epoch * len(val_loader)))

            # Sum up scores
            val_loss += loss.item()
            val_acc += acc.item()

    # Return average scores
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    return avg_val_loss, avg_val_acc


def k_fold_cross_validation(config: configs.Config, dir_path: str) -> None:
    # Initialize logging
    summary_dir = os.path.join(dir_path, "summary")
    summary_writer = tensorboard.SummaryWriter(summary_dir)
    config.to_tensorboard(summary_writer)
    config.to_file(dir_path)

    # Set seeds
    torch.manual_seed(config.runner.seed)
    random.seed(config.runner.seed)
    np.random.seed(config.runner.seed)

    # Move model to device
    config.model.to(config.runner.device)

    # Initialize folds and scores
    k_fold = KFold(n_splits=config.runner.num_folds, shuffle=True)
    scores_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(config.data), start=1):
        # Initialize logging of fold
        fold_dir = os.path.join(dir_path, f"fold_{fold}")
        fold_writer = tensorboard.SummaryWriter(fold_dir)

        # Initialize data loader
        train_subset = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subset = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(dataset=config.data,
                                                   batch_size=config.runner.batch_size,
                                                   sampler=train_subset)
        val_loader = torch.utils.data.DataLoader(dataset=config.data,
                                                 batch_size=config.runner.batch_size,
                                                 sampler=val_subset)

        # Reset model weights and scores
        config.model.apply(config.weight_init)
        best_val_acc = 0

        # Train and validate
        for epoch in range(1, config.runner.num_epochs + 1):
            sleep(0.25)
            print(f"Fold {fold}/{config.runner.num_folds}, Epoch {epoch}/{config.runner.num_epochs}:")
            sleep(0.25)

            train(fold_writer, train_loader, epoch, config)
            val_loss, val_acc = validate(fold_writer, val_loader, epoch, config)

            # Track the best performance, and save the model's state
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(fold_dir, "best_model.pth")
                torch.save({"epoch": epoch,
                            "model_state_dict": config.model.state_dict(),
                            "optimizer_state_dict": config.optimizer.state_dict()},
                           model_path)

        # Save the best score of this fold
        scores_per_fold.append(best_val_acc)

        # Save the last model
        model_path = os.path.join(fold_dir, "last_model.pth")
        torch.save({"model_state_dict": config.model.state_dict(),
                    "optimizer_state_dict": config.optimizer.state_dict()},
                   model_path)

        # Close logging of this fold
        fold_writer.close()

    # Report the result
    result = f"| Parameter        | Value                          |  \n" \
             f"| ---------------- | ------------------------------ |  \n" \
             f"| Average Accuracy | {np.mean(scores_per_fold):.5f} |  \n" \
             f"| Number of Folds  | {config.runner.num_folds}             |  \n"

    for fold in range(config.runner.num_folds):
        result += f"| Accuracy of Fold {fold} | {scores_per_fold[fold-1]:.5f} |  \n"

    summary_writer.add_text(tag="Result", text_string=result)
    summary_writer.close()


if __name__ == "__main__":
    # Create config for first run
    config_a = configs.create_config()

    # Set output directory
    output_a = os.path.join(constants.ROOT, r"models\run_a")

    # Start training
    k_fold_cross_validation(config_a, output_a)

    # Load config from previous run
    config_b = configs.load_config(r"C:\Users\E8J0G0K\Documents\Repos\ml-template\models\run_a\config.pt")

    # Set new output directory
    output_b = os.path.join(constants.ROOT, r"models\run_b")

    # Start training
    k_fold_cross_validation(config_b, output_b)
