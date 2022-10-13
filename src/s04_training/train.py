"""This module contains the training in form of a k-fold-cross-validation."""

import os
import random
import shutil
from time import sleep

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.s00_utils import constants
from src.s03_prototyping import configs


def train(writer: SummaryWriter, train_loader: DataLoader, epoch: int, config: configs.Config) -> None:
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


def validate(writer: SummaryWriter, val_loader: DataLoader, epoch: int, config: configs.Config) -> (float, float):
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


def k_fold_cross_validation(config: configs.Config, dir_path: str, clear_dir=True) -> (float, float):
    """
    Executes training and validating a model on k different splits.
    """
    # Clean up
    if clear_dir:
        shutil.rmtree(dir_path, ignore_errors=True)

    # Initialize logging
    summary_dir = os.path.join(dir_path, "summary")
    summary_writer = SummaryWriter(summary_dir)
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
    accuracy_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(config.data), start=1):
        # Initialize logging of fold
        fold_dir = os.path.join(dir_path, f"fold_{fold}")
        fold_writer = SummaryWriter(fold_dir)

        # Initialize data loader
        train_subset = SubsetRandomSampler(train_idx)
        val_subset = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset=config.data,
                                  batch_size=config.runner.batch_size,
                                  sampler=train_subset)
        val_loader = DataLoader(dataset=config.data,
                                batch_size=config.runner.batch_size,
                                sampler=val_subset)

        # Reset model weights and scores
        config.model.apply(config.weight_init)
        best_val_acc = 0

        # Train and validate
        for epoch in range(1, config.runner.num_epochs + 1):
            # Print status to console (sleep needed to avoid conflict with tqdm progress bars)
            sleep(0.25)
            run_name = os.path.basename(dir_path)
            print(f"Run {run_name}, Fold {fold}/{config.runner.num_folds}, Epoch {epoch}/{config.runner.num_epochs}:")
            sleep(0.25)

            # Execute training and validating
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
        accuracy_per_fold.append(best_val_acc)

        # Save the last model
        model_path = os.path.join(fold_dir, "last_model.pth")
        torch.save({"model_state_dict": config.model.state_dict(),
                    "optimizer_state_dict": config.optimizer.state_dict()},
                   model_path)

        # Close logging of this fold
        fold_writer.close()

    # Report the result as Markdown table to tensorboard
    avg = np.mean(accuracy_per_fold)
    std = np.std(accuracy_per_fold)

    result = f"| Parameter          | Value     |  \n" \
             f"| ------------------ | --------- |  \n" \
             f"| Average Accuracy   | {avg:.5f} |  \n" \
             f"| Standard Deviation | {std:.5f} |  \n"

    for fold in range(config.runner.num_folds):
        result += f"| Accuracy of Fold {fold} | {accuracy_per_fold[fold - 1]:.5f} |  \n"

    summary_writer.add_text(tag="Result", text_string=result)
    summary_writer.close()

    return avg, std


if __name__ == "__main__":
    # Create config or load one
    train_config = configs.create_config()
    # train_config = configs.load_config(r"C:\Users\E8J0G0K\Documents\Repos\ml-template\models\run_a\config.pt")

    # Set output directory
    output_dir = os.path.join(constants.ROOT, r"models\run")

    # Start training
    k_fold_cross_validation(train_config, output_dir)

