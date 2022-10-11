import os
from time import sleep

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.s00_utils import configs, constants
from src.s01_data import datasets
from src.s03_modelling import models


def train(writer, model, train_loader, epoch, optimizer, config):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        # Send data to GPU
        data = data.to(config.device)
        target = target.to(config.device)

        # Reset optimizer
        optimizer.zero_grad()

        # Calculate loss and update weights accordingly
        output = model(data)
        loss = config.loss_function(output, target)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        acc_function = config.accuracy_function.to(config.device)
        acc = acc_function(output, target)

        # Log to tensorboard
        writer.add_scalar("Loss/Train", loss.item(), (batch_idx + epoch * len(train_loader)))
        writer.add_scalar("Accuracy/Train", acc.item(), (batch_idx + epoch * len(train_loader)))


def validate(writer, model, val_loader, epoch, config):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc="Validating")):
            # Send data to GPU
            data = data.to(config.device)
            target = target.to(config.device)

            # Calculate loss
            output = model(data)
            loss = config.loss_function(output, target)

            # Calculate accuracy
            acc_function = config.accuracy_function.to(config.device)
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


def k_fold_cross_validation(model, config):
    # Initialize logging
    config.to_file()
    summary_dir = os.path.join(config.output_dir, "summary")
    summary_writer = SummaryWriter(summary_dir)
    summary_writer.add_text(tag="Config", text_string=config.as_table())

    # Initialize model
    model.to(config.device)

    # Initialize optimizer
    optimizer = config.optimizer(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    # Initialize folds and scores
    k_fold = KFold(n_splits=config.num_folds, shuffle=True)
    scores_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(config.data)):
        sleep(0.25)
        print(f"==========Fold-{fold}==========")
        sleep(0.25)

        # Initialize logging
        fold_dir = os.path.join(config.output_dir, f"fold_{fold}")
        fold_writer = SummaryWriter(fold_dir)

        # Initialize data loader
        train_subset = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subset = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(config.data, batch_size=config.batch_size, sampler=train_subset)
        val_loader = torch.utils.data.DataLoader(config.data, batch_size=config.batch_size, sampler=val_subset)

        # Reset model and scores
        model.apply(models.reset_weights)
        best_val_acc = 0

        # Train and validate
        for epoch in range(1, config.num_epochs + 1):
            sleep(0.25)
            print(f"Epoch {epoch}:")
            sleep(0.25)

            train(fold_writer, model, train_loader, epoch, optimizer, config)
            val_loss, val_acc = validate(fold_writer, model, val_loader, epoch, config)

            # Track the best performance, and save the model's state
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(fold_dir, "best_model.pth")
                torch.save({"epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()},
                           model_path)

        # Save the best score of this fold
        scores_per_fold.append(best_val_acc)

        # Save the last model
        model_path = os.path.join(fold_dir, "last_model.pth")
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()},
                   model_path)

        # Close logging of this fold
        fold_writer.close()

    # Report the result
    result = f"| Parameter        | Value                          |  \n" \
             f"| ---------------- | ------------------------------ |  \n" \
             f"| Average Accuracy | {np.mean(scores_per_fold):.5f} |  \n" \
             f"| Number of Folds  | {config.num_folds}             |  \n"

    for fold in range(config.num_folds):
        result += f"| Accuracy of Fold {fold} | {scores_per_fold[fold]:.5f} |  \n"

    summary_writer.add_text(tag="Result", text_string=result)
    summary_writer.close()


if __name__ == "__main__":
    # Initialize model
    train_model = models.SimpleNet()

    # Initialize config
    train_config = configs.Config(
        num_epochs=20,
        batch_size=32,
        learning_rate=0.01,
        momentum=0.5,
        num_folds=10,
        device=torch.device("cpu"),
        data=datasets.MNIST_Dataset,
        loss_function=torch.nn.CrossEntropyLoss(),
        accuracy_function=torchmetrics.Accuracy(),
        optimizer=torch.optim.SGD,
        output_dir=os.path.join(constants.ROOT, r"models\my_test"),
    )
    # Start training
    k_fold_cross_validation(train_model, train_config)
