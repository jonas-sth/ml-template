import os
import time

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


def train(writer, model, train_loader, epoch, loss_function, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # Send data to GPU
        data = data.to(torch.device("cuda:0"))
        target = target.to(torch.device("cuda:0"))

        # Reset optimizer
        optimizer.zero_grad()

        # Calculate loss and update weights accordingly
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        acc_function = torchmetrics.Accuracy().to(torch.device("cuda:0"))
        acc = acc_function(output, target)

        # Log to tensorboard
        writer.add_scalar("Loss/Train", loss.item(), (batch_idx + epoch * len(train_loader)))
        writer.add_scalar("Accuracy/Train", acc.item(), (batch_idx + epoch * len(train_loader)))


def validate(writer, model, val_loader, epoch, loss_function):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
            # Send data to GPU
            data = data.to(torch.device("cuda:0"))
            target = target.to(torch.device("cuda:0"))

            # Calculate loss
            output = model(data)
            loss = loss_function(output, target)

            # Calculate accuracy
            acc_function = torchmetrics.Accuracy().to(torch.device("cuda:0"))
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


def k_fold_cross_validation(model, data, config):
    # Initialize logging
    output_dir = os.path.join(constants.ROOT, rf"models\run_{time.strftime('%Y%m%d-%H%M%S')}")
    config.to_file(os.path.join(output_dir, "config.json"))

    # Initialize model
    model.cuda()

    # Initialize optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    # Initialize folds and scores
    k_fold = KFold(n_splits=config.num_folds, shuffle=True)
    scores_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(data)):
        print(f"==========Fold-{fold}==========")

        # Initialize logging
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        writer = SummaryWriter(fold_dir)
        writer.add_text(tag="config", text_string=config.as_dict())

        # Initialize data loader
        train_subset = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subset = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, sampler=train_subset)
        val_loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, sampler=val_subset)

        # Reset model and scores
        model.apply(models.reset_weights)
        best_val_acc = 0

        # Train and validate
        for epoch in range(1, config.num_epochs + 1):
            print(f"----------Epoch-{epoch}----------")
            train(writer, model, train_loader, epoch, cross_entropy_loss, optimizer)
            val_loss, val_acc = validate(writer, model, val_loader, epoch, cross_entropy_loss)

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
        
    # Report the summary
    print(f"Average score over {config.num_folds} folds: {np.mean(scores_per_fold)}")


if __name__ == "__main__":
    # Initialize model
    train_model = models.SimpleNet()

    # Initialize data
    train_data = datasets.MNIST_Dataset

    # Initialize config
    train_config = configs.Config(
        num_epochs=4,
        batch_size=1024,
        learning_rate=0.01,
        momentum=0.5,
        num_folds=2
    )

    # Start training
    k_fold_cross_validation(train_model, train_data, train_config)
