import os
from time import sleep

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from sklearn.model_selection import KFold
from torchvision.transforms import transforms
from tqdm import tqdm

from src.s00_utils import configs, constants
from src.s01_data import datasets
from src.s03_modelling import models
from src.s04_training import runners


def train(writer: torch.utils.tensorboard.writer.SummaryWriter,
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


def validate(writer: torch.utils.tensorboard.writer.SummaryWriter,
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
    summary_writer = SummaryWriter(summary_dir)
    config.to_tensorboard(summary_writer)
    config.to_file(dir_path)

    # Move model to device
    config.model.to(config.runner.device)

    # Initialize folds and scores
    k_fold = KFold(n_splits=config.runner.num_folds, shuffle=True)
    scores_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(config.data)):
        sleep(0.25)
        print(f"==========Fold-{fold}==========")
        sleep(0.25)

        # Initialize logging of fold
        fold_dir = os.path.join(dir_path, f"fold_{fold}")
        fold_writer = SummaryWriter(fold_dir)

        # Initialize data loader
        train_subset = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subset = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(dataset=config.data,
                                                   batch_size=config.runner.batch_size,
                                                   sampler=train_subset)
        val_loader = torch.utils.data.DataLoader(dataset=config.data,
                                                 batch_size=config.runner.batch_size,
                                                 sampler=val_subset)

        # Reset model and scores
        config.model.apply(models.reset_weights)
        best_val_acc = 0

        # Train and validate
        for epoch in range(1, config.runner.num_epochs + 1):
            sleep(0.25)
            print(f"Epoch {epoch}:")
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
        result += f"| Accuracy of Fold {fold} | {scores_per_fold[fold]:.5f} |  \n"

    summary_writer.add_text(tag="Result", text_string=result)
    summary_writer.close()


def create_config():
    # Initialize runner
    runner = runners.CustomKFoldRunner(num_folds=5,
                                       num_epochs=20,
                                       batch_size=64,
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
    model = models.CustomConvNet()  # TODO: add dropout parameter, weight init and activation function

    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.001)

    # Initialize loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # Initialize accuracy function
    accuracy_function = torchmetrics.Accuracy()

    # Combine into one config
    config = configs.Config(runner=runner,
                            data=data,
                            model=model,
                            optimizer=optimizer,
                            loss_function=loss_function,
                            accuracy_function=accuracy_function
                            )

    return config


if __name__ == "__main__":
    # Create config
    # new_config = create_config()
    loaded_config = configs.Config.from_file(
        r"C:\Users\E8J0G0K\Documents\Repos\ml-template\models\testing\config.pt"
    )

    # Set output directory
    output_dir = os.path.join(constants.ROOT, r"models\testing2")

    # Start training
    k_fold_cross_validation(loaded_config, output_dir)
