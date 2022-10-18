"""This module contains custom trainers implementing training routines."""

import os
import random
import shutil
import time
from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.s00_utils import basics


class CustomKFoldTrainer:
    """
    This class enables k-fold-cross-validation given a modular set of parameters.
    It supports the saving of the configuration to a file and rich logging to tensorboard.
    """

    def __init__(self,
                 num_folds: int,
                 num_epochs: int,
                 batch_size: int,
                 device: torch.device,
                 data: torch.utils.data.Dataset,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_function: torch.nn.Module,
                 accuracy_function: torch.nn.Module,
                 weight_init: Callable = None,
                 lr_scheduler=None,  # :torch.optim.lr_scheduler._LRScheduler
                 seed: int = None):

        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.data = data
        self.model = model
        self.weight_init = weight_init
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.accuracy_function = accuracy_function
        self.seed = seed if seed is not None else int(time.time())

    @classmethod
    def from_file(cls, file_path):
        """
        Loads the object from the given file path via torch and returns it, if it is of this class.
        """
        trainer = torch.load(file_path)
        if isinstance(trainer, cls):
            return trainer
        else:
            return None

    def to_file(self, dir_path):
        """
        Saves this object with its parameters in the given directory via torch.
        """
        # Create necessary directories
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save this config object via torch
        output_path = os.path.join(dir_path, "config.pt")
        torch.save(self, output_path)

    def to_tensorboard(self, writer: SummaryWriter) -> None:
        """
        Writes the parameters of this trainer as formatted Markdown text to the given tensorboard writer.
        """
        # Handle optional config parameters
        weight_init_string = basics.get_method_as_string(self.weight_init) if self.weight_init is not None else "None"
        lr_scheduler_string = basics.get_state_as_string(self.lr_scheduler) if self.lr_scheduler is not None else "None"

        # Add every component of this config with a unique tag to tensorboard as Markdown text.
        writer.add_text(tag="Base Parameters", text_string=basics.markdown(self._get_base_parameters_string()))
        writer.add_text(tag="Dataset", text_string=basics.markdown(str(self.data)))
        writer.add_text(tag="Model", text_string=basics.markdown(str(self.model)))
        writer.add_text(tag="Weight Initialization", text_string=basics.markdown(weight_init_string))
        writer.add_text(tag="Optimizer", text_string=basics.markdown(str(self.optimizer)))
        writer.add_text(tag="Learning Rate Scheduler", text_string=basics.markdown(lr_scheduler_string))
        writer.add_text(tag="Loss Function", text_string=basics.markdown(str(self.loss_function)))
        writer.add_text(tag="Accuracy Function", text_string=basics.markdown(str(self.accuracy_function)))

        # Get one sample of the data with the right dimensions and use it to log the graph to tensorboard
        data_sample = self.data[0][0][None, :]
        writer.add_graph(self.model, data_sample)

    def _get_base_parameters_string(self):
        """
        Returns the basic parameters of this trainer as string. Used for cleaner logging to tensorboard.
        """
        return f"{self.__class__.__name__}(\n" \
               f"  num_folds: {self.num_folds}\n" \
               f"  num_epochs: {self.num_epochs}\n" \
               f"  batch_size: {self.batch_size}\n" \
               f"  seed: {self.seed}\n" \
               f"  device: {self.device}\n" \
               f")"

    def k_fold_cross_validation(self, dir_path: str, clear_dir=True, run_info=None) -> (float, float):
        """
        Executes training and validating a model on k different splits.
        """
        # Clean up
        if clear_dir:
            shutil.rmtree(dir_path, ignore_errors=True)

        # Initialize logging
        summary_dir = os.path.join(dir_path, "summary")
        summary_writer = SummaryWriter(summary_dir)
        self.to_tensorboard(summary_writer)
        self.to_file(dir_path)

        # Set seeds
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Move model to device
        self.model.to(self.device)

        # Initialize folds and scores
        k_fold = KFold(n_splits=self.num_folds, shuffle=True)
        accuracy_per_fold = []

        for fold, (train_idx, val_idx) in enumerate(k_fold.split(self.data), start=1):
            # Initialize logging of fold
            fold_dir = os.path.join(dir_path, f"fold_{fold}")
            fold_writer = SummaryWriter(fold_dir)

            # Initialize data loader
            train_subset = SubsetRandomSampler(train_idx)
            val_subset = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset=self.data,
                                      batch_size=self.batch_size,
                                      sampler=train_subset)
            val_loader = DataLoader(dataset=self.data,
                                    batch_size=self.batch_size,
                                    sampler=val_subset)

            # Reset model, optimizer, learning rate scheduler and score
            self.model.apply(basics.weight_reset)
            self.optimizer = self.optimizer.__class__(self.model.parameters(), **self.optimizer.defaults)
            if self.lr_scheduler is not None:
                self.lr_scheduler = self.lr_scheduler.__class__(self.optimizer,
                                                                **basics.get_lr_scheduler_params(self.lr_scheduler))
            best_val_acc = 0

            # Initialize weights
            if self.weight_init is not None:
                self.model.apply(self.weight_init)

            # Set up profiler
            """
            Attention:  The tensorboard plugin: torch-tb-profiler doesn't work natively on Windows.
                        A dirty workaround to fix the invalid escapes in the saved jsons is the following:
                        Go to /path_to_python_installation/Lib/site-packages/torch_tb_profiler/profiler/data.py
                        and change the loading of the json file in line 127 (might change depending on version) to: 
                        trace_json = json.loads(data.replace(b"\\", b"\\\\"), strict=False).
                        More at: https://github.com/pytorch/kineto/issues/336
            """
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(fold_dir),
                record_shapes=True,
                with_stack=True)
            prof.start()

            # Train and validate
            for epoch in range(1, self.num_epochs + 1):
                # Print status to console (sleep needed to avoid conflict with tqdm progress bars)
                time.sleep(0.25)
                if run_info is not None:
                    print(f"Run {run_info}, Fold {fold}/{self.num_folds}, Epoch {epoch}/{self.num_epochs}:")
                else:
                    print(f"Fold {fold}/{self.num_folds}, Epoch {epoch}/{self.num_epochs}:")
                time.sleep(0.25)

                # Execute training and validating
                self._train(fold_writer, train_loader, epoch, prof)
                val_loss, val_acc = self._validate(fold_writer, val_loader, epoch)

                # Update learning rate
                if self.lr_scheduler is not None:
                    fold_writer.add_scalar("Learning rate", self.optimizer.param_groups[0]["lr"], epoch)
                    self.lr_scheduler.step()

                # Track the best performance, and save the model's state
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_path = os.path.join(fold_dir, "best_model.pth")
                    torch.save({"epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict()},
                               model_path)

            # Save the best score of this fold
            accuracy_per_fold.append(best_val_acc)

            # Save the last model
            model_path = os.path.join(fold_dir, "last_model.pth")
            torch.save({"model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()},
                       model_path)

            # Close logging of this fold
            fold_writer.close()
            prof.stop()

        # Report the result as Markdown table to tensorboard
        avg = np.mean(accuracy_per_fold)
        std = np.std(accuracy_per_fold)

        result = f"| Parameter          | Value     |  \n" \
                 f"| ------------------ | --------- |  \n" \
                 f"| Average Accuracy   | {avg:.5f} |  \n" \
                 f"| Standard Deviation | {std:.5f} |  \n"

        for fold in range(self.num_folds):
            result += f"| Accuracy of Fold {fold} | {accuracy_per_fold[fold - 1]:.5f} |  \n"

        summary_writer.add_text(tag="Result", text_string=result)
        summary_writer.close()

        return avg, std

    def _train(self, writer: SummaryWriter, train_loader: DataLoader, epoch: int, prof) -> None:
        """
        Trains one epoch.
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            # Send data to GPU
            data = data.to(self.device)
            target = target.to(self.device)

            # Reset optimizer
            self.optimizer.zero_grad()

            # Calculate loss and update weights accordingly
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()
            prof.step()

            # Calculate accuracy
            acc_function = self.accuracy_function.to(self.device)
            acc = acc_function(output, target)

            # Log to tensorboard
            writer.add_scalar("Loss/Train", loss.item(), (batch_idx + epoch * len(train_loader)))
            writer.add_scalar("Accuracy/Train", acc.item(), (batch_idx + epoch * len(train_loader)))

            # Log images one time to tensorboard
            if epoch == 1 and batch_idx == 0:
                writer.add_images(tag="First Training Batch", img_tensor=data, global_step=0)
                writer.add_image(tag="One Image", img_tensor=data[0], global_step=0)

    def _validate(self, writer: SummaryWriter, val_loader: DataLoader, epoch: int) -> (float, float):
        """
        Validates one epoch.
        """
        self.model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc="Validating")):
                # Send data to GPU
                data = data.to(self.device)
                target = target.to(self.device)

                # Calculate loss
                output = self.model(data)
                loss = self.loss_function(output, target)

                # Calculate accuracy
                acc_function = self.accuracy_function.to(self.device)
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
