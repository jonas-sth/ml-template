import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.s00_utils import configs, constants
from src.s01_data import datasets
from src.s03_modelling import models


def train(model, data, config):
    # Initialize logging
    output_dir = os.path.join(constants.ROOT, rf"models\run_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(output_dir)
    config.to_file(os.path.join(output_dir, "config.json"))
    writer.add_text(tag="config", text_string=config.as_dict())

    # Initialize model
    model.cuda()
    model.train()

    # Initialize optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    # Initialize data loader
    train_loader = DataLoader(data)

    # Train
    for epoch in range(config.num_epochs):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")):
            # Send data to GPU
            data_gpu = data.to(torch.device("cuda:0"))
            target_gpu = target.to(torch.device("cuda:0"))

            # Reset optimizer
            optimizer.zero_grad()

            # Calculate loss and update weights accordingly
            output = model(data_gpu)
            loss = cross_entropy_loss(output, target_gpu)
            loss.backward()
            optimizer.step()

            # Log to tensorboard
            writer.add_scalar("Loss/train", loss.item(), (batch_idx + epoch * len(train_loader)))
            writer.add_scalar("Accuracy/train", 0, (batch_idx + epoch * len(train_loader)))

        # Save snapshot after each epoch?
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss}, PATH)

    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))

    # Save best model only
    # todo


if __name__ == "__main__":
    # Initialize data
    train_data = datasets.MNIST_Dataset

    # Initialize config
    train_config = configs.Config(
        num_epochs=2,
        batch_size=256,
        learning_rate=0.01,
        momentum=0.5
    )

    # Initialize model
    train_model = models.SimpleNet()

    # Start training
    train(train_model, train_data, train_config)
