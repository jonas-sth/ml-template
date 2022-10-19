"""This module contains custom models that are used within the training."""

import torch
from torch import nn


class CustomConvNet(nn.Module):
    """
    Simple Convolutional Neural Network
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10, 5, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24 * 24 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        """
        Forward pass
        """
        return self.layers(x)


class CustomConvEncoder(nn.Module):
    def __init__(self, n_image_channels, noise_size):
        super().__init__()

        n_feature_maps = 64

        self.layers = nn.Sequential(
            # input is (image_channels) x 32 x 32
            nn.Conv2d(n_image_channels, n_feature_maps, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            # state size: (n_feature_maps) x 16 x 16
            nn.Conv2d(n_feature_maps, n_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 2),
            nn.ReLU(inplace=True),
            # state size: (n_feature_maps*2) x 8 x 8
            nn.Conv2d(n_feature_maps * 2, noise_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(noise_size),
            nn.ReLU(inplace=True),
            # state size: noise_size x 4 x 4
            nn.MaxPool2d(4, 1, 0)
            # state size: noise_size x 1 x 1
        )

    def forward(self, x):
        return self.layers(x)


class CustomConvDecoder(nn.Module):
    """
    Simple Convolutional Decoder Neural Network.
    Can be used to train a Convolutional Net in form of an autoencoder.
    """
    def __init__(self, noise_size, num_classes, n_image_channels):
        super().__init__()

        input_length = noise_size + num_classes
        n_channels = 64

        self.layers = nn.Sequential(
            # input is noise and class as one-hot
            nn.ConvTranspose2d(input_length, n_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_channels * 8),
            nn.ReLU(True),
            # state size. (n_channels*8) x 4 x 4
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels * 4),
            nn.ReLU(True),
            # state size. (n_channels*4) x 8 x 8
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels * 2),
            nn.ReLU(True),
            # state size. (n_channels*2) x 16 x 16
            nn.ConvTranspose2d(n_channels * 2, n_image_channels, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(n_image_channels),
            nn.Tanh(),
            # state size. 3 x 32 x 32
        )

    def forward(self, x):
        """
        Forward pass
        """
        return self.layers(x)


if __name__ == '__main__':
    random_images = torch.rand((10, 3, 32, 32))

    n_image_channels = random_images.shape[1]
    noise_size = 10

    encoder = CustomConvEncoder(n_image_channels=n_image_channels, noise_size=noise_size)
    decoder = CustomConvDecoder(n_image_channels=n_image_channels, noise_size=noise_size, num_classes=0)

    encoding = encoder(random_images)
    output = decoder(encoding)


