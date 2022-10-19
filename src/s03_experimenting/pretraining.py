import os

import torch
from torchvision.transforms import transforms

from src.s00_utils import constants
from src.s02_customizing import datasets, models, pretrainers


def pretrain(dir_path, clear_dir=True):
    """
    Runs an experiment with the specified parameters.
    Use this method to create new configurations.
    """
    # Initialize data
    data = datasets.CustomImageDataset(image_dir=os.path.join(constants.ROOT, r"data\d01_raw\mnist\train"),
                                       label_path=os.path.join(constants.ROOT, r"data\d01_raw\mnist\train\labels.csv"),
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Grayscale(),
                                                                     transforms.Normalize(0.5, 0.2)]
                                                                    )
                                       )

    # Initialize models
    n_image_channels = data[0][0].shape[1]
    noise_size = 10
    encoder = models.CustomConvEncoder(n_image_channels=n_image_channels, noise_size=noise_size)
    decoder = models.CustomConvDecoder(n_image_channels=n_image_channels, noise_size=noise_size, num_classes=0)

    # Initialize optimizer
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.1)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.1)

    # Initialize loss function
    loss_function = torch.nn.MSELoss()

    # Initialize trainer
    pre_trainer = pretrainers.CustomAutoencoderPreTrainer(
        num_epochs=50,
        encoder=encoder,
        decoder=decoder,
        encoder_optimizer=encoder_optimizer,
        decoder_optimizer=decoder_optimizer,
        data=data,
        batch_size=64,
        loss_function=loss_function,
        device=torch.device("cpu")
    )

    # Run experiment
    pre_trainer.pretrain(dir_path)


if __name__ == '__main__':
    output_path = os.path.join(constants.ROOT, r"models\pretrained")
    pretrain(output_path)
