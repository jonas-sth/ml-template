import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class CustomAutoencoderPreTrainer:
    """
    This class enables pretraining of a model by incorporating it as encoder in an autoencoder structure.
    """

    def __init__(self, num_epochs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 data, batch_size, loss_function, device):
        self.num_epochs = num_epochs
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.data = data
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.device = device

    def pretrain(self, dir_path):

        data_loader = DataLoader(self.data, batch_size=64, shuffle=True)

        self.encoder.train()
        self.encoder.train()
        
        # Train
        for epoch in tqdm(range(1, self.num_epochs + 1), desc="Epochs", leave=False):
            for batch_idx, (input_data, _) in enumerate(tqdm(data_loader, desc="Training", leave=False)):
                # Send data to GPU
                input_data = input_data.to(self.device)

                # Reset optimizer
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                # Calculate loss and update weights accordingly
                encoding = self.encoder(input_data)
                output_data = self.decoder(encoding)

                loss = self.loss_function(output_data, input_data)
                loss.backward()

                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

            if epoch % 10 == 0:
                path = os.path.join(dir_path, f"gan_after_epoch_{epoch}")
                torch.save({
                    "encoder_state_dict": self.encoder.state_dict(),
                    "decoder_state_dict": self.decoder.state_dict(),
                }, path)

        path = os.path.join(dir_path, "latest")
        torch.save({
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
        }, path)
