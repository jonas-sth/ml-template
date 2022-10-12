import dataclasses

import torch


@dataclasses.dataclass
class CustomKFoldRunner:
    num_folds: int
    num_epochs: int
    batch_size: int
    # seed:
    device: torch.device

    def __repr__(self):
        return f"{self.__class__.__name__}(\n" \
               f"  num_folds: {self.num_folds}\n" \
               f"  num_epochs: {self.num_epochs}\n" \
               f"  batch_size: {self.batch_size}\n" \
               f"  device: {self.device}\n" \
               f")"
