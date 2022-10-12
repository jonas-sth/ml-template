import dataclasses

import torch


@dataclasses.dataclass
class CustomKFoldRunner:
    num_folds: int
    num_epochs: int
    batch_size: int
    # seed:
    device: torch.device
