import torch
from torch import nn 

class BarlowTwins(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)