import torch
from torch import nn

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BTLoss(nn.Module):
    def __init__(self, bt_lambd = 0.005, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bt_lambd = bt_lambd
    
    def forward(self, z1, z2) -> torch.Tensor:
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.bt_lambd * off_diag
        return loss