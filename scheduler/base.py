import torch
from torch.optim.lr_scheduler import *

scheduler_map = {
    'cosine' : CosineAnnealingLR
}

def get_sche(sche_name:str, optimizer: torch.optim.Optimizer, T_max:int):
    if sche_name not in list(scheduler_map.keys()):
        raise Exception(f'There is no optim key {sche_name} supported')
    
    return scheduler_map[sche_name](
        optimizer=optimizer,
        T_max=T_max
    )