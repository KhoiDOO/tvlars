import torch
from torch import nn 
from torchvision import models

model_map = {
    'resnet18' : models.resnet18,
    'resnet50' : models.resnet50,
    'effb0' : models.efficientnet_b0
}

def get_model(model:str, num_classes:int = None) -> nn.Module:
    if model not in list(model_map.keys()):
        raise Exception(f'the model {model} is current not supported')
    
    if num_classes is not None:
        return model_map[model](num_classes = num_classes)
    else:
        return model_map[model]()