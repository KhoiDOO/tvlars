import torch
from torch import nn 
from torchvision import models

model_map = {
    'resnet18' : models.resnet18
}

def get_model(model:str, num_classes:int):
    if model not in list(model_map.keys()):
        raise Exception(f'the model {model} is current not supported')
    
    return model_map[model](num_classes = num_classes)