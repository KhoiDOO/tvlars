import torch
from torch import nn 
from torchvision import models

model_map = {
    'resnet18' : models.resnet18,
    'resnet34' : models.resnet34,
    'resnet50' : models.resnet50,
    'effb0' : models.efficientnet_b0
}

winit_map = {
    'xavier_uniform' : nn.init.xavier_uniform_,
    'xavier_normal' : nn.init.xavier_normal_,
    'kaiming_uniform' : nn.init.kaiming_uniform_,
    'kaiming_normal' : nn.init.kaiming_normal_
}

def get_model(model:str, num_classes:int = None, winit:str = 'xavier_uniform') -> nn.Module:
    if model not in list(model_map.keys()):
        raise Exception(f'the model {model} is current not supported')
    
    if num_classes is not None:
        model = model_map[model](num_classes = num_classes)
    else:
        model = model_map[model]()
    
    for name, param in model.named_parameters():
        if 'bn' or 'bias' in name:
            continue
        else:
            param = winit_map[winit](param)
    
    return model