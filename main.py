import os, sys
import argparse

import random
import numpy as np
import torch

from train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Optimizer BenchMark',
                    description='This project takes into considering the performance comparison between optimizers',
                    epilog='ENJOY!!!')
    
    parser.add_argument('--bs', type = int, default=32,
                    help='batch size')
    parser.add_argument('--workers', type = int, default=4,
                    help='Number of processor used in data loader')
    parser.add_argument('--epochs', type = int, default=1,
                    help='# Epochs used in training')
    parser.add_argument('--lr', type=float, default=0.01, 
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--port', type=int, default=8080, help='Multi-GPU Training Port.')
    parser.add_argument('--wd', default=5e-4, type=float, metavar='W',
                    help='weight decay')
    
    parser.add_argument('--ds', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Data set name')
    parser.add_argument('--model', type=str, default='resnet18', choices= ['resnet18'],
                        help='model used in training')
    parser.add_argument('--opt', type=str, default='lars', choices=['adam', 'adamw', 'adagrad', 'rmsprop', 'lars', 'tvlars'],
                        help='optimizer used in training')
    parser.add_argument('--sd', type=str, default='cosine', choices=['cosine', 'lars-warm'],
                        help='Learning rate scheduler used in training')
    
    parser.add_argument('--lmbda', type=float, default=0.001,
                        help='Delay factor used in TVLARS')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    main(args=args)