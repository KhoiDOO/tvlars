import os, sys
import argparse

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
    
    parser.add_argument('--ds', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Data set name')
    parser.add_argument('--model', type=str, default='resnet18', choices= ['resnet18', 'resnet50', 'effb0'],
                        help='model used in training')
    parser.add_argument('--opt', type=str, default='lars', choices=['adam', 'adamw', 'adagrad', 'rmsprop', 'lars', 'tvlars', 'clars', 'lamb'],
                        help='optimizer used in training')
    parser.add_argument('--sd', type=str, default="None", choices=["None", 'cosine', 'lars-warm'],
                        help='Learning rate scheduler used in training')
    
    parser.add_argument('--lmbda', type=float, default=0.001,
                        help='Delay factor used in TVLARS')
    
    parser.add_argument('--dv', nargs='+', 
                        help='List of devices used in training', required=True)
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.dv)
    
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    from train import main
    main(args=args)