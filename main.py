import os, sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Optimizer BenchMark',
                    description='This project takes into considering the performance comparison between optimizers',
                    epilog='ENJOY!!!')
    
    # MAIN
    
    ## ALL OPT
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
    
    parser.add_argument('--ds', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet'],
                        help='Data set name')
    parser.add_argument('--model', type=str, default='resnet18', choices= ['resnet18', 'resnet50', 'effb0'],
                        help='model used in training')
    parser.add_argument('--opt', type=str, default='lars', choices=['adam', 'adamw', 'adagrad', 'rmsprop', 'lars', 'tvlars', 'clars', 'lamb', 'khlars'],
                        help='optimizer used in training')
    parser.add_argument('--sd', type=str, default="None", choices=["None", 'cosine', 'lars-warm'],
                        help='Learning rate scheduler used in training')
    parser.add_argument('--dv', nargs='+', default=-1,
                        help='List of devices used in training', required=True)
    
    ## TVLARS
    parser.add_argument('--lmbda', type=float, default=0.001,
                        help='Delay factor used in TVLARS')
    
    ## BARLOW TWINS
    parser.add_argument('--btlmbda', type=float, default=0.005,
                        help='Lambda factor used in Barlow Twins')
    parser.add_argument('--vs', type=int, default=128,
                        help='Vector size')
    parser.add_argument('--lr_classifier', type=float, default=0.3,    
                        help='classifier learning rate')
    parser.add_argument('--lr_backbone', type=float, default=0,    
                        help='backbone learning rate')
    
    # MODE
    parser.add_argument('--mode', type=str, default='clf', choices=['clf', 'bt'],
                        help='Experiment Mode')
    
    args = parser.parse_args()
    
    if args.dv != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.dv)
    else:
        print("All GPU in use")
    
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    if args.mode == 'clf':
        print(f"SIMPLE CLASSFICATION EXPERIMENT")
        from clf import main
        main(args=args)
    elif args.mode == 'bt':
        print(f"BARLOW TWINS - SELF SUPERVISED LEARNING")
        from self_sl import main
        main(args=args)