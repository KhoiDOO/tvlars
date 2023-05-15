import os, sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from dataset.bench import get_dataset
from model.base import get_model
from opt.base import get_opt
from opt.lars import LARS
from opt.tvlars import TVLARS
from scheduler.base import get_sche
from scheduler.lars_warmup import adjust_learning_rate

def folder_setup(args: argparse):
    runs_dir = os.getcwd() + "/runs"
    
    data_model_dir = runs_dir + f"/{args.ds}_{args.model}"
    if not os.path.exists(data_model_dir):
        os.mkdir(data_model_dir)
    
    if args.opt in ['adam', 'adamw', 'adagrad', 'rmsprop', 'lars']:
        opt_dir = data_model_dir + f"/{args.opt}"
    elif args.opt == 'tvlars':
        opt_dir = data_model_dir + f"/{args.opt}_{args.lmbda}"
    
    if not os.path.exists(opt_dir):
        os.mkdir(opt_dir)
    
    return opt_dir

def main(args: argparse):
    # Device Conversion
    args.dv_inuse = [int(x) for x in args.dv]
    
    # Setup folder
    args.log_dir = folder_setup(args=args)
    
    # Setup Multi GPU Training
    args.ngpus_per_node = torch.cuda.device_count()
    args.rank = 0
    args.dist_url = f'tcp://localhost:{args.port}'
    args.world_size = args.ngpus_per_node
    mp.spawn(main_worker, (args,), args.ngpus_per_node)

def main_worker(gpu, args):
    args.rank += gpu
    
    print(f"GPU: {gpu} - Integrated - Rank {args.rank}")
    
    dist.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    
    if args.rank == 0:
        log = {
            "train_loss" : [],
            "train_acc" : [],
            "test_loss" : [],
            "test_acc" : []
        }
        
        log_path = args.log_dir + f"/{args.bs}_{args.lr}_{args.sd}.parquet"
    
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    # Data Loader
    num_classes, train_dataset, test_dataset = get_dataset(dataset_name=args.ds)
    
    assert args.bs % args.world_size == 0
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)
    per_device_batch_size = args.bs // args.world_size

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=per_device_batch_size, num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=per_device_batch_size, num_workers=args.workers, pin_memory=True, sampler=test_sampler
    )
    
    # Model 
    model = get_model(model=args.model, num_classes=num_classes).cuda(gpu)
    model = DDP(model, device_ids=[gpu])
    
    # Optimizer
    if args.opt not in ['lars', 'tvlars', 'khlars']:
        optimizer = get_opt(
            opt_name=args.opt,
            params=model.parameters(),
            learning_rate=args.lr,
            weight_decay=args.wd
        )
    elif args.opt == 'lars':
        optimizer = LARS(
            params=model.parameters(), 
            weight_decay=args.wd, 
            lr=args.lr, 
            weight_decay_filter=True, 
            lars_adaptation_filter=True
        )
    elif args.opt == 'tvlars':
        optimizer = TVLARS(
            params=model.parameters(), 
            weight_decay=args.wd, 
            lmbda=args.lmbda,
            lr=args.lr, 
            weight_decay_filter=True, 
            lars_adaptation_filter=True,
        )
    
    # Learning Scheduler / Warm Up
    if args.sd == 'lars-warm':
        pass
    elif args.sd == 'cosine':
        scheduler = get_sche(
            sche_name=args.sd,
            optimizer=optimizer,
            T_max=args.epochs
        )
    
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    
    # Training and Evaluation
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_loss = 0
        correct = 0
        total = 0
        batch_count = 0
        for step, (train_img, train_label) in tqdm(enumerate(train_loader, start=epoch * len(train_loader))):
            if args.sd == 'lars-warm':
                adjust_learning_rate(args, optimizer, train_loader, step)
            batch_count = step
            train_img = train_img.cuda(gpu, non_blocking=True)
            train_label = train_label.cuda(gpu, non_blocking=True)
            logits = model(train_img)
            loss = criterion(logits, train_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.sd == 'cosine':
                scheduler.step()
            
            if args.rank == 0:
                train_loss += loss.item()
                _, predicted = logits.max(1)
                total += train_label.size(0)
                correct += predicted.eq(train_label).sum().item()
        
        if args.rank == 0:
            log["train_loss"].append(train_loss/(batch_count+1))
            log["train_acc"].append(100.*correct/total)
        
        if args.rank == 0:
            test_sampler.set_epoch(epoch)
            with torch.no_grad():
                test_loss = 0
                correct = 0
                total = 0
                batch_count = 0
                for step, (val_img, val_label) in tqdm(enumerate(test_loader)):
                    batch_count = step
                    val_img = val_img.cuda(gpu, non_blocking=True)
                    val_label = val_label.cuda(gpu, non_blocking=True)
                    logits = model(val_img)
                    loss = criterion(logits, val_label)
                
                    test_loss += loss.item()
                    _, predicted = logits.max(1)
                    total += val_label.size(0)
                    correct += predicted.eq(val_label).sum().item()
                
                log["test_loss"].append(test_loss/(batch_count+1))
                log["test_acc"].append(100.*correct/total)   
        
            print(f"Epoch: {epoch} - " + " - ".join([f"{key}: {log[key][epoch]}" for key in log]))
    
    if args.rank == 0:
        log_df = pd.DataFrame(log)
        log_df.to_parquet(log_path)
    
    dist.destroy_process_group()