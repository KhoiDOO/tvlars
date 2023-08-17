import os
import argparse
import pandas as pd
import pickle
from tqdm import tqdm
from util import *

import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from dataset import *
from model.base import get_model
from opt import *
from scheduler.base import get_sche
from scheduler.lars_warmup import adjust_learning_rate

def main(args: argparse):
    
    # Setup folder
    args.log_dir = folder_setup(args=args)
    check_exp_exist(args=args)
    
    # Setup Multi GPU Training
    args.ngpus = torch.cuda.device_count()
    args.rank = 0
    args.dist_url = f'tcp://localhost:{args.port}'
    args.world_size = args.ngpus
    
    mp.spawn(main_worker, (args,), args.ngpus)

def main_worker(gpu, args):
    args.rank += gpu
    
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
    num_classes, train_dataset, test_dataset = get_dataset(args=args)
    
    assert args.bs % args.world_size == 0
    train_sampler = DistributedSampler(train_dataset) if args.opt != 'khlars' else None
    test_sampler = DistributedSampler(test_dataset) if args.opt != 'khlars' else None
    per_device_batch_size = args.bs // args.world_size

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=per_device_batch_size, num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=per_device_batch_size, num_workers=args.workers, pin_memory=True, sampler=test_sampler
    )
    
    # Model 
    model = get_model(model=args.model, num_classes=num_classes).cuda(gpu)
    if args.sd == "lars-warm":
        param_weights = []
        param_biases = []
        for param in model.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{'params': param_weights}, {'params': param_biases}]
    if args.opt != 'khlars':
        model = DDP(model, device_ids=[gpu])
    
    # Optimizer
    if args.opt not in ['lars', 'tvlars', 'khlars', 'lamb', 'clars']:
        optimizer = get_opt(
            opt_name=args.opt,
            params=model.parameters(),
            learning_rate=args.lr,
            weight_decay=args.wd
        )
    elif args.opt == 'lars':
        optimizer = LARS(
            params=parameters if args.sd == "lars-warm" else model.parameters(), 
            weight_decay=args.wd, 
            lr=args.lr, 
            weight_decay_filter=True, 
            lars_adaptation_filter=True
        )
    elif args.opt == 'khlars':
        optimizer = KHLARS(
            params=model.parameters(), 
            weight_decay=args.wd, 
            lr=args.lr, 
            weight_decay_filter=True, 
            lars_adaptation_filter=True
        )
    elif args.opt == 'clars':
        optimizer = CLARS(
            params=parameters, 
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
    elif args.opt == 'lamb':
        optimizer = LAMB(
            params=parameters if args.sd == "lars-warm" else model.parameters(),
            lr = args.lr,
            weight_decay=args.wd,
            adam=False
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
        model.train()
        if args.opt != 'khlars':
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
            loss.backward(
                retain_graph = True if args.opt == 'khlars' else False,
                create_graph = True if args.opt == 'khlars' else False
            )
            
            if args.opt == 'tvlars':
                optimizer.step(unit_step_cnt = len(train_loader), epoch_delay_cnt = 10)
            else:
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
        
            if args.opt != 'khlars':
                test_sampler.set_epoch(epoch)
            model.eval()
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
        
        if args.opt in ['lars', 'tvlars', 'khlars', 'clars', 'lamb']:
            ratio_log = optimizer.ratio_log
            weight_log = optimizer.weight_log
            gradient_log = optimizer.gradient_log
            
            ratio_log_path = args.log_dir + f"/ratio_{args.bs}_{args.lr}_{args.sd}.pickle"
            weight_log_path = args.log_dir + f"/weight_{args.bs}_{args.lr}_{args.sd}.pickle"
            gradient_log_path = args.log_dir + f"/gradient_{args.bs}_{args.lr}_{args.sd}.pickle"
            
            with open(ratio_log_path, 'wb') as handle:
                pickle.dump(ratio_log, handle)
                
            with open(weight_log_path, 'wb') as handle:
                pickle.dump(weight_log, handle)
            
            with open(gradient_log_path, 'wb') as handle:
                pickle.dump(gradient_log, handle)
                
            if args.opt == 'khlars':
                hessian_log = optimizer.hessian_log
                hessian_log_path = args.log_dir + f"/hessian_{args.bs}_{args.lr}_{args.sd}.pickle"
                
                with open(hessian_log_path, 'wb') as handle:
                    pickle.dump(hessian_log, handle)
    
    dist.destroy_process_group()