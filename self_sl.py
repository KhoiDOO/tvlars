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
from model import get_model, BarlowTwins
from opt import *
from scheduler.base import get_sche
from scheduler.lars_warmup import adjust_learning_rate
from metric import accuracy

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
            "cl_loss" : [],
            "train_loss" : [],
            "train_acc_1" : [],
            "train_acc_5" : [],
            "test_loss" : [],
            "test_acc_1" : [],
            "test_acc_5" : []
        }
        
        log_path = args.log_dir + f"/{args.bs}_{args.lr}_{args.sd}.parquet"
    
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    # Data Loader
    num_classes, train_dataset, test_dataset = get_dataset(
        args=args,
        bt_stage=0
    )
    
    assert args.bs % args.world_size == 0
    train_sampler = DistributedSampler(train_dataset) if args.opt != 'khlars' else None
    per_device_batch_size = args.bs // args.world_size

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=per_device_batch_size, num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )
    
    # Model 
    model = BarlowTwins(args=args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
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
            adam=True
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
    
    # Training Constrastive Learning
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):
        if args.opt != 'khlars':
            train_sampler.set_epoch(epoch)
        train_loss = 0
        batch_count = 0
        for step, ((img1, img2), _) in tqdm(enumerate(train_loader, start=epoch * len(train_loader))):
            if args.sd == 'lars-warm':
                adjust_learning_rate(args, optimizer, train_loader, step)
            batch_count = step
            optimizer.zero_grad()
            img1 = img1.cuda(gpu, non_blocking=True)
            img2 = img2.cuda(gpu, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                loss = model(img1, img2)
            
            scaler.scale(loss).backward(
                retain_graph = True if args.opt == 'khlars' else False,
                create_graph = True if args.opt == 'khlars' else False
            )
            scaler.step(optimizer)
            scaler.update()
            
            if args.sd == 'cosine':
                scheduler.step()
            
            if args.rank == 0:
                train_loss += loss.item()
        
        if args.rank == 0:
            log["cl_loss"].append(train_loss/(batch_count+1))
            curr_loss = log["cl_loss"][-1]
            print(f"Epoch: {epoch} - CL Loss: {curr_loss}")
    
    # Training Classification
    
    # resetup model
    clf_model = get_model(model=args.model, num_classes=num_classes).cuda(gpu)
    missing_keys, unexpected_keys = clf_model.load_state_dict(model.module.backbone.state_dict(), strict=False)
    if missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []:
        clf_model.fc = nn.Linear(512, num_classes)
    clf_model.fc.weight.data.normal_(mean=0.0, std=0.01)
    clf_model.fc.bias.data.zero_()
    model.requires_grad_(False)
    model.fc.requires_grad_(True)   
    classifier_parameters, model_parameters = [], []
    for name, param in clf_model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)
    clf_model = DDP(clf_model, device_ids=[gpu], find_unused_parameters=True)
    del model
        
    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    clf_optimizer = torch.optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.wd)
    clf_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(clf_optimizer, args.epochs)
    
    # resetup data set
    num_classes, train_dataset, test_dataset = get_dataset(
        args=args,
        bt_stage=1
    )
    
    # assert args.bs % args.world_size == 0
    train_sampler = DistributedSampler(train_dataset) if args.opt != 'khlars' else None
    test_sampler = DistributedSampler(test_dataset) if args.opt != 'khlars' else None
    per_device_batch_size = 256 // args.world_size

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=per_device_batch_size, num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=per_device_batch_size, num_workers=args.workers, pin_memory=True, sampler=test_sampler
    )
    
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    
    # CLF train 
    for epoch in range(args.epochs):
        if args.opt != 'khlars':
            train_sampler.set_epoch(epoch)
        train_loss = 0
        train_acc_1 = 0
        train_acc_5 = 0
        batch_count = 0
        for step, (train_img, train_label) in tqdm(enumerate(train_loader, start=epoch * len(train_loader))):
            batch_count = step
            train_img = train_img.cuda(gpu, non_blocking=True)
            train_label = train_label.cuda(gpu, non_blocking=True)
            logits = clf_model(train_img)
            loss = criterion(logits, train_label)
            acc1, acc5 = accuracy(logits, train_label, topk=(1, 5))
            
            clf_optimizer.zero_grad()
            loss.backward(
                retain_graph = True if args.opt == 'khlars' else False,
                create_graph = True if args.opt == 'khlars' else False
            )
            clf_optimizer.step()
            clf_scheduler.step()
            
            if args.rank == 0:
                train_loss += loss.item()
                train_acc_1 += acc1.item()
                train_acc_5 += acc5.item()
        
        if args.rank == 0:
            log["train_loss"].append(train_loss/(batch_count+1))
            log["train_acc_1"].append(train_acc_1/(batch_count+1))
            log["train_acc_5"].append(train_acc_5/(batch_count+1))
        
            if args.opt != 'khlars':
                test_sampler.set_epoch(epoch)
            with torch.no_grad():
                test_loss = 0
                test_acc_1 = 0
                test_acc_5 = 0
                batch_count = 0
                for step, (val_img, val_label) in tqdm(enumerate(test_loader)):
                    batch_count = step
                    val_img = val_img.cuda(gpu, non_blocking=True)
                    val_label = val_label.cuda(gpu, non_blocking=True)
                    logits = clf_model(val_img)
                    loss = criterion(logits, val_label)
                    acc1, acc5 = accuracy(logits, val_label, topk=(1, 5))
                
                    test_loss += loss.item()
                    test_acc_1 += acc1.item()
                    test_acc_5 += acc5.item()
                
                log["test_loss"].append(test_loss/(batch_count+1))
                log["test_acc_1"].append(test_acc_1/(batch_count+1))
                log["test_acc_5"].append(test_acc_1/(batch_count+1))
        
            print(f"Epoch: {epoch} - " + " - ".join([f"{key}: {log[key][epoch]}" for key in log]))
    
    if args.rank == 0:
        log_df = pd.DataFrame(log)
        log_df.to_parquet(log_path)
        
        if args.opt in ['lars', 'tvlars', 'khlars', 'clars', 'lamb']:
            ratio_log = optimizer.ratio_log
            
            ratio_log_path = args.log_dir + f"/{args.bs}_{args.lr}_{args.sd}.pickle"
            
            with open(ratio_log_path, 'wb') as handle:
                pickle.dump(ratio_log, handle)
                
            if args.opt == 'khlars':
                hessian_log = optimizer.hessian_log
                hessian_log_path = args.log_dir + f"/hessian_{args.bs}_{args.lr}_{args.sd}.pickle"
                
                with open(hessian_log_path, 'wb') as handle:
                    pickle.dump(hessian_log, handle)
    
    dist.destroy_process_group()