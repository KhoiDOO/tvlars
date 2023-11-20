import os
import argparse

def folder_setup(args: argparse):
    if args.mode == 'clf':
        exp_dir_name = 'runs'
    elif args.mode == 'bt':
        exp_dir_name = 'cl_runs'
    runs_dir = os.getcwd() + f"/{exp_dir_name}"
    if not os.path.exists(runs_dir):
        os.mkdir(runs_dir)
    
    data_model_dir = runs_dir + f"/{args.ds}_{args.model}_{args.winit}"
    if not os.path.exists(data_model_dir):
        os.mkdir(data_model_dir)
    
    if args.opt in ['adam', 'adamw', 'adagrad', 'rmsprop', 'lars', 'clars', 'khlars', 'lamb']:
        opt_dir = data_model_dir + f"/{args.opt}"
    elif args.opt == 'tvlars':
        opt_dir = data_model_dir + f"/{args.opt}_{args.lmbda}"
    
    if not os.path.exists(opt_dir):
        os.mkdir(opt_dir)
    
    return opt_dir

def check_exp_exist(args):
    check_log_path = args.log_dir + f"/{args.bs}_{args.lr}_{args.sd}.parquet"
    check_ratio_path = args.log_dir + f"/{args.bs}_{args.lr}_{args.sd}.pickle"
    filename = "/".join(check_log_path.split("/")[-3:]).replace('.parquet', '')
    if os.path.exists(check_log_path) and os.path.exists(check_ratio_path):
        print(f"{filename}: Existed -> Skipped")
        exit(0)
    else:
        print(f"{filename}: Non-existed -> Conducted")