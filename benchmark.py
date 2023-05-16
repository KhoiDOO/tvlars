import os, sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from glob import glob

def get_report_temp(opt_split):
    control = None
    opt_name = opt_split[-1]
    if "_" in opt_name:
        full_opt_name = opt_name.split("_")
        if full_opt_name[0] == 'lars':
            control = 0
        elif full_opt_name[0] == 'tvlars':
            control = 1
    else:
        control = 0
    
    if control == 0:
        return {
            "bs" : [],
            "lr" : [],
            "sd" : [],
            "max_train_acc" : [],
            "max_test_acc" : [],
            "mean_train_acc" : [],
            "mean_test_acc" : [],
        }
    elif control == 1:
        return {
            "lambda" : [full_opt_name[1]]*len(glob("/".join(opt_split) + "/*.parquet")),
            "bs" : [],
            "lr" : [],
            "sd" : [],
            "max_train_acc" : [],
            "max_test_acc" : [],
            "mean_train_acc" : [],
            "mean_test_acc" : [],
        }
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="OPTIMIZER BENCHMARK"
    )
    
    parser.add_argument('--mode', type=str, default='table', choices=['table', 'plot'],
                        help='benchmark mode')
    
    args = parser.parse_args()
    
    if args.mode != 'table':
        raise NotImplementedError(f'mode {args.mode} is not exist or not implemented yet')
     
    results_dir = os.getcwd() + "/results"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        
    runs_dir = os.getcwd() + "/runs"
    data_model_dirs = glob(runs_dir + "/*")
    for data_model_dir in data_model_dirs:
        if len(os.listdir(data_model_dir)) == 0:
            continue
        data_model_result_dir = results_dir + f'/{data_model_dir.split("/")[-1]}'
        if not os.path.exists(data_model_result_dir):
            os.mkdir(data_model_result_dir)
            
    for data_model_dir in data_model_dirs:
        if len(os.listdir(data_model_dir)) == 0:
            continue
        data_model_result_dir = results_dir + f'/{data_model_dir.split("/")[-1]}'
        
        opt_dirs = glob(data_model_dir + "/*")
        for opt_dir in opt_dirs:
            opt_split = opt_dir.split("/")
            log_files = glob(opt_dir + "/*.parquet")
            if len(log_files) == 0:
                continue
            
            log_template = get_report_temp(opt_split=opt_split)
            
            for log_file in log_files:
                filename = log_file.split("/")[-1].replace('.parquet', '')
                filename_split = tuple(filename.split("_"))
                bs, lr, sd = filename_split
                log_template["bs"].append(int(bs))
                log_template["lr"].append(float(lr))
                log_template["sd"].append(sd)
                
                base_df = pd.read_parquet(log_file)
                base_train_acc = base_df["train_acc"].values.tolist()
                base_test_acc = base_df["test_acc"].values.tolist()
                log_template["max_train_acc"].append(
                    max(base_train_acc)
                )
                log_template["max_test_acc"].append(
                    max(base_test_acc)
                )
                log_template["mean_train_acc"].append(
                    sum(base_train_acc)/len(base_train_acc)
                )
                log_template["mean_test_acc"].append(
                    sum(base_test_acc)/len(base_test_acc)
                )
            
            base_log_df = pd.DataFrame(log_template).sort_values(by='bs')
            base_log_df.to_csv(data_model_result_dir + f"/{opt_split[-1]}.csv")