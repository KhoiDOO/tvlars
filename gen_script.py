import os, sys
from glob import glob
import json
from random import randint
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="Script Generation"
    )
    
    parser.add_argument('--dv', nargs='+', 
                        help='List of devices used in training', required=True)
    
    args = parser.parse_args()
    
    available_device = " ".join(args.dv)
    
    run_file = os.getcwd() + "/run.sh"
    
    scripts = glob(os.getcwd() + "/script/*/script.json")
    
    cmd_lst = []
    
    for script in scripts:
        opt_name = script.split("/")[-2]
        
        script_data = json.load(open(script, mode='r'))
            
        datasets = script_data["ds"]
        models = script_data["model"]
        epoch = script_data["epochs"]
        wd = script_data["wd"]
        bss = script_data["bs"]
        sds = script_data["sd"]
        
        if opt_name == "lars":
            for dataset in datasets:
                for model in models:
                    for w in wd:
                        for bs in bss:
                            for lr in script_data["non-warm"]["lr"][str(bs)]:
                                cmd_lst.append(f"python main.py --bs {bs} --epochs 100 --lr {lr} --port {randint(3333, 8889)} --wd {w} --ds {dataset} --model {model} --opt lars --sd None --dv {available_device}\n")
                            for lr in script_data["warm"]["lr"]:
                                cmd_lst.append(f"python main.py --bs {bs} --epochs 100 --lr {lr} --port {randint(3333, 8889)} --wd {w} --ds {dataset} --model {model} --opt lars --sd 'lars-warm' --dv {available_device}\n")
        
        if opt_name == 'tvlars':
            for dataset in datasets:
                for model in models:
                    for w in wd:
                        for bs in bss:
                            for lr in script_data["lr"][str(bs)]:
                                for lmd in script_data["lmbda"][str(bs)]:
                                    for sd in sds:
                                        cmd_lst.append(f"python main.py --bs {bs} --epochs 100 --lr {lr} --port {randint(3333, 8889)} --wd {w} --ds {dataset} --model {model} --opt tvlars --sd {sd} --lmbda {lmd} --dv {available_device}\n")
                                        
    
    # Generate script
    with open(run_file, mode='w') as file:
        file.writelines(
            cmd_lst
        )
        file.close()