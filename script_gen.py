import os, sys
from glob import glob
import json
from random import randint

if __name__ == "__main__":
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
                            # for lr in script["non-warm"]["lr"][str(bs)]:
                            #     for sd in sds:
                            #         filepath = report_temp.format(dataset, model, bs, lr, sd)
                            #         filename = "/".join(filepath.split("/")[-3:])
                            #         if os.path.exists(filepath):
                            #             print(f"{filename}: Existed -> Skipped")
                            #         else:
                            #             print(f"{filename}: Non-Existed -> Conducted")
                            for lr in script_data["warm"]["lr"]:
                                cmd_lst.append(f"python main.py --bs {bs} --epochs 100 --lr {lr} --port {randint(3333, 8889)} --wd {w} --ds {dataset} --model {model} --opt lars --sd 'lars-warm'\n")
        
        if opt_name == 'tvlars':
            for dataset in datasets:
                for model in models:
                    for w in wd:
                        for bs in bss:
                            for lr in script_data["lr"][str(bs)]:
                                for lmd in script_data["lmbda"][str(bs)]:
                                    for sd in sds:
                                        cmd_lst.append(f"python main.py --bs {bs} --epochs 100 --lr {lr} --port {randint(3333, 8889)} --wd {w} --ds {dataset} --model {model} --opt tvlars --sd {sd} --lmbda {lmd}\n")
                                        
    
    # Generate script
    with open(run_file, mode='w') as file:
        file.writelines(
            cmd_lst
        )
        file.close()