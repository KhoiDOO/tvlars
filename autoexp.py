import os, sys
import subprocess
import argparse
import json
from random import randint
import signal

parser = argparse.ArgumentParser(
    prog="Automation Experiment Optimizer Benchmark"
)
parser.add_argument("--opt", type=str, default='lars', choices=['lars', 'tvlars'],
                    help='Optimizer used in automation experiment')

args = parser.parse_args()

if __name__ == "__main__":
    script_path = os.getcwd() + f"/script/{args.opt}/script.json"
    
    script = json.load(open(script_path, mode='r'))
    
    datasets = script["ds"]
    models = script["model"]
    epoch = script["epochs"]
    wd = script["wd"]
    bss = script["bs"]
    sds = script["sd"]
    
    if args.opt == 'lars':
        report_temp = os.getcwd() + "/{0}_{1}/lars/{2}_{3}_{4}.parquet"
        
        for dataset in datasets:
            for model in models:
                for w in wd:
                    for bs in bss:
                        for lr in script["non-warm"]["lr"][str(bs)]:
                            for sd in sds:
                                filepath = report_temp.format(dataset, model, bs, lr, sd)
                                filename = "/".join(filepath.split("/")[-3:])
                                if os.path.exists(filepath):
                                    print(f"{filename}: Existed -> Skipped")
                                else:
                                    print(f"{filename}: Non-Existed -> Conducted")
                                    conduct = True
                                    while(conduct):
                                        try:
                                            process = subprocess.Popen([
                                                "python", "single.py", 
                                                "--bs", str(bs), 
                                                "--workers", "4",
                                                "--epochs", str(epoch),
                                                "--port", str(randint(4444, 8889)),
                                                "--wd", str(w),
                                                "--ds", dataset,
                                                "--model", model,
                                                "--opt", "lars",
                                                "--sd", sd
                                            ],
                                                                       stdin=subprocess.PIPE,
                                                                       stdout=subprocess.PIPE,
                                                                       stderr=subprocess.PIPE)
                                            conduct = False
                                        except RuntimeError as e:
                                            print(e)
                                            process.send_signal(signal.CTRL_C_EVENT)
                                            continue
                
    elif args.opt == 'tvlars':
        report_temp = os.getcwd() + "/{0}_{1}/tvlars_{2}/{3}_{4}_{5}.parquet"
        
        for dataset in datasets:
            for model in models:
                for w in wd:
                    for bs in bss:
                        for lr in script["lr"][str(bs)]:
                            for lmd in script["lmbda"][str(bs)]:
                                for sd in sds:
                                    filepath = report_temp.format(dataset, model, lmd, bs, lr, sd)
                                    filename = "/".join(filepath.split("/")[-4:])
                                    if os.path.exists(filepath):
                                        print(f"{filename}: Existed -> Skipped")
                                    else:
                                        print(f"{filename}: Non-Existed -> Conducted")
                                        conduct = True
                                        while(conduct):
                                            try:
                                                process = subprocess.Popen([
                                                    "python", "single.py", 
                                                    "--bs", str(bs), 
                                                    "--workers", "4",
                                                    "--epochs", str(epoch),
                                                    "--port", str(randint(4444, 8889)),
                                                    "--wd", str(w),
                                                    "--ds", dataset,
                                                    "--model", model,
                                                    "--opt", "tvlars",
                                                    "--sd", sd,
                                                    "--lmbda", str(lmd)
                                                ],
                                                                        stdin=subprocess.PIPE,
                                                                        stdout=subprocess.PIPE,
                                                                        stderr=subprocess.PIPE)
                                                conduct = False
                                            except RuntimeError as e:
                                                print(e)
                                                process.send_signal(signal.CTRL_C_EVENT)
                                                continue