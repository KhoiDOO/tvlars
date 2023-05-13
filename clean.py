import os, sys
from glob import glob
import pandas as pd
import json

if __name__ == "__main__":
    runs_dir = os.getcwd() + "/runs"
    report_path = os.getcwd() + '/clean_report.json'
    
    data_model_dir = glob(runs_dir + "/*")
    
    report = {
        "empty_folder" : [],
        "empty_opt_folder" : []
    }
    
    for dm_dir in data_model_dir:
        
        opt_dirs = glob(dm_dir + "/*")
        
        if len(opt_dirs) == 0:
            report["empty_folder"].append(dm_dir)
            continue
        
        for opt_dir in opt_dirs:
            
            log_files = glob(opt_dir + "/*.parquet")
            
            if len(log_files) == 0:
                report["empty_opt_folder"].append(opt_dir)
                continue
                
            for log_file in log_files:
                base_df = pd.read_parquet(log_file)
                
                if base_df.shape[0] < 10:
                    os.remove(
                        path=log_file
                    )
                    
    with open(file=report_path, mode='w') as file:
        json.dump(report, file)