import os
import io
import pandas as pd
import PIL.Image as Image
from tqdm import tqdm
import json
import math

if __name__ == '__main__':
    save_dir = os.getcwd() + "/src"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    train_save_dir = save_dir + "/train"
    test_save_dir = save_dir + "/test"
    
    for folder in [train_save_dir, test_save_dir]:
        if not os.path.exists(folder):
            os.mkdir(folder)
            
    with open('dataset_infos.json', mode='r') as file:
        ds_info = json.load(file)
    
    cls_lst = ds_info['Maysee--tiny-imagenet']['features']['label']['names']
    
    train_df = pd.read_parquet('train-00000-of-00001-1359597a978bc4fa.parquet')
    test_df = pd.read_parquet('valid-00000-of-00001-70d52db3c749a935.parquet')
    
    for base_idx, base_df in enumerate([train_df, test_df]):
        if base_idx == 0:
            base_save_dir = train_save_dir
        elif base_idx == 1:
            base_save_dir = test_save_dir
        label_value_cnt = base_df['label'].value_counts()
        with tqdm(total=len(label_value_cnt)) as pbar:
            for index, cls_idx in enumerate(label_value_cnt):
                cls = cls_lst[index]
                
                cls_save_dir = base_save_dir + f"/{cls}"
                if not os.path.exists(cls_save_dir):
                    os.mkdir(cls_save_dir)
                
                cls_base_df = base_df[base_df['label'] == index]
                for idx, row in cls_base_df.iterrows():
                    save_path = cls_save_dir + f"/{idx}.png"
                    
                    img = Image.open(io.BytesIO(row['image']['bytes']))
                    img.save(save_path)
                pbar.update(1)