from dependencies.prd_curves import prd_score
from PIL import Image
import glob
import timm
import torch
import numpy as np
from prdc import compute_prdc
from tqdm import tqdm
# from dataset import get_dataloader
# from feature_extractor import calculate_features
# from subprocess import call


def pr_values(real_dataloader, fake_dataloader):
    nearest_k = 5

    metrics = compute_prdc(real_features=real_dataloader,
                           fake_features=fake_dataloader,
                           nearest_k=nearest_k)
    return metrics

def evaluate_datasets(datasets:dict, model, real_data, inception_pretrained=False):
    report = dict()
    classes = ["NORMAL", "CNV", "DME", "DRUSEN"]
    for cl in classes:
        print(f"Class: {cl}")
        
        for key, value in datasets.items():
            print(f"Dataset: {key}")
            real_dataloader = get_dataloader(f"{real_data}/{cl}/")
            fake_dataloader = get_dataloader(f"{value}/{cl}/")

            real = calculate_features(real_dataloader, pretrained=inception_pretrained)
            fake = calculate_features(fake_dataloader, pretrained=inception_pretrained)

            values = pr_values(real,fake)
            print(values)
            

            report[f"{key}-{cl}"] = values
            
            call("python -m pytorch_fid {value}/{cl} {real_data}/{cl} --device cuda:0")
                
            print("-----------------------------------------------------")
    
    return report