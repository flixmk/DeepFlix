from DeepFlix.generative.diffusion.evaluation.dependencies.prd_curves import prd_score
from DeepFlix.generative.diffusion.evaluation.dependencies.fid import fid_score
from PIL import Image
import glob
import timm
import torch
import numpy as np
from prdc import compute_prdc
from tqdm import tqdm
from DeepFlix.generative.diffusion.evaluation.dataset import get_dataloader
from DeepFlix.generative.diffusion.evaluation.feature_extractor import calculate_features
import shlex, subprocess
import re


def pr_values(real_dataloader, fake_dataloader):
    nearest_k = 5

    metrics = compute_prdc(real_features=real_dataloader,
                           fake_features=fake_dataloader,
                           nearest_k=nearest_k)
    return metrics

def evaluate_datasets(datasets:dict, real_data, model=None, inception_pretrained=False):
    report = dict()
    classes = ["NORMAL", "CNV", "DME", "DRUSEN"]
    print(inception_pretrained)
    for cl in classes:
        print(f"Class: {cl}")
        
        for key, value in datasets.items():
            print(f"Dataset: {key}")
            real_dataloader = get_dataloader(f"{real_data}/{cl}/")
            fake_dataloader = get_dataloader(f"{value}/{cl}/")

            real = calculate_features(real_dataloader, model=model, pretrained=inception_pretrained)
            fake = calculate_features(fake_dataloader, model=model, pretrained=inception_pretrained)

            values = pr_values(real,fake)
            print(values)
            

            report[f"{key}-{cl}"] = values
            
            fid_score.main(f"{real_data}/{cl}/", f"{value}/{cl}/", 8, dims=2048, num_workers=None, pretrained=inception_pretrained)

            print("-----------------------------------------------------")
    
    return report