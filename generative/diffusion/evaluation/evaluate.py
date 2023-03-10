
from PIL import Image
import glob
import timm
import torch
import numpy as np
from prdc import compute_prdc
from tqdm import tqdm
import pandas as pd
import os
import shutil
from DeepFlix.generative.diffusion.evaluation.dataset import get_dataloader
from DeepFlix.generative.diffusion.evaluation.feature_extractor import calculate_features
from DeepFlix.generative.diffusion.evaluation.dependencies.prd_curves import prd_score
from DeepFlix.generative.diffusion.evaluation.dependencies.fid import fid_score


def pr_values(real_dataloader, fake_dataloader):
    nearest_k = 5

    metrics = compute_prdc(real_features=real_dataloader,
                           fake_features=fake_dataloader,
                           nearest_k=nearest_k)
    return metrics

def evaluate_datasets(datasets:dict, real_data, sd_settings:dict=None, model=None, inception_pretrained=False):
    report_class_pr, report_class_fid  = dict(), dict()
    report_overall_pr , report_overall_fid = dict(), dict()

    classes = ["NORMAL", "CNV", "DME", "DRUSEN"]

    eval_dict = dict()

    eval_df = pd.DataFrame(columns=["Model", 
                                "inf_steps", 
                                "guidance_scale", 
                                "neg_prompt", 
                                "P_CNV", 
                                "R_CNV",
                                "D_CNV",
                                "C_CNV",
                                "FID_CNV",
                                "P_DRUSEN", 
                                "R_DRUSEN",
                                "D_DRUSEN",
                                "C_DRUSEN",
                                "FID_DRUSEN"
                                "P_NORMAL", 
                                "R_NORMAL",
                                "D_NORMAL",
                                "C_NORMAL",
                                "FID_NORMAL"
                                "P_DME", 
                                "R_DME",
                                "D_DME",
                                "C_DME",
                                "FID_DME",
                                "Precision",
                                "Recall",
                                "Density",
                                "Coverage",
                                "FID"])

    eval_df["Model"]= list(datasets.keys())


    for model_id, (key, value) in enumerate(datasets.items()):
        eval_df.loc[eval_df["Model"] == key, "inf_steps"] = sd_settings[key][0]
        eval_df.loc[eval_df["Model"] == key, "guidance_scale"] = sd_settings[key][1]
        eval_df.loc[eval_df["Model"] == key, "neg_prompt"] = sd_settings[key][2]

    print(f"Evaluating individual classes")
    for cl in tqdm(classes):
        real_dataloader = get_dataloader(f"{real_data}/{cl}/")
        for model_id, (key, value) in enumerate(datasets.items()):

            fake_dataloader = get_dataloader(f"{value}/{cl}/")

            real = calculate_features(real_dataloader, model=model, pretrained=inception_pretrained)
            fake = calculate_features(fake_dataloader, model=model, pretrained=inception_pretrained)

            values = pr_values(real,fake)
            fid = fid_score.main(f"{real_data}/{cl}/", f"{value}/{cl}/", 8, dims=2048, num_workers=None, pretrained=inception_pretrained)

            eval_df.loc[eval_df["Model"] == key, f"P_{cl}"] = values["precision"]
            eval_df.loc[eval_df["Model"] == key, f"R_{cl}"] = values["recall"]
            eval_df.loc[eval_df["Model"] == key, f"D_{cl}"] = values["density"]
            eval_df.loc[eval_df["Model"] == key, f"C_{cl}"] = values["coverage"]
            eval_df.loc[eval_df["Model"] == key, f"FID_{cl}"] = fid

    print(f"Evaluating overall performance")
    FOLDER_NAME_OVERALL_PERFORMANCE = "ALL"
    for cl in classes:
        # synthetic data
        for key, value in datasets.items():
            path = f"{value}/{FOLDER_NAME_OVERALL_PERFORMANCE}/"
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            shutil.copytree(f"{value}/{cl}/", f"{value}/{FOLDER_NAME_OVERALL_PERFORMANCE}/", dirs_exist_ok = True)

        # real data
        path = f"{real_data}/{FOLDER_NAME_OVERALL_PERFORMANCE}/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        shutil.copytree(f"{real_data}/{cl}/", f"{real_data}/{FOLDER_NAME_OVERALL_PERFORMANCE}/", dirs_exist_ok = True)

    # evaluating performance on overall data
    real_dataloader = get_dataloader(f"{real_data}/{FOLDER_NAME_OVERALL_PERFORMANCE}/")
    for key, value in datasets.items():
        
        fake_dataloader = get_dataloader(f"{value}/{FOLDER_NAME_OVERALL_PERFORMANCE}/")

        real = calculate_features(real_dataloader, model=model, pretrained=inception_pretrained)
        fake = calculate_features(fake_dataloader, model=model, pretrained=inception_pretrained)

        values = pr_values(real,fake)
        fid = fid_score.main(
            f"{real_data}/{FOLDER_NAME_OVERALL_PERFORMANCE}/", 
            f"{value}/{FOLDER_NAME_OVERALL_PERFORMANCE}/", 
            batch_size=8, 
            dims=2048, 
            num_workers=None, 
            pretrained=inception_pretrained)
        

        eval_df.loc[eval_df["Model"] == key, f"Precision"] = values["precision"]
        eval_df.loc[eval_df["Model"] == key, f"Recall"] = values["recall"]
        eval_df.loc[eval_df["Model"] == key, f"Density"] = values["density"]
        eval_df.loc[eval_df["Model"] == key, f"Coverage"] = values["coverage"]
        eval_df.loc[eval_df["Model"] == key, f"FID"] = fid
        
    return eval_df