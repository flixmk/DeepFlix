from dependencies.prd_curves import prd_score
from PIL import Image
import glob
import timm
import torch
import numpy as np
from prdc import compute_prdc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm