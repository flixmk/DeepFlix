import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_files = None
        self.len = None
        self.images = list()
        
        
        self.img_files = glob.glob(self.img_dir + "/*.png")
        self.len = len(self.img_files)
        for file in self.img_files:
            img = Image.open(file)
            img = img.convert(mode='RGB')
            img = np.array(img).astype(np.uint8)
            self.images.append(img)
        self.images = np.array(self.images)
        self.images = torch.from_numpy(self.images).squeeze().permute(0,3,2,1)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.images[idx]
    
def get_dataloader(img_dir):
    
    dataset = ImageDataset(img_dir)
    return DataLoader(dataset, batch_size=8)