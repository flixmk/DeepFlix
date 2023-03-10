import timm
import torch
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_features(dataloader, model=None, pretrained=None):

    if model is None:
        if pretrained:
            model = timm.create_model('inception_v3', pretrained=True, num_classes=4).to(device)
            model.load_state_dict(torch.load("../models/finetuned_best.pt"))
            model.eval()
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
        elif pretrained == False:
            model = timm.create_model('inception_v3', pretrained=True, num_classes=0).to(device)
            model.eval()
        else:
            print("No model configured")

    dummy_output = model(torch.randn(1, 3, 512, 512).to(device))
    features = np.empty((0,int(dummy_output.shape[1]))), np.empty((0,int(dummy_output.shape[1])))
    for count, image in tqdm(enumerate(dataloader)):
        output = model(image.float().to(device))
        features = np.concatenate((features, output.cpu().detach().numpy()), axis=0)
    return features