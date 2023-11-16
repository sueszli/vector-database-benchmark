from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch

class imageDataset(Dataset):
    def __init__(self, filelist_path, img_dir, transform=None,
        target_transform=None) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.filelist_path = filelist_path
        self.filelist = pd.read_csv(filelist_path, header=None, on_bad_lines='warn')
        self.img_labels = self.filelist.iloc[:, 1:].values
        self.transform = transform
        self.target_transform = target_transform
        self.labels = self.collectLabels()
        self.num_lables = len(self.labels)
        self.oneHot = self.initOnehot()
    
    def img_loader(self, path):
        return Image.open(path).convert('RGB')
    
    def initOnehot(self):
        oneHot = np.zeros(self.num_lables)
        return oneHot
    
    def makeOnehot(self, labels):
        onehotLabel = np.zeros(self.num_lables)
        for lab in labels:
            onehotLabel[self.labels.index(lab)] = 1
        return onehotLabel
        
    def collectLabels(self):
        img_labels = self.img_labels
        img_labels = img_labels.reshape(1, -1).tolist()[0]
        while np.nan in img_labels:
            img_labels.remove(np.nan)
        labels = list(set(img_labels))
        return labels

    def getNumLabels(self):            
        return self.num_lables
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filelist.iloc[idx, 0])
        label = [i for i in self.filelist.iloc[idx, 1:] if type(i) == str]
        label = self.makeOnehot(label)
        image = self.img_loader(img_path)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.FloatTensor(label)
        return image, label