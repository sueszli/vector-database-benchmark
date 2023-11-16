import numpy as np
import pickle
from numpy.core.overrides import os
import pandas as pd
from PIL import Image

class dataLoader:
    def __init__(self, filelist_path, img_dir, img_fm_path):
        self.img_dir = img_dir
        self.filelist_path = filelist_path
        self.img_featuremaps_path = img_fm_path
        self.dataLoadObjectSavingPath = './objects'
        
        self.filelist_raw = pd.read_csv(self.filelist_path, header=None, on_bad_lines='warn')
        self.filelist = self.cleanFileList()    # 去掉nan
        
        self.items_list = self.getImgNames()    # index2item
        self.item2index = self.item2index()
        
        self.labels_list = self.getAllLabels()     # index2labels
        self.label2index = self.label2index()
        self.labels_byItem = list(self.filelist.values())
        self.label_freq = self.getLabelFreq()
        
        self.img_featuremaps = self.getImageFeaturemaps()
    
    def getLabels(self, name):
        if type(name) == str:
            return self.filelist[name]
        elif type(name) == int:
            return self.filelist[self.items_list[name]]
        
    def cleanFileList(self):
        file_dict = {}
        filelist_raw = self.filelist_raw.values
        for line in filelist_raw:
            file_dict[line[0]] = [i for i in line[1:] if type(i) == str]
        return file_dict
    
    def getAllLabels(self):
        tmp = self.filelist.values()
        labels = set()
        for item in tmp:
            for label in item:
                labels.add(label)
        return list(labels)

    def getImgNames(self):
        tmp = self.filelist.keys()
        img_names = []
        for item in tmp:
            img_names.append(item)
        return img_names
    
    def label2index(self):
        label2index = dict()
        for idx, label in enumerate(self.labels_list):
            label2index[label] = idx
        return label2index
    
    def getLabelFreq(self):
        lbs = {}
        for line in self.labels_byItem:
            for label in line:
                if label not in lbs.keys():
                    lbs[label] = 1
                else:
                    lbs[label] += 1
        return lbs
    
    def item2index(self):
        items = {}
        for idx, line in enumerate(self.items_list):
            items[line] = idx
        return items
    
    def getImage(self, name):
        if type(name) == str:
            return os.path.join(self.img_dir, name)
        
        elif type(name) == int:
            img_name = self.items_list[name]
            return os.path.join(self.img_dir, img_name)
    
    def getImageFeaturemaps(self):
        img_fm = {}
        with open(self.img_featuremaps_path, 'rb') as f:
            img_fm = pickle.load(f)
        return img_fm
        
    def __labelSize__(self):
        return len(self.labels_list)
    
    def __len__(self):
        return len(self.filelist)
    

# dl = dataLoader('./filelist/filelist.csv', './imgs', './features/features.json')