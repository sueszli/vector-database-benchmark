import os
import pickle
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from .util import add_gen_flag, normalize_per_sample, renormalize_per_sample

class DoppelGANgerDataModule(LightningDataModule):
    """
    Note that for now, we will still follow the Dataset format stated in
    https://github.com/fjxmlzn/DoppelGANger#dataset-format.

    Please notice that this module can not work alone without doppelganger_torch.
    """

    def __init__(self, sample_len, real_data, feature_outputs, attribute_outputs, batch_size=32):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.sample_len = sample_len
        self.batch_size = batch_size
        data_all = real_data['data_feature']
        data_attribute = real_data['data_attribute']
        data_gen_flag = real_data['data_gen_flag']
        data_feature_outputs = feature_outputs
        data_attribute_outputs = attribute_outputs
        self.num_real_attribute = len(data_attribute_outputs)
        self.num_feature_dim = len(data_feature_outputs)
        (data_feature, data_attribute, data_attribute_outputs, real_attribute_mask) = normalize_per_sample(data_all, data_attribute, data_feature_outputs, data_attribute_outputs)
        (data_feature, data_feature_outputs) = add_gen_flag(data_feature, data_gen_flag, data_feature_outputs, self.sample_len)
        self.data_feature_outputs = data_feature_outputs
        self.data_attribute_outputs = data_attribute_outputs
        self.real_attribute_mask = real_attribute_mask
        total_generate_num_sample = data_feature.shape[0]
        from bigdl.nano.utils.common import invalidInputError
        if data_feature.shape[1] % self.sample_len != 0:
            invalidInputError(False, 'length must be a multiple of sample_len')
        self.length = int(data_feature.shape[1] / self.sample_len)
        self.data_feature = data_feature
        self.data_attribute = data_attribute

    def train_dataloader(self):
        if False:
            i = 10
            return i + 15
        self.data_feature = torch.from_numpy(self.data_feature).float()
        self.data_attribute = torch.from_numpy(self.data_attribute).float()
        dataset = CustomizedDataset(self.data_feature, self.data_attribute)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

class CustomizedDataset(Dataset):

    def __init__(self, data_feature, data_attribute):
        if False:
            for i in range(10):
                print('nop')
        self.data_feature = data_feature
        self.data_attribute = data_attribute

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self.data_feature.shape[0]

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        return (self.data_feature[index], self.data_attribute[index])