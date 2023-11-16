import cv2
import numpy as np
import torch
from modelscope.metainfo import Models
from modelscope.msdatasets.dataset_cls.custom_datasets import CUSTOM_DATASETS, TorchCustomDataset
from modelscope.msdatasets.dataset_cls.custom_datasets.video_frame_interpolation.data_utils import img2tensor, img_padding
from modelscope.utils.constant import Tasks

def default_loader(path):
    if False:
        for i in range(10):
            print('nop')
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

@CUSTOM_DATASETS.register_module(Tasks.video_frame_interpolation, module_name=Models.video_frame_interpolation)
class VideoFrameInterpolationDataset(TorchCustomDataset):
    """Dataset for video frame-interpolation.
    """

    def __init__(self, dataset, opt):
        if False:
            print('Hello World!')
        self.dataset = dataset
        self.opt = opt

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.dataset)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        item_dict = self.dataset[index]
        img0 = default_loader(item_dict['Input1:FILE'])
        img1 = default_loader(item_dict['Input2:FILE'])
        img2 = default_loader(item_dict['Input3:FILE'])
        img3 = default_loader(item_dict['Input4:FILE'])
        gt = default_loader(item_dict['Output:FILE'])
        (img0, img1, img2, img3, gt) = img2tensor([img0, img1, img2, img3, gt], bgr2rgb=False, float32=True)
        imgs = torch.cat((img0, img1, img2, img3), dim=0)
        (height, width) = (imgs.size(1), imgs.size(2))
        imgs = img_padding(imgs, height, width, pad_num=32)
        return {'input': imgs, 'target': gt / 255.0}