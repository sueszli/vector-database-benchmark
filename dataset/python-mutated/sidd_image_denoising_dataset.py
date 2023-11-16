import cv2
import numpy as np
from modelscope.metainfo import CustomDatasets
from modelscope.msdatasets.dataset_cls.custom_datasets import CUSTOM_DATASETS, TorchCustomDataset
from modelscope.utils.constant import Tasks
from .data_utils import img2tensor, padding
from .transforms import augment, paired_random_crop

def default_loader(path):
    if False:
        print('Hello World!')
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

@CUSTOM_DATASETS.register_module(Tasks.image_denoising, module_name=CustomDatasets.SiddDataset)
class SiddImageDenoisingDataset(TorchCustomDataset):
    """Paired image dataset for image restoration.
    """

    def __init__(self, dataset, opt, is_train):
        if False:
            return 10
        self.dataset = dataset
        self.opt = opt
        self.is_train = is_train

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.dataset)

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        item_dict = self.dataset[index]
        gt_path = item_dict['Clean Image:FILE']
        img_gt = default_loader(gt_path)
        lq_path = item_dict['Noisy Image:FILE']
        img_lq = default_loader(lq_path)
        if self.is_train:
            gt_size = self.opt.gt_size
            (img_gt, img_lq) = padding(img_gt, img_lq, gt_size)
            (img_gt, img_lq) = paired_random_crop(img_gt, img_lq, gt_size, scale=1)
            (img_gt, img_lq) = augment([img_gt, img_lq], self.opt.use_flip, self.opt.use_rot)
        (img_gt, img_lq) = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        return {'input': img_lq, 'target': img_gt}