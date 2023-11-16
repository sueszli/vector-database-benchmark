import numpy as np
from neon.data.dataloader_transformers import DataLoaderTransformer
from neon.data.dataloader_transformers import TypeCast, BGRMeanSubtract
from aeon import DataLoader
from neon.data.dataloaderadapter import DataLoaderAdapter

class ObjectLocalization(DataLoaderTransformer):

    def __init__(self, dataloader, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ObjectLocalization, self).__init__(dataloader, None, *args, **kwargs)
        self.img_shape = dataloader.shapes()[0]
        self.dataloader = dataloader

    def transform(self, t):
        if False:
            while True:
                i = 10
        (im_shape, gt_boxes, num_gt_boxes, gt_classes, difficult, img) = t
        self.img_shape = im_shape
        gt_boxes = gt_boxes.get().reshape((-1, 4, self.be.bsz))
        gt_boxes = gt_boxes.reshape((-1, self.be.bsz))
        return (img, (gt_boxes, gt_classes.get(), num_gt_boxes.get(), difficult.get(), im_shape.get()))

    def get_img_shape(self):
        if False:
            return 10
        return self.img_shape

    def set_classes(self, classes):
        if False:
            for i in range(10):
                print('nop')
        self.CLASSES = classes
        self.num_classes = len(self.CLASSES)

def build_dataloader(config, manifest_root, batch_size, subset_pct=100, PIXEL_MEANS=np.array([104, 117, 123])):
    if False:
        for i in range(10):
            print('nop')
    '\n    Builds the dataloader for the Faster-RCNN network using our aeon loader.\n    Besides, the base loader, we add several operations:\n    1. Cast the image data into float32 format\n    2. Subtract the BGRMean from the image. We used pre-defined means from training\n       the VGG network.\n    3. Repack the data for Faster-RCNN model. This model has several nested branches, so\n       The buffers have to repacked into nested tuples to match the branch leafs. Additionally,\n       buffers for training the RCNN portion of the model are also allocated and provisioned\n       to the model.\n\n    Arguments:\n        config (dict): dataloader configuration\n        be (backend): compute backend\n\n    Returns:\n        dataloader object.\n    '
    config['manifest_root'] = manifest_root
    config['batch_size'] = batch_size
    config['subset_fraction'] = float(subset_pct / 100.0)
    dl = DataLoaderAdapter(DataLoader(config))
    dl = TypeCast(dl, index=5, dtype=np.float32)
    dl = BGRMeanSubtract(dl, index=5, pixel_mean=PIXEL_MEANS)
    dl = ObjectLocalization(dl)
    dl.set_classes(config['etl'][0]['class_names'])
    dl.shape = dl.shapes()[5]
    return dl