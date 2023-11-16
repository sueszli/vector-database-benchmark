from modelscope.models.cv.tinynas_detection.damo.structures.image_list import to_image_list

class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        if False:
            while True:
                i = 10
        self.size_divisible = size_divisible

    def __call__(self, batch):
        if False:
            return 10
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return (images, targets, img_ids)

class TTACollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        if False:
            print('Hello World!')
        return list(zip(*batch))