import mmcv
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class ResizeToMultiple(object):
    """Resize images & seg to multiple of divisor.

    Args:
        size_divisor (int): images and gt seg maps need to resize to multiple
            of size_divisor. Default: 32.
        interpolation (str, optional): The interpolation mode of image resize.
            Default: None
    """

    def __init__(self, size_divisor=32, interpolation=None):
        if False:
            for i in range(10):
                print('nop')
        self.size_divisor = size_divisor
        self.interpolation = interpolation

    def __call__(self, results):
        if False:
            for i in range(10):
                print('nop')
        "Call function to resize images, semantic segmentation map to\n        multiple of size divisor.\n\n        Args:\n            results (dict): Result dict from loading pipeline.\n\n        Returns:\n            dict: Resized results, 'img_shape', 'pad_shape' keys are updated.\n        "
        img = results['img']
        img = mmcv.imresize_to_multiple(img, self.size_divisor, scale_factor=1, interpolation=self.interpolation if self.interpolation else 'bilinear')
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            gt_seg = mmcv.imresize_to_multiple(gt_seg, self.size_divisor, scale_factor=1, interpolation='nearest')
            results[key] = gt_seg
        return results

    def __repr__(self):
        if False:
            print('Hello World!')
        repr_str = self.__class__.__name__
        repr_str += f'(size_divisor={self.size_divisor}, interpolation={self.interpolation})'
        return repr_str