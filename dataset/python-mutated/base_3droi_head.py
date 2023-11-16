from abc import ABCMeta, abstractmethod
from mmcv.runner import BaseModule

class Base3DRoIHead(BaseModule, metaclass=ABCMeta):
    """Base class for 3d RoIHeads."""

    def __init__(self, bbox_head=None, mask_roi_extractor=None, mask_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        if False:
            i = 10
            return i + 15
        super(Base3DRoIHead, self).__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if bbox_head is not None:
            self.init_bbox_head(bbox_head)
        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)
        self.init_assigner_sampler()

    @property
    def with_bbox(self):
        if False:
            return 10
        'bool: whether the RoIHead has box head'
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        if False:
            i = 10
            return i + 15
        'bool: whether the RoIHead has mask head'
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def init_bbox_head(self):
        if False:
            print('Hello World!')
        'Initialize the box head.'
        pass

    @abstractmethod
    def init_mask_head(self):
        if False:
            i = 10
            return i + 15
        'Initialize maek head.'
        pass

    @abstractmethod
    def init_assigner_sampler(self):
        if False:
            print('Hello World!')
        'Initialize assigner and sampler.'
        pass

    @abstractmethod
    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        if False:
            return 10
        'Forward function during training.\n\n        Args:\n            x (dict): Contains features from the first stage.\n            img_metas (list[dict]): Meta info of each image.\n            proposal_list (list[dict]): Proposal information from rpn.\n            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]):\n                GT bboxes of each sample. The bboxes are encapsulated\n                by 3D box structures.\n            gt_labels (list[torch.LongTensor]): GT labels of each sample.\n            gt_bboxes_ignore (list[torch.Tensor], optional):\n                Ground truth boxes to be ignored.\n\n        Returns:\n            dict[str, torch.Tensor]: Losses from each head.\n        '
        pass

    def simple_test(self, x, proposal_list, img_metas, proposals=None, rescale=False, **kwargs):
        if False:
            return 10
        'Test without augmentation.'
        pass

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Test with augmentations.\n\n        If rescale is False, then returned bboxes and masks will fit the scale\n        of imgs[0].\n        '
        pass