try:
    import MinkowskiEngine as ME
except ImportError:
    pass
from mmdet3d.core import bbox3d2result
from mmdet3d.models import DETECTORS, build_backbone, build_head
from .base import Base3DDetector

@DETECTORS.register_module()
class MinkSingleStage3DDetector(Base3DDetector):
    """Single stage detector based on MinkowskiEngine `GSDN
    <https://arxiv.org/abs/2006.12356>`_.

    Args:
        backbone (dict): Config of the backbone.
        head (dict): Config of the head.
        voxel_size (float): Voxel size in meters.
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
        pretrained (str, optional): Deprecated initialization parameter.
            Defaults to None.
    """

    def __init__(self, backbone, head, voxel_size, train_cfg=None, test_cfg=None, init_cfg=None, pretrained=None):
        if False:
            return 10
        super(MinkSingleStage3DDetector, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        self.voxel_size = voxel_size
        self.init_weights()

    def extract_feat(self, points):
        if False:
            i = 10
            return i + 15
        'Extract features from points.\n\n        Args:\n            points (list[Tensor]): Raw point clouds.\n\n        Returns:\n            SparseTensor: Voxelized point clouds.\n        '
        (coordinates, features) = ME.utils.batch_sparse_collate([(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points], device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x)
        return x

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, img_metas):
        if False:
            while True:
                i = 10
        'Forward of training.\n\n        Args:\n            points (list[Tensor]): Raw point clouds.\n            gt_bboxes (list[BaseInstance3DBoxes]): Ground truth\n                bboxes of each sample.\n            gt_labels(list[torch.Tensor]): Labels of each sample.\n            img_metas (list[dict]): Contains scene meta infos.\n\n        Returns:\n            dict: Centerness, bbox and classification loss values.\n        '
        x = self.extract_feat(points)
        losses = self.head.forward_train(x, gt_bboxes_3d, gt_labels_3d, img_metas)
        return losses

    def simple_test(self, points, img_metas, *args, **kwargs):
        if False:
            return 10
        'Test without augmentations.\n\n        Args:\n            points (list[torch.Tensor]): Points of each sample.\n            img_metas (list[dict]): Contains scene meta infos.\n\n        Returns:\n            list[dict]: Predicted 3d boxes.\n        '
        x = self.extract_feat(points)
        bbox_list = self.head.forward_test(x, img_metas)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for (bboxes, scores, labels) in bbox_list]
        return bbox_results

    def aug_test(self, points, img_metas, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Test with augmentations.\n\n        Args:\n            points (list[list[torch.Tensor]]): Points of each sample.\n            img_metas (list[dict]): Contains scene meta infos.\n\n        Returns:\n            list[dict]: Predicted 3d boxes.\n        '
        raise NotImplementedError