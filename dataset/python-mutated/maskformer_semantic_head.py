import torch
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from .base_panoptic_fusion_head import BasePanopticFusionHead

@HEADS.register_module()
class MaskFormerSemanticHead(BasePanopticFusionHead):

    def __init__(self, num_things_classes=80, num_stuff_classes=53, test_cfg=None, loss_panoptic=None, init_cfg=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(num_things_classes, num_stuff_classes, test_cfg, loss_panoptic, init_cfg, **kwargs)

    def forward_train(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'MaskFormerFusionHead has no training loss.'
        return dict()

    def simple_test(self, mask_cls_results, mask_pred_results, img_metas, rescale=False, **kwargs):
        if False:
            while True:
                i = 10
        results = []
        for (mask_cls_result, mask_pred_result, meta) in zip(mask_cls_results, mask_pred_results, img_metas):
            (img_height, img_width) = meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]
            if rescale:
                (ori_height, ori_width) = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(mask_pred_result[:, None], size=(ori_height, ori_width), mode='bilinear', align_corners=False)[:, 0]
            cls_score = F.softmax(mask_cls_result, dim=-1)[..., :-1]
            mask_pred = mask_pred_result.sigmoid()
            seg_mask = torch.einsum('qc,qhw->chw', cls_score, mask_pred)
            seg_logit = F.softmax(seg_mask, dim=0)
            seg_pred = seg_logit.argmax(dim=0)
            seg_pred = seg_pred.cpu().numpy()
            results.append(seg_pred)
        return results