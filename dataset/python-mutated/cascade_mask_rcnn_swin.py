import os
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn as nn
from modelscope.models.cv.image_instance_segmentation.backbones import SwinTransformer
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
logger = get_logger()

def build_backbone(cfg):
    if False:
        while True:
            i = 10
    assert isinstance(cfg, dict)
    cfg = cfg.copy()
    type = cfg.pop('type')
    if type == 'SwinTransformer':
        return SwinTransformer(**cfg)
    else:
        raise ValueError(f"backbone '{type}' is not supported.")

def build_neck(cfg):
    if False:
        while True:
            i = 10
    assert isinstance(cfg, dict)
    cfg = cfg.copy()
    type = cfg.pop('type')
    if type == 'FPN':
        from mmdet.models import FPN
        return FPN(**cfg)
    else:
        raise ValueError(f"neck '{type}' is not supported.")

def build_rpn_head(cfg):
    if False:
        i = 10
        return i + 15
    assert isinstance(cfg, dict)
    cfg = cfg.copy()
    type = cfg.pop('type')
    if type == 'RPNHead':
        from mmdet.models import RPNHead
        return RPNHead(**cfg)
    else:
        raise ValueError(f"rpn head '{type}' is not supported.")

def build_roi_head(cfg):
    if False:
        print('Hello World!')
    assert isinstance(cfg, dict)
    cfg = cfg.copy()
    type = cfg.pop('type')
    if type == 'CascadeRoIHead':
        from mmdet.models import CascadeRoIHead
        return CascadeRoIHead(**cfg)
    else:
        raise ValueError(f"roi head '{type}' is not supported.")

class CascadeMaskRCNNSwin(nn.Module):

    def __init__(self, backbone, neck, rpn_head, roi_head, pretrained=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Args:\n            backbone (dict): backbone config.\n            neck (dict): neck config.\n            rpn_head (dict): rpn_head config.\n            roi_head (dict): roi_head config.\n            pretrained (bool): whether to use pretrained model\n        '
        super(CascadeMaskRCNNSwin, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.rpn_head = build_rpn_head(rpn_head)
        self.roi_head = build_roi_head(roi_head)
        self.classes = kwargs.pop('classes', None)
        if pretrained:
            assert 'model_dir' in kwargs, 'pretrained model dir is missing.'
            model_path = os.path.join(kwargs['model_dir'], ModelFile.TORCH_MODEL_FILE)
            logger.info(f'loading model from {model_path}')
            weight = torch.load(model_path, map_location='cpu')['state_dict']
            tgt_weight = self.state_dict()
            for name in list(weight.keys()):
                if name in tgt_weight:
                    load_size = weight[name].size()
                    tgt_size = tgt_weight[name].size()
                    mis_match = False
                    if len(load_size) != len(tgt_size):
                        mis_match = True
                    else:
                        for (n1, n2) in zip(load_size, tgt_size):
                            if n1 != n2:
                                mis_match = True
                                break
                    if mis_match:
                        logger.info(f'size mismatch for {name}, skip loading.')
                        del weight[name]
            self.load_state_dict(weight, strict=False)
            logger.info('load model done')
        from mmcv.parallel import DataContainer, scatter
        self.data_container = DataContainer
        self.scatter = scatter

    def extract_feat(self, img):
        if False:
            print('Hello World!')
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Args:\n            img (Tensor): of shape (N, C, H, W) encoding input images.\n                Typically these should be mean centered and std scaled.\n\n            img_metas (list[dict]): list of image info dict where each dict\n                has: 'img_shape', 'scale_factor', 'flip', and may also contain\n                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.\n                For details on the values of these keys see\n                `mmdet/datasets/pipelines/formatting.py:Collect`.\n\n            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with\n                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.\n\n            gt_labels (list[Tensor]): class indices corresponding to each box\n\n            gt_bboxes_ignore (None | list[Tensor]): specify which bounding\n                boxes can be ignored when computing the loss.\n\n            gt_masks (None | Tensor) : true segmentation masks for each box\n                used if the architecture supports a segmentation task.\n\n            proposals : override rpn proposals with custom proposals. Use when\n                `with_rpn` is False.\n\n        Returns:\n            dict[str, Tensor]: a dictionary of loss components\n        "
        x = self.extract_feat(img)
        losses = dict()
        proposal_cfg = self.rpn_head.train_cfg.get('rpn_proposal', self.rpn_head.test_cfg)
        (rpn_losses, proposal_list) = self.rpn_head.forward_train(x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg, **kwargs)
        losses.update(rpn_losses)
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)
        return losses

    def forward_test(self, img, img_metas, proposals=None, rescale=True):
        if False:
            print('Hello World!')
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        result = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
        return dict(eval_result=result, img_metas=img_metas)

    def forward(self, img, img_metas, **kwargs):
        if False:
            while True:
                i = 10
        if isinstance(img, self.data_container):
            img = img.data[0]
        if isinstance(img_metas, self.data_container):
            img_metas = img_metas.data[0]
        for (k, w) in kwargs.items():
            if isinstance(w, self.data_container):
                w = w.data[0]
            kwargs[k] = w
        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).device
            img = self.scatter(img, [device])[0]
            img_metas = self.scatter(img_metas, [device])[0]
            for (k, w) in kwargs.items():
                kwargs[k] = self.scatter(w, [device])[0]
        if self.training:
            losses = self.forward_train(img, img_metas, **kwargs)
            (loss, log_vars) = self._parse_losses(losses)
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(img_metas))
            return outputs
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        if False:
            while True:
                i = 10
        log_vars = OrderedDict()
        for (loss_name, loss_value) in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum((_loss.mean() for _loss in loss_value))
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')
        loss = sum((_value for (_key, _value) in log_vars.items() if 'loss' in _key))
        log_vars['loss'] = loss
        for (loss_name, loss_value) in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        return (loss, log_vars)

    def train_step(self, data, optimizer):
        if False:
            i = 10
            return i + 15
        losses = self(**data)
        (loss, log_vars) = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs

    def val_step(self, data, optimizer=None):
        if False:
            i = 10
            return i + 15
        losses = self(**data)
        (loss, log_vars) = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs