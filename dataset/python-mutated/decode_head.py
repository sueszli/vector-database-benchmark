from abc import ABCMeta, abstractmethod
from mmcv.cnn import normal_init
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch import nn as nn
from mmseg.models.builder import build_loss

class Base3DDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float, optional): Ratio of dropout layer. Default: 0.5.
        conv_cfg (dict, optional): Config of conv layers.
            Default: dict(type='Conv1d').
        norm_cfg (dict, optional): Config of norm layers.
            Default: dict(type='BN1d').
        act_cfg (dict, optional): Config of activation layers.
            Default: dict(type='ReLU').
        loss_decode (dict, optional): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int, optional): The label index to be ignored.
            When using masked BCE loss, ignore_index should be set to None.
            Default: 255.
    """

    def __init__(self, channels, num_classes, dropout_ratio=0.5, conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'), act_cfg=dict(type='ReLU'), loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, class_weight=None, loss_weight=1.0), ignore_index=255, init_cfg=None):
        if False:
            return 10
        super(Base3DDecodeHead, self).__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.conv_seg = nn.Conv1d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def init_weights(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize weights of classification layer.'
        super().init_weights()
        normal_init(self.conv_seg, mean=0, std=0.01)

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        if False:
            return 10
        'Placeholder of forward function.'
        pass

    def forward_train(self, inputs, img_metas, pts_semantic_mask, train_cfg):
        if False:
            i = 10
            return i + 15
        'Forward function for training.\n\n        Args:\n            inputs (list[torch.Tensor]): List of multi-level point features.\n            img_metas (list[dict]): Meta information of each sample.\n            pts_semantic_mask (torch.Tensor): Semantic segmentation masks\n                used if the architecture supports semantic segmentation task.\n            train_cfg (dict): The training config.\n\n        Returns:\n            dict[str, Tensor]: a dictionary of loss components\n        '
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, pts_semantic_mask)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        if False:
            for i in range(10):
                print('nop')
        'Forward function for testing.\n\n        Args:\n            inputs (list[Tensor]): List of multi-level point features.\n            img_metas (list[dict]): Meta information of each sample.\n            test_cfg (dict): The testing config.\n\n        Returns:\n            Tensor: Output segmentation map.\n        '
        return self.forward(inputs)

    def cls_seg(self, feat):
        if False:
            for i in range(10):
                print('nop')
        'Classify each points.'
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        if False:
            print('Hello World!')
        'Compute semantic segmentation loss.\n\n        Args:\n            seg_logit (torch.Tensor): Predicted per-point segmentation logits\n                of shape [B, num_classes, N].\n            seg_label (torch.Tensor): Ground-truth segmentation label of\n                shape [B, N].\n        '
        loss = dict()
        loss['loss_sem_seg'] = self.loss_decode(seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss