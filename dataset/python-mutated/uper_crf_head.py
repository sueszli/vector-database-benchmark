import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .newcrf_utils import normal_init, resize

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg, act_cfg, align_corners):
        if False:
            for i in range(10):
                print('nop')
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            if pool_scale == 1:
                norm_cfg = dict(type='GN', requires_grad=True, num_groups=256)
            self.append(nn.Sequential(nn.AdaptiveAvgPool2d(pool_scale), ConvModule(self.in_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=norm_cfg, act_cfg=self.act_cfg)))

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        'Forward function.'
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(ppm_out, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

class BaseDecodeHead(nn.Module):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self, in_channels, channels, *, num_classes, dropout_ratio=0.1, conv_cfg=None, norm_cfg=None, act_cfg=dict(type='ReLU'), in_index=-1, input_transform=None, loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), ignore_index=255, sampler=None, align_corners=False):
        if False:
            for i in range(10):
                print('nop')
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        if False:
            while True:
                i = 10
        'Extra repr.'
        s = f'input_transform={self.input_transform}, ignore_index={self.ignore_index}, align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        if False:
            for i in range(10):
                print('nop')
        "Check and initialize input transforms.\n\n        The in_channels, in_index and input_transform must match.\n        Specifically, when input_transform is None, only single feature map\n        will be selected. So in_channels and in_index must be of type int.\n        When input_transform\n\n        Args:\n            in_channels (int|Sequence[int]): Input channels.\n            in_index (int|Sequence[int]): Input feature index.\n            input_transform (str|None): Transformation type of input features.\n                Options: 'resize_concat', 'multiple_select', None.\n                'resize_concat': Multiple feature maps will be resize to the\n                    same size as first one and than concat together.\n                    Usually used in FCN head of HRNet.\n                'multiple_select': Multiple feature maps will be bundle into\n                    a list and passed into decode head.\n                None: Only one select feature map is allowed.\n        "
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        if False:
            while True:
                i = 10
        'Initialize weights of classification layer.'

    def _transform_inputs(self, inputs):
        if False:
            print('Hello World!')
        'Transform inputs for decoder.\n\n        Args:\n            inputs (list[Tensor]): List of multi-level img features.\n\n        Returns:\n            Tensor: The transformed inputs\n        '
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [resize(input=x, size=inputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners) for x in inputs]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        'Placeholder of forward function.'
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        if False:
            i = 10
            return i + 15
        "Forward function for training.\n        Args:\n            inputs (list[Tensor]): List of multi-level img features.\n            img_metas (list[dict]): List of image info dict where each dict\n                has: 'img_shape', 'scale_factor', 'flip', and may also contain\n                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.\n                For details on the values of these keys see\n                `mmseg/datasets/pipelines/formatting.py:Collect`.\n            gt_semantic_seg (Tensor): Semantic segmentation masks\n                used if the architecture supports semantic segmentation task.\n            train_cfg (dict): The training config.\n\n        Returns:\n            dict[str, Tensor]: a dictionary of loss components\n        "
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        if False:
            i = 10
            return i + 15
        "Forward function for testing.\n\n        Args:\n            inputs (list[Tensor]): List of multi-level img features.\n            img_metas (list[dict]): List of image info dict where each dict\n                has: 'img_shape', 'scale_factor', 'flip', and may also contain\n                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.\n                For details on the values of these keys see\n                `mmseg/datasets/pipelines/formatting.py:Collect`.\n            test_cfg (dict): The testing config.\n\n        Returns:\n            Tensor: Output segmentation map.\n        "
        return self.forward(inputs)

class UPerHead(BaseDecodeHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        if False:
            while True:
                i = 10
        super(UPerHead, self).__init__(input_transform='multiple_select', **kwargs)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels:
            l_conv = ConvModule(in_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=True)
            fpn_conv = ConvModule(self.channels, self.channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=True)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        'Forward function.'
        inputs = self._transform_inputs(inputs)
        laterals = [lateral_conv(inputs[i]) for (i, lateral_conv) in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(laterals[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        fpn_outs.append(laterals[-1])
        return fpn_outs[0]

class PSP(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        if False:
            i = 10
            return i + 15
        super(PSP, self).__init__(input_transform='multiple_select', **kwargs)
        self.psp_modules = PPM(pool_scales, self.in_channels[-1], self.channels, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, align_corners=self.align_corners)
        self.bottleneck = ConvModule(self.in_channels[-1] + len(pool_scales) * self.channels, self.channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        if False:
            i = 10
            return i + 15
        'Forward function of PSP module.'
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        'Forward function.'
        inputs = self._transform_inputs(inputs)
        return self.psp_forward(inputs)