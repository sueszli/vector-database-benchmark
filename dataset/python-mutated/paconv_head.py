from mmcv.cnn.bricks import ConvModule
from ..builder import HEADS
from .pointnet2_head import PointNet2Head

@HEADS.register_module()
class PAConvHead(PointNet2Head):
    """PAConv decoder head.

    Decoder head used in `PAConv <https://arxiv.org/abs/2103.14635>`_.
    Refer to the `official code <https://github.com/CVMI-Lab/PAConv>`_.

    Args:
        fp_channels (tuple[tuple[int]]): Tuple of mlp channels in FP modules.
        fp_norm_cfg (dict): Config of norm layers used in FP modules.
            Default: dict(type='BN2d').
    """

    def __init__(self, fp_channels=((768, 256, 256), (384, 256, 256), (320, 256, 128), (128 + 6, 128, 128, 128)), fp_norm_cfg=dict(type='BN2d'), **kwargs):
        if False:
            i = 10
            return i + 15
        super(PAConvHead, self).__init__(fp_channels, fp_norm_cfg, **kwargs)
        self.pre_seg_conv = ConvModule(fp_channels[-1][-1], self.channels, kernel_size=1, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, feat_dict):
        if False:
            print('Hello World!')
        'Forward pass.\n\n        Args:\n            feat_dict (dict): Feature dict from backbone.\n\n        Returns:\n            torch.Tensor: Segmentation map of shape [B, num_classes, N].\n        '
        (sa_xyz, sa_features) = self._extract_input(feat_dict)
        fp_feature = sa_features[-1]
        for i in range(self.num_fp):
            fp_feature = self.FP_modules[i](sa_xyz[-(i + 2)], sa_xyz[-(i + 1)], sa_features[-(i + 2)], fp_feature)
        output = self.pre_seg_conv(fp_feature)
        output = self.cls_seg(output)
        return output