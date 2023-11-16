from typing import Any, Dict, List
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from modelscope.models.cv.image_instance_segmentation.maskdino.maskdino_encoder import MSDeformAttnTransformerEncoderOnly
from modelscope.models.cv.image_instance_segmentation.maskdino.position_encoding import PositionEmbeddingSine
from modelscope.models.cv.image_instance_segmentation.maskdino.utils import Conv2d

class MSDeformAttnPixelDecoder(nn.Module):

    def __init__(self, input_shape: Dict[str, Any], *, transformer_dropout: float, transformer_nheads: int, transformer_dim_feedforward: int, transformer_enc_layers: int, conv_dim: int, mask_dim: int, transformer_in_features: List[str], common_stride: int):
        if False:
            return 10
        '\n        NOTE: this interface is experimental.\n        Args:\n            input_shape: shapes (channels and stride) of the input features\n            transformer_dropout: dropout probability in transformer\n            transformer_nheads: number of heads in transformer\n            transformer_dim_feedforward: dimension of feedforward network\n            transformer_enc_layers: number of transformer encoder layers\n            conv_dim: number of output channels for the intermediate conv layers.\n            mask_dim: number of output channels for the final conv layer.\n        '
        super().__init__()
        self.conv_dim = conv_dim
        transformer_input_shape = {k: v for (k, v) in input_shape.items() if k in transformer_in_features}
        input_shape = sorted(input_shape.items(), key=lambda x: x[1]['stride'])
        self.in_features = [k for (k, v) in input_shape]
        self.feature_strides = [v['stride'] for (k, v) in input_shape]
        self.feature_channels = [v['channels'] for (k, v) in input_shape]
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1]['stride'])
        self.transformer_in_features = [k for (k, v) in transformer_input_shape]
        transformer_in_channels = [v['channels'] for (k, v) in transformer_input_shape]
        self.transformer_feature_strides = [v['stride'] for (k, v) in transformer_input_shape]
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim)))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim))])
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.transformer = MSDeformAttnTransformerEncoderOnly(d_model=conv_dim, dropout=transformer_dropout, nhead=transformer_nheads, dim_feedforward=transformer_dim_feedforward, num_encoder_layers=transformer_enc_layers, num_feature_levels=self.transformer_num_feature_levels)
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.mask_dim = mask_dim
        self.mask_features = Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        self.maskformer_num_feature_levels = 3
        self.common_stride = common_stride
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs = []
        output_convs = []
        use_bias = False
        for (idx, in_channels) in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = nn.GroupNorm(32, conv_dim)
            output_norm = nn.GroupNorm(32, conv_dim)
            lateral_conv = Conv2d(in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm)
            output_conv = Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=output_norm, activation=F.relu)
            self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
            self.add_module('layer_{}'.format(idx + 1), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @autocast(enabled=False)
    def forward_features(self, features):
        if False:
            while True:
                i = 10
        srcs = []
        pos = []
        for (idx, f) in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
        (y, spatial_shapes, level_start_index) = self.transformer(srcs, None, pos)
        bs = y.shape[0]
        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)
        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for (i, z) in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        for (idx, f) in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode='bilinear', align_corners=False)
            y = output_conv(y)
            out.append(y)
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        return (self.mask_features(out[-1]), out[0], multi_scale_features)