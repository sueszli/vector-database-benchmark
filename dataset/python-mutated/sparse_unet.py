import torch
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential
from mmcv.runner import BaseModule, auto_fp16
from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops.sparse_block import replace_feature
from ..builder import MIDDLE_ENCODERS

@MIDDLE_ENCODERS.register_module()
class SparseUNet(BaseModule):
    """SparseUNet for PartA^2.

    See the `paper <https://arxiv.org/abs/1907.03670>`_ for more details.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        norm_cfg (dict): Config of normalization layer.
        base_channels (int): Out channels for conv_input layer.
        output_channels (int): Out channels for conv_out layer.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
        decoder_channels (tuple[tuple[int]]):
            Convolutional channels of each decode block.
        decoder_paddings (tuple[tuple[int]]): Paddings of each decode block.
    """

    def __init__(self, in_channels, sparse_shape, order=('conv', 'norm', 'act'), norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01), base_channels=16, output_channels=128, encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)), encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)), decoder_channels=((64, 64, 64), (64, 64, 32), (32, 32, 16), (16, 16, 16)), decoder_paddings=((1, 0), (1, 0), (0, 0), (0, 1)), init_cfg=None):
        if False:
            while True:
                i = 10
        super().__init__(init_cfg=init_cfg)
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.decoder_channels = decoder_channels
        self.decoder_paddings = decoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}
        if self.order[0] != 'conv':
            self.conv_input = make_sparse_convmodule(in_channels, self.base_channels, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm1', conv_type='SubMConv3d', order=('conv',))
        else:
            self.conv_input = make_sparse_convmodule(in_channels, self.base_channels, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm1', conv_type='SubMConv3d')
        encoder_out_channels = self.make_encoder_layers(make_sparse_convmodule, norm_cfg, self.base_channels)
        self.make_decoder_layers(make_sparse_convmodule, norm_cfg, encoder_out_channels)
        self.conv_out = make_sparse_convmodule(encoder_out_channels, self.output_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), norm_cfg=norm_cfg, padding=0, indice_key='spconv_down2', conv_type='SparseConv3d')

    @auto_fp16(apply_to=('voxel_features',))
    def forward(self, voxel_features, coors, batch_size):
        if False:
            i = 10
            return i + 15
        'Forward of SparseUNet.\n\n        Args:\n            voxel_features (torch.float32): Voxel features in shape [N, C].\n            coors (torch.int32): Coordinates in shape [N, 4],\n                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).\n            batch_size (int): Batch size.\n\n        Returns:\n            dict[str, torch.Tensor]: Backbone features.\n        '
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)
        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()
        (N, C, D, H, W) = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        decode_features = []
        x = encode_features[-1]
        for i in range(self.stage_num, 0, -1):
            x = self.decoder_layer_forward(encode_features[i - 1], x, getattr(self, f'lateral_layer{i}'), getattr(self, f'merge_layer{i}'), getattr(self, f'upsample_layer{i}'))
            decode_features.append(x)
        seg_features = decode_features[-1].features
        ret = dict(spatial_features=spatial_features, seg_features=seg_features)
        return ret

    def decoder_layer_forward(self, x_lateral, x_bottom, lateral_layer, merge_layer, upsample_layer):
        if False:
            print('Hello World!')
        'Forward of upsample and residual block.\n\n        Args:\n            x_lateral (:obj:`SparseConvTensor`): Lateral tensor.\n            x_bottom (:obj:`SparseConvTensor`): Feature from bottom layer.\n            lateral_layer (SparseBasicBlock): Convolution for lateral tensor.\n            merge_layer (SparseSequential): Convolution for merging features.\n            upsample_layer (SparseSequential): Convolution for upsampling.\n\n        Returns:\n            :obj:`SparseConvTensor`: Upsampled feature.\n        '
        x = lateral_layer(x_lateral)
        x = replace_feature(x, torch.cat((x_bottom.features, x.features), dim=1))
        x_merge = merge_layer(x)
        x = self.reduce_channel(x, x_merge.features.shape[1])
        x = replace_feature(x, x_merge.features + x.features)
        x = upsample_layer(x)
        return x

    @staticmethod
    def reduce_channel(x, out_channels):
        if False:
            while True:
                i = 10
        'reduce channel for element-wise addition.\n\n        Args:\n            x (:obj:`SparseConvTensor`): Sparse tensor, ``x.features``\n                are in shape (N, C1).\n            out_channels (int): The number of channel after reduction.\n\n        Returns:\n            :obj:`SparseConvTensor`: Channel reduced feature.\n        '
        features = x.features
        (n, in_channels) = features.shape
        assert in_channels % out_channels == 0 and in_channels >= out_channels
        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def make_encoder_layers(self, make_block, norm_cfg, in_channels):
        if False:
            i = 10
            return i + 15
        'make encoder layers using sparse convs.\n\n        Args:\n            make_block (method): A bounded function to build blocks.\n            norm_cfg (dict[str]): Config of normalization layer.\n            in_channels (int): The number of encoder input channels.\n\n        Returns:\n            int: The number of encoder output channels.\n        '
        self.encoder_layers = SparseSequential()
        for (i, blocks) in enumerate(self.encoder_channels):
            blocks_list = []
            for (j, out_channels) in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                if i != 0 and j == 0:
                    blocks_list.append(make_block(in_channels, out_channels, 3, norm_cfg=norm_cfg, stride=2, padding=padding, indice_key=f'spconv{i + 1}', conv_type='SparseConv3d'))
                else:
                    blocks_list.append(make_block(in_channels, out_channels, 3, norm_cfg=norm_cfg, padding=padding, indice_key=f'subm{i + 1}', conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels

    def make_decoder_layers(self, make_block, norm_cfg, in_channels):
        if False:
            i = 10
            return i + 15
        'make decoder layers using sparse convs.\n\n        Args:\n            make_block (method): A bounded function to build blocks.\n            norm_cfg (dict[str]): Config of normalization layer.\n            in_channels (int): The number of encoder input channels.\n\n        Returns:\n            int: The number of encoder output channels.\n        '
        block_num = len(self.decoder_channels)
        for (i, block_channels) in enumerate(self.decoder_channels):
            paddings = self.decoder_paddings[i]
            setattr(self, f'lateral_layer{block_num - i}', SparseBasicBlock(in_channels, block_channels[0], conv_cfg=dict(type='SubMConv3d', indice_key=f'subm{block_num - i}'), norm_cfg=norm_cfg))
            setattr(self, f'merge_layer{block_num - i}', make_block(in_channels * 2, block_channels[1], 3, norm_cfg=norm_cfg, padding=paddings[0], indice_key=f'subm{block_num - i}', conv_type='SubMConv3d'))
            if block_num - i != 1:
                setattr(self, f'upsample_layer{block_num - i}', make_block(in_channels, block_channels[2], 3, norm_cfg=norm_cfg, indice_key=f'spconv{block_num - i}', conv_type='SparseInverseConv3d'))
            else:
                setattr(self, f'upsample_layer{block_num - i}', make_block(in_channels, block_channels[2], 3, norm_cfg=norm_cfg, padding=paddings[1], indice_key='subm1', conv_type='SubMConv3d'))
            in_channels = block_channels[2]