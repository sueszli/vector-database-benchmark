import torch.nn as nn

class FPNFusionModule(nn.Module):
    """ This is a fpn-style cross-scale feature fusion module" """

    def __init__(self, embed_dims, fuse_dim=256, n_block=4, use_bn=False):
        if False:
            print('Hello World!')
        super().__init__()
        ' Initializes the model.\n        Args:\n            embed_dims: the list of channel dim for different scale feature maps (i.e., the input)\n            fuse_dim: the channel dim of the fused feature map (i.e., the output)\n            n_block: the number of multi-scale features (default=4)\n            use_bn: whether to use bn\n        '
        self.embed_dims = embed_dims
        self.fuse_dim = fuse_dim
        self.n_block = n_block
        self.multi_scaler = _make_multi_scale_layers(embed_dims, fuse_dim, use_bn=use_bn, n_block=n_block)

    def forward(self, x_blocks):
        if False:
            while True:
                i = 10
        x_blocks = x_blocks
        for idx in range(self.n_block - 1, -1, -1):
            x_blocks[idx] = getattr(self.multi_scaler, f'layer_{idx}_rn')(x_blocks[idx])
            x_blocks[idx] = getattr(self.multi_scaler, f'p_norm_{idx}')(x_blocks[idx])
        refined_embeds = []
        for idx in range(self.n_block - 1, -1, -1):
            if idx == self.n_block - 1:
                path = getattr(self.multi_scaler, f'refinenet_{idx}')([x_blocks[idx]], None)
            else:
                path = getattr(self.multi_scaler, f'refinenet_{idx}')([path, x_blocks[idx]], x_blocks[idx].size()[2:])
            refined_embeds.append(path)
        return refined_embeds

def _make_multi_scale_layers(in_shape, out_shape, n_block=4, groups=1, use_bn=False):
    if False:
        while True:
            i = 10
    out_shapes = [out_shape for _ in range(n_block)]
    multi_scaler = nn.Module()
    for idx in range(n_block - 1, -1, -1):
        '\n          1 x 1 conv for dim reduction -> group norm\n        '
        layer_name = f'layer_{idx}_rn'
        multi_scaler.add_module(layer_name, nn.Conv2d(in_shape[idx], out_shapes[idx], kernel_size=1))
        layer_name = f'p_norm_{idx}'
        multi_scaler.add_module(layer_name, nn.GroupNorm(32, out_shapes[idx]))
        layer_name = f'refinenet_{idx}'
        multi_scaler.add_module(layer_name, _make_fusion_block(out_shape, use_bn))
        nn.init.xavier_uniform_(getattr(multi_scaler, f'layer_{idx}_rn').weight, gain=1)
        nn.init.constant_(getattr(multi_scaler, f'layer_{idx}_rn').bias, 0)
    return multi_scaler

def _make_fusion_block(features, use_bn):
    if False:
        for i in range(10):
            print('nop')
    ' We use a resnet bottleneck structure for fpn '
    return FeatureFusionBlock(features, nn.ReLU(False), bn=use_bn, expand=False, align_corners=True)

class FeatureFusionBlock(nn.Module):
    """ Feature fusion block """

    def __init__(self, features, activation, bn=False, expand=False, align_corners=True):
        if False:
            while True:
                i = 10
        'Init.\n        Args:\n            features (int): channel dim of the input feature\n            activation: activation function to use\n            bn: whether to use bn\n            expand: whether to exapnd feature or not\n            align_corners: wheter to use align_corners for interpolation\n        '
        super(FeatureFusionBlock, self).__init__()
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features
        if self.expand is True:
            out_features = features // 2
        self.smoothing = nn.Conv2d(features, out_features, kernel_size=1, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, xs, up_size):
        if False:
            return 10
        ' Forward pass.\n        Args\n            xs: xs[0]: the feature refined from the previous step, xs[1]: the next scale features to fuse\n            up_size: the size for upsampling; xs[0] is upsampled before merging with xs[1]\n        Returns:\n            output: the fused feature, which is fed to the next fusion step as an input\n        '
        output = xs[0]
        if len(xs) == 2:
            output = nn.functional.interpolate(output, size=up_size, mode='bilinear', align_corners=self.align_corners)
            output = self.smoothing(output)
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        return output

class ResidualConvUnit(nn.Module):
    """ Residual convolution module. """

    def __init__(self, features, activation, bn):
        if False:
            print('Hello World!')
        'Init.\n        Args:\n            features (int): channel dim of the input\n            activation: activation function\n            bn: whether to use bn\n        '
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, 64, kernel_size=1, stride=1, bias=not self.bn, groups=self.groups)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=not self.bn, groups=self.groups)
        self.conv3 = nn.Conv2d(64, features, kernel_size=1, stride=1, bias=not self.bn, groups=self.groups)
        if self.bn is True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
            self.bn3 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if False:
            print('Hello World!')
        ' Forward pass\n\n        Args:\n            x (tensor): input feature\n\n        Returns:\n            tensor: output feature\n        '
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn is True:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn is True:
            out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        if self.bn is True:
            out = self.bn3(out)
        if self.groups > 1:
            out = self.conv_merge(out)
        return self.skip_add.add(out, x)