import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.models.cv.video_super_resolution.common import PixelShufflePack, ResidualBlockNoBN, flow_warp, make_layer

class ConvModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act_cfg=dict(type='ReLU'), inplace=True):
        if False:
            i = 10
            return i + 15
        super(ConvModule, self).__init__()
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_activation = act_cfg is not None
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if self.with_activation:
            self.activate = getattr(nn, act_cfg['type'])(self.inplace)

    def forward(self, x):
        if False:
            return 10
        x = self.conv(x)
        if self.with_activation:
            x = self.activate(x)
        return x

class BasicVSRNet(nn.Module):
    """BasicVSR network structure for video super-resolution.
    Support only x4 upsampling.
    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021
    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None):
        if False:
            return 10
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet(pretrained=spynet_pretrained)
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def check_if_mirror_extended(self, lrs):
        if False:
            i = 10
            return i + 15
        'Check whether the input is a mirror-extended sequence.\n        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the\n        (t-1-i)-th frame.\n        Args:\n            lrs (tensor): Input LR images with shape (n, t, c, h, w)\n        '
        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            (lrs_1, lrs_2) = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        if False:
            i = 10
            return i + 15
        "Compute optical flow using SPyNet for feature warping.\n        Note that if the input is an mirror-extended sequence, 'flows_forward'\n        is not needed, since it is equal to 'flows_backward.flip(1)'.\n        Args:\n            lrs (tensor): Input LR images with shape (n, t, c, h, w)\n        Return:\n            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the\n                flows used for forward-time propagation (current to previous).\n                'flows_backward' corresponds to the flows used for\n                backward-time propagation (current to next).\n        "
        (n, t, c, h, w) = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        if self.is_mirror_extended:
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        return (flows_forward, flows_backward)

    def forward(self, lrs):
        if False:
            print('Hello World!')
        'Forward function for BasicVSR.\n        Args:\n            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).\n        Returns:\n            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).\n        '
        (n, t, c, h, w) = lrs.size()
        assert h >= 64 and w >= 64, f'The height and width of inputs should be at least 64, but got {h} and {w}.'
        self.check_if_mirror_extended(lrs)
        (flows_forward, flows_backward) = self.compute_flow(lrs)
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)
            outputs.append(feat_prop)
        outputs = outputs[::-1]
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            out += base
            outputs[i] = out
        return torch.stack(outputs, dim=1)

class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        if False:
            while True:
                i = 10
        super().__init__()
        main = []
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        main.append(make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels))
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        if False:
            i = 10
            return i + 15
        'Forward function for ResidualBlocksWithInputConv.\n        Args:\n            feat (Tensor): Input feature with shape (n, in_channels, h, w)\n        Returns:\n            Tensor: Output feature with shape (n, out_channels, h, w)\n        '
        return self.main(feat)

class SPyNet(nn.Module):
    """SPyNet network structure.
    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        if isinstance(pretrained, str):
            self.load_state_dict(torch.load(pretrained), strict=True)
        elif pretrained is not None:
            raise TypeError(f'[pretrained] should be str or None, but got {type(pretrained)}.')
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        if False:
            i = 10
            return i + 15
        'Compute flow from ref to supp.\n        Note that in this function, the images are already resized to a\n        multiple of 32.\n        Args:\n            ref (Tensor): Reference image with shape of (n, 3, h, w).\n            supp (Tensor): Supporting image with shape of (n, 3, h, w).\n        Returns:\n            Tensor: Estimated optical flow: (n, 2, h, w).\n        '
        (n, _, h, w) = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for level in range(5):
            ref.append(F.avg_pool2d(input=ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(input=supp[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            flow = flow_up + self.basic_module[level](torch.cat([ref[level], flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode='border'), flow_up], 1))
        return flow

    def forward(self, ref, supp):
        if False:
            while True:
                i = 10
        'Forward function of SPyNet.\n        This function computes the optical flow from ref to supp.\n        Args:\n            ref (Tensor): Reference image with shape of (n, 3, h, w).\n            supp (Tensor): Supporting image with shape of (n, 3, h, w).\n        Returns:\n            Tensor: Estimated optical flow: (n, 2, h, w).\n        '
        (h, w) = ref.shape[2:4]
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_up, w_up), mode='bilinear', align_corners=False)
        flow = F.interpolate(input=self.compute_flow(ref, supp), size=(h, w), mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow

class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.basic_module = nn.Sequential(ConvModule(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3, act_cfg=dict(type='ReLU')), ConvModule(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3, act_cfg=dict(type='ReLU')), ConvModule(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3, act_cfg=dict(type='ReLU')), ConvModule(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3, act_cfg=dict(type='ReLU')), ConvModule(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3, act_cfg=None))

    def forward(self, tensor_input):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).\n                8 channels contain:\n                [reference image (3), neighbor image (3), initial flow (2)].\n        Returns:\n            Tensor: Refined flow with shape (b, 2, h, w)\n        '
        return self.basic_module(tensor_input)