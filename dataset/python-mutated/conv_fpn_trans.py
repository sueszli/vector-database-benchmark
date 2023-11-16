import random
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from modelscope.models.multi_modal.vldoc.convnext import convnext_tiny
from modelscope.utils.logger import get_logger
try:
    import apex
    import apex.normalization
    LN = apex.normalization.FusedLayerNorm
except ImportError:
    LN = torch.nn.LayerNorm
logging = get_logger()

class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor=None, expand_ratio=4.0, init_values: float=None):
        if False:
            while True:
                i = 10
        '\n        The implementation of the transformer block refers to:\n        https://github.com/openai/CLIP/blob/b46f5ac7587d2e1862f8b7b1573179d80dcdd620/clip/model.py\n        '
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LN(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * expand_ratio)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * expand_ratio, d_model))]))
        self.ln_2 = LN(d_model)
        self.attn_mask = attn_mask
        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(d_model), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(d_model), requires_grad=True)
        else:
            (self.gamma_1, self.gamma_2) = (1.0, 1.0)

    def attention(self, x: torch.Tensor):
        if False:
            while True:
                i = 10
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        x = x + self.gamma_1 * self.attention(self.ln_1(x))
        x = x + self.gamma_2 * self.mlp(self.ln_2(x))
        return x

class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        if False:
            i = 10
            return i + 15
        return x * torch.sigmoid(1.702 * x)

def drop_grid(grid_map, drop_range=(0.3, 0.8), training=False):
    if False:
        print('Hello World!')
    '\n    only drop in the training phase.\n    grid_map: [N, D, T1, ...]\n    '
    if training:
        drop_ratio = random.random() * (drop_range[1] - drop_range[0]) + drop_range[0]
        mask = (torch.rand_like(grid_map[:, 0]) < drop_ratio).bool()
        grid_map = grid_map.masked_fill(mask.unsqueeze(1), 0.0)
    return grid_map

class GumbelSample(nn.Module):

    def __init__(self, in_dim, num_keep):
        if False:
            print('Hello World!')
        super(GumbelSample, self).__init__()
        self.keep_layer = nn.Sequential(nn.Conv2d(in_dim, 256, 3, stride=1, padding=3, dilation=3, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, stride=1, padding=2, dilation=2, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 1, 3, stride=1, padding=1, dilation=1), nn.Sigmoid())
        self.num_keep = num_keep
        self.diffusion = nn.Conv2d(in_dim, in_dim, 3, padding=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, tau=1):
        if False:
            print('Hello World!')
        '\n        x: [N, C, H, W]\n        '
        N = x.size(0)
        keep_score = self.keep_layer(x)
        keep_score = torch.clamp(keep_score, min=0.0, max=1.0)
        keep_score = torch.cat([keep_score, 1 - keep_score], dim=1) + 1e-05
        gumbel_score = F.gumbel_softmax(keep_score.log(), tau=tau, hard=False, dim=1)
        index = gumbel_score.max(dim=1, keepdim=True)[1]
        gumbel_hard = torch.zeros_like(gumbel_score, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
        gumbel_hard = gumbel_hard - gumbel_score.detach() + gumbel_score
        gumbel_score = gumbel_score[:, 0].contiguous().view(N, -1)
        gumbel_hard = gumbel_hard[:, 0].contiguous().view(N, -1)
        idx_true = torch.topk(gumbel_score, self.num_keep, dim=1)[1]
        topk_mask = torch.zeros_like(gumbel_score).bool().fill_(False).scatter_(1, idx_true, True)
        return (topk_mask, gumbel_hard, keep_score[:, 0])

    def sample(self, x, topk_mask, gumbel_hard):
        if False:
            for i in range(10):
                print('nop')
        (N, D, H, W) = x.size()
        x = x.contiguous().view(N, D, -1)
        x = x * gumbel_hard.unsqueeze(1)
        x = x.transpose(1, 2)
        x = x[topk_mask].contiguous().view(N, -1, D)
        x = drop_grid(x.transpose(1, 2), drop_range=(0.0, 0.2), training=self.training).transpose(1, 2)
        return x

    def random_sample(self, x):
        if False:
            i = 10
            return i + 15
        (N, D, H, W) = x.size()
        x = x.contiguous().view(N, D, -1)
        x = x.transpose(1, 2)
        idx_true = torch.topk(torch.rand_like(x[:, :, 0]), self.num_keep, dim=1)[1]
        topk_mask = torch.zeros_like(x[:, :, 0]).bool().fill_(False).scatter_(1, idx_true, True)
        x = x[topk_mask].contiguous().view(N, -1, D)
        x = drop_grid(x.transpose(1, 2), drop_range=(0.0, 0.2), training=self.training).transpose(1, 2)
        return (x, topk_mask)

    def restore(self, x, topk_mask, src):
        if False:
            while True:
                i = 10
        '\n        x: [N, D, H, W]\n        topk_mask: [N, HxW]\n        src: [N, num_keep, D]\n        '
        (N, D, H, W) = x.size()
        x = drop_grid(x, drop_range=(0.2, 0.8), training=self.training)
        x = x.contiguous().view(N, D, -1).transpose(1, 2)
        x = x.masked_scatter(topk_mask.unsqueeze(-1), src)
        x = x.transpose(1, 2).contiguous().view(N, D, H, W)
        x = self.dropout(self.diffusion(x))
        return x

class FPNTrans(nn.Module):

    def __init__(self, trans_layers=2, inner_channels=256, img_size=(896, 896), inner_vit=False, out_sampling=False):
        if False:
            for i in range(10):
                print('nop')
        super(FPNTrans, self).__init__()
        self.cnn = convnext_tiny(pretrained=True, in_22k=True)
        self.dims = self.cnn.dims
        self.img_size = img_size
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.in5 = nn.Conv2d(self.dims[-1], inner_channels, 1, bias=False)
        self.in4 = nn.Conv2d(self.dims[-2], inner_channels, 1, bias=False)
        self.in3 = nn.Conv2d(self.dims[-3], inner_channels, 1, bias=False)
        self.in2 = nn.Conv2d(self.dims[-4], inner_channels, 1, bias=False)
        self.out5 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=False), nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=False), nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=False), nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=False)
        self.inner_vit = inner_vit
        if inner_vit:
            self.num_keep1 = (self.img_size[0] // 64) ** 2
            self.gumble_sample1 = GumbelSample(inner_channels, num_keep=self.num_keep1)
            self.pos_emb1 = nn.Parameter(torch.randn(inner_channels, self.img_size[0] // 32, self.img_size[1] // 32))
            trunc_normal_(self.pos_emb1, std=0.02)
            self.mini_vit = nn.Sequential(*[ResidualAttentionBlock(inner_channels, 4, expand_ratio=2, init_values=0.1) for _ in range(trans_layers)])
        self.dropout_pos = nn.Dropout(0.1)
        if out_sampling:
            self.num_keep2 = (self.img_size[0] // 64) ** 2
            self.gumble_sample2 = GumbelSample(inner_channels, num_keep=self.num_keep2)
            self.pos_emb2 = nn.Parameter(torch.randn(inner_channels, self.img_size[0] // 4, self.img_size[1] // 4))
            trunc_normal_(self.pos_emb2, std=0.02)
        self.out_sampling = out_sampling
        self.drop_path = DropPath(0.1)

    def forward(self, x):
        if False:
            print('Hello World!')
        ms_features = self.cnn(x)
        (c2, c3, c4, c5) = ms_features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)
        (N, D5, H5, W5) = in5.size()
        if self.inner_vit:
            keep_score = None
            in5_pos = self.dropout_pos(in5 + self.pos_emb1.unsqueeze(0))
            (in5_pos_in, topk_mask) = self.gumble_sample1.random_sample(in5_pos)
            in5_pos_in = in5_pos_in.transpose(0, 1)
            in5_pos_out = self.mini_vit(in5_pos_in).permute(1, 2, 0)
            in5 = self.gumble_sample1.restore(in5, topk_mask, in5_pos_out.transpose(1, 2))
        else:
            keep_score = None
        out4 = self.up5(in5) + self.drop_path(in4)
        out3 = self.up4(out4) + self.drop_path(in3)
        out2 = self.up3(out3) + self.drop_path(in2)
        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)
        feat_ms = torch.cat((p5, p4, p3, p2), 1)
        ret_dict = dict(feat_ms=feat_ms, keep_score=keep_score)
        if self.out_sampling:
            (topk_mask2, gumbel_hard2, keep_score2) = self.gumble_sample2(feat_ms)
            feat_ms_pos = self.dropout_pos(feat_ms + self.pos_emb2.unsqueeze(0))
            feat_ms_pos_sampled = self.gumble_sample2.sample(feat_ms_pos, topk_mask2, gumbel_hard2).transpose(0, 1)
            ret_dict_sup = dict(feat_ms_pos_sampled=feat_ms_pos_sampled, sampler=self.gumble_sample2, keep_score2=keep_score2)
            ret_dict.update(ret_dict_sup)
        return ret_dict