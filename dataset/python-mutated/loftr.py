import torch
import torch.nn as nn
from einops.einops import rearrange
from .backbone import build_backbone
from .loftr_module import FinePreprocess, LocalFeatureTransformer
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from .utils.position_encoding import PositionEncodingSine

class LoFTR(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'], temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config['fine'])
        self.fine_matching = FineMatching()

    def forward(self, data):
        if False:
            i = 10
            return i + 15
        "\n        Update:\n            data (dict): {\n                'image0': (torch.Tensor): (N, 1, H, W)\n                'image1': (torch.Tensor): (N, 1, H, W)\n                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position\n                'mask1'(optional) : (torch.Tensor): (N, H, W)\n            }\n        "
        data.update({'bs': data['image0'].size(0), 'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]})
        if data['hw0_i'] == data['hw1_i']:
            (feats_c, feats_f) = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            ((feat_c0, feat_c1), (feat_f0, feat_f1)) = (feats_c.split(data['bs']), feats_f.split(data['bs']))
        else:
            ((feat_c0, feat_f0), (feat_c1, feat_f1)) = (self.backbone(data['image0']), self.backbone(data['image1']))
        data.update({'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:], 'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]})
        feat_c0 = self.pos_encoding(feat_c0)
        feat_c1 = self.pos_encoding(feat_c1)
        mask_c0 = mask_c1 = None
        if 'mask0' in data:
            (mask_c0, mask_c1) = (data['mask0'].flatten(-2), data['mask1'].flatten(-2))
        (feat_c0, feat_c1) = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
        (feat_f0_unfold, feat_f1_unfold) = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:
            (feat_f0_unfold, feat_f1_unfold) = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        if False:
            while True:
                i = 10
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)