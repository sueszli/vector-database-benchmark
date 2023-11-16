import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp
from .attn import Attention

def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    if False:
        while True:
            i = 10
    '\n    Eliminate potential background candidates for computation reduction and noise cancellation.\n    Args:\n        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights\n        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens\n        lens_t (int): length of template\n        keep_ratio (float): keep ratio of search region tokens (candidates)\n        global_index (torch.Tensor): global index of search region tokens\n        box_mask_z (torch.Tensor): template mask used to accumulate attention weights\n\n    Returns:\n        tokens_new (torch.Tensor): tokens after candidate elimination\n        keep_index (torch.Tensor): indices of kept search region tokens\n        removed_index (torch.Tensor): indices of removed search region tokens\n    '
    lens_s = attn.shape[-1] - lens_t
    (bs, hn, _, _) = attn.shape
    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return (tokens, global_index, None)
    attn_t = attn[:, :, :lens_t, lens_t:]
    if box_mask_z is not None:
        if not isinstance(box_mask_z, list):
            box_mask_z = [box_mask_z]
        box_mask_z_cat = torch.stack(box_mask_z, dim=1)
        box_mask_z = box_mask_z_cat.flatten(1)
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)
    (sorted_attn, indices) = torch.sort(attn_t, dim=1, descending=True)
    (_, topk_idx) = (sorted_attn[:, :lens_keep], indices[:, :lens_keep])
    (_, non_topk_idx) = (sorted_attn[:, lens_keep:], indices[:, lens_keep:])
    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]
    (B, L, C) = tokens_s.shape
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)
    return (tokens_new, keep_index, removed_index)

class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0):
        if False:
            return 10
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
        if False:
            for i in range(10):
                print('nop')
        (x_attn, attn) = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)
        lens_t = global_index_template.shape[1]
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            (x, global_index_search, removed_index_search) = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return (x, global_index_template, global_index_search, removed_index_search, attn)