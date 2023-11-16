from __future__ import annotations
from typing import Optional
import torch
from torch import nn
from kornia.contrib.models.common import MLP, ConvNormAct
from kornia.core import Module, Tensor, concatenate

def _inverse_sigmoid(x: torch.Tensor, eps: float=1e-05) -> torch.Tensor:
    if False:
        while True:
            i = 10
    'Inverse sigmoid function.\n\n    Args:\n        x: input tensor\n        eps: epsilon value for numerical stability\n\n    Returns:\n        output tensor\n    '
    out = x.clip(min=0.0, max=1.0)
    return torch.log(out.clip(min=eps) / (1.0 - out).clip(min=eps))

def _deformable_attention_kernel(value: Tensor, value_spatial_shapes: list[tuple[int, int]], sampling_locations: Tensor, attention_weights: Tensor) -> Tensor:
    if False:
        while True:
            i = 10
    'Deformable Attention Kernel used in Deformable DETR.\n\n    Described in https://arxiv.org/abs/2010.04159.\n\n    Args:\n        value: shape (N, Lv, n_head * C)\n        value_spatial_shapes: [(H0, W0), (H1, W1), ...]\n        sampling_locations: shape (N, Lq, n_head, n_levels, n_points, 2)\n        attention_weights: shape (N, Lq, n_head, n_levels, n_points)\n\n    Returns:\n        output, shape (N, Lq, n_head * C)\n    '
    (bs, _, n_head, c) = value.shape
    (_, Len_q, _, n_levels, n_points, _) = sampling_locations.shape
    split_shape: list[int] = [h * w for (h, w) in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list: list[Tensor] = []
    for (level, (h, w)) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampling_value_l_ = torch.nn.functional.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(bs * n_head, 1, Len_q, n_levels * n_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)
    return output.permute(0, 2, 1)

class MultiScaleDeformableAttention(Module):
    """Multi-scale Deformable Attention used in Deformable DETR.

    Described in https://arxiv.org/abs/2010.04159.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points
        self.head_dim = embed_dim // num_heads
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: Tensor, reference_points: Tensor, value: Tensor, value_spatial_shapes: list[tuple[int, int]]) -> Tensor:
        if False:
            while True:
                i = 10
        '\n        Args:\n            query: shape (N, Lq, C)\n            reference_points: shape (N, Lq, n_levels, 4)\n            value: shape (N, Lv, C)\n            value_spatial_shapes: [(H0, W0), (H1, W1), ...]\n\n        Returns:\n            output, shape (N, Lq, C)\n        '
        (N, Lenq, _) = query.shape
        (_, Len_v, _) = value.shape
        sampling_offsets = self.sampling_offsets(query).reshape(N, Lenq, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(N, Lenq, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).reshape(N, Lenq, self.num_heads, self.num_levels, self.num_points)
        reference_points_cxcy = reference_points[:, :, None, :, None, :2]
        reference_points_wh = reference_points[:, :, None, :, None, 2:]
        sampling_locations = reference_points_cxcy + sampling_offsets / self.num_points * reference_points_wh * 0.5
        value_buf = self.value_proj(value).reshape(N, Len_v, self.num_heads, self.head_dim)
        out = _deformable_attention_kernel(value_buf, value_spatial_shapes, sampling_locations, attention_weights)
        out = self.output_proj(out)
        return out

class TransformerDecoderLayer(Module):
    """Deformable Transformer Decoder layer used in Deformable DETR.

    Described in: https://arxiv.org/abs/2010.04159.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, num_levels: int, num_points: int) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, num_heads, num_levels, num_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.activation = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

    def _ffn(self, x: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        return self.linear2(self.dropout3(self.activation(self.linear1(x))))

    def forward(self, tgt: Tensor, ref_points: Tensor, memory: Tensor, memory_spatial_shapes: list[tuple[int, int]], memory_level_start_index: Optional[list[int]]=None, attn_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, query_pos_embed: Optional[Tensor]=None) -> Tensor:
        if False:
            return 10
        q = k = tgt + query_pos_embed
        (out, _) = self.self_attn(q, k, value=tgt)
        tgt = self.norm1(tgt + self.dropout1(out))
        out = self.cross_attn(tgt + query_pos_embed, ref_points, memory, memory_spatial_shapes)
        tgt = self.norm2(tgt + self.dropout2(out))
        out = self.norm3(tgt + self.dropout4(self._ffn(tgt)))
        return out

class TransformerDecoder:

    def __init__(self, hidden_dim: int, decoder_layers: nn.ModuleList, num_layers: int, eval_idx: int=-1) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.layers = decoder_layers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self, tgt: Tensor, ref_points_unact: Tensor, memory: Tensor, memory_spatial_shapes: list[tuple[int, int]], memory_level_start_index: list[int], bbox_head: nn.ModuleList, score_head: nn.ModuleList, query_pos_head: nn.Module, attn_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None) -> tuple[Tensor, Tensor]:
        if False:
            return 10
        output: Tensor = tgt
        dec_out_bboxes: list[Tensor] = []
        dec_out_logits: list[Tensor] = []
        ref_points_detach = torch.sigmoid(ref_points_unact)
        for (i, layer) in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed: Tensor = query_pos_head(ref_points_detach)
            output = layer(output, ref_points_input, memory, memory_spatial_shapes, memory_level_start_index, attn_mask, memory_mask, query_pos_embed)
            inter_ref_bbox = torch.sigmoid(bbox_head[i](output) + _inverse_sigmoid(ref_points_detach))
            if i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break
            ref_points_detach = inter_ref_bbox
        return (torch.stack(dec_out_bboxes), torch.stack(dec_out_logits))

class RTDETRHead(Module):

    def __init__(self, num_classes: int, hidden_dim: int, num_queries: int, in_channels: list[int], num_decoder_layers: int, num_heads: int=8, num_decoder_points: int=4, dropout: float=0.0) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.num_queries = num_queries
        self.num_levels = len(in_channels)
        self.input_proj = nn.ModuleList()
        for ch_in in in_channels:
            self.input_proj.append(ConvNormAct(ch_in, hidden_dim, 1, act='none'))
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, num_levels=len(in_channels), num_points=num_decoder_points) for _ in range(num_decoder_layers)])
        self.decoder = TransformerDecoder(hidden_dim=hidden_dim, decoder_layers=self.decoder_layers, num_layers=num_decoder_layers)
        self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)])
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)])

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor]:
        if False:
            print('Hello World!')
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
        (denoising_class, denoising_bbox_unact, attn_mask) = (None, None, None)
        (target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits) = self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)
        (out_bboxes, out_logits) = self.decoder.forward(target, init_ref_points_unact, memory, spatial_shapes, level_start_index, self.dec_bbox_head, self.dec_score_head, self.query_pos_head, attn_mask=attn_mask)
        return (out_logits[-1], out_bboxes[-1])

    def _get_encoder_input(self, feats: Tensor) -> tuple[Tensor, list[tuple[int, int]], list[int]]:
        if False:
            print('Hello World!')
        proj_feats: list[Tensor] = [self.input_proj[i](feat) for (i, feat) in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))
        feat_flatten_list: list[Tensor] = []
        spatial_shapes: list[tuple[int, int]] = []
        level_start_index: list[int] = [0]
        for (i, feat) in enumerate(proj_feats):
            (_, _, h, w) = feat.shape
            feat_flatten_list.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append((h, w))
            level_start_index.append(h * w + level_start_index[-1])
        feat_flatten: Tensor = torch.concat(feat_flatten_list, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_decoder_input(self, memory: Tensor, spatial_shapes: list[tuple[int, int]], denoising_class: Optional[Tensor]=None, denoising_bbox_unact: Optional[Tensor]=None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if False:
            for i in range(10):
                print('nop')
        (anchors, valid_mask) = self._generate_anchors(spatial_shapes, device=memory.device, dtype=memory.dtype)
        memory = valid_mask.to(memory) * memory
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        (_, topk_ind) = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))
        enc_topk_bboxes = torch.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)
        enc_topk_logits = enc_outputs_class.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))
        target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)
        return (target.detach(), reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits)

    @staticmethod
    def _generate_anchors(spatial_shapes: list[tuple[int, int]], grid_size: float=0.05, eps: float=0.01, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None) -> tuple[Tensor, Tensor]:
        if False:
            i = 10
            return i + 15
        'Generate anchors for RT-DETR.\n\n        Args:\n            spatial_shapes: shape (width, height) of the feature maps\n            grid_size: size of the grid\n            eps: specify the minimum and maximum size of the anchors\n            device: device to place the anchors\n            dtype: data type for the anchors\n\n        Returns:\n            logit of anchors and mask\n        '
        anchors_list: list[Tensor] = []
        for (i, (h, w)) in enumerate(spatial_shapes):
            (grid_y, grid_x) = torch.meshgrid(torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            wh = torch.empty(2, device=device, dtype=dtype)
            wh[0] = w
            wh[1] = h
            grid_xy = (grid_xy + 0.5) / wh
            grid_wh = torch.ones_like(grid_xy) * grid_size * 2.0 ** i
            anchors_list.append(concatenate([grid_xy, grid_wh], -1).reshape(-1, h * w, 4))
        anchors = concatenate(anchors_list, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        inf_t = torch.empty(1, device=device, dtype=dtype)
        inf_t[0] = torch.inf
        anchors = torch.where(valid_mask, anchors, inf_t)
        return (anchors, valid_mask)