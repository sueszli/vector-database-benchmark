from functools import partial
import torch
from torch import Tensor
import math
import torch.nn.functional as F
from . import register_monotonic_attention
from .monotonic_multihead_attention import MonotonicAttention, MonotonicInfiniteLookbackAttention, WaitKAttention
from typing import Dict, Optional

def fixed_pooling_monotonic_attention(monotonic_attention):
    if False:
        print('Hello World!')

    def create_model(monotonic_attention, klass):
        if False:
            print('Hello World!')

        class FixedStrideMonotonicAttention(monotonic_attention):

            def __init__(self, args):
                if False:
                    for i in range(10):
                        print('nop')
                self.waitk_lagging = 0
                self.num_heads = 0
                self.noise_mean = 0.0
                self.noise_var = 0.0
                super().__init__(args)
                self.pre_decision_type = args.fixed_pre_decision_type
                self.pre_decision_ratio = args.fixed_pre_decision_ratio
                self.pre_decision_pad_threshold = args.fixed_pre_decision_pad_threshold
                assert self.pre_decision_ratio > 1
                if args.fixed_pre_decision_type == 'average':
                    self.pooling_layer = torch.nn.AvgPool1d(kernel_size=self.pre_decision_ratio, stride=self.pre_decision_ratio, ceil_mode=True)
                elif args.fixed_pre_decision_type == 'last':

                    def last(key):
                        if False:
                            i = 10
                            return i + 15
                        if key.size(2) < self.pre_decision_ratio:
                            return key
                        else:
                            k = key[:, :, self.pre_decision_ratio - 1::self.pre_decision_ratio].contiguous()
                            if key.size(-1) % self.pre_decision_ratio != 0:
                                k = torch.cat([k, key[:, :, -1:]], dim=-1).contiguous()
                            return k
                    self.pooling_layer = last
                else:
                    raise NotImplementedError

            @staticmethod
            def add_args(parser):
                if False:
                    print('Hello World!')
                super(FixedStrideMonotonicAttention, FixedStrideMonotonicAttention).add_args(parser)
                parser.add_argument('--fixed-pre-decision-ratio', type=int, required=True, help='Ratio for the fixed pre-decision,indicating how many encoder steps will startsimultaneous decision making process.')
                parser.add_argument('--fixed-pre-decision-type', default='average', choices=['average', 'last'], help='Pooling type')
                parser.add_argument('--fixed-pre-decision-pad-threshold', type=float, default=0.3, help='If a part of the sequence has pad,the threshold the pooled part is a pad.')

            def insert_zeros(self, x):
                if False:
                    print('Hello World!')
                (bsz_num_heads, tgt_len, src_len) = x.size()
                stride = self.pre_decision_ratio
                weight = F.pad(torch.ones(1, 1, 1).to(x), (stride - 1, 0))
                x_upsample = F.conv_transpose1d(x.view(-1, src_len).unsqueeze(1), weight, stride=stride, padding=0)
                return x_upsample.squeeze(1).view(bsz_num_heads, tgt_len, -1)

            def p_choose(self, query: Optional[Tensor], key: Optional[Tensor], key_padding_mask: Optional[Tensor]=None, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]=None):
                if False:
                    return 10
                assert key is not None
                assert query is not None
                src_len = key.size(0)
                tgt_len = query.size(0)
                batch_size = query.size(1)
                key_pool = self.pooling_layer(key.transpose(0, 2)).transpose(0, 2)
                if key_padding_mask is not None:
                    key_padding_mask_pool = self.pooling_layer(key_padding_mask.unsqueeze(0).float()).squeeze(0).gt(self.pre_decision_pad_threshold)
                    key_padding_mask_pool[:, 0] = 0
                else:
                    key_padding_mask_pool = None
                if incremental_state is not None:
                    if max(1, math.floor(key.size(0) / self.pre_decision_ratio)) < key_pool.size(0):
                        key_pool = key_pool[:-1]
                        if key_padding_mask_pool is not None:
                            key_padding_mask_pool = key_padding_mask_pool[:-1]
                p_choose_pooled = self.p_choose_from_qk(query, key_pool, key_padding_mask_pool, incremental_state=incremental_state)
                p_choose = self.insert_zeros(p_choose_pooled)
                if p_choose.size(-1) < src_len:
                    p_choose = torch.cat([p_choose, torch.zeros(p_choose.size(0), tgt_len, src_len - p_choose.size(-1)).to(p_choose)], dim=2)
                else:
                    p_choose = p_choose[:, :, :src_len]
                    p_choose[:, :, -1] = p_choose_pooled[:, :, -1]
                assert list(p_choose.size()) == [batch_size * self.num_heads, tgt_len, src_len]
                return p_choose
        FixedStrideMonotonicAttention.__name__ = klass.__name__
        return FixedStrideMonotonicAttention
    return partial(create_model, monotonic_attention)

@register_monotonic_attention('waitk_fixed_pre_decision')
@fixed_pooling_monotonic_attention(WaitKAttention)
class WaitKAttentionFixedStride:
    pass

@register_monotonic_attention('hard_aligned_fixed_pre_decision')
@fixed_pooling_monotonic_attention(MonotonicAttention)
class MonotonicAttentionFixedStride:
    pass

@register_monotonic_attention('infinite_lookback_fixed_pre_decision')
@fixed_pooling_monotonic_attention(MonotonicInfiniteLookbackAttention)
class MonotonicInfiniteLookbackAttentionFixedStride:
    pass