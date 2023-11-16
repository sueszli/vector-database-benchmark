"""
Multi-perspective matching layer
"""
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import FromParams
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, masked_max, masked_mean, masked_softmax, tiny_value_of_dtype

def multi_perspective_match(vector1: torch.Tensor, vector2: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate multi-perspective cosine matching between time-steps of vectors\n    of the same length.\n\n    # Parameters\n\n    vector1 : `torch.Tensor`\n        A tensor of shape `(batch, seq_len, hidden_size)`\n    vector2 : `torch.Tensor`\n        A tensor of shape `(batch, seq_len or 1, hidden_size)`\n    weight : `torch.Tensor`\n        A tensor of shape `(num_perspectives, hidden_size)`\n\n    # Returns\n\n    `torch.Tensor` :\n        Shape `(batch, seq_len, 1)`.\n    `torch.Tensor` :\n        Shape `(batch, seq_len, num_perspectives)`.\n    '
    assert vector1.size(0) == vector2.size(0)
    assert weight.size(1) == vector1.size(2) == vector1.size(2)
    similarity_single = F.cosine_similarity(vector1, vector2, 2).unsqueeze(2)
    weight = weight.unsqueeze(0).unsqueeze(0)
    vector1 = weight * vector1.unsqueeze(2)
    vector2 = weight * vector2.unsqueeze(2)
    similarity_multi = F.cosine_similarity(vector1, vector2, dim=3)
    return (similarity_single, similarity_multi)

def multi_perspective_match_pairwise(vector1: torch.Tensor, vector2: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
    '\n    Calculate multi-perspective cosine matching between each time step of\n    one vector and each time step of another vector.\n\n    # Parameters\n\n    vector1 : `torch.Tensor`\n        A tensor of shape `(batch, seq_len1, hidden_size)`\n    vector2 : `torch.Tensor`\n        A tensor of shape `(batch, seq_len2, hidden_size)`\n    weight : `torch.Tensor`\n        A tensor of shape `(num_perspectives, hidden_size)`\n\n    # Returns\n\n    `torch.Tensor` :\n        A tensor of shape `(batch, seq_len1, seq_len2, num_perspectives)` consisting\n        multi-perspective matching results\n    '
    num_perspectives = weight.size(0)
    weight = weight.unsqueeze(0).unsqueeze(2)
    vector1 = weight * vector1.unsqueeze(1).expand(-1, num_perspectives, -1, -1)
    vector2 = weight * vector2.unsqueeze(1).expand(-1, num_perspectives, -1, -1)
    vector1_norm = vector1.norm(p=2, dim=3, keepdim=True)
    vector2_norm = vector2.norm(p=2, dim=3, keepdim=True)
    mul_result = torch.matmul(vector1, vector2.transpose(2, 3))
    norm_value = vector1_norm * vector2_norm.transpose(2, 3)
    return (mul_result / norm_value.clamp(min=tiny_value_of_dtype(norm_value.dtype))).permute(0, 2, 3, 1)

class BiMpmMatching(nn.Module, FromParams):
    """
    This `Module` implements the matching layer of BiMPM model described in [Bilateral
    Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/abs/1702.03814)
    by Zhiguo Wang et al., 2017.
    Also please refer to the [TensorFlow implementation](https://github.com/zhiguowang/BiMPM/) and
    [PyTorch implementation](https://github.com/galsang/BIMPM-pytorch).

    # Parameters

    hidden_dim : `int`, optional (default = `100`)
        The hidden dimension of the representations
    num_perspectives : `int`, optional (default = `20`)
        The number of perspectives for matching
    share_weights_between_directions : `bool`, optional (default = `True`)
        If True, share weight between matching from sentence1 to sentence2 and from sentence2
        to sentence1, useful for non-symmetric tasks
    is_forward : `bool`, optional (default = `None`)
        Whether the matching is for forward sequence or backward sequence, useful in finding last
        token in full matching. It can not be None if with_full_match is True.
    with_full_match : `bool`, optional (default = `True`)
        If True, include full match
    with_maxpool_match : `bool`, optional (default = `True`)
        If True, include max pool match
    with_attentive_match : `bool`, optional (default = `True`)
        If True, include attentive match
    with_max_attentive_match : `bool`, optional (default = `True`)
        If True, include max attentive match
    """

    def __init__(self, hidden_dim: int=100, num_perspectives: int=20, share_weights_between_directions: bool=True, is_forward: bool=None, with_full_match: bool=True, with_maxpool_match: bool=True, with_attentive_match: bool=True, with_max_attentive_match: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_perspectives = num_perspectives
        self.is_forward = is_forward
        self.with_full_match = with_full_match
        self.with_maxpool_match = with_maxpool_match
        self.with_attentive_match = with_attentive_match
        self.with_max_attentive_match = with_max_attentive_match
        if not (with_full_match or with_maxpool_match or with_attentive_match or with_max_attentive_match):
            raise ConfigurationError('At least one of the matching method should be enabled')

        def create_parameter():
            if False:
                return 10
            param = nn.Parameter(torch.zeros(num_perspectives, hidden_dim))
            torch.nn.init.kaiming_normal_(param)
            return param

        def share_or_create(weights_to_share):
            if False:
                while True:
                    i = 10
            return weights_to_share if share_weights_between_directions else create_parameter()
        output_dim = 2
        if with_full_match:
            if is_forward is None:
                raise ConfigurationError('Must specify is_forward to enable full matching')
            self.full_match_weights = create_parameter()
            self.full_match_weights_reversed = share_or_create(self.full_match_weights)
            output_dim += num_perspectives + 1
        if with_maxpool_match:
            self.maxpool_match_weights = create_parameter()
            output_dim += num_perspectives * 2
        if with_attentive_match:
            self.attentive_match_weights = create_parameter()
            self.attentive_match_weights_reversed = share_or_create(self.attentive_match_weights)
            output_dim += num_perspectives + 1
        if with_max_attentive_match:
            self.max_attentive_match_weights = create_parameter()
            self.max_attentive_match_weights_reversed = share_or_create(self.max_attentive_match_weights)
            output_dim += num_perspectives + 1
        self.output_dim = output_dim

    def get_output_dim(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.output_dim

    def forward(self, context_1: torch.Tensor, mask_1: torch.BoolTensor, context_2: torch.Tensor, mask_2: torch.BoolTensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if False:
            return 10
        '\n        Given the forward (or backward) representations of sentence1 and sentence2, apply four bilateral\n        matching functions between them in one direction.\n\n        # Parameters\n\n        context_1 : `torch.Tensor`\n            Tensor of shape (batch_size, seq_len1, hidden_dim) representing the encoding of the first sentence.\n        mask_1 : `torch.BoolTensor`\n            Boolean Tensor of shape (batch_size, seq_len1), indicating which\n            positions in the first sentence are padding (0) and which are not (1).\n        context_2 : `torch.Tensor`\n            Tensor of shape (batch_size, seq_len2, hidden_dim) representing the encoding of the second sentence.\n        mask_2 : `torch.BoolTensor`\n            Boolean Tensor of shape (batch_size, seq_len2), indicating which\n            positions in the second sentence are padding (0) and which are not (1).\n\n        # Returns\n\n        `Tuple[List[torch.Tensor], List[torch.Tensor]]` :\n            A tuple of matching vectors for the two sentences. Each of which is a list of\n            matching vectors of shape (batch, seq_len, num_perspectives or 1)\n        '
        assert not mask_2.requires_grad and (not mask_1.requires_grad)
        assert context_1.size(-1) == context_2.size(-1) == self.hidden_dim
        len_1 = get_lengths_from_binary_sequence_mask(mask_1)
        len_2 = get_lengths_from_binary_sequence_mask(mask_2)
        context_1 = context_1 * mask_1.unsqueeze(-1)
        context_2 = context_2 * mask_2.unsqueeze(-1)
        matching_vector_1: List[torch.Tensor] = []
        matching_vector_2: List[torch.Tensor] = []
        cosine_sim = F.cosine_similarity(context_1.unsqueeze(-2), context_2.unsqueeze(-3), dim=3)
        cosine_max_1 = masked_max(cosine_sim, mask_2.unsqueeze(-2), dim=2, keepdim=True)
        cosine_mean_1 = masked_mean(cosine_sim, mask_2.unsqueeze(-2), dim=2, keepdim=True)
        cosine_max_2 = masked_max(cosine_sim.permute(0, 2, 1), mask_1.unsqueeze(-2), dim=2, keepdim=True)
        cosine_mean_2 = masked_mean(cosine_sim.permute(0, 2, 1), mask_1.unsqueeze(-2), dim=2, keepdim=True)
        matching_vector_1.extend([cosine_max_1, cosine_mean_1])
        matching_vector_2.extend([cosine_max_2, cosine_mean_2])
        if self.with_full_match:
            if self.is_forward:
                last_position_1 = (len_1 - 1).clamp(min=0)
                last_position_1 = last_position_1.view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
                last_position_2 = (len_2 - 1).clamp(min=0)
                last_position_2 = last_position_2.view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
                context_1_last = context_1.gather(1, last_position_1)
                context_2_last = context_2.gather(1, last_position_2)
            else:
                context_1_last = context_1[:, 0:1, :]
                context_2_last = context_2[:, 0:1, :]
            matching_vector_1_full = multi_perspective_match(context_1, context_2_last, self.full_match_weights)
            matching_vector_2_full = multi_perspective_match(context_2, context_1_last, self.full_match_weights_reversed)
            matching_vector_1.extend(matching_vector_1_full)
            matching_vector_2.extend(matching_vector_2_full)
        if self.with_maxpool_match:
            matching_vector_max = multi_perspective_match_pairwise(context_1, context_2, self.maxpool_match_weights)
            matching_vector_1_max = masked_max(matching_vector_max, mask_2.unsqueeze(-2).unsqueeze(-1), dim=2)
            matching_vector_1_mean = masked_mean(matching_vector_max, mask_2.unsqueeze(-2).unsqueeze(-1), dim=2)
            matching_vector_2_max = masked_max(matching_vector_max.permute(0, 2, 1, 3), mask_1.unsqueeze(-2).unsqueeze(-1), dim=2)
            matching_vector_2_mean = masked_mean(matching_vector_max.permute(0, 2, 1, 3), mask_1.unsqueeze(-2).unsqueeze(-1), dim=2)
            matching_vector_1.extend([matching_vector_1_max, matching_vector_1_mean])
            matching_vector_2.extend([matching_vector_2_max, matching_vector_2_mean])
        att_2 = context_2.unsqueeze(-3) * cosine_sim.unsqueeze(-1)
        att_1 = context_1.unsqueeze(-2) * cosine_sim.unsqueeze(-1)
        if self.with_attentive_match:
            att_mean_2 = masked_softmax(att_2.sum(dim=2), mask_1.unsqueeze(-1))
            att_mean_1 = masked_softmax(att_1.sum(dim=1), mask_2.unsqueeze(-1))
            matching_vector_1_att_mean = multi_perspective_match(context_1, att_mean_2, self.attentive_match_weights)
            matching_vector_2_att_mean = multi_perspective_match(context_2, att_mean_1, self.attentive_match_weights_reversed)
            matching_vector_1.extend(matching_vector_1_att_mean)
            matching_vector_2.extend(matching_vector_2_att_mean)
        if self.with_max_attentive_match:
            att_max_2 = masked_max(att_2, mask_2.unsqueeze(-2).unsqueeze(-1), dim=2)
            att_max_1 = masked_max(att_1.permute(0, 2, 1, 3), mask_1.unsqueeze(-2).unsqueeze(-1), dim=2)
            matching_vector_1_att_max = multi_perspective_match(context_1, att_max_2, self.max_attentive_match_weights)
            matching_vector_2_att_max = multi_perspective_match(context_2, att_max_1, self.max_attentive_match_weights_reversed)
            matching_vector_1.extend(matching_vector_1_att_max)
            matching_vector_2.extend(matching_vector_2_att_max)
        return (matching_vector_1, matching_vector_2)