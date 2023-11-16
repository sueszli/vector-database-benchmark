from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from modelscope.metainfo import Heads
from modelscope.models.base import TorchHead
from modelscope.models.builder import HEADS
from modelscope.outputs import AttentionTokenClassificationModelOutput, ModelOutputBase, OutputKeys, TokenClassificationModelOutput
from modelscope.utils.constant import Tasks

@HEADS.register_module(Tasks.token_classification, module_name=Heads.lstm_crf)
@HEADS.register_module(Tasks.named_entity_recognition, module_name=Heads.lstm_crf)
@HEADS.register_module(Tasks.word_segmentation, module_name=Heads.lstm_crf)
@HEADS.register_module(Tasks.part_of_speech, module_name=Heads.lstm_crf)
class LSTMCRFHead(TorchHead):

    def __init__(self, hidden_size=100, num_labels=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(hidden_size=hidden_size, num_labels=num_labels)
        assert num_labels is not None
        self.ffn = nn.Linear(hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, inputs: ModelOutputBase, attention_mask=None, label=None, label_mask=None, offset_mapping=None, **kwargs):
        if False:
            print('Hello World!')
        logits = self.ffn(inputs.last_hidden_state)
        return TokenClassificationModelOutput(loss=None, logits=logits)

    def decode(self, logits, label_mask):
        if False:
            i = 10
            return i + 15
        seq_lens = label_mask.sum(-1).long()
        mask = torch.arange(label_mask.shape[1], device=seq_lens.device)[None, :] < seq_lens[:, None]
        predicts = self.crf.decode(logits, mask).squeeze(0)
        return predicts

@HEADS.register_module(Tasks.transformer_crf, module_name=Heads.transformer_crf)
@HEADS.register_module(Tasks.token_classification, module_name=Heads.transformer_crf)
@HEADS.register_module(Tasks.named_entity_recognition, module_name=Heads.transformer_crf)
@HEADS.register_module(Tasks.word_segmentation, module_name=Heads.transformer_crf)
@HEADS.register_module(Tasks.part_of_speech, module_name=Heads.transformer_crf)
class TransformersCRFHead(TorchHead):

    def __init__(self, hidden_size, num_labels, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(hidden_size=hidden_size, num_labels=num_labels, **kwargs)
        self.linear = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, inputs: ModelOutputBase, attention_mask=None, label=None, label_mask=None, offset_mapping=None, **kwargs):
        if False:
            print('Hello World!')
        logits = self.linear(inputs.last_hidden_state)
        if label_mask is not None:
            mask = label_mask
            masked_lengths = mask.sum(-1).long()
            masked_logits = torch.zeros_like(logits)
            for i in range(mask.shape[0]):
                masked_logits[i, :masked_lengths[i], :] = logits[i].masked_select(mask[i].unsqueeze(-1)).view(masked_lengths[i], -1)
            logits = masked_logits
        return AttentionTokenClassificationModelOutput(loss=None, logits=logits, hidden_states=inputs.hidden_states, attentions=inputs.attentions)

    def decode(self, logits, label_mask):
        if False:
            print('Hello World!')
        seq_lens = label_mask.sum(-1).long()
        mask = torch.arange(label_mask.shape[1], device=seq_lens.device)[None, :] < seq_lens[:, None]
        predicts = self.crf.decode(logits, mask).squeeze(0)
        return predicts

class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm

    """

    def __init__(self, num_tags: int, batch_first: bool=False) -> None:
        if False:
            print('Hello World!')
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize the transition parameters.\n        The parameters will be initialized randomly from a uniform distribution\n        between -0.1 and 0.1.\n        '
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: Optional[torch.ByteTensor]=None, reduction: str='mean') -> torch.Tensor:
        if False:
            while True:
                i = 10
        'Compute the conditional log likelihood of a sequence of tags given emission scores.\n        Args:\n            emissions (`~torch.Tensor`): Emission score tensor of size\n                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,\n                ``(batch_size, seq_length, num_tags)`` otherwise.\n            tags (`~torch.LongTensor`): Sequence of tags tensor of size\n                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,\n                ``(batch_size, seq_length)`` otherwise.\n            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``\n                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.\n            reduction: Specifies  the reduction to apply to the output:\n                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.\n                ``sum``: the output will be summed over batches. ``mean``: the output will be\n                averaged over batches. ``token_mean``: the output will be averaged over tokens.\n        Returns:\n            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if\n            reduction is ``none``, ``()`` otherwise.\n        '
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.ByteTensor]=None, nbest: Optional[int]=None, pad_tag: Optional[int]=None) -> List[List[List[int]]]:
        if False:
            for i in range(10):
                print('nop')
        'Find the most likely tag sequence using Viterbi algorithm.\n        Args:\n            emissions (`~torch.Tensor`): Emission score tensor of size\n                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,\n                ``(batch_size, seq_length, num_tags)`` otherwise.\n            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``\n                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.\n            nbest (`int`): Number of most probable paths for each sequence\n            pad_tag (`int`): Tag at padded positions. Often input varies in length and\n                the length will be padded to the maximum length in the batch. Tags at\n                the padded positions will be assigned with a padding tag, i.e. `pad_tag`\n        Returns:\n            A PyTorch tensor of the best tag sequence for each batch of shape\n            (nbest, batch_size, seq_length)\n        '
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(self, emissions: torch.Tensor, tags: Optional[torch.LongTensor]=None, mask: Optional[torch.ByteTensor]=None) -> None:
        if False:
            i = 10
            return i + 15
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(f'expected last dimension of emissions is {self.num_tags}, got {emissions.size(2)}')
        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(f'the first two dimensions of emissions and tags must match, got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(f'the first two dimensions of emissions and mask must match, got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and (not no_empty_seq_bf):
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor) -> torch.Tensor:
        if False:
            return 10
        (seq_length, batch_size) = tags.shape
        mask = mask.float()
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        seq_length = emissions.size(0)
        score = self.start_transitions + emissions[0]
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.ByteTensor, pad_tag: Optional[int]=None) -> List[List[int]]:
        if False:
            for i in range(10):
                print('nop')
        if pad_tag is None:
            pad_tag = 0
        device = emissions.device
        (seq_length, batch_size) = mask.shape
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags), dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags), dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size), pad_tag, dtype=torch.long, device=device)
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            (next_score, indices) = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(-1).bool(), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).bool(), indices, oor_idx)
            history_idx[i - 1] = indices
        end_score = score + self.end_transitions
        (_, end_tag) = end_score.max(dim=1)
        seq_ends = mask.long().sum(dim=0) - 1
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags), end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags))
        history_idx = history_idx.transpose(1, 0).contiguous()
        best_tags_arr = torch.zeros((seq_length, batch_size), dtype=torch.long, device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)
        return torch.where(mask.bool(), best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor, mask: torch.ByteTensor, nbest: int, pad_tag: Optional[int]=None) -> List[List[List[int]]]:
        if False:
            i = 10
            return i + 15
        if pad_tag is None:
            pad_tag = 0
        device = emissions.device
        (seq_length, batch_size) = mask.shape
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size, nbest), pad_tag, dtype=torch.long, device=device)
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission
            (next_score, indices) = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)
            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)
            score = torch.where(mask[i].unsqueeze(-1).bool().unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1).bool(), indices, oor_idx)
            history_idx[i - 1] = indices
        end_score = score + self.end_transitions.unsqueeze(-1)
        (_, end_tag) = end_score.view(batch_size, -1).topk(nbest, dim=1)
        seq_ends = mask.long().sum(dim=0) - 1
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest), end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        history_idx = history_idx.transpose(1, 0).contiguous()
        best_tags_arr = torch.zeros((seq_length, batch_size, nbest), dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest
        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)