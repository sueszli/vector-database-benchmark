from typing import Dict, Optional
import torch
import torch.nn as nn
from torch import Tensor

class MultichannelSearch(nn.Module):

    def __init__(self, tgt_dicts):
        if False:
            return 10
        super().__init__()
        tgt_dict = list(tgt_dicts.values())[0]
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        for tgt_dict in tgt_dicts.values():
            assert self.pad == tgt_dict.pad()
            assert self.unk == tgt_dict.unk()
            assert self.eos == tgt_dict.eos()
        self.vocab_sizes = {channel: len(tgt_dicts[channel]) for channel in tgt_dicts}
        self.src_lengths = torch.tensor(-1)
        self.supports_constraints = False
        self.stop_on_max_len = False

    def step(self, step, lprobs, scores, prev_output_tokens=None, original_batch_idxs=None):
        if False:
            print('Hello World!')
        "Take a single search step.\n\n        Args:\n            step: the current search step, starting at 0\n            lprobs: dictionary of channels {channel : (bsz x input_beam_size x vocab_size_channel)}\n                the model's log-probabilities over the vocabulary at the current step\n            scores: {channel : (bsz x input_beam_size x step)}\n                the historical model scores of each hypothesis up to this point\n            prev_output_tokens: {channel : (bsz x step)}\n                the previously generated oputput tokens\n            original_batch_idxs: (bsz)\n                the tensor with the batch indices, in the range [0, bsz)\n                this is useful in case there has been applied a re-ordering\n                and we need to know the orignal indices\n\n        Return: A tuple of (scores, indices, beams) where:\n            scores: {channel : (bsz x output_beam_size)}\n                the scores of the chosen elements; output_beam_size can be\n                larger than input_beam_size, e.g., we may return\n                2*input_beam_size to account for EOS\n            indices: {channel : (bsz x output_beam_size)}\n                the indices of the chosen elements\n            beams: (bsz x output_beam_size)\n                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)\n        "
        raise NotImplementedError

    @torch.jit.export
    def set_src_lengths(self, src_lengths):
        if False:
            i = 10
            return i + 15
        self.src_lengths = src_lengths

    @torch.jit.export
    def init_constraints(self, batch_constraints: Optional[Tensor], beam_size: int):
        if False:
            i = 10
            return i + 15
        'Initialize constraint states for constrained decoding (if supported).\n\n        Args:\n            batch_constraints: (torch.Tensor, optional)\n                the list of constraints, in packed form\n            beam_size: (int)\n                the beam size\n        Returns:\n            *encoder_out* rearranged according to *new_order*\n        '
        pass

    def prune_sentences(self, batch_idxs: Tensor):
        if False:
            while True:
                i = 10
        '\n        Removes constraint states for completed sentences (if supported).\n        This is called from sequence_generator._generate() when sentences are\n        deleted from the batch.\n\n        Args:\n            batch_idxs: Indices of *sentences* whose constraint state should be *kept*.\n        '
        pass

    def update_constraints(self, active_hypos: Tensor):
        if False:
            for i in range(10):
                print('nop')
        '\n        Updates the constraint states by selecting the beam items that are retained.\n        This is called at each time step of sequence_generator._generate() when\n        the set of 2 * {beam_size} candidate hypotheses are reduced to the beam size.\n\n        Args:\n            active_hypos: (batch size, beam size)\n              list of integers denoting, for each sentence, which beam candidate items\n              should be kept.\n        '
        pass

def unravel_index(index, shape):
    if False:
        for i in range(10):
            print('nop')
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return torch.stack(tuple(reversed(out)), dim=-1)

def topk_sum(lprobs_list, k):
    if False:
        print('Hello World!')
    '\n    lprobs_list = [lprobs_1,...,lprobs_n], where:\n        lprobs_1 : (batch_size x beam_size x vocab_1)\n        ...\n        lprobs_n : (batch_size x beam_size x vocab_n)\n\n    Return:\n        - topk_values : (batch_size x k)\n            values of the topk sum of the form :\n                lprobs_1[bsz, beam_idx, vocab_1_idx] + ... + lprobs_n[bsz, beam_idx, vocab_n_idx]\n        - topk_idxs : (batch_size x k x n+1)\n            each (n+1)-tensor being [beam_idx, vocab_1_idx, ..., vocab_n_idx]\n    '
    lprobs_topk_list = []
    lprobs_topk_indices_list = []
    for lprobs in lprobs_list:
        k_i = min(k, lprobs.size(-1))
        (topk_values, topk_indices) = torch.topk(lprobs, k=k_i)
        lprobs_topk_list.append(topk_values)
        lprobs_topk_indices_list.append(topk_indices)
    sum_lprobs_topk = lprobs_topk_list[0]
    for i in range(1, len(lprobs_topk_list)):
        unsqueezed_lprobs = lprobs_topk_list[i]
        for _ in range(i):
            unsqueezed_lprobs = unsqueezed_lprobs.unsqueeze(-2)
        sum_lprobs_topk = sum_lprobs_topk.unsqueeze(-1) + unsqueezed_lprobs
    (topk_sum_values, topk_sum_indices) = torch.topk(sum_lprobs_topk.view(sum_lprobs_topk.size(0), -1), k=k)
    topk_sum_indices = unravel_index(topk_sum_indices, tuple(sum_lprobs_topk.shape[1:]))
    for i_batch in range(topk_sum_indices.size(0)):
        for i_cand in range(topk_sum_indices.size(1)):
            (i_beam, *transformed_vocab_indices) = topk_sum_indices[i_batch, i_cand]
            true_vocab_indices = [i_beam]
            for (j, transformed_vocab_j_idx) in enumerate(transformed_vocab_indices):
                true_vocab_j_idx = lprobs_topk_indices_list[j][i_batch, i_beam, transformed_vocab_j_idx]
                true_vocab_indices.append(true_vocab_j_idx)
            topk_sum_indices[i_batch, i_cand] = torch.tensor(true_vocab_indices)
    topk_sum_beams = topk_sum_indices[:, :, 0]
    topk_sum_indices = topk_sum_indices[:, :, 1:]
    return (topk_sum_values, topk_sum_indices, topk_sum_beams)

class MultichannelBeamSearch(MultichannelSearch):

    def __init__(self, tgt_dicts):
        if False:
            print('Hello World!')
        super().__init__(tgt_dicts)
        self.constraint_states = None

    @torch.jit.export
    def step(self, step: int, lprobs, scores: Optional[Dict[str, Tensor]], prev_output_tokens: Optional[Dict[str, Tensor]]=None, original_batch_idxs: Optional[Tensor]=None):
        if False:
            while True:
                i = 10
        channels = list(lprobs.keys())
        (bsz, beam_size, _) = lprobs[channels[0]].size()
        lprobs_list = []
        if step == 0:
            for channel in channels:
                lprobs_list.append(lprobs[channel][:, ::beam_size, :].contiguous())
        else:
            assert scores is not None
            for channel in channels:
                lprobs_list.append(lprobs[channel] + scores[channel][:, :, step - 1].unsqueeze(-1))
        (topk_sum_values, topk_sum_indices, topk_sum_beams) = topk_sum(lprobs_list, k=beam_size * 2)
        beams_buf = topk_sum_beams
        scores_buf = {}
        indices_buf = {}
        for (i, channel) in enumerate(channels):
            indices_buf[channel] = topk_sum_indices[:, :, i]
            scores_buf[channel] = torch.tensor([lprobs_list[i][i_batch, i_beam, i_index] for i_batch in range(bsz) for (i_beam, i_index) in zip(beams_buf[i_batch], indices_buf[channel][i_batch])]).view(bsz, -1).to(lprobs_list[i].device)
        return (scores_buf, indices_buf, beams_buf)

class ContiguousMultichannelBeamSearch(MultichannelSearch):

    def __init__(self, tgt_dicts):
        if False:
            print('Hello World!')
        super().__init__(tgt_dicts)
        self.constraint_states = None

    @torch.jit.export
    def step(self, step: int, lprobs, scores: Optional[Tensor], prev_output_tokens: Optional[Tensor]=None, original_batch_idxs: Optional[Tensor]=None):
        if False:
            for i in range(10):
                print('nop')
        n_channels = len(lprobs)
        (bsz, beam_size, _) = lprobs[0].size()
        lprobs_list = []
        if step == 0:
            for i in range(n_channels):
                lprobs_list.append(lprobs[i][:, ::beam_size, :].contiguous())
        else:
            assert scores is not None
            for i in range(n_channels):
                lprobs_list.append(lprobs[i] + scores[:, :, step - 1, i].unsqueeze(-1))
        (topk_sum_values, topk_sum_indices, topk_sum_beams) = topk_sum(lprobs_list, k=beam_size * 2)
        beams_buf = topk_sum_beams
        indices_buf = topk_sum_indices
        scores_buf = torch.tensor([lprobs_list[i][i_batch, i_beam, i_index] for i in range(len(lprobs_list)) for i_batch in range(bsz) for (i_beam, i_index) in zip(beams_buf[i_batch], indices_buf[i_batch, :, i])]).view(len(lprobs_list), bsz, -1).permute(1, 2, 0).to(lprobs_list[0].device)
        return (scores_buf, indices_buf, beams_buf)

class ContiguousMultichannelSampling(MultichannelSearch):
    sampling_topk: int
    sampling_topp: float

    def __init__(self, tgt_dicts, sampling_topk=-1, sampling_topp=-1.0):
        if False:
            while True:
                i = 10
        super().__init__(tgt_dicts)
        self.sampling_topk = sampling_topk
        self.sampling_topp = sampling_topp

    def _sample_topp(self, lprobs):
        if False:
            i = 10
            return i + 15
        'Sample among the smallest set of elements whose cumulative probability mass exceeds p.\n\n        See `"The Curious Case of Neural Text Degeneration"\n        (Holtzman et al., 2019) <https://arxiv.org/abs/1904.09751>`_.\n\n        Args:\n            lprobs: (bsz x input_beam_size x vocab_size)\n                the model\'s log-probabilities over the vocabulary at the current step\n\n        Return: A tuple of (trimed_probs, truncated_indices) where:\n            trimed_probs: (bsz x input_beam_size x ?)\n                the model\'s probabilities over the elements selected to sample from. The\n                width of the third dimension is determined by top-P.\n            truncated_indices: (bsz x input_beam_size x ?)\n                the indices of the chosen elements.\n        '
        probs = lprobs.exp_()
        (sorted_probs, sorted_indices) = probs.sort(descending=True)
        cumsum_probs = sorted_probs.cumsum(dim=2)
        mask = cumsum_probs.lt(self.sampling_topp)
        cumsum_mask = mask.cumsum(dim=2)
        last_included = cumsum_mask[:, :, -1:]
        last_included.clamp_(0, mask.size()[2] - 1)
        mask = mask.scatter_(2, last_included, 1)
        max_dim = last_included.max()
        truncated_mask = mask[:, :, :max_dim + 1]
        truncated_probs = sorted_probs[:, :, :max_dim + 1]
        truncated_indices = sorted_indices[:, :, :max_dim + 1]
        trim_mask = ~truncated_mask
        trimed_probs = truncated_probs.masked_fill_(trim_mask, 0)
        return (trimed_probs, truncated_indices)

    @torch.jit.export
    def step(self, step: int, lprobs, scores, prev_output_tokens: Optional[Tensor]=None, original_batch_idxs: Optional[Tensor]=None):
        if False:
            print('Hello World!')
        n_channels = len(lprobs)
        (bsz, beam_size, vocab_size) = lprobs[0].size()
        if step == 0:
            for i in range(n_channels):
                lprobs[i] = lprobs[i][:, ::beam_size, :].contiguous()
        probs = []
        top_indices = []
        for i in range(n_channels):
            if self.sampling_topp > 0:
                (probs_i, top_indices_i) = self._sample_topp(lprobs[i])
            elif self.sampling_topk > 0:
                (lprobs[i], top_indices_i) = lprobs[i].topk(min(self.sampling_topk, lprobs[i].size(-1)))
                probs_i = lprobs[i].exp_()
            else:
                probs_i = lprobs[i].exp_()
                top_indices_i = torch.empty(0).to(probs_i)
            probs.append(probs_i)
            top_indices.append(top_indices_i)
        indices_buf = []
        for i in range(n_channels):
            if step == 0:
                indices_buf.append(torch.multinomial(probs[i].view(bsz, -1), beam_size, replacement=True).view(bsz, beam_size))
            else:
                indices_buf.append(torch.multinomial(probs[i].view(bsz * beam_size, -1), 1, replacement=True).view(bsz, beam_size))
        if step == 0:
            for i in range(n_channels):
                probs[i] = probs[i].expand(bsz, beam_size, -1)
        scores_buf = []
        for i in range(n_channels):
            scores_buf.append(torch.gather(probs[i], dim=2, index=indices_buf[i].unsqueeze(-1)))
            scores_buf[i] = scores_buf[i].log_().view(bsz, -1)
        if self.sampling_topk > 0 or self.sampling_topp > 0:
            for i in range(n_channels):
                indices_buf[i] = torch.gather(top_indices[i].expand(bsz, beam_size, -1), dim=2, index=indices_buf[i].unsqueeze(-1)).squeeze(2)
        if step == 0:
            beams_buf = indices_buf[0].new_zeros(bsz, beam_size)
        else:
            beams_buf = torch.arange(0, beam_size).to(indices_buf[0]).repeat(bsz, 1)
            for i in range(n_channels):
                scores_buf[i].add_(torch.gather(scores[:, :, step - 1, i], dim=1, index=beams_buf))
        scores_buf = torch.stack(scores_buf, dim=-1)
        indices_buf = torch.stack(indices_buf, dim=-1)
        return (scores_buf, indices_buf, beams_buf)