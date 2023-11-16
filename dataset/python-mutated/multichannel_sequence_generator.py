import math
from typing import Dict, List, Optional
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
import torch
import torch.nn as nn
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from fairseq.ngram_repeat_block import NGramRepeatBlock
from .multichannel_search import ContiguousMultichannelBeamSearch
from fairseq.models.speech_dlm import SpeechDLM

class MultichannelSequenceGenerator(nn.Module):

    def __init__(self, models, tgt_dicts, beam_size=1, max_len_a=0, max_len_b=200, min_len=1, normalize_scores=True, len_penalty=1.0, unk_penalty=0.0, temperature=1.0, match_source_len=False, no_repeat_ngram_size=0, search_strategy=None, eos=None, symbols_to_strip_from_output=None, lm_model=None, lm_weight=1.0, duration_temperature=1.0):
        if False:
            while True:
                i = 10
        'Generate multi-channel parallel units with the SpeechDLM model\n        as described in the paper: https://arxiv.org/pdf/2203.16502.pdf;\n\n        Args:\n            models (List[~fairseq.models.FairseqModel]): ensemble of models,\n                currently support fairseq.models.TransformerModel for scripting\n            beam_size (int, optional): beam width (default: 1)\n            max_len_a/b (int, optional): generate sequences of maximum length\n                ax + b, where x is the source length\n            min_len (int, optional): the minimum length of the generated output\n                (not including end-of-sentence)\n            normalize_scores (bool, optional): normalize scores by the length\n                of the output (default: True)\n            len_penalty (float, optional): length penalty, where <1.0 favors\n                shorter, >1.0 favors longer sentences (default: 1.0)\n            unk_penalty (float, optional): unknown word penalty, where <0\n                produces more unks, >0 produces fewer (default: 0.0)\n            temperature (float, optional): temperature, where values\n                >1.0 produce more uniform samples and values <1.0 produce\n                sharper samples (default: 1.0)\n            match_source_len (bool, optional): outputs should match the source\n                length (default: False)\n            duration_temperature (float, optional): rate of the duration prediction,\n                higher rate induces a faster generated wav (default: 1.0)\n        '
        super().__init__()
        if isinstance(models, MultichannelEnsembleModel):
            self.model = models
        else:
            self.model = MultichannelEnsembleModel(models)
        self.tgt_dicts = tgt_dicts
        self.pad = list(tgt_dicts.values())[0].pad()
        self.unk = list(tgt_dicts.values())[0].unk()
        self.eos = list(tgt_dicts.values())[0].eos() if eos is None else eos
        self.symbols_to_strip_from_output = symbols_to_strip_from_output.union({self.eos}) if symbols_to_strip_from_output is not None else {self.eos}
        self.channels = list(tgt_dicts.keys())
        self.n_channels = len(self.channels)
        self.vocab_sizes = [len(tgt_dicts[channel]) for channel in self.channels]
        max_possible_beam_size = 1
        for i in self.vocab_sizes:
            max_possible_beam_size *= i - 1
        self.beam_size = min(beam_size, max_possible_beam_size)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        if isinstance(temperature, (int, float)):
            temperature = {channel: temperature for channel in self.channels}
        elif isinstance(temperature, ListConfig) or isinstance(temperature, list):
            temperature = {channel: temperature[i] for (i, channel) in enumerate(self.channels)}
        assert isinstance(temperature, DictConfig) or isinstance(temperature, dict), f'temperature: expected dict, but found {type(temperature)}'
        self.temperature = temperature
        self.match_source_len = match_source_len
        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None
        for channel in temperature:
            assert temperature[channel] > 0, '--temperature must be greater than 0'
        if search_strategy is None:
            self.search = ContiguousMultichannelBeamSearch(tgt_dicts)
        else:
            self.search = search_strategy
        self.should_set_src_lengths = hasattr(self.search, 'needs_src_lengths') and self.search.needs_src_lengths
        self.model.eval()
        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()
        self.duration_prediction = bool(str(getattr(models[0].decoder.args, 'duration_prediction', 'false')).lower() == 'true')
        self.delayed_duration = bool(str(getattr(models[0].decoder.args, 'delayed_duration_target', 'false')).lower() == 'true')
        self.duration_temperature = duration_temperature

    def cuda(self):
        if False:
            for i in range(10):
                print('nop')
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(self, sample: Dict[str, Dict[str, Tensor]], prefix_tokens: Optional[Dict[str, Tensor]]=None, bos_token: Optional[int]=None):
        if False:
            return 10
        'Generate a batch of translations.\n\n        Args:\n            sample (dict): batch\n            prefix_tokens (dict of torch.LongTensor, optional): force decoder to begin\n                with these tokens\n            bos_token (int, optional): beginning of sentence token\n                (default: self.eos)\n        '
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        if False:
            return 10
        'Generate translations. Match the api of other fairseq generators.\n\n        Args:\n            models (List[~fairseq.models.FairseqModel]): ensemble of models\n            sample (dict): batch\n            prefix_tokens (dict of torch.LongTensor, optional): force decoder to begin\n                with these tokens\n            constraints (torch.LongTensor, optional): force decoder to include\n                the list of constraints\n            bos_token (int, optional): beginning of sentence token\n                (default: self.eos)\n        '
        return self._generate(sample, **kwargs)

    def _generate(self, sample: Dict[str, Dict[str, Tensor]], prefix_tokens: Optional[Dict[str, Tensor]]=None, constraints: Optional[Tensor]=None, bos_token: Optional[int]=None):
        if False:
            while True:
                i = 10
        "\n        Here sample is expected to have the following form\n            {\n                'id': index,\n                'net_input': {\n                    'src_tokens': {\n                        'channel1' : tensor((batch x src_length)),\n                        'channel2' : tensor((batch x src_length)),\n                    },\n                    ...\n                },\n            }\n        and prefix_tokens\n            {\n                'channel1' : tensor((batch x prefix_length)),\n                'channel2' : tensor((batch x prefix_length)),\n            }\n        "
        if self.model.is_speech_dlm:
            incremental_states = torch.jit.annotate(List[Dict[str, Dict[str, Optional[Tensor]]]], [torch.jit.annotate(List[Dict[str, Dict[str, Optional[Tensor]]]], [{} for _ in range(self.n_channels)]) for i in range(self.model.models_size)])
        else:
            incremental_states = torch.jit.annotate(List[Dict[str, Dict[str, Optional[Tensor]]]], [torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}) for i in range(self.model.models_size)])
        net_input = sample['net_input']
        src_tokens = torch.stack([net_input['src_tokens'][channel] for channel in self.channels], dim=-1)
        prefix_tokens = torch.stack([prefix_tokens[channel] for channel in self.channels], dim=-1)
        src_lengths = (src_tokens[..., 0].ne(self.eos) & src_tokens[..., 0].ne(self.pad)).long().sum(dim=1)
        (bsz, src_len) = src_tokens.size()[:2]
        beam_size = self.beam_size
        if constraints is not None and (not self.search.supports_constraints):
            raise NotImplementedError("Target-side constraints were provided, but search method doesn't support them")
        self.search.init_constraints(constraints, beam_size)
        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(int(self.max_len_a * src_len + self.max_len_b), self.model.max_decoder_positions() - 1)
        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!'
        encoder_outs = self.model.forward_encoder(net_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        assert encoder_outs is not None
        scores = torch.zeros(bsz * beam_size, max_len + 1, self.n_channels).to(src_tokens).float()
        tokens = torch.zeros(bsz * beam_size, max_len + 2, self.n_channels).to(src_tokens).long().fill_(self.pad)
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None
        cands_to_ignore = torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        finalized = torch.jit.annotate(List[List[Dict[str, Tensor]]], [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)])
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz
        cand_size = 2 * beam_size
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens).to(src_tokens.device)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)
        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None
        original_batch_idxs: Optional[Tensor] = None
        if 'id' in sample and isinstance(sample['id'], Tensor):
            original_batch_idxs = sample['id']
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)
        if self.duration_prediction:
            dur_counter = torch.ones(bsz * beam_size, self.n_channels).to(src_tokens)
            dur_counter_jump_indices = None
        for step in range(max_len + 1):
            if reorder_state is not None:
                if batch_idxs is not None:
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(encoder_outs, reorder_state)
            input_tokens = {channel: tokens[:, :step + 1, i] for (i, channel) in enumerate(self.channels)}
            (lprobs_dict, avg_attn_scores) = self.model.forward_decoder(input_tokens, encoder_outs, incremental_states, self.temperature)
            if not self.duration_prediction:
                lprobs_list = list(lprobs_dict.values())
            else:
                lprobs_list = [net_output['pred_token'] for net_output in lprobs_dict.values()]
                dur_preds = torch.stack([net_output['pred_duration'] for net_output in lprobs_dict.values()]).squeeze(-1).T
                dur_preds = dur_preds / self.duration_temperature
                dur_preds = dur_preds.round().long()
                dur_preds[dur_preds < 1] = 1
                if step > 0:
                    non_edge_indices = tokens[:, step, :] == tokens[:, step - 1, :]
                    if self.delayed_duration:
                        dur_preds[non_edge_indices] = 1
                    elif dur_counter_jump_indices is not None:
                        dur_counter[dur_counter_jump_indices & non_edge_indices] = 2
                if step > 0:
                    if self.delayed_duration:
                        dur_counter -= ((dur_counter == 1) | (tokens[:, step, :] == tokens[:, step - 1, :])).int()
                        dur_counter[dur_counter < 0] = 0
                    else:
                        dur_counter -= (tokens[:, step, :] == tokens[:, step - 1, :]).int()
                        dur_counter[dur_counter < 1] = 1
                if self.delayed_duration:
                    dur_counter_jump_indices = dur_counter == 0
                    dur_counter[dur_counter_jump_indices] = dur_preds[dur_counter_jump_indices]
                copy_prev_token = dur_counter != 1
                if self.delayed_duration is False:
                    dur_counter_jump_indices = dur_counter == 1
                    dur_counter[dur_counter_jump_indices] = dur_preds[dur_counter_jump_indices]
            if self.lm_model is not None:
                assert False, 'Currently not supported in multichannelLM case'
            for i in range(self.n_channels):
                lprobs_list[i][lprobs_list[i] != lprobs_list[i]] = torch.tensor(-math.inf).to(lprobs_list[i])
                lprobs_list[i][:, self.pad] = -math.inf
                lprobs_list[i][:, self.unk] -= self.unk_penalty
                if step >= max_len:
                    lprobs_list[i][:, :self.eos] = -math.inf
                    lprobs_list[i][:, self.eos + 1:] = -math.inf
                else:
                    lprobs_list[i][:, self.eos] = -math.inf
                if prefix_tokens is not None and step < prefix_tokens.size(1) and (step < max_len):
                    (lprobs_list[i], tokens[..., i], scores[..., i]) = self._prefix_tokens(step, lprobs_list[i], scores[..., i], tokens[..., i], prefix_tokens[..., i], beam_size)
                    if self.duration_prediction:
                        can_copy_mask = (prefix_tokens[:, step, i].eq(self.pad) | prefix_tokens[:, step, i].eq(self.unk)).repeat_interleave(beam_size)
                        copy_prev_token[:, i] &= can_copy_mask
                elif step < self.min_len:
                    lprobs_list[i][:, self.eos] = -math.inf
                if self.duration_prediction:
                    if step < max_len:
                        for j in range(copy_prev_token.size(0)):
                            if copy_prev_token[j, i]:
                                prev_token = tokens[j, step, i]
                                lprobs_list[i][j, :prev_token] = -math.inf
                                lprobs_list[i][j, prev_token + 1:] = -math.inf
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(bsz * beam_size, avg_attn_scores.size(1), max_len + 2).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)
            scores = scores.type_as(lprobs_list[0])
            eos_bbsz_idx = torch.empty(0).to(tokens)
            eos_scores = torch.empty(0).to(scores)
            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)
            if self.repeat_ngram_blocker is not None:
                for i in range(self.n_channels):
                    lprobs_list[i] = self.repeat_ngram_blocker(tokens, lprobs_list[i], bsz, beam_size, step)
            (cand_scores, cand_indices, cand_beams) = self.search.step(step, [lprobs_list[i].view(bsz, -1, self.vocab_sizes[i]) for i in range(self.n_channels)], scores.view(bsz, beam_size, -1, self.n_channels)[:, :, :step, :], tokens[:, :step + 1], original_batch_idxs)
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask = torch.any(eos_mask, dim=-1, keepdim=False)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)
            eos_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size])
            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.stack([torch.masked_select(cand_scores[:, :beam_size, i], mask=eos_mask[:, :beam_size]) for i in range(self.n_channels)], dim=-1)
                finalized_sents = self.finalize_hypos(step, eos_bbsz_idx, eos_scores, tokens, scores, finalized, finished, beam_size, attn, src_lengths, max_len)
                num_remaining_sent -= len(finalized_sents)
            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f'{step} < {max_len}'
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)
                batch_mask = torch.ones(bsz, dtype=torch.bool, device=cand_indices.device)
                batch_mask[finalized_sents] = False
                batch_idxs = torch.arange(bsz, device=cand_indices.device).masked_select(batch_mask)
                self.search.prune_sentences(batch_idxs)
                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]
                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1, self.n_channels)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1, self.n_channels)
                if self.duration_prediction:
                    dur_counter = dur_counter.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, self.n_channels)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                bsz = new_bsz
            else:
                batch_idxs = None
            eos_mask[:, :beam_size] = ~(~cands_to_ignore & ~eos_mask[:, :beam_size])
            active_mask = torch.add(eos_mask.type_as(cand_offsets) * cand_size, cand_offsets[:eos_mask.size(1)])
            (new_cands_to_ignore, active_hypos) = torch.topk(active_mask, k=beam_size, dim=1, largest=False)
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            assert (~cands_to_ignore).any(dim=1).all()
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_bbsz_idx = active_bbsz_idx.view(-1)
            tokens[:, :step + 1] = torch.index_select(tokens[:, :step + 1], dim=0, index=active_bbsz_idx)
            for i in range(self.n_channels):
                tokens.view(bsz, beam_size, -1, self.n_channels)[:, :, step + 1, i] = torch.gather(cand_indices[..., i], dim=1, index=active_hypos)
            if step > 0:
                scores[:, :step] = torch.index_select(scores[:, :step], dim=0, index=active_bbsz_idx)
            for i in range(self.n_channels):
                scores.view(bsz, beam_size, -1, self.n_channels)[:, :, step, i] = torch.gather(cand_scores[..., i], dim=1, index=active_hypos)
            if self.duration_prediction:
                dur_counter = torch.index_select(dur_counter, dim=0, index=active_bbsz_idx)
            self.search.update_constraints(active_hypos)
            if attn is not None:
                attn[:, :, :step + 2] = torch.index_select(attn[:, :, :step + 2], dim=0, index=active_bbsz_idx)
            reorder_state = active_bbsz_idx
        for sent in range(len(finalized)):
            scores = torch.tensor([float(elem['score'].item()) for elem in finalized[sent]])
            (_, sorted_scores_indices) = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(List[Dict[str, Tensor]], finalized[sent])
        return finalized

    def _prefix_tokens(self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int):
        if False:
            for i in range(10):
                print('nop')
        'Handle prefix tokens'
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        prefix_mask &= prefix_toks.ne(self.unk)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(-1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask])
        unk_mask = prefix_toks.eq(self.unk)
        if len(lprobs[unk_mask]) > 0:
            copy_lprobs = lprobs[unk_mask][:, :]
            copy_lprobs[:, self.eos] = -math.inf
            lprobs[unk_mask] = copy_lprobs
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return (lprobs, tokens, scores)

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        if False:
            for i in range(10):
                print('nop')
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(self, step: int, bbsz_idx, eos_scores, tokens, scores, finalized: List[List[Dict[str, Tensor]]], finished: List[bool], beam_size: int, attn: Optional[Tensor], src_lengths, max_len: int):
        if False:
            while True:
                i = 10
        'Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.\n        A sentence is finalized when {beam_size} finished items have been collected for it.\n\n        Returns number of sentences (not beam items) being finalized.\n        These will be removed from the batch and not processed further.\n        Args:\n            bbsz_idx (Tensor):\n        '
        assert bbsz_idx.numel() == eos_scores.size(0)
        tokens_clone = tokens.index_select(0, bbsz_idx)[:, 1:step + 2]
        tokens_clone[:, step] = self.eos
        attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2] if attn is not None else None
        pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
        pos_scores[:, step, :] = eos_scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        sents_seen: Dict[str, Optional[Tensor]] = {}
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i].sum()
            unfin_idx = idx // beam_size
            sent = unfin_idx + cum_unfin[unfin_idx]
            seen = str(sent.item()) + '_' + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None
            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf).to(score)
            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)
                finalized[sent].append({'tokens': tokens_clone[i], 'score': score, 'attention': hypo_attn, 'alignment': torch.empty(0), 'positional_scores': pos_scores[i]})
        newly_finished: List[int] = []
        for seen in sents_seen.keys():
            sent: int = int(float(seen.split('_')[0]))
            unfin_idx: int = int(float(seen.split('_')[1]))
            if not finished[sent] and self.is_finished(step, unfin_idx, max_len, len(finalized[sent]), beam_size):
                finished[sent] = True
                newly_finished.append(unfin_idx)
        return newly_finished

    def is_finished(self, step: int, unfin_idx: int, max_len: int, finalized_sent_len: int, beam_size: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check whether decoding for a sentence is finished, which\n        occurs when the list of finalized sentences has reached the\n        beam size, or when we reach the maximum length.\n        '
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False

class MultichannelEnsembleModel(nn.Module):
    """A wrapper around an ensemble of SpeechDLM models."""

    def __init__(self, models):
        if False:
            print('Hello World!')
        super().__init__()
        self.models_size = len(models)
        self.single_model = models[0]
        self.models = nn.ModuleList(models)
        self.has_incremental: bool = False
        if all((hasattr(m, 'decoder') and isinstance(m.decoder, FairseqIncrementalDecoder) for m in models)):
            self.has_incremental = True
        if isinstance(models[0], SpeechDLM):
            self.is_speech_dlm = True
        else:
            self.is_speech_dlm = False
        if getattr(models[0].decoder.args, 'duration_prediction', False):
            self.is_duration_prediction = True
        else:
            self.is_duration_prediction = False

    def forward(self):
        if False:
            while True:
                i = 10
        pass

    def has_encoder(self):
        if False:
            return 10
        return hasattr(self.single_model, 'encoder')

    def has_incremental_states(self):
        if False:
            print('Hello World!')
        return self.has_incremental

    def max_decoder_positions(self):
        if False:
            for i in range(10):
                print('nop')
        return min([m.max_decoder_positions() for m in self.models])

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if False:
            print('Hello World!')
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(self, tokens, encoder_outs: List[Dict[str, List[Tensor]]], incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]], temperature: Dict[str, float]=1.0):
        if False:
            i = 10
            return i + 15
        if isinstance(temperature, (float, int)):
            temperature = {channel: temperature for channel in tokens}
        log_probs = {channel: [] for channel in tokens}
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for (i, model) in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out, incremental_state=incremental_states[i])
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]['attn']
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]
            if self.is_speech_dlm:
                if self.is_duration_prediction:
                    decoder_out_divided_by_temperature = {channel_src: {channel_pred: {'pred_token': decoder_out[0][channel_src][channel_pred]['pred_token'][:, -1:, :].div_(temperature[channel_pred]), 'pred_duration': decoder_out[0][channel_src][channel_pred]['pred_duration'][:, -1:, :]} for channel_pred in decoder_out[0][channel_src]} for channel_src in decoder_out[0]}
                else:
                    decoder_out_divided_by_temperature = {channel_src: {channel_pred: decoder_out[0][channel_src][channel_pred][:, -1:, :].div_(temperature[channel_pred]) for channel_pred in decoder_out[0][channel_src]} for channel_src in decoder_out[0]}
            else:
                decoder_out_divided_by_temperature = {channel: decoder_out[0][channel][:, -1:, :].div_(temperature[channel]) for channel in decoder_out[0]}
            decoder_out_tuple = (decoder_out_divided_by_temperature, None if decoder_len <= 1 else decoder_out[1])
            probs = model.get_normalized_probs(decoder_out_tuple, log_probs=True, sample=None)
            if self.is_speech_dlm:
                if self.is_duration_prediction:
                    probs = {channel: {'pred_token': probs[channel][channel]['pred_token'][:, -1, :], 'pred_duration': probs[channel][channel]['pred_duration'][:, -1, :]} for channel in probs}
                else:
                    probs = {channel: probs[channel][channel][:, -1, :] for channel in probs}
            else:
                probs = {channel: probs[channel][:, -1, :] for channel in probs}
            if self.models_size == 1:
                return (probs, attn)
            for channel in probs:
                log_probs[channel].append(probs[channel])
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = {}
        for channel in log_probs:
            avg_probs[channel] = torch.logsumexp(torch.stack(log_probs[channel], dim=0), dim=0) - math.log(self.models_size)
        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return (avg_probs, avg_attn)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order):
        if False:
            print('Hello World!')
        '\n        Reorder encoder output according to *new_order*.\n\n        Args:\n            encoder_out: output from the ``forward()`` method\n            new_order (LongTensor): desired order\n\n        Returns:\n            *encoder_out* rearranged according to *new_order*\n        '
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for (i, model) in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(model.encoder.reorder_encoder_out(encoder_outs[i], new_order))
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(self, incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]], new_order):
        if False:
            i = 10
            return i + 15
        if not self.has_incremental_states():
            return
        for (i, model) in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(incremental_states[i], new_order)