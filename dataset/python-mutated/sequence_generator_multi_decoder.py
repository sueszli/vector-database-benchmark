from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from fairseq import search

class MultiDecoderSequenceGenerator(nn.Module):

    def __init__(self, models, tgt_dict, tgt_dict_mt, beam_size=1, beam_size_mt=1, max_len_a=0, max_len_b=200, max_len_a_mt=0, max_len_b_mt=200, max_len=0, min_len=1, normalize_scores=True, len_penalty=1.0, len_penalty_mt=1.0, unk_penalty=0.0, temperature=1.0, match_source_len=False, no_repeat_ngram_size=0, eos=None, eos_mt=None, symbols_to_strip_from_output=None, lm_model=None, lm_weight=1.0):
        if False:
            i = 10
            return i + 15
        'Generates translations of a given source sentence.\n\n        Args:\n            models (List[~fairseq.models.FairseqModel]): ensemble of models,\n                currently support fairseq.models.TransformerModel for scripting\n            beam_size (int, optional): beam width (default: 1)\n            max_len_a/b (int, optional): generate sequences of maximum length\n                ax + b, where x is the source length for the second pass\n            max_len_a_mt/b_mt (int, optional): generate sequences of maximum length\n                ax + b, where x is the source length for the first pass\n            max_len (int, optional): the maximum length of the generated output\n                (not including end-of-sentence)\n            min_len (int, optional): the minimum length of the generated output\n                (not including end-of-sentence)\n            normalize_scores (bool, optional): normalize scores by the length\n                of the output (default: True)\n            len_penalty (float, optional): length penalty in the second pass, where <1.0 favors\n                shorter, >1.0 favors longer sentences (default: 1.0)\n            len_penalty (float, optional): length penalty in the first pass, where <1.0 favors\n                shorter, >1.0 favors longer sentences (default: 1.0)\n            unk_penalty (float, optional): unknown word penalty, where <0\n                produces more unks, >0 produces fewer (default: 0.0)\n            temperature (float, optional): temperature, where values\n                >1.0 produce more uniform samples and values <1.0 produce\n                sharper samples (default: 1.0)\n            match_source_len (bool, optional): outputs should match the source\n                length (default: False)\n        '
        super().__init__()
        from examples.speech_to_speech.unity.sequence_generator import SequenceGenerator
        self.generator = SequenceGenerator(models, tgt_dict, beam_size=beam_size, max_len_a=max_len_a, max_len_b=max_len_b, max_len=max_len, min_len=min_len, normalize_scores=normalize_scores, len_penalty=len_penalty, unk_penalty=unk_penalty, temperature=temperature, match_source_len=match_source_len, no_repeat_ngram_size=no_repeat_ngram_size, search_strategy=search.BeamSearch(tgt_dict), eos=eos, symbols_to_strip_from_output=symbols_to_strip_from_output, lm_model=lm_model, lm_weight=lm_weight)
        self.eos = self.generator.eos
        self.generator_mt = SequenceGenerator(models, tgt_dict_mt, beam_size=beam_size_mt, max_len_a=max_len_a_mt, max_len_b=max_len_b_mt, max_len=max_len, min_len=min_len, normalize_scores=normalize_scores, len_penalty=len_penalty_mt, unk_penalty=unk_penalty, temperature=temperature, match_source_len=match_source_len, no_repeat_ngram_size=no_repeat_ngram_size, search_strategy=search.BeamSearch(tgt_dict_mt), eos=eos_mt, symbols_to_strip_from_output=symbols_to_strip_from_output)

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs) -> List[List[Dict[str, Tensor]]]:
        if False:
            print('Hello World!')
        'Generate translations. Match the api of other fairseq generators.\n\n        Args:\n            models (List[~fairseq.models.FairseqModel]): ensemble of models\n            sample (dict): batch\n            prefix_tokens (torch.LongTensor, optional): force decoder to begin\n                with these tokens\n            constraints (torch.LongTensor, optional): force decoder to include\n                the list of constraints\n            bos_token (int, optional): beginning of sentence token\n                (default: self.eos)\n        '
        return self._generate(sample, **kwargs)

    def _generate(self, sample: Dict[str, Dict[str, Tensor]], prefix_tokens: Optional[Tensor]=None, constraints: Optional[Tensor]=None, bos_token: Optional[int]=None):
        if False:
            while True:
                i = 10
        net_input = sample['net_input']
        if 'src_tokens' in net_input:
            src_tokens = net_input['src_tokens']
            if 'src_lengths' in net_input:
                src_lengths = net_input['src_lengths']
            else:
                src_lengths = (src_tokens.ne(self.generator.eos) & src_tokens.ne(self.generator.pad)).long().sum(dim=1)
        else:
            raise Exception('expected src_tokens or source in net input. input keys: ' + str(net_input.keys()))
        if constraints is not None and (not self.generator.search.supports_constraints):
            raise NotImplementedError("Target-side constraints were provided, but search method doesn't support them")
        self.generator.search.init_constraints(constraints, self.generator.beam_size)
        self.generator_mt.search.init_constraints(constraints, self.generator_mt.beam_size)
        with torch.autograd.profiler.record_function('EnsembleModel: forward_encoder'):
            encoder_outs = self.generator.model.forward_encoder(net_input)
        single_model = self.generator.model.single_model
        mt_decoder = getattr(single_model, f'{single_model.mt_task_name}_decoder')
        finalized_mt = self.generator_mt.generate_decoder(encoder_outs, src_tokens, src_lengths, sample, prefix_tokens, constraints, bos_token, aux_task_name=single_model.mt_task_name)
        max_tgt_len = max([len(hypo[0]['tokens']) for hypo in finalized_mt])
        prev_output_tokens_mt = src_tokens.new_zeros(src_tokens.shape[0], max_tgt_len).fill_(mt_decoder.padding_idx).int()
        for (i, hypo) in enumerate(finalized_mt):
            i_beam = 0
            tmp = hypo[i_beam]['tokens'].int()
            prev_output_tokens_mt[i, 0] = self.generator_mt.eos
            if tmp[-1] == self.generator_mt.eos:
                tmp = tmp[:-1]
            prev_output_tokens_mt[i, 1:len(tmp) + 1] = tmp
            text = ''.join([self.generator_mt.tgt_dict[c] for c in tmp])
            text = text.replace('_', ' ')
            text = text.replace('▁', ' ')
            text = text.replace('<unk>', ' ')
            text = text.replace('<s>', '')
            text = text.replace('</s>', '')
            if len(text) > 0 and text[0] == ' ':
                text = text[1:]
            sample_id = sample['id'].tolist()[i]
            print('{} (None-{})'.format(text, sample_id))
        x = mt_decoder(prev_output_tokens_mt, encoder_out=encoder_outs[0], features_only=True)[0].transpose(0, 1)
        if getattr(single_model, 'proj', None) is not None:
            x = single_model.proj(x)
        mt_decoder_padding_mask = None
        if prev_output_tokens_mt.eq(mt_decoder.padding_idx).any():
            mt_decoder_padding_mask = prev_output_tokens_mt.eq(mt_decoder.padding_idx)
        if getattr(single_model, 'synthesizer_encoder', None) is not None:
            t2u_encoder_out = single_model.synthesizer_encoder(x, mt_decoder_padding_mask)
        else:
            t2u_encoder_out = {'encoder_out': [x], 'encoder_padding_mask': [mt_decoder_padding_mask] if mt_decoder_padding_mask is not None else [], 'encoder_embedding': [], 'encoder_states': [], 'src_tokens': [], 'src_lengths': []}
        if getattr(single_model, 't2u_augmented_cross_attn', False):
            encoder_outs_aug = [t2u_encoder_out]
        else:
            encoder_outs = [t2u_encoder_out]
            encoder_outs_aug = None
        finalized = self.generator.generate_decoder(encoder_outs, src_tokens, src_lengths, sample, prefix_tokens, constraints, bos_token, encoder_outs_aug=encoder_outs_aug)
        return finalized