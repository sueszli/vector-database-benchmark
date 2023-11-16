import json
import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import Dictionary, IdDataset, ListDataset, NestedDictionaryDataset, NumelDataset, NumSamplesDataset, PadDataset, SortDataset, data_utils, encoders
from fairseq.tasks import LegacyFairseqTask, register_task
from . import wsc_utils

@register_task('wsc')
class WSCTask(LegacyFairseqTask):
    """Task to finetune RoBERTa for Winograd Schemas."""

    @staticmethod
    def add_args(parser):
        if False:
            while True:
                i = 10
        'Add task-specific arguments to the parser.'
        parser.add_argument('data', metavar='DIR', help='path to data directory; we load <split>.jsonl')
        parser.add_argument('--init-token', type=int, default=None, help='add token at the beginning of each batch item')

    def __init__(self, args, vocab):
        if False:
            print('Hello World!')
        super().__init__(args)
        self.vocab = vocab
        self.mask = vocab.add_symbol('<mask>')
        self.bpe = encoders.build_bpe(args)
        self.tokenizer = encoders.build_tokenizer(args)
        if args.bpe == 'gpt2':
            self.leading_space = True
            self.trailing_space = False
        else:
            self.leading_space = False
            self.trailing_space = True

    @classmethod
    def load_dictionary(cls, filename):
        if False:
            i = 10
            return i + 15
        'Load the dictionary from the filename\n\n        Args:\n            filename (str): the filename\n        '
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        if False:
            while True:
                i = 10
        assert args.criterion == 'wsc', 'Must set --criterion=wsc'
        vocab = cls.load_dictionary(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(vocab)))
        return cls(args, vocab)

    def binarize(self, s: str, append_eos: bool=False):
        if False:
            while True:
                i = 10
        if self.tokenizer is not None:
            s = self.tokenizer.encode(s)
        if self.bpe is not None:
            s = self.bpe.encode(s)
        tokens = self.vocab.encode_line(s, append_eos=append_eos, add_if_not_exist=False).long()
        if self.args.init_token is not None:
            tokens = torch.cat([tokens.new([self.args.init_token]), tokens])
        return tokens

    def binarize_with_mask(self, txt, prefix, suffix, leading_space, trailing_space):
        if False:
            while True:
                i = 10
        toks = self.binarize(prefix + leading_space + txt + trailing_space + suffix, append_eos=True)
        mask = torch.zeros_like(toks, dtype=torch.bool)
        mask_start = len(self.binarize(prefix))
        mask_size = len(self.binarize(leading_space + txt))
        mask[mask_start:mask_start + mask_size] = 1
        return (toks, mask)

    def load_dataset(self, split, epoch=1, combine=False, data_path=None, return_only=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Load a given dataset split.\n\n        Args:\n            split (str): name of the split (e.g., train, valid, test)\n        '
        if data_path is None:
            data_path = os.path.join(self.args.data, split + '.jsonl')
        if not os.path.exists(data_path):
            raise FileNotFoundError('Cannot find data: {}'.format(data_path))
        query_tokens = []
        query_masks = []
        query_lengths = []
        candidate_tokens = []
        candidate_masks = []
        candidate_lengths = []
        labels = []
        for (sentence, pronoun_span, query, label) in wsc_utils.jsonl_iterator(data_path):
            prefix = sentence[:pronoun_span.start].text
            suffix = sentence[pronoun_span.end:].text_with_ws
            leading_space = ' ' if sentence[:pronoun_span.start].text_with_ws.endswith(' ') else ''
            trailing_space = ' ' if pronoun_span.text_with_ws.endswith(' ') else ''
            cand_spans = wsc_utils.filter_noun_chunks(wsc_utils.extended_noun_chunks(sentence), exclude_pronouns=True, exclude_query=query, exact_match=False)
            if query is not None:
                (query_toks, query_mask) = self.binarize_with_mask(query, prefix, suffix, leading_space, trailing_space)
                query_len = len(query_toks)
            else:
                (query_toks, query_mask, query_len) = (None, None, 0)
            query_tokens.append(query_toks)
            query_masks.append(query_mask)
            query_lengths.append(query_len)
            (cand_toks, cand_masks) = ([], [])
            for cand_span in cand_spans:
                (toks, mask) = self.binarize_with_mask(cand_span.text, prefix, suffix, leading_space, trailing_space)
                cand_toks.append(toks)
                cand_masks.append(mask)
            cand_toks = data_utils.collate_tokens(cand_toks, pad_idx=self.vocab.pad())
            cand_masks = data_utils.collate_tokens(cand_masks, pad_idx=0)
            assert cand_toks.size() == cand_masks.size()
            candidate_tokens.append(cand_toks)
            candidate_masks.append(cand_masks)
            candidate_lengths.append(cand_toks.size(1))
            labels.append(label)
        query_lengths = np.array(query_lengths)
        query_tokens = ListDataset(query_tokens, query_lengths)
        query_masks = ListDataset(query_masks, query_lengths)
        candidate_lengths = np.array(candidate_lengths)
        candidate_tokens = ListDataset(candidate_tokens, candidate_lengths)
        candidate_masks = ListDataset(candidate_masks, candidate_lengths)
        labels = ListDataset(labels, [1] * len(labels))
        dataset = {'id': IdDataset(), 'query_tokens': query_tokens, 'query_masks': query_masks, 'candidate_tokens': candidate_tokens, 'candidate_masks': candidate_masks, 'labels': labels, 'nsentences': NumSamplesDataset(), 'ntokens': NumelDataset(query_tokens, reduce=True)}
        nested_dataset = NestedDictionaryDataset(dataset, sizes=[query_lengths])
        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(query_tokens))
        dataset = SortDataset(nested_dataset, sort_order=[shuffle])
        if return_only:
            return dataset
        self.datasets[split] = dataset
        return self.datasets[split]

    def build_dataset_for_inference(self, sample_json):
        if False:
            return 10
        with tempfile.NamedTemporaryFile(buffering=0) as h:
            h.write((json.dumps(sample_json) + '\n').encode('utf-8'))
            dataset = self.load_dataset('disambiguate_pronoun', data_path=h.name, return_only=True)
        return dataset

    def disambiguate_pronoun(self, model, sentence, use_cuda=False):
        if False:
            return 10
        sample_json = wsc_utils.convert_sentence_to_json(sentence)
        dataset = self.build_dataset_for_inference(sample_json)
        sample = dataset.collater([dataset[0]])
        if use_cuda:
            sample = utils.move_to_cuda(sample)

        def get_masked_input(tokens, mask):
            if False:
                print('Hello World!')
            masked_tokens = tokens.clone()
            masked_tokens[mask.bool()] = self.mask
            return masked_tokens

        def get_lprobs(tokens, mask):
            if False:
                i = 10
                return i + 15
            (logits, _) = model(src_tokens=get_masked_input(tokens, mask))
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float)
            scores = lprobs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
            mask = mask.type_as(scores)
            scores = (scores * mask).sum(dim=-1) / mask.sum(dim=-1)
            return scores
        cand_lprobs = get_lprobs(sample['candidate_tokens'][0], sample['candidate_masks'][0])
        if sample['query_tokens'][0] is not None:
            query_lprobs = get_lprobs(sample['query_tokens'][0].unsqueeze(0), sample['query_masks'][0].unsqueeze(0))
            return (query_lprobs >= cand_lprobs).all().item() == 1
        else:
            best_idx = cand_lprobs.argmax().item()
            full_cand = sample['candidate_tokens'][0][best_idx]
            mask = sample['candidate_masks'][0][best_idx]
            toks = full_cand[mask.bool()]
            return self.bpe.decode(self.source_dictionary.string(toks)).strip()

    @property
    def source_dictionary(self):
        if False:
            for i in range(10):
                print('nop')
        return self.vocab

    @property
    def target_dictionary(self):
        if False:
            i = 10
            return i + 15
        return self.vocab

@register_task('winogrande')
class WinograndeTask(WSCTask):
    """
    Task for WinoGrande dataset. Efficient implementation for Winograd schema
    tasks with exactly two candidates, one of which is correct.
    """

    @classmethod
    def setup_task(cls, args, **kwargs):
        if False:
            print('Hello World!')
        assert args.criterion == 'winogrande', 'Must set --criterion=winogrande'
        vocab = cls.load_dictionary(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(vocab)))
        return cls(args, vocab)

    def load_dataset(self, split, epoch=1, combine=False, data_path=None, return_only=False, **kwargs):
        if False:
            print('Hello World!')
        'Load a given dataset split.\n\n        Args:\n            split (str): name of the split (e.g., train, valid, test)\n        '
        if data_path is None:
            data_path = os.path.join(self.args.data, split + '.jsonl')
        if not os.path.exists(data_path):
            raise FileNotFoundError('Cannot find data: {}'.format(data_path))
        query_tokens = []
        query_masks = []
        query_lengths = []
        candidate_tokens = []
        candidate_masks = []
        candidate_lengths = []
        itr = wsc_utils.winogrande_jsonl_iterator(data_path, eval=split == 'test')
        for sample in itr:
            (sentence, pronoun_span, query, cand_text) = sample
            prefix = sentence[:pronoun_span[0]].rstrip()
            suffix = sentence[pronoun_span[1]:]
            leading_space = ' ' if sentence[:pronoun_span[0]].endswith(' ') else ''
            trailing_space = ''
            if query is not None:
                (query_toks, query_mask) = self.binarize_with_mask(query, prefix, suffix, leading_space, trailing_space)
                query_len = len(query_toks)
            else:
                (query_toks, query_mask, query_len) = (None, None, 0)
            query_tokens.append(query_toks)
            query_masks.append(query_mask)
            query_lengths.append(query_len)
            (cand_toks, cand_mask) = self.binarize_with_mask(cand_text, prefix, suffix, leading_space, trailing_space)
            candidate_tokens.append(cand_toks)
            candidate_masks.append(cand_mask)
            candidate_lengths.append(cand_toks.size(0))
        query_lengths = np.array(query_lengths)

        def get_pad_dataset_fn(tokens, length, pad_idx):
            if False:
                return 10
            return PadDataset(ListDataset(tokens, length), pad_idx=pad_idx, left_pad=False)
        query_tokens = get_pad_dataset_fn(query_tokens, query_lengths, self.vocab.pad())
        query_masks = get_pad_dataset_fn(query_masks, query_lengths, 0)
        candidate_lengths = np.array(candidate_lengths)
        candidate_tokens = get_pad_dataset_fn(candidate_tokens, candidate_lengths, self.vocab.pad())
        candidate_masks = get_pad_dataset_fn(candidate_masks, candidate_lengths, 0)
        dataset = {'id': IdDataset(), 'query_tokens': query_tokens, 'query_masks': query_masks, 'candidate_tokens': candidate_tokens, 'candidate_masks': candidate_masks, 'nsentences': NumSamplesDataset(), 'ntokens': NumelDataset(query_tokens, reduce=True)}
        nested_dataset = NestedDictionaryDataset(dataset, sizes=[query_lengths])
        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(query_tokens))
        dataset = SortDataset(nested_dataset, sort_order=[shuffle])
        if return_only:
            return dataset
        self.datasets[split] = dataset
        return self.datasets[split]