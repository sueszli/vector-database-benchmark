import numpy as np
import torch
from . import FairseqDataset, data_utils

def collate(samples, pad_idx, eos_idx, fixed_pad_length=None, pad_to_bsz=None):
    if False:
        print('Hello World!')
    if len(samples) == 0:
        return {}

    def merge(key, is_list=False):
        if False:
            print('Hello World!')
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                res.append(data_utils.collate_tokens([s[key][i] for s in samples], pad_idx, eos_idx, left_pad=False, pad_to_length=fixed_pad_length, pad_to_bsz=pad_to_bsz))
            return res
        else:
            return data_utils.collate_tokens([s[key] for s in samples], pad_idx, eos_idx, left_pad=False, pad_to_length=fixed_pad_length, pad_to_bsz=pad_to_bsz)
    src_tokens = merge('source')
    if samples[0]['target'] is not None:
        is_target_list = isinstance(samples[0]['target'], list)
        target = merge('target', is_target_list)
    else:
        target = src_tokens
    return {'id': torch.LongTensor([s['id'] for s in samples]), 'nsentences': len(samples), 'ntokens': sum((len(s['source']) for s in samples)), 'net_input': {'src_tokens': src_tokens, 'src_lengths': torch.LongTensor([s['source'].numel() for s in samples])}, 'target': target}

class MonolingualDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching
            (default: True).
    """

    def __init__(self, dataset, sizes, src_vocab, tgt_vocab=None, add_eos_for_other_targets=False, shuffle=False, targets=None, add_bos_token=False, fixed_pad_length=None, pad_to_bsz=None, src_lang_idx=None, tgt_lang_idx=None):
        if False:
            i = 10
            return i + 15
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = src_vocab
        self.tgt_vocab = tgt_vocab or src_vocab
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle
        self.add_bos_token = add_bos_token
        self.fixed_pad_length = fixed_pad_length
        self.pad_to_bsz = pad_to_bsz
        self.src_lang_idx = src_lang_idx
        self.tgt_lang_idx = tgt_lang_idx
        assert targets is None or all((t in {'self', 'future', 'past'} for t in targets)), "targets must be none or one of 'self', 'future', 'past'"
        if targets is not None and len(targets) == 0:
            targets = None
        self.targets = targets

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        if self.targets is not None:
            (source, future_target, past_target) = self.dataset[index]
            (source, target) = self._make_source_target(source, future_target, past_target)
        else:
            source = self.dataset[index]
            target = None
        (source, target) = self._maybe_add_bos(source, target)
        return {'id': index, 'source': source, 'target': target}

    def __len__(self):
        if False:
            return 10
        return len(self.dataset)

    def _make_source_target(self, source, future_target, past_target):
        if False:
            i = 10
            return i + 15
        if self.targets is not None:
            target = []
            if self.add_eos_for_other_targets and ('self' in self.targets or 'past' in self.targets) and (source[-1] != self.vocab.eos()):
                source = torch.cat([source, source.new([self.vocab.eos()])])
                if 'future' in self.targets:
                    future_target = torch.cat([future_target, future_target.new([self.vocab.pad()])])
                if 'past' in self.targets:
                    past_target = torch.cat([past_target.new([self.vocab.pad()]), past_target[1:], source[-2, None]])
            for t in self.targets:
                if t == 'self':
                    target.append(source)
                elif t == 'future':
                    target.append(future_target)
                elif t == 'past':
                    target.append(past_target)
                else:
                    raise Exception('invalid target ' + t)
            if len(target) == 1:
                target = target[0]
        else:
            target = future_target
        return (source, self._filter_vocab(target))

    def _maybe_add_bos(self, source, target):
        if False:
            print('Hello World!')
        if self.add_bos_token:
            source = torch.cat([source.new([self.vocab.bos()]), source])
            if target is not None:
                target = torch.cat([target.new([self.tgt_vocab.bos()]), target])
        return (source, target)

    def num_tokens_vec(self, indices):
        if False:
            i = 10
            return i + 15
        'Return the number of tokens for a set of positions defined by indices.\n        This value is used to enforce ``--max-tokens`` during batching.'
        return self.sizes[indices]

    def _filter_vocab(self, target):
        if False:
            return 10
        if len(self.tgt_vocab) != len(self.vocab):

            def _filter(target):
                if False:
                    print('Hello World!')
                mask = target.ge(len(self.tgt_vocab))
                if mask.any():
                    target[mask] = self.tgt_vocab.unk()
                return target
            if isinstance(target, list):
                return [_filter(t) for t in target]
            return _filter(target)
        return target

    def collater(self, samples):
        if False:
            i = 10
            return i + 15
        'Merge a list of samples to form a mini-batch.\n\n        Args:\n            samples (List[dict]): samples to collate\n\n        Returns:\n            dict: a mini-batch with the following keys:\n\n                - `id` (LongTensor): example IDs in the original input order\n                - `ntokens` (int): total number of tokens in the batch\n                - `net_input` (dict): the input to the Model, containing keys:\n\n                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in\n                    the source sentence of shape `(bsz, src_len)`. Padding will\n                    appear on the right.\n\n                - `target` (LongTensor): a padded 2D Tensor of tokens in the\n                  target sentence of shape `(bsz, tgt_len)`. Padding will appear\n                  on the right.\n        '
        return collate(samples, self.vocab.pad(), self.vocab.eos(), self.fixed_pad_length, self.pad_to_bsz)

    def num_tokens(self, index):
        if False:
            return 10
        'Return the number of tokens in a sample. This value is used to\n        enforce ``--max-tokens`` during batching.'
        return self.sizes[index]

    def size(self, index):
        if False:
            return 10
        "Return an example's size as a float or tuple. This value is used when\n        filtering a dataset with ``--max-positions``."
        return self.sizes[index]

    def ordered_indices(self):
        if False:
            i = 10
            return i + 15
        'Return an ordered list of indices. Batches will be constructed based\n        on this order.'
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    @property
    def supports_prefetch(self):
        if False:
            print('Hello World!')
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        if False:
            print('Hello World!')
        self.dataset.prefetch(indices)