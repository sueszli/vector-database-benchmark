import numpy as np
import torch
from . import Dictionary, FairseqDataset, data_utils

def collate(samples, pad_idx, eos_idx, vocab, left_pad_source=False, left_pad_target=False, input_feeding=True, pad_to_length=None):
    if False:
        while True:
            i = 10
    assert input_feeding
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        if False:
            return 10
        return data_utils.collate_tokens([s[key] for s in samples], pad_idx, eos_idx=None, left_pad=left_pad, move_eos_to_beginning=move_eos_to_beginning, pad_to_length=pad_to_length)
    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source, pad_to_length=pad_to_length['source'] if pad_to_length is not None else None)
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    (src_lengths, sort_order) = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target, pad_to_length=pad_to_length['target'] if pad_to_length is not None else None)
        target = target.index_select(0, sort_order)
        ntokens = sum((len(s['target']) for s in samples))
        if input_feeding:
            prev_output_tokens = merge('target', left_pad=left_pad_target, move_eos_to_beginning=True, pad_to_length=pad_to_length['target'] if pad_to_length is not None else None)
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum((len(s['source']) for s in samples))
    batch = {'id': id, 'ntokens': ntokens, 'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}, 'target': target, 'target_lengths': torch.LongTensor([len(t) for t in target]), 'nsentences': samples[0]['source'].size(0), 'sort_order': sort_order}
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch

class SpanMaskedTokensDataset(FairseqDataset):
    """
    A wrapper around TokenBlockDataset for T5 dataset.

    Args:
        dataset (~torch.utils.data.Dataset): dataset to wrap
        vocab (~fairseq.data.Dictionary): vocabulary
        noise_density (float): fraction of the tokens to select as noise.
        mean_noise_span_length (float): mean noise span length.
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
        seed: Seed for random number generator for reproducibility.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, vocab: Dictionary, noise_density: float, mean_noise_span_length: float, shuffle: bool, seed: int=1):
        if False:
            i = 10
            return i + 15
        self.dataset = dataset
        self.vocab = vocab
        self.seed = seed
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.shuffle = shuffle
        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        if False:
            print('Hello World!')
        return True

    def set_epoch(self, epoch, **unused):
        if False:
            i = 10
            return i + 15
        self.epoch = epoch

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            assert item[-1] == self.vocab.eos()
            noise_mask = self.random_spans_noise_mask(len(item))
            source_sentinel_ids = self.create_sentinel_ids(noise_mask.astype(np.int8))
            source = self.filter_input_ids(item, source_sentinel_ids)
            target_sentinel_ids = self.create_sentinel_ids((~noise_mask).astype(np.int8))
            target = self.filter_input_ids(item, target_sentinel_ids)
        return {'id': index, 'source': torch.from_numpy(source), 'target': torch.from_numpy(target)}

    def random_spans_noise_mask(self, length):
        if False:
            i = 10
            return i + 15
        '\n        This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .\n        Noise mask consisting of random spans of noise tokens.\n        The number of noise tokens and the number of noise spans and non-noise spans\n        are determined deterministically as follows:\n        num_noise_tokens = round(length * noise_density)\n        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)\n        Spans alternate between non-noise and noise, beginning with non-noise.\n        Subject to the above restrictions, all masks are equally likely.\n        Args:\n            length: an int32 scalar (length of the incoming token sequence)\n        Returns:\n            a boolean tensor with shape [length]\n        '
        orig_length = length
        num_noise_tokens = int(np.round(length * self.noise_density))
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        def _random_segmentation(num_items, num_segments):
            if False:
                print('Hello World!')
            '\n            Partition a sequence of items randomly into non-empty segments.\n            Args:\n                num_items: an integer scalar > 0\n                num_segments: an integer scalar in [1, num_items]\n            Returns:\n                a Tensor with shape [num_segments] containing positive integers that add up to num_items\n            '
            mask_indices = np.arange(num_items - 1) < num_segments - 1
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            (_, segment_length) = np.unique(segment_id, return_counts=True)
            return segment_length
        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = np.reshape(np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2])
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)
        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        if False:
            print('Hello World!')
        '\n        Sentinel ids creation given the indices that should be masked.\n        The start indices of each mask are replaced by the sentinel ids in increasing\n        order. Consecutive mask indices to be deleted are replaced with `-1`.\n        '
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, len(self.vocab) - sentinel_ids, 0)
        sentinel_ids -= mask_indices - start_indices
        return sentinel_ids

    @staticmethod
    def filter_input_ids(input_ids, sentinel_ids):
        if False:
            i = 10
            return i + 15
        '\n        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.\n        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.\n        '
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        return input_ids_full[input_ids_full >= 0]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.dataset)

    def collater(self, samples, pad_to_length=None):
        if False:
            print('Hello World!')
        '\n        Merge a list of samples to form a mini-batch.\n        Args:\n            samples (List[dict]): samples to collate\n        Returns:\n            dict: a mini-batch of data\n        '
        return collate(samples, self.vocab.pad(), self.vocab.eos(), self.vocab, pad_to_length=pad_to_length)

    def num_tokens(self, index):
        if False:
            print('Hello World!')
        'Return the number of tokens in a sample. This value is used to\n        enforce ``--max-tokens`` during batching.'
        return self.dataset.sizes[index]

    def size(self, index):
        if False:
            i = 10
            return i + 15
        "Return an example's size as a float or tuple. This value is used when\n        filtering a dataset with ``--max-positions``."
        return self.dataset.sizes[index]

    def ordered_indices(self):
        if False:
            return 10
        'Return an ordered list of indices. Batches will be constructed based\n        on this order.'
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.dataset.sizes[indices], kind='mergesort')]

    def prefetch(self, indices):
        if False:
            return 10
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    @property
    def supports_prefetch(self):
        if False:
            for i in range(10):
                print('nop')
        return hasattr(self.src, 'supports_prefetch') and self.src.supports_prefetch and hasattr(self.tgt, 'supports_prefetch') and self.tgt.supports_prefetch