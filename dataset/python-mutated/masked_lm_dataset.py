import math
from typing import Dict, List, Tuple
import numpy as np
import torch
from fairseq.data import Dictionary, FairseqDataset, data_utils
from fairseq.data.concat_dataset import ConcatDataset
from fairseq.data.legacy.block_pair_dataset import BlockPairDataset
from fairseq.data.token_block_dataset import TokenBlockDataset

class MaskedLMDataset(FairseqDataset):
    """
    A wrapper Dataset for masked language modelling. The dataset
    wraps around TokenBlockDataset or BlockedPairDataset and creates a batch
    where the input blocks are masked according to the specified masking
    probability. Additionally the batch can also contain sentence level targets
    if this is specified.

    Args:
        dataset: Dataset which generates blocks of data. Only BlockPairDataset
            and TokenBlockDataset are supported.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of padding token in dictionary
        mask_idx: Id of mask token in dictionary
        classif_token_idx: Id of classification token in dictionary. This is the
            token associated with the sentence embedding (Eg: CLS for BERT)
        sep_token_idx: Id of separator token in dictionary
            (Eg: SEP in BERT)
        seed: Seed for random number generator for reproducibility.
        shuffle: Shuffle the elements before batching.
        has_pairs: Specifies whether the underlying dataset
            generates a pair of blocks along with a sentence_target or not.
            Setting it to True assumes that the underlying dataset generates a
            label for the pair of sentences which is surfaced as
            sentence_target. The default value assumes a single block with no
            sentence target.
        segment_id: An optional segment id for filling in the segment labels
            when we are in the single block setting (Eg: XLM). Default is 0.
        masking_ratio: specifies what percentage of the blocks should be masked.
        masking_prob: specifies the probability of a given token being
            replaced with the "MASK" token.
        random_token_prob: specifies the probability of a given token being
            replaced by a random token from the vocabulary.
    """

    def __init__(self, dataset: FairseqDataset, sizes: np.ndarray, vocab: Dictionary, pad_idx: int, mask_idx: int, classif_token_idx: int, sep_token_idx: int, seed: int=1, shuffle: bool=True, has_pairs: bool=True, segment_id: int=0, masking_ratio: float=0.15, masking_prob: float=0.8, random_token_prob: float=0.1):
        if False:
            print('Hello World!')
        assert isinstance(dataset, TokenBlockDataset) or isinstance(dataset, BlockPairDataset) or isinstance(dataset, ConcatDataset), 'MaskedLMDataset only wraps TokenBlockDataset or BlockPairDataset or ConcatDataset'
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.classif_token_idx = classif_token_idx
        self.sep_token_idx = sep_token_idx
        self.shuffle = shuffle
        self.seed = seed
        self.has_pairs = has_pairs
        self.segment_id = segment_id
        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.random_token_prob = random_token_prob
        if not has_pairs:
            self.sizes = self.sizes + 1

    def __getitem__(self, index: int):
        if False:
            while True:
                i = 10
        if self.has_pairs:
            (block_one, block_two, sentence_target) = self.dataset[index]
        else:
            block_one = self.dataset[index]
        return {'id': index, 'block_one': block_one, 'block_two': block_two if self.has_pairs else None, 'sentence_target': sentence_target if self.has_pairs else None}

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.dataset)

    def _mask_block(self, sentence: np.ndarray, mask_idx: int, pad_idx: int, dictionary_token_range: Tuple):
        if False:
            for i in range(10):
                print('nop')
        "\n        Mask tokens for Masked Language Model training\n        Samples mask_ratio tokens that will be predicted by LM.\n\n        Note:This function may not be efficient enough since we had multiple\n        conversions between np and torch, we can replace them with torch\n        operators later.\n\n        Args:\n            sentence: 1d tensor to be masked\n            mask_idx: index to use for masking the sentence\n            pad_idx: index to use for masking the target for tokens we aren't\n                predicting\n            dictionary_token_range: range of indices in dictionary which can\n                be used for random word replacement\n                (e.g. without special characters)\n        Return:\n            masked_sent: masked sentence\n            target: target with words which we are not predicting replaced\n                by pad_idx\n        "
        masked_sent = np.copy(sentence)
        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * self.masking_ratio)
        mask = np.random.choice(sent_length, mask_num, replace=False)
        target = np.copy(sentence)
        for i in range(sent_length):
            if i in mask:
                rand = np.random.random()
                if rand < self.masking_prob:
                    masked_sent[i] = mask_idx
                elif rand < self.masking_prob + self.random_token_prob:
                    masked_sent[i] = np.random.randint(dictionary_token_range[0], dictionary_token_range[1])
            else:
                target[i] = pad_idx
        return (masked_sent, target)

    def _collate(self, samples: List[Dict], pad_idx: int, eos_idx: int):
        if False:
            print('Hello World!')
        '\n        Does the heavy lifting for creating a batch from the input list of\n        examples. The logic is as follows:\n            1. Mask the input blocks. In case has_pair is True then we have 2\n               blocks to mask.\n            2. Prepend the first masked block tensor with the special token\n               used as sentence embedding. Eg: CLS in BERT. This happens\n               irrespective of the value of has_pair.\n            3. If has_pair is True, then append the first masked block with the\n               special separator token (eg: SEP for BERT) and compute segment\n               label accordingly. In this case, also append the second masked\n               block with this special separator token and compute its segment\n               label.\n            4. For the targets tensor, prepend and append with padding index\n               accordingly.\n            5. Concatenate all tensors.\n        '
        if len(samples) == 0:
            return {}
        with data_utils.numpy_seed(self.seed + samples[0]['id']):
            for s in samples:
                token_range = (self.vocab.nspecial, len(self.vocab))
                (masked_blk_one, masked_tgt_one) = self._mask_block(s['block_one'], self.mask_idx, self.pad_idx, token_range)
                tokens = np.concatenate([[self.classif_token_idx], masked_blk_one])
                targets = np.concatenate([[self.pad_idx], masked_tgt_one])
                segments = np.ones(len(tokens)) * self.segment_id
                if self.has_pairs:
                    tokens_one = np.concatenate([tokens, [self.sep_token_idx]])
                    targets_one = np.concatenate([targets, [self.pad_idx]])
                    (masked_blk_two, masked_tgt_two) = self._mask_block(s['block_two'], self.mask_idx, self.pad_idx, token_range)
                    tokens_two = np.concatenate([masked_blk_two, [self.sep_token_idx]])
                    targets_two = np.concatenate([masked_tgt_two, [self.pad_idx]])
                    segments_one = np.zeros(len(tokens_one))
                    segments_two = np.ones(len(tokens_two))
                    tokens = np.concatenate([tokens_one, tokens_two])
                    targets = np.concatenate([targets_one, targets_two])
                    segments = np.concatenate([segments_one, segments_two])
                s['source'] = torch.LongTensor(tokens)
                s['segment_labels'] = torch.LongTensor(segments)
                s['lm_target'] = torch.LongTensor(targets)

        def merge(key):
            if False:
                print('Hello World!')
            return data_utils.collate_tokens([s[key] for s in samples], pad_idx, eos_idx, left_pad=False)
        return {'id': torch.LongTensor([s['id'] for s in samples]), 'ntokens': sum((len(s['source']) for s in samples)), 'net_input': {'src_tokens': merge('source'), 'segment_labels': merge('segment_labels')}, 'lm_target': merge('lm_target'), 'sentence_target': torch.LongTensor([s['sentence_target'] for s in samples]) if self.has_pairs else None, 'nsentences': len(samples)}

    def collater(self, samples: List[Dict]):
        if False:
            return 10
        'Merge a list of samples to form a mini-batch.\n\n        Args:\n            samples (List[dict]): samples to collate\n\n        Returns:\n            dict: a mini-batch of data\n        '
        return self._collate(samples, self.vocab.pad(), self.vocab.eos())

    def num_tokens(self, index: int):
        if False:
            while True:
                i = 10
        '\n        Return the number of tokens in a sample. This value is used to\n        enforce max-tokens during batching.\n        '
        return self.sizes[index]

    def size(self, index: int):
        if False:
            return 10
        "\n        Return an example's size as a float or tuple. This value is used when\n        filtering a dataset with max-positions.\n        "
        return self.sizes[index]

    def ordered_indices(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an ordered list of indices. Batches will be constructed based\n        on this order.\n        '
        if self.shuffle:
            return np.random.permutation(len(self))
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
            i = 10
            return i + 15
        self.dataset.prefetch(indices)