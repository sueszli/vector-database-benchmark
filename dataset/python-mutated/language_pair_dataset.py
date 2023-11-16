import logging
import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
logger = logging.getLogger(__name__)

def collate(samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False, input_feeding=True, pad_to_length=None, pad_to_multiple=1):
    if False:
        for i in range(10):
            print('nop')
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        if False:
            for i in range(10):
                print('nop')
        return data_utils.collate_tokens([s[key] for s in samples], pad_idx, eos_idx, left_pad, move_eos_to_beginning, pad_to_length=pad_to_length, pad_to_multiple=pad_to_multiple)

    def check_alignment(alignment, src_len, tgt_len):
        if False:
            return 10
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning('alignment size mismatch found, skipping alignment!')
            return False
        return True

    def compute_alignment_weights(alignments):
        if False:
            print('Hello World!')
        '\n        Given a tensor of shape [:, 2] containing the source-target indices\n        corresponding to the alignments, a weight vector containing the\n        inverse frequency of each target index is computed.\n        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then\n        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target\n        index 3 is repeated twice)\n        '
        align_tgt = alignments[:, 1]
        (_, align_tgt_i, align_tgt_c) = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()
    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source, pad_to_length=pad_to_length['source'] if pad_to_length is not None else None)
    src_lengths = torch.LongTensor([s['source'].ne(pad_idx).long().sum() for s in samples])
    (src_lengths, sort_order) = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target, pad_to_length=pad_to_length['target'] if pad_to_length is not None else None)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].ne(pad_idx).long().sum() for s in samples]).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()
        if samples[0].get('prev_output_tokens', None) is not None:
            prev_output_tokens = merge('prev_output_tokens', left_pad=left_pad_target)
        elif input_feeding:
            prev_output_tokens = merge('target', left_pad=left_pad_target, move_eos_to_beginning=True, pad_to_length=pad_to_length['target'] if pad_to_length is not None else None)
    else:
        ntokens = src_lengths.sum().item()
    batch = {'id': id, 'nsentences': len(samples), 'ntokens': ntokens, 'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}, 'target': target}
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens.index_select(0, sort_order)
    if samples[0].get('alignment', None) is not None:
        (bsz, tgt_sz) = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]
        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths
        alignments = [alignment + offset for (align_idx, offset, src_len, tgt_len) in zip(sort_order, offsets, src_lengths, tgt_lengths) for alignment in [samples[align_idx]['alignment'].view(-1, 2)] if check_alignment(alignment, src_len, tgt_len)]
        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)
            batch['alignments'] = alignments
            batch['align_weights'] = align_weights
    if samples[0].get('constraints', None) is not None:
        lens = [sample.get('constraints').size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for (i, sample) in enumerate(samples):
            constraints[i, 0:lens[i]] = samples[i].get('constraints')
        batch['constraints'] = constraints.index_select(0, sort_order)
    return batch

class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(self, src, src_sizes, src_dict, tgt=None, tgt_sizes=None, tgt_dict=None, left_pad_source=True, left_pad_target=False, shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False, align_dataset=None, constraints=None, append_bos=False, eos=None, num_buckets=0, src_lang_id=None, tgt_lang_id=None, pad_to_multiple=1):
        if False:
            return 10
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(tgt), 'Source and target must contain the same number of examples'
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = np.vstack((self.src_sizes, self.tgt_sizes)).T if self.tgt_sizes is not None else self.src_sizes
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, 'Both source and target needed when alignments are provided'
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset
            self.src = BucketPadLengthDataset(self.src, sizes=self.src_sizes, num_buckets=num_buckets, pad_idx=self.src_dict.pad(), left_pad=self.left_pad_source)
            self.src_sizes = self.src.sizes
            logger.info('bucketing source lengths: {}'.format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(self.tgt, sizes=self.tgt_sizes, num_buckets=num_buckets, pad_idx=self.tgt_dict.pad(), left_pad=self.left_pad_target)
                self.tgt_sizes = self.tgt.sizes
                logger.info('bucketing target lengths: {}'.format(list(self.tgt.buckets)))
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.compat.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [(None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        if False:
            return 10
        return self.buckets

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])
        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])
            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])
        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
        example = {'id': index, 'source': src_item, 'target': tgt_item}
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        if self.constraints is not None:
            example['constraints'] = self.constraints[index]
        return example

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        if False:
            return 10
        "Merge a list of samples to form a mini-batch.\n\n        Args:\n            samples (List[dict]): samples to collate\n            pad_to_length (dict, optional): a dictionary of\n                {'source': source_pad_to_length, 'target': target_pad_to_length}\n                to indicate the max length to pad to in source and target respectively.\n\n        Returns:\n            dict: a mini-batch with the following keys:\n\n                - `id` (LongTensor): example IDs in the original input order\n                - `ntokens` (int): total number of tokens in the batch\n                - `net_input` (dict): the input to the Model, containing keys:\n\n                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in\n                    the source sentence of shape `(bsz, src_len)`. Padding will\n                    appear on the left if *left_pad_source* is ``True``.\n                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded\n                    lengths of each source sentence of shape `(bsz)`\n                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of\n                    tokens in the target sentence, shifted right by one\n                    position for teacher forcing, of shape `(bsz, tgt_len)`.\n                    This key will not be present if *input_feeding* is\n                    ``False``.  Padding will appear on the left if\n                    *left_pad_target* is ``True``.\n                  - `src_lang_id` (LongTensor): a long Tensor which contains source\n                    language IDs of each sample in the batch\n\n                - `target` (LongTensor): a padded 2D Tensor of tokens in the\n                  target sentence of shape `(bsz, tgt_len)`. Padding will appear\n                  on the left if *left_pad_target* is ``True``.\n                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language\n                   IDs of each sample in the batch\n        "
        res = collate(samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos, left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target, input_feeding=self.input_feeding, pad_to_length=pad_to_length, pad_to_multiple=self.pad_to_multiple)
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res['net_input']['src_tokens']
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res['net_input']['src_lang_id'] = torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
            if self.tgt_lang_id is not None:
                res['tgt_lang_id'] = torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
        return res

    def num_tokens(self, index):
        if False:
            return 10
        'Return the number of tokens in a sample. This value is used to\n        enforce ``--max-tokens`` during batching.'
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def num_tokens_vec(self, indices):
        if False:
            print('Hello World!')
        'Return the number of tokens for a set of positions defined by indices.\n        This value is used to enforce ``--max-tokens`` during batching.'
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        if False:
            while True:
                i = 10
        "Return an example's size as a float or tuple. This value is used when\n        filtering a dataset with ``--max-positions``."
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        if False:
            return 10
        'Return an ordered list of indices. Batches will be constructed based\n        on this order.'
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
            return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        else:
            return indices[np.argsort(self.bucketed_num_tokens[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        if False:
            while True:
                i = 10
        return getattr(self.src, 'supports_prefetch', False) and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)

    def prefetch(self, indices):
        if False:
            return 10
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        if False:
            while True:
                i = 10
        'Filter a list of sample indices. Remove those that are longer\n            than specified in max_sizes.\n\n        Args:\n            indices (np.array): original array of sample indices\n            max_sizes (int or list[int] or tuple[int]): max sample size,\n                can be defined separately for src and tgt (then list or tuple)\n\n        Returns:\n            np.array: filtered sample array\n            list: list of removed indices\n        '
        return data_utils.filter_paired_dataset_indices_by_size(self.src_sizes, self.tgt_sizes, indices, max_sizes)