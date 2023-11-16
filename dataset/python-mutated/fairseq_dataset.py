import logging
import numpy as np
import torch.utils.data
from fairseq.data import data_utils
logger = logging.getLogger(__name__)

class EpochListening:
    """Mixin for receiving updates whenever the epoch increments."""

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        if False:
            print('Hello World!')
        '\n        Whether we can reuse the :class:`fairseq.data.EpochBatchIterator` for\n        this dataset across epochs.\n\n        This needs to return ``False`` if the sample sizes can change across\n        epochs, in which case we may need to regenerate batches at each epoch.\n        If your dataset relies in ``set_epoch`` then you should consider setting\n        this to ``False``.\n        '
        return True

    def set_epoch(self, epoch):
        if False:
            i = 10
            return i + 15
        'Will receive the updated epoch number at the beginning of the epoch.'
        pass

class FairseqDataset(torch.utils.data.Dataset, EpochListening):
    """A dataset that provides helpers for batching."""

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def __len__(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def collater(self, samples):
        if False:
            return 10
        'Merge a list of samples to form a mini-batch.\n\n        Args:\n            samples (List[dict]): samples to collate\n\n        Returns:\n            dict: a mini-batch suitable for forwarding with a Model\n        '
        raise NotImplementedError

    def num_tokens(self, index):
        if False:
            return 10
        'Return the number of tokens in a sample. This value is used to\n        enforce ``--max-tokens`` during batching.'
        raise NotImplementedError

    def num_tokens_vec(self, indices):
        if False:
            i = 10
            return i + 15
        'Return the number of tokens for a set of positions defined by indices.\n        This value is used to enforce ``--max-tokens`` during batching.'
        raise NotImplementedError

    def size(self, index):
        if False:
            return 10
        "Return an example's size as a float or tuple. This value is used when\n        filtering a dataset with ``--max-positions``."
        raise NotImplementedError

    def ordered_indices(self):
        if False:
            return 10
        'Return an ordered list of indices. Batches will be constructed based\n        on this order.'
        return np.arange(len(self), dtype=np.int64)

    @property
    def supports_prefetch(self):
        if False:
            return 10
        'Whether this dataset supports prefetching.'
        return False

    def attr(self, attr: str, index: int):
        if False:
            print('Hello World!')
        return getattr(self, attr, None)

    def prefetch(self, indices):
        if False:
            for i in range(10):
                print('nop')
        'Prefetch the data required for this epoch.'
        raise NotImplementedError

    def get_batch_shapes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a list of valid batch shapes, for example::\n\n            [(8, 512), (16, 256), (32, 128)]\n\n        The first dimension of each tuple is the batch size and can be ``None``\n        to automatically infer the max batch size based on ``--max-tokens``.\n        The second dimension of each tuple is the max supported length as given\n        by :func:`fairseq.data.FairseqDataset.num_tokens`.\n\n        This will be used by :func:`fairseq.data.FairseqDataset.batch_by_size`\n        to restrict batch shapes. This is useful on TPUs to avoid too many\n        dynamic shapes (and recompilations).\n        '
        return None

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        if False:
            return 10
        '\n        Given an ordered set of indices, return batches according to\n        *max_tokens*, *max_sentences* and *required_batch_size_multiple*.\n        '
        from fairseq.data import data_utils
        fixed_shapes = self.get_batch_shapes()
        if fixed_shapes is not None:

            def adjust_bsz(bsz, num_tokens):
                if False:
                    return 10
                if bsz is None:
                    assert max_tokens is not None, 'Must specify --max-tokens'
                    bsz = max_tokens // num_tokens
                if max_sentences is not None:
                    bsz = min(bsz, max_sentences)
                elif bsz >= required_batch_size_multiple and bsz % required_batch_size_multiple != 0:
                    bsz -= bsz % required_batch_size_multiple
                return bsz
            fixed_shapes = np.array([[adjust_bsz(bsz, num_tokens), num_tokens] for (bsz, num_tokens) in fixed_shapes])
        try:
            num_tokens_vec = self.num_tokens_vec(indices).astype('int64')
        except NotImplementedError:
            num_tokens_vec = None
        return data_utils.batch_by_size(indices, num_tokens_fn=self.num_tokens, num_tokens_vec=num_tokens_vec, max_tokens=max_tokens, max_sentences=max_sentences, required_batch_size_multiple=required_batch_size_multiple, fixed_shapes=fixed_shapes)

    def filter_indices_by_size(self, indices, max_sizes):
        if False:
            return 10
        "\n        Filter a list of sample indices. Remove those that are longer than\n        specified in *max_sizes*.\n\n        WARNING: don't update, override method in child classes\n\n        Args:\n            indices (np.array): original array of sample indices\n            max_sizes (int or list[int] or tuple[int]): max sample size,\n                can be defined separately for src and tgt (then list or tuple)\n\n        Returns:\n            np.array: filtered sample array\n            list: list of removed indices\n        "
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, 'sizes') and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif hasattr(self, 'sizes') and isinstance(self.sizes, list) and (len(self.sizes) == 1):
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                (indices, ignored) = data_utils._filter_by_size_dynamic(indices, self.size, max_sizes)
        else:
            (indices, ignored) = data_utils._filter_by_size_dynamic(indices, self.size, max_sizes)
        return (indices, ignored)

    @property
    def supports_fetch_outside_dataloader(self):
        if False:
            i = 10
            return i + 15
        'Whether this dataset supports fetching outside the workers of the dataloader.'
        return True

class FairseqIterableDataset(torch.utils.data.IterableDataset, EpochListening):
    """
    For datasets that need to be read sequentially, usually because the data is
    being streamed or otherwise can't be manipulated on a single machine.
    """

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError