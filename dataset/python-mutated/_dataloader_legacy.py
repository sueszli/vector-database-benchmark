from __future__ import annotations
from typing import Any
from pytorch_lightning.trainer.supporters import CombinedLoader, CombinedLoaderIterator
__all__ = ['ConcatLoader']

class ConcatLoader(CombinedLoader):
    """This loader is same as CombinedLoader in PyTorch-Lightning, but concatenate sub-loaders
    instead of loading them in parallel.

    Parameters
    ----------
    loaders
        For example, ::

            {
                "train": DataLoader(train_dataset),
                "val": DataLoader(val_dataset)
            }

        In this example, the loader will first produce the batches from "train", then "val".

    mode
        Only support "min_size" for now.
    """

    def __init__(self, loaders: dict[str, Any], mode: str='min_size'):
        if False:
            i = 10
            return i + 15
        if mode != 'min_size':
            raise ValueError('Only min_size mode is supported now.')
        super().__init__(loaders, mode)

    def __iter__(self) -> Any:
        if False:
            while True:
                i = 10
        'Replace the super-class iterator with ours.'
        self._try_to_patch_pytorch_dataloader()
        iterator = ConcatLoaderIterator(self.loaders)
        self.on_restart(iterator)
        self._iterator = iterator
        return iterator

    @staticmethod
    def _try_to_patch_pytorch_dataloader():
        if False:
            print('Hello World!')
        'Copied from CombinedLoader.'
        from torch.utils.data.dataloader import _BaseDataLoaderIter

        def __getstate__patch__(*_):
            if False:
                while True:
                    i = 10
            return {}
        _BaseDataLoaderIter.__getstate__ = __getstate__patch__

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return int(sum((self._calc_num_batches(loader) for loader in self.loaders.values())))

class ConcatLoaderIterator(CombinedLoaderIterator):
    """Similar to CombinedLoaderIterator in Lightning, but in a concat manner."""

    def __next__(self) -> Any:
        if False:
            i = 10
            return i + 15
        "Fetches the next batch from multiple data loaders,\n        by looking for the first iterator that isn't exhausted yet.\n        "
        if not len(self.loader_iters) == len(self.loaders):
            raise RuntimeError('loader_iters must have the same length as loaders.')
        for (i, (loader_name, iterator)) in enumerate(self.loader_iters.items()):
            try:
                return (self.request_next_batch(iterator), loader_name)
            except StopIteration:
                if i + 1 == len(self.loader_iters):
                    raise