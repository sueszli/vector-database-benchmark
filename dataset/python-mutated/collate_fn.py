import collections.abc
from torch.utils.data.dataloader import default_collate

class FilteringCollateFn:
    """
    Callable object doing job of ``collate_fn`` like ``default_collate``,
    but does not cast batch items with specified key to `torch.Tensor`.

    Only adds them to list.
    Supports only key-value format batches
    """

    def __init__(self, *keys):
        if False:
            return 10
        '\n        Args:\n            keys: Keys for values that will not be converted to tensor and stacked\n        '
        self.keys = keys

    def __call__(self, batch):
        if False:
            while True:
                i = 10
        '\n        Args:\n            batch: current batch\n\n        Returns:\n            batch values filtered by `keys`\n        '
        if isinstance(batch[0], collections.abc.Mapping):
            result = {}
            for key in batch[0]:
                items = [d[key] for d in batch]
                if key not in self.keys:
                    items = default_collate(items)
                result[key] = items
            return result
        else:
            return default_collate(batch)
__all__ = ['FilteringCollateFn']