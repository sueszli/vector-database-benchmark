import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from ..utils.generic import ModelOutput

class PipelineDataset(Dataset):

    def __init__(self, dataset, process, params):
        if False:
            print('Hello World!')
        self.dataset = dataset
        self.process = process
        self.params = params

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.dataset)

    def __getitem__(self, i):
        if False:
            return 10
        item = self.dataset[i]
        processed = self.process(item, **self.params)
        return processed

class PipelineIterator(IterableDataset):

    def __init__(self, loader, infer, params, loader_batch_size=None):
        if False:
            while True:
                i = 10
        '\n        Roughly equivalent to\n\n        ```\n        for item in loader:\n            yield infer(item, **params)\n        ```\n\n                Arguments:\n                    loader (`torch.utils.data.DataLoader` or any iterator):\n                        The iterator that will be used to apply `infer` on.\n                    infer (any function):\n                        The function to apply of each element of `loader`.\n                    params (`dict`):\n                        The parameters passed to `infer` along with every item\n                    loader_batch_size (`int`, *optional*):\n                        If specified, the items of `loader` are supposed to come as batch, and are loader_batched here\n                        making it roughly behave as\n\n\n        ```\n        for items in loader:\n            for i in loader_batch_size:\n                item = items[i]\n                yield infer(item, **params)\n        ```'
        self.loader = loader
        self.infer = infer
        self.params = params
        if loader_batch_size == 1:
            loader_batch_size = None
        self.loader_batch_size = loader_batch_size
        self._loader_batch_index = None
        self._loader_batch_data = None

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.loader)

    def __iter__(self):
        if False:
            print('Hello World!')
        self.iterator = iter(self.loader)
        return self

    def loader_batch_item(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return item located at `loader_batch_index` within the current `loader_batch_data`.\n        '
        if isinstance(self._loader_batch_data, torch.Tensor):
            result = self._loader_batch_data[self._loader_batch_index]
        else:
            loader_batched = {}
            for (k, element) in self._loader_batch_data.items():
                if isinstance(element, ModelOutput):
                    element = element.to_tuple()
                    if isinstance(element[0], torch.Tensor):
                        loader_batched[k] = tuple((el[self._loader_batch_index].unsqueeze(0) for el in element))
                    elif isinstance(element[0], np.ndarray):
                        loader_batched[k] = tuple((np.expand_dims(el[self._loader_batch_index], 0) for el in element))
                    continue
                if k in {'hidden_states', 'past_key_values', 'attentions'} and isinstance(element, tuple):
                    if isinstance(element[0], torch.Tensor):
                        loader_batched[k] = tuple((el[self._loader_batch_index].unsqueeze(0) for el in element))
                    elif isinstance(element[0], np.ndarray):
                        loader_batched[k] = tuple((np.expand_dims(el[self._loader_batch_index], 0) for el in element))
                    continue
                if element is None:
                    loader_batched[k] = None
                elif isinstance(element[self._loader_batch_index], torch.Tensor):
                    loader_batched[k] = element[self._loader_batch_index].unsqueeze(0)
                elif isinstance(element[self._loader_batch_index], np.ndarray):
                    loader_batched[k] = np.expand_dims(element[self._loader_batch_index], 0)
                else:
                    loader_batched[k] = element[self._loader_batch_index]
            result = self._loader_batch_data.__class__(loader_batched)
        self._loader_batch_index += 1
        return result

    def __next__(self):
        if False:
            while True:
                i = 10
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            return self.loader_batch_item()
        item = next(self.iterator)
        processed = self.infer(item, **self.params)
        if self.loader_batch_size is not None:
            if isinstance(processed, torch.Tensor):
                first_tensor = processed
            else:
                key = list(processed.keys())[0]
                first_tensor = processed[key]
            if isinstance(first_tensor, list):
                observed_batch_size = len(first_tensor)
            else:
                observed_batch_size = first_tensor.shape[0]
            if 0 < observed_batch_size < self.loader_batch_size:
                self.loader_batch_size = observed_batch_size
            self._loader_batch_data = processed
            self._loader_batch_index = 0
            return self.loader_batch_item()
        else:
            return processed

class PipelineChunkIterator(PipelineIterator):

    def __init__(self, loader, infer, params, loader_batch_size=None):
        if False:
            i = 10
            return i + 15
        '\n        Roughly equivalent to\n\n        ```\n        for iterator in loader:\n            for item in iterator:\n                yield infer(item, **params)\n        ```\n\n                Arguments:\n                    loader (`torch.utils.data.DataLoader` or any iterator):\n                        The iterator that will be used to apply `infer` on.\n                    infer (any function):\n                        The function to apply of each element of `loader`.\n                    params (`dict`):\n                        The parameters passed to `infer` along with every item\n        '
        super().__init__(loader, infer, params)

    def __iter__(self):
        if False:
            return 10
        self.iterator = iter(self.loader)
        self.subiterator = None
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        if self.subiterator is None:
            "Subiterator None means we haven't started a `preprocess` iterator. so start it"
            self.subiterator = self.infer(next(self.iterator), **self.params)
        try:
            processed = next(self.subiterator)
        except StopIteration:
            self.subiterator = self.infer(next(self.iterator), **self.params)
            processed = next(self.subiterator)
        return processed

class PipelinePackIterator(PipelineIterator):
    """
    Roughly equivalent to

    ```
    packed =  []
    for item in loader:
        packed.append(item)
        if item["is_last"]:
            yield packed
            packed = []
    ```

        but it also handles cases where `item` are batched (meaning it's a dict of Tensor with first dimension > 1. In
        that case it does

    ```
    packed =  []
    for batch in loader:
        # item is batched
        for item in batch:
            packed.append(item)
            if item["is_last"]:
                yield packed
                packed = []
    ```

        Arguments:
            loader (`torch.utils.data.DataLoader` or any iterator):
                The iterator that will be used to apply `infer` on.
            infer (any function):
                The function to apply of each element of `loader`.
            params (`dict`):
                The parameters passed to `infer` along with every item
            loader_batch_size (`int`, *optional*):
                If specified, the items of `loader` are supposed to come as batch, and are loader_batched here making
                it roughly behave as


    ```
    for items in loader:
        for i in loader_batch_size:
            item = items[i]
            yield infer(item, **params)
    ```"""

    def __iter__(self):
        if False:
            return 10
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        is_last = False
        accumulator = []
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            while self._loader_batch_index < self.loader_batch_size:
                item = self.loader_batch_item()
                is_last = item.pop('is_last')
                accumulator.append(item)
                if is_last:
                    return accumulator
        while not is_last:
            processed = self.infer(next(self.iterator), **self.params)
            if self.loader_batch_size is not None:
                if isinstance(processed, torch.Tensor):
                    first_tensor = processed
                else:
                    key = list(processed.keys())[0]
                    first_tensor = processed[key]
                if isinstance(first_tensor, list):
                    observed_batch_size = len(first_tensor)
                else:
                    observed_batch_size = first_tensor.shape[0]
                if 0 < observed_batch_size < self.loader_batch_size:
                    self.loader_batch_size = observed_batch_size
                self._loader_batch_data = processed
                self._loader_batch_index = 0
                while self._loader_batch_index < self.loader_batch_size:
                    item = self.loader_batch_item()
                    is_last = item.pop('is_last')
                    accumulator.append(item)
                    if is_last:
                        return accumulator
            else:
                item = processed
                is_last = item.pop('is_last')
                accumulator.append(item)
        return accumulator

class KeyDataset(Dataset):

    def __init__(self, dataset: Dataset, key: str):
        if False:
            return 10
        self.dataset = dataset
        self.key = key

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.dataset)

    def __getitem__(self, i):
        if False:
            while True:
                i = 10
        return self.dataset[i][self.key]

class KeyPairDataset(Dataset):

    def __init__(self, dataset: Dataset, key1: str, key2: str):
        if False:
            return 10
        self.dataset = dataset
        self.key1 = key1
        self.key2 = key2

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.dataset)

    def __getitem__(self, i):
        if False:
            for i in range(10):
                print('nop')
        return {'text': self.dataset[i][self.key1], 'text_pair': self.dataset[i][self.key2]}