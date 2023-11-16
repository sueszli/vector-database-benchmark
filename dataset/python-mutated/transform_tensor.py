from deeplake.util.exceptions import TensorDoesNotExistError, SampleAppendError
from deeplake.core.linked_tiled_sample import LinkedTiledSample
from deeplake.core.partial_sample import PartialSample
from deeplake.core.linked_sample import LinkedSample
from deeplake.core.sample import Sample
from deeplake.core.tensor import Tensor
from typing import Union, List, Any
from itertools import chain
import numpy as np
import posixpath
import bisect

class TransformTensor:

    def __init__(self, dataset, name, is_group=False):
        if False:
            print('Hello World!')
        self.items = []
        self.dataset = dataset
        self.name = name
        self.is_group = is_group
        self.idx = slice(None, None, None)
        self.numpy_only = True
        self.cum_sizes = []

    def __len__(self):
        if False:
            i = 10
            return i + 15
        if self.numpy_only:
            return 0 if not self.cum_sizes else self.cum_sizes[-1]
        return len(self.items)

    def __getattr__(self, item):
        if False:
            return 10
        return self.dataset[posixpath.join(self.name, item)][self.idx]

    def __getitem__(self, item):
        if False:
            i = 10
            return i + 15
        if isinstance(item, str):
            return self.__getattr__(item)
        self.idx = item
        return self

    def _get_output_sample(self, item):
        if False:
            while True:
                i = 10
        if isinstance(item, Sample):
            return item.array
        return_as_is = (LinkedSample, Tensor, type(None), PartialSample, LinkedTiledSample)
        if isinstance(item, return_as_is):
            return item
        return np.asarray(item)

    def _numpy_only_data(self):
        if False:
            print('Hello World!')
        idx = self.idx
        if isinstance(idx, int):
            i = bisect.bisect_right(self.cum_sizes, idx)
            if i == 0:
                j = idx
            else:
                j = idx - self.cum_sizes[i - 1]
            return self.items[i][j]
        return self.items[idx]

    def numpy(self) -> Union[List, np.ndarray]:
        if False:
            i = 10
            return i + 15
        if self.numpy_only:
            return self._numpy_only_data()
        if isinstance(self.idx, int):
            items = [self.numpy_compressed()]
            squeeze = True
        else:
            items = self.numpy_compressed()
            squeeze = False
        values: List[Any] = []
        for item in items:
            values.append(self._get_output_sample(item))
        if squeeze:
            values = values[0]
        return values

    def numpy_compressed(self):
        if False:
            for i in range(10):
                print('nop')
        if self.numpy_only:
            return self._numpy_only_data()
        idx = self.idx
        return self.items[idx]

    def non_numpy_only(self):
        if False:
            return 10
        if self.numpy_only:
            items = list(chain(*self.items[:]))
            self.items.clear()
            self.items += items
            self.cum_sizes.clear()
            self.numpy_only = False

    def _item_added(self, item):
        if False:
            while True:
                i = 10
        if self.dataset.all_chunk_engines:
            self.dataset.item_added(item, self.name)

    def _verify_item(self, item):
        if False:
            print('Hello World!')
        if not isinstance(item, (LinkedSample, LinkedTiledSample, Tensor)) and item is not None:
            shape = getattr(item, 'shape', None)

    def append(self, item):
        if False:
            print('Hello World!')
        'Adds an item to the tensor.'
        if self.is_group:
            raise TensorDoesNotExistError(self.name)
        try:
            self.non_numpy_only()
            self._verify_item(item)
            self.items.append(item)
            self._item_added(item)
        except Exception as e:
            self.items.clear()
            raise SampleAppendError(self.name, item) from e

    def _extend_numpy(self, items):
        if False:
            return 10
        'Extend tensor with a numpy array in a numpy-only tensor.\n        Returns ``True`` if successful, ``False`` otherwise.\n        '
        if isinstance(items, np.ndarray):
            self.items.append(items)
            if len(self.cum_sizes) == 0:
                self.cum_sizes.append(len(items))
            else:
                self.cum_sizes.append(self.cum_sizes[-1] + len(items))
            self._item_added(items)
            return True
        else:
            self.non_numpy_only()
            return False

    def extend(self, items):
        if False:
            return 10
        if self.numpy_only:
            if self._extend_numpy(items):
                return
        for item in items:
            self.append(item)