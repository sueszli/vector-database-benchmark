from chainer.dataset.tabular import tabular_dataset

class _Astuple(tabular_dataset.TabularDataset):

    def __init__(self, dataset):
        if False:
            print('Hello World!')
        self._dataset = dataset

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._dataset)

    @property
    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dataset.keys

    @property
    def mode(self):
        if False:
            while True:
                i = 10
        return tuple

    def get_examples(self, indices, key_indices):
        if False:
            return 10
        return self._dataset.get_examples(indices, key_indices)

    def convert(self, data):
        if False:
            print('Hello World!')
        return self._dataset.convert(data)

class _Asdict(tabular_dataset.TabularDataset):

    def __init__(self, dataset):
        if False:
            i = 10
            return i + 15
        self._dataset = dataset

    def __len__(self):
        if False:
            return 10
        return len(self._dataset)

    @property
    def keys(self):
        if False:
            return 10
        return self._dataset.keys

    @property
    def mode(self):
        if False:
            return 10
        return dict

    def get_examples(self, indices, key_indices):
        if False:
            i = 10
            return i + 15
        return self._dataset.get_examples(indices, key_indices)

    def convert(self, data):
        if False:
            i = 10
            return i + 15
        return self._dataset.convert(data)