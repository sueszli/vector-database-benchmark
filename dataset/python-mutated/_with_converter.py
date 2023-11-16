from chainer.dataset.tabular import tabular_dataset

class _WithConverter(tabular_dataset.TabularDataset):

    def __init__(self, dataset, converter):
        if False:
            return 10
        self._dataset = dataset
        self._converter = converter

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._dataset)

    @property
    def keys(self):
        if False:
            i = 10
            return i + 15
        return self._dataset.keys

    @property
    def mode(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dataset.mode

    def get_examples(self, indices, key_indices):
        if False:
            while True:
                i = 10
        return self._dataset.get_examples(indices, key_indices)

    def convert(self, data):
        if False:
            i = 10
            return i + 15
        if isinstance(data, tuple):
            return self._converter(*data)
        elif isinstance(data, dict):
            return self._converter(**data)
        else:
            return self._converter(data)