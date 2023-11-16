import six

class DictDataset(object):
    """Dataset of a dictionary of datasets.

    It combines multiple datasets into one dataset. Each example is represented
    by a dictionary mapping a key to an example of the corresponding dataset.

    Args:
        datasets: Underlying datasets. The keys are used as the keys of each
            example. All datasets must have the same length.

    """

    def __init__(self, **datasets):
        if False:
            while True:
                i = 10
        if not datasets:
            raise ValueError('no datasets are given')
        length = None
        for (key, dataset) in six.iteritems(datasets):
            if length is None:
                length = len(dataset)
            elif length != len(dataset):
                raise ValueError('dataset length conflicts at "{}"'.format(key))
        self._datasets = datasets
        self._length = length

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        batches = {key: dataset[index] for (key, dataset) in six.iteritems(self._datasets)}
        if isinstance(index, slice):
            length = len(six.next(six.itervalues(batches)))
            return [{key: batch[i] for (key, batch) in six.iteritems(batches)} for i in six.moves.range(length)]
        else:
            return batches

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self._length