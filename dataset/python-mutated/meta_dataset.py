import bisect
from abc import ABC, abstractmethod
from typing import Tuple

class Dataset(ABC):
    """An abstract base class for all map-style datasets.

    .. admonition:: Abstract methods
    
       All subclasses should overwrite these two methods:

       * ``__getitem__()``: fetch a data sample for a given key.
       * ``__len__()``: return the size of the dataset.

       They play roles in the data pipeline, see the description below.

    .. admonition:: Dataset in the Data Pipline

       Usually a dataset works with :class:`~.DataLoader`, :class:`~.Sampler`, :class:`~.Collator` and other components.

       For example, the sampler generates **indexes** of batches in advance according to the size of the dataset (calling ``__len__``),
       When dataloader need to yield a batch of data, pass indexes into the ``__getitem__`` method, then collate them to a batch.

       * Highly recommended reading :ref:`dataset-guide` for more details;
       * It might helpful to read the implementation of :class:`~.MNIST`, :class:`~.CIFAR10` and other existed subclass.

    .. warning::

       By default, all elements in a dataset would be :class:`numpy.ndarray`.
       It means that if you want to do Tensor operations, it's better to do the conversion explicitly, such as:

       .. code-block:: python

          dataset = MyCustomDataset()  # A subclass of Dataset
          data, label = MyCustomDataset[0]  # equals to MyCustomDataset.__getitem__[0]
          data = Tensor(data, dtype="float32")  # convert to MegEngine Tensor explicitly

          megengine.functional.ops(data)

       Tensor ops on ndarray directly are undefined behaviors.
    """

    @abstractmethod
    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def __getitem__(self, index):
        if False:
            return 10
        pass

    @abstractmethod
    def __len__(self):
        if False:
            while True:
                i = 10
        pass

class StreamDataset(Dataset):
    """All datasets that represent an iterable of data samples should subclass it. 
    Such form of datasets is particularly useful when data come from a stream.
    All subclasses should overwrite __iter__(), which would return an iterator of samples in this dataset.
    
    Returns:
        Dataset: An iterable Dataset.

    Examples:

        .. code-block:: python

            from megengine.data.dataset import StreamDataset
            from megengine.data.dataloader import DataLoader, get_worker_info
            from megengine.data.sampler import StreamSampler

            class MyStream(StreamDataset):
                def __init__(self):
                    self.data = [iter([1, 2, 3]), iter([4, 5, 6]), iter([7, 8, 9])]
                def __iter__(self):
                    worker_info = get_worker_info()
                    data_iter = self.data[worker_info.idx]
                    while True:
                        yield next(data_iter)

            dataloader = DataLoader(
                dataset = MyStream(),
                sampler = StreamSampler(batch_size=2),
                num_workers=3,
                parallel_stream = True,
            )

            for step, data in enumerate(dataloader):
                print(data)
    """

    @abstractmethod
    def __init__(self):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def __iter__(self):
        if False:
            i = 10
            return i + 15
        pass

    def __getitem__(self, idx):
        if False:
            for i in range(10):
                print('nop')
        raise AssertionError('can not get item from StreamDataset by index')

    def __len__(self):
        if False:
            return 10
        raise AssertionError('StreamDataset does not have length')

class ArrayDataset(Dataset):
    """ArrayDataset is a dataset for numpy array data.

    One or more numpy arrays are needed to initiate the dataset.
    And the dimensions represented sample number are expected to be the same.

    Args:
        Arrays(dataset and labels): the datas and labels to be returned iteratively.

    Returns:
        Tuple: A set of raw data and corresponding label.


    Examples:

        .. code-block:: python

            from megengine.data.dataset import ArrayDataset
            from megengine.data.dataloader import DataLoader
            from megengine.data.sampler import SequentialSampler

            rand_data = np.random.randint(0, 255, size=(sample_num, 1, 32, 32), dtype=np.uint8)
            label = np.random.randint(0, 10, size=(sample_num,), dtype=int)
            dataset = ArrayDataset(rand_data, label)
            seque_sampler = SequentialSampler(dataset, batch_size=2)

            dataloader = DataLoader(
                dataset,
                sampler = seque_sampler,
                num_workers=3,
            )

            for step, data in enumerate(dataloader):
                print(data)

    """

    def __init__(self, *arrays):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if not all((len(arrays[0]) == len(array) for array in arrays)):
            raise ValueError('lengths of input arrays are inconsistent')
        self.arrays = arrays

    def __getitem__(self, index: int) -> Tuple:
        if False:
            while True:
                i = 10
        return tuple((array[index] for array in self.arrays))

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self.arrays[0])

class ConcatDataset(Dataset):
    """ConcatDataset is a concatenation of multiple datasets.

    This dataset is used for assembleing multiple map-style
    datasets.

    Args:
        datasets(list of Dataset): list of datasets to be composed.

    Returns:
        Dataset: A Dataset which composes fields of multiple datasets.

    Examples:

        .. code-block:: python

            from megengine.data.dataset import ArrayDataset, ConcatDataset

            data1 = np.random.randint(0, 255, size=(2, 1, 32, 32), dtype=np.uint8)
            data2 = np.random.randint(0, 255, size=(2, 1, 32, 32), dtype=np.uint8)
            label = np.random.randint(0, 10, size=(2,), dtype=int)
            labe2 = np.random.randint(0, 10, size=(2,), dtype=int)
            dataset1 = ArrayDataset(data1, label1)
            dataset2 = ArrayDataset(data2, label2)
            dataset = ConcatDataset([dataset1, dataset2])
            seque_sampler = SequentialSampler(dataset, batch_size=2)

            dataloader = DataLoader(
                dataset,
                sampler = seque_sampler,
                num_workers=3,
            )

            for step, data in enumerate(dataloader):
                print(data)

    """

    def __init__(self, datasets):
        if False:
            for i in range(10):
                print('nop')
        super(ConcatDataset, self).__init__()
        self.datasets = datasets

        def cumsum(datasets):
            if False:
                return 10
            (r, s) = ([], 0)
            for e in datasets:
                l = len(e)
                r.append(l + s)
                s += l
            return r
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        for d in self.datasets:
            assert not isinstance(d, StreamDataset), 'ConcatDataset does not support StreamDataset'
        self.datasets = list(datasets)
        self.cumulative_sizes = cumsum(self.datasets)

    def __getitem__(self, idx):
        if False:
            return 10
        if idx < 0:
            if -idx > len(self):
                raise ValueError('absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        if False:
            return 10
        return self.cumulative_sizes[-1]