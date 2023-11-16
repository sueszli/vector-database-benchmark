import math
import pickle
from typing import Any, NamedTuple, Sequence, get_args
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.utilities.combined_loader import _LITERAL_SUPPORTED_MODES, _SUPPORTED_MODES, CombinedLoader, _MaxSize, _MaxSizeCycle, _MinSize, _Sequential
from torch import Tensor
from torch.utils._pytree import tree_flatten
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tests_pytorch.helpers.runif import RunIf

@pytest.mark.parametrize(('dataset_1', 'dataset_2'), [(list(range(10)), list(range(20))), (range(10), range(20)), (torch.randn(10, 3, 2), torch.randn(20, 5, 6)), (TensorDataset(torch.randn(10, 3, 2)), TensorDataset(torch.randn(20, 5, 6)))])
def test_combined_dataset(dataset_1, dataset_2):
    if False:
        i = 10
        return i + 15
    datasets = [DataLoader(dataset_1), DataLoader(dataset_2)]
    combined_loader = CombinedLoader(datasets, 'max_size_cycle')
    assert combined_loader._dataset_length() == 20

def test_combined_dataset_no_length():
    if False:
        i = 10
        return i + 15

    class Foo:

        def __len__(self):
            if False:
                for i in range(10):
                    print('nop')
            return 5

    class Bar:
        ...

    class Baz:

        def __len__(self):
            if False:
                return 10
            pass
    cl = CombinedLoader([DataLoader(Foo()), DataLoader(Bar()), DataLoader(Baz())])
    assert cl._dataset_length() == 5
    cl = CombinedLoader(DataLoader(Bar()))
    with pytest.raises(NotImplementedError, match='All datasets are iterable-style'):
        cl._dataset_length()

def test_combined_loader_length_must_call_iter_first():
    if False:
        return 10
    loader = CombinedLoader([1, 2, 3])
    with pytest.raises(RuntimeError, match='Please call `iter.*` first'):
        len(loader)

def test_combined_loader_modes_for_dict():
    if False:
        for i in range(10):
            print('nop')
    'Test `CombinedLoaderIterator` given mapping iterables.'
    iterables = {'a': torch.utils.data.DataLoader(range(10), batch_size=4), 'b': torch.utils.data.DataLoader(range(20), batch_size=5)}
    lengths = [len(v) for v in iterables.values()]
    min_len = min(lengths)
    combined_loader = CombinedLoader(iterables, 'min_size')
    iter(combined_loader)
    assert combined_loader._iterator is not None
    assert len(combined_loader) == min_len
    for (item, idx, _) in combined_loader:
        assert isinstance(combined_loader._iterator, _MinSize)
        assert isinstance(item, dict)
        assert list(item) == ['a', 'b']
    assert idx == min_len - 1
    assert idx == len(combined_loader) - 1
    max_len = max(lengths)
    combined_loader = CombinedLoader(iterables, 'max_size_cycle')
    iter(combined_loader)
    assert combined_loader._iterator is not None
    assert len(combined_loader) == max_len
    for (item, idx, _) in combined_loader:
        assert isinstance(combined_loader._iterator, _MaxSizeCycle)
        assert isinstance(item, dict)
        assert list(item) == ['a', 'b']
    assert idx == max_len - 1
    assert idx == len(combined_loader) - 1
    combined_loader = CombinedLoader(iterables, 'max_size')
    iter(combined_loader)
    assert len(combined_loader) == max_len
    for (item, idx, _) in combined_loader:
        assert isinstance(combined_loader._iterator, _MaxSize)
        assert isinstance(item, dict)
        assert list(item) == ['a', 'b']
        are_nones = [x is None for x in item.values()]
        should_be_nones = [idx >= length for length in lengths]
        assert are_nones == should_be_nones
    assert idx == max_len - 1
    assert idx == len(combined_loader) - 1
    sum_len = sum(lengths)
    combined_loader = CombinedLoader(iterables, 'sequential')
    iter(combined_loader)
    assert combined_loader._iterator is not None
    assert len(combined_loader) == sum_len
    for (total_idx, (item, batch_idx, dataloader_idx)) in enumerate(combined_loader):
        assert isinstance(combined_loader._iterator, _Sequential)
        assert isinstance(batch_idx, int)
        assert isinstance(item, Tensor)
    assert idx == lengths[-1] - 1
    assert total_idx == sum_len - 1
    assert total_idx == len(combined_loader) - 1
    assert dataloader_idx == len(iterables) - 1

def test_combined_loader_modes_for_list():
    if False:
        for i in range(10):
            print('nop')
    'Test `CombinedLoaderIterator` given list of iterables.'
    iterables = [torch.utils.data.DataLoader(range(10), batch_size=4), torch.utils.data.DataLoader(range(20), batch_size=5)]
    lengths = [len(v) for v in iterables]
    min_len = min(lengths)
    combined_loader = CombinedLoader(iterables, 'min_size')
    iter(combined_loader)
    assert len(combined_loader) == min_len
    for (item, idx, _) in combined_loader:
        assert isinstance(combined_loader._iterator, _MinSize)
        assert isinstance(item, list)
        assert len(item) == 2
    assert idx == min_len - 1
    assert idx == len(combined_loader) - 1
    max_len = max(lengths)
    combined_loader = CombinedLoader(iterables, 'max_size_cycle')
    iter(combined_loader)
    assert len(combined_loader) == max_len
    for (item, idx, _) in combined_loader:
        assert isinstance(combined_loader._iterator, _MaxSizeCycle)
        assert isinstance(item, list)
        assert len(item) == 2
    assert idx == max_len - 1
    assert idx == len(combined_loader) - 1
    combined_loader = CombinedLoader(iterables, 'max_size')
    iter(combined_loader)
    assert len(combined_loader) == max_len
    for (item, idx, _) in combined_loader:
        assert isinstance(combined_loader._iterator, _MaxSize)
        assert isinstance(item, list)
        assert len(item) == 2
        are_nones = [x is None for x in item]
        should_be_nones = [idx >= length for length in lengths]
        assert are_nones == should_be_nones
    assert idx == max_len - 1
    assert idx == len(combined_loader) - 1
    sum_len = sum(lengths)
    combined_loader = CombinedLoader(iterables, 'sequential')
    iter(combined_loader)
    assert combined_loader._iterator is not None
    assert len(combined_loader) == sum_len
    for (total_idx, (item, batch_idx, dataloader_idx)) in enumerate(combined_loader):
        assert isinstance(combined_loader._iterator, _Sequential)
        assert isinstance(batch_idx, int)
        assert isinstance(item, Tensor)
    assert idx == lengths[-1] - 1
    assert total_idx == sum_len - 1
    assert total_idx == len(combined_loader) - 1
    assert dataloader_idx == len(iterables) - 1

def test_combined_loader_modes_for_namedtuple():
    if False:
        print('Hello World!')
    'Test `CombinedLoaderIterator` given a namedtuple of iterables.'

    class IterablesNamedTuple(NamedTuple):
        a: Any
        b: Any
    iterables = IterablesNamedTuple(a=torch.utils.data.DataLoader(range(10), batch_size=4), b=torch.utils.data.DataLoader(range(20), batch_size=5))
    lengths = [len(v) for v in iterables]
    min_len = min(lengths)
    combined_loader = CombinedLoader(iterables, 'min_size')
    iter(combined_loader)
    assert len(combined_loader) == min_len
    for (item, idx, _) in combined_loader:
        assert isinstance(combined_loader._iterator, _MinSize)
        assert isinstance(item, IterablesNamedTuple)
    assert idx == min_len - 1
    assert idx == len(combined_loader) - 1
    max_len = max(lengths)
    combined_loader = CombinedLoader(iterables, 'max_size_cycle')
    iter(combined_loader)
    assert len(combined_loader) == max_len
    for (item, idx, _) in combined_loader:
        assert isinstance(combined_loader._iterator, _MaxSizeCycle)
        assert isinstance(item, IterablesNamedTuple)
    assert idx == max_len - 1
    assert idx == len(combined_loader) - 1
    combined_loader = CombinedLoader(iterables, 'max_size')
    iter(combined_loader)
    assert len(combined_loader) == max_len
    for (item, idx, _) in combined_loader:
        assert isinstance(combined_loader._iterator, _MaxSize)
        assert isinstance(item, IterablesNamedTuple)
        are_nones = [x is None for x in item]
        should_be_nones = [idx >= length for length in lengths]
        assert are_nones == should_be_nones
    assert idx == max_len - 1
    assert idx == len(combined_loader) - 1
    sum_len = sum(lengths)
    combined_loader = CombinedLoader(iterables, 'sequential')
    iter(combined_loader)
    assert combined_loader._iterator is not None
    assert len(combined_loader) == sum_len
    for (total_idx, (item, batch_idx, dataloader_idx)) in enumerate(combined_loader):
        assert isinstance(combined_loader._iterator, _Sequential)
        assert isinstance(batch_idx, int)
        assert isinstance(item, Tensor)
    assert idx == lengths[-1] - 1
    assert total_idx == sum_len - 1
    assert total_idx == len(combined_loader) - 1
    assert dataloader_idx == len(iterables) - 1

def test_combined_loader_raises():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match="Unsupported mode 'testtt'"):
        CombinedLoader([range(10)], 'testtt')

class TestIterableDataset(IterableDataset):

    def __init__(self, size: int=10):
        if False:
            print('Hello World!')
        self.size = size

    def __iter__(self):
        if False:
            print('Hello World!')
        self.sampler = SequentialSampler(range(self.size))
        self.sampler_iter = iter(self.sampler)
        return self

    def __next__(self):
        if False:
            return 10
        return next(self.sampler_iter)

@pytest.mark.parametrize('mode', ['min_size', 'max_size_cycle', 'max_size', 'sequential'])
@pytest.mark.parametrize('use_multiple_dataloaders', [False, True])
def test_combined_loader_sequence_iterable_dataset(mode, use_multiple_dataloaders):
    if False:
        while True:
            i = 10
    "Test `CombinedLoader` of mode 'min_size' given sequence iterables."
    if use_multiple_dataloaders:
        loaders = [torch.utils.data.DataLoader(TestIterableDataset(10), batch_size=2), torch.utils.data.DataLoader(TestIterableDataset(20), batch_size=2)]
    else:
        loaders = [torch.utils.data.DataLoader(TestIterableDataset(10), batch_size=2)]
    combined_loader = CombinedLoader(loaders, mode)
    has_break = False
    for (idx, item) in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        if not use_multiple_dataloaders and idx == 4:
            has_break = True
            break
    if mode == 'max_size_cycle':
        assert all(combined_loader._iterator._consumed) == (not has_break)
    expected = 5
    if use_multiple_dataloaders:
        if mode in ['max_size_cycle', 'max_size']:
            expected = 10
        elif mode == 'sequential':
            expected = 15
    assert idx == expected - 1

@pytest.mark.parametrize('mode', ['min_size', 'max_size_cycle', 'max_size', 'sequential'])
def test_combined_loader_simultaneous_workers(mode):
    if False:
        return 10
    'Test `CombinedLoader` to check how it initializes dataloader workers.'

    class TestDataLoader(DataLoader):

        def __init__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            super().__init__(*args, **kwargs)
            self.workers_active = False

        def _get_iterator(self):
            if False:
                for i in range(10):
                    print('nop')
            self.workers_active = True
            return super()._get_iterator()

        def _shutdown_workers(self):
            if False:
                for i in range(10):
                    print('nop')
            self.workers_active = False
            super()._shutdown_workers()
    loaders = [TestDataLoader(range(10), batch_size=2, num_workers=0), TestDataLoader(range(20), batch_size=2, num_workers=0)]
    combined_loader = CombinedLoader(loaders, mode)
    _ = iter(combined_loader)
    workers_active = []
    for loader in loaders:
        workers_active.append(loader.workers_active)
    expected = [True, False] if mode == 'sequential' else [True, True]
    assert workers_active == expected

@pytest.mark.parametrize(('limits', 'expected'), [(None, [('a', 0, 0), ('b', 1, 0), ('c', 2, 0), ('d', 0, 1), ('e', 1, 1)]), ([1, 0], [('a', 0, 0)]), ([0, float('inf')], [('d', 0, 1), ('e', 1, 1)]), ([1, 1], [('a', 0, 0), ('d', 0, 1)])])
def test_sequential_mode_limits(limits, expected):
    if False:
        print('Hello World!')
    iterable1 = ['a', 'b', 'c']
    iterable2 = ['d', 'e']
    iterator = _Sequential([iterable1, iterable2], limits)
    assert list(iterator) == expected

@pytest.mark.parametrize('iterator_cls', [_Sequential, _MinSize, _MaxSize, _MaxSizeCycle])
def test_iterator_mode_limits_raises(iterator_cls):
    if False:
        print('Hello World!')
    with pytest.raises(ValueError, match='number of limits \\(0\\) and number of iterables \\(2\\)'):
        iterator_cls([0, 1], [])

def test_combined_loader_flattened_setter():
    if False:
        i = 10
        return i + 15
    iterables = [[0], [[1], [[2]]]]
    combined_loader = CombinedLoader(iterables)
    with pytest.raises(ValueError, match='Mismatch in flattened length \\(1\\) and existing length \\(3\\)'):
        combined_loader.flattened = [2]
    assert combined_loader.flattened == [[0], [1], [2]]
    combined_loader.flattened = [[3], [2], [1]]
    assert combined_loader.iterables == [[3], [[2], [[1]]]]

@pytest.mark.parametrize('lengths', [[4, 6], [5, 5], [6, 4]])
def test_combined_loader_sequence_with_map_and_iterable(lengths):
    if False:
        while True:
            i = 10

    class MyIterableDataset(IterableDataset):

        def __init__(self, size: int=10):
            if False:
                i = 10
                return i + 15
            self.size = size

        def __iter__(self):
            if False:
                i = 10
                return i + 15
            self.sampler = SequentialSampler(range(self.size))
            self.iter_sampler = iter(self.sampler)
            return self

        def __next__(self):
            if False:
                print('Hello World!')
            return next(self.iter_sampler)

    class MyMapDataset(Dataset):

        def __init__(self, size: int=10):
            if False:
                print('Hello World!')
            self.size = size

        def __getitem__(self, index):
            if False:
                for i in range(10):
                    print('nop')
            return index

        def __len__(self):
            if False:
                print('Hello World!')
            return self.size
    (x, y) = lengths
    loaders = [DataLoader(MyIterableDataset(x)), DataLoader(MyMapDataset(y))]
    dataloader = CombinedLoader(loaders, mode='max_size_cycle')
    seen = sum((1 for _ in dataloader))
    assert seen == max(x, y)

@pytest.mark.parametrize('use_distributed_sampler', [False, True])
def test_combined_data_loader_validation_test(use_distributed_sampler):
    if False:
        for i in range(10):
            print('nop')
    'This test makes sure distributed sampler has been properly injected in dataloaders when using CombinedLoader.'

    class CustomDataset(Dataset):

        def __init__(self, data):
            if False:
                for i in range(10):
                    print('nop')
            self.data = data

        def __len__(self):
            if False:
                print('Hello World!')
            return len(self.data)

        def __getitem__(self, index):
            if False:
                print('Hello World!')
            return self.data[index]

    class CustomSampler(RandomSampler):

        def __init__(self, data_source, name) -> None:
            if False:
                print('Hello World!')
            super().__init__(data_source)
            self.name = name
    dataset = CustomDataset(range(10))
    combined_loader = CombinedLoader({'a': DataLoader(CustomDataset(range(10))), 'b': DataLoader(dataset, sampler=CustomSampler(dataset, 'custom_sampler')), 'c': {'c': DataLoader(CustomDataset(range(10))), 'd': DataLoader(CustomDataset(range(10)))}, 'd': [DataLoader(CustomDataset(range(10))), DataLoader(CustomDataset(range(10)))]})
    model = BoringModel()
    trainer = Trainer(use_distributed_sampler=use_distributed_sampler, strategy='ddp', accelerator='cpu', devices=2)
    trainer.strategy.connect(model)
    trainer._data_connector.attach_data(model, train_dataloaders=combined_loader)
    trainer.fit_loop.setup_data()
    samplers_flattened = tree_flatten(combined_loader.sampler)[0]
    assert len(samplers_flattened) == 6
    if use_distributed_sampler:
        assert all((isinstance(s, DistributedSampler) for s in samplers_flattened))
    else:
        assert all((isinstance(s, (SequentialSampler, CustomSampler)) for s in samplers_flattened))
    datasets_flattened = [dl.dataset for dl in combined_loader.flattened]
    assert len(datasets_flattened) == 6
    assert all((isinstance(ds, CustomDataset) for ds in datasets_flattened))

@pytest.mark.parametrize('accelerator', ['cpu', pytest.param('gpu', marks=RunIf(min_cuda_gpus=2))])
@pytest.mark.parametrize('use_distributed_sampler', [False, True])
def test_combined_data_loader_with_max_size_cycle_and_ddp(monkeypatch, accelerator, use_distributed_sampler):
    if False:
        i = 10
        return i + 15
    'This test makes sure distributed sampler has been properly injected in dataloaders when using CombinedLoader\n    with ddp and `max_size_cycle` mode.'
    trainer = Trainer(strategy='ddp', accelerator=accelerator, devices=2, use_distributed_sampler=use_distributed_sampler)
    model = BoringModel()
    combined_loader = CombinedLoader({'a': DataLoader(RandomDataset(32, 8), batch_size=1), 'b': DataLoader(RandomDataset(32, 8), batch_size=1)})
    trainer.strategy.connect(model)
    trainer._data_connector.attach_data(model, train_dataloaders=combined_loader)
    trainer.fit_loop.setup_data()
    assert len(combined_loader) == 4 if use_distributed_sampler else 8
    for a_length in [6, 8, 10]:
        combined_loader = CombinedLoader({'a': DataLoader(range(a_length), batch_size=1), 'b': DataLoader(range(8), batch_size=1)}, mode='max_size_cycle')
        iter(combined_loader)
        length = max(a_length, 8)
        assert len(combined_loader) == length
        trainer._data_connector.attach_data(model, train_dataloaders=combined_loader)
        original_process_dataloader = trainer._data_connector._prepare_dataloader

        def non_shuffle_process_dataloader(dl, shuffle, mode):
            if False:
                print('Hello World!')
            return original_process_dataloader(dl, False, mode)
        monkeypatch.setattr(trainer._data_connector, '_prepare_dataloader', non_shuffle_process_dataloader)
        trainer.fit_loop.setup_data()
        monkeypatch.undo()
        assert len(combined_loader) == length // 2 if use_distributed_sampler else length
        if use_distributed_sampler:
            last_batch = list(combined_loader)[-1][0]
            if a_length == 6:
                assert last_batch == {'a': torch.tensor([0]), 'b': torch.tensor([6])}
            elif a_length == 8:
                assert last_batch == {'a': torch.tensor([6]), 'b': torch.tensor([6])}
            elif a_length == 10:
                assert last_batch == {'a': torch.tensor([8]), 'b': torch.tensor([0])}

    class InfiniteDataset(IterableDataset):

        def __iter__(self):
            if False:
                return 10
            while True:
                yield 1
    combined_loader = CombinedLoader({'a': DataLoader(InfiniteDataset(), batch_size=1), 'b': DataLoader(range(8), batch_size=1)}, mode='max_size_cycle')
    assert len(combined_loader.iterables['b']) == 8
    trainer._data_connector.attach_data(model, train_dataloaders=combined_loader)
    trainer.fit_loop.setup_data()
    assert len(combined_loader.iterables['b']) == 4 if use_distributed_sampler else 8

@pytest.mark.parametrize('use_distributed_sampler', [False, True])
@pytest.mark.parametrize('mode', ['min_size', 'max_size_cycle', 'max_size', 'sequential'])
def test_combined_dataloader_for_training_with_ddp(use_distributed_sampler, mode, mps_count_0):
    if False:
        i = 10
        return i + 15
    'When providing a CombinedLoader as the training data, it should be correctly receive the distributed\n    samplers.'
    dim = 3
    n1 = 8
    n2 = 6
    dataloader = {'a': DataLoader(RandomDataset(dim, n1), batch_size=1), 'b': DataLoader(RandomDataset(dim, n2), batch_size=1)}
    if mode != 'max_size_cycle':
        dataloader = CombinedLoader(dataloader, mode=mode)
    model = BoringModel()
    trainer = Trainer(strategy='ddp', accelerator='auto', devices='auto', use_distributed_sampler=use_distributed_sampler)
    trainer.strategy.connect(model)
    trainer._data_connector.attach_data(model=model, train_dataloaders=dataloader)
    fn = _SUPPORTED_MODES[mode]['fn']
    expected_length_before_ddp = fn([n1, n2])
    expected_length_after_ddp = math.ceil(expected_length_before_ddp / trainer.num_devices) if use_distributed_sampler else expected_length_before_ddp
    trainer.fit_loop.setup_data()
    assert trainer.train_dataloader is not None
    assert isinstance(trainer.fit_loop._combined_loader, CombinedLoader)
    assert trainer.fit_loop._combined_loader._mode == mode
    assert trainer.num_training_batches == expected_length_after_ddp

def test_supported_modes():
    if False:
        for i in range(10):
            print('nop')
    assert set(_SUPPORTED_MODES) == set(get_args(_LITERAL_SUPPORTED_MODES))

def test_combined_loader_can_be_pickled():
    if False:
        print('Hello World!')
    dataloader = DataLoader([0, 1, 2, 3])
    iterator = iter(dataloader)
    with pytest.raises(NotImplementedError, match='cannot be pickled'):
        pickle.dumps(iterator)
    numbers = list(range(10))
    cl = CombinedLoader([dataloader, numbers])
    iter(cl)
    iterator = cl._iterator
    assert iterator.__getstate__() == {'iterables': [dataloader, numbers], 'iterators': [None, iterator.iterators[1]], 'limits': None, '_idx': 0}
    pickle.dumps(cl)