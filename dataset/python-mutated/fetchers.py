from typing import Any, Iterator, List, Optional
from typing_extensions import override
from lightning.fabric.utilities.data import sized_len
from lightning.pytorch.utilities.combined_loader import _ITERATOR_RETURN, CombinedLoader
from lightning.pytorch.utilities.exceptions import MisconfigurationException

def _profile_nothing() -> None:
    if False:
        while True:
            i = 10
    pass

class _DataFetcher(Iterator):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._combined_loader: Optional[CombinedLoader] = None
        self.iterator: Optional[Iterator] = None
        self.fetched: int = 0
        self.done: bool = False
        self.length: Optional[int] = None
        self._start_profiler = _profile_nothing
        self._stop_profiler = _profile_nothing

    @property
    def combined_loader(self) -> CombinedLoader:
        if False:
            for i in range(10):
                print('nop')
        if self._combined_loader is None:
            raise MisconfigurationException(f'`{self.__class__.__name__}` should have been `setup` with a `CombinedLoader`.')
        return self._combined_loader

    def setup(self, combined_loader: CombinedLoader) -> None:
        if False:
            print('Hello World!')
        self._combined_loader = combined_loader

    @override
    def __iter__(self) -> '_DataFetcher':
        if False:
            while True:
                i = 10
        self.iterator = iter(self.combined_loader)
        self.reset()
        return self

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        if False:
            print('Hello World!')
        assert self.iterator is not None
        self._start_profiler()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.done = True
            raise
        finally:
            self._stop_profiler()
        self.fetched += 1
        if self.length is not None:
            self.done = self.fetched >= self.length
        return batch

    def reset(self) -> None:
        if False:
            print('Hello World!')
        self.fetched = 0
        if self._combined_loader is not None:
            self.length = sized_len(self.combined_loader)
            self.done = self.length == 0

    def teardown(self) -> None:
        if False:
            return 10
        self.reset()
        if self._combined_loader is not None:
            self._combined_loader.reset()
        self.iterator = None

class _PrefetchDataFetcher(_DataFetcher):
    """This class is used to control batch fetching flow.

    Args:
        prefetch_batches: Number of batches to pre-fetch. Pre-fetching at least 1 batch is necessary to properly track
            whether a batch is the last one (available with :attr:`self.done`) when the length is not available. The
            value of this argument is ignored when the length is available.

    """

    def __init__(self, prefetch_batches: int=1) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if prefetch_batches < 0:
            raise ValueError('`prefetch_batches` should at least be 0.')
        self.prefetch_batches = prefetch_batches
        self.batches: List[Any] = []

    @override
    def __iter__(self) -> '_PrefetchDataFetcher':
        if False:
            return 10
        super().__iter__()
        if self.length is not None:
            return self
        for _ in range(self.prefetch_batches):
            try:
                batch = super().__next__()
                self.batches.append(batch)
            except StopIteration:
                break
        return self

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        if False:
            for i in range(10):
                print('nop')
        if self.batches:
            batch = self.batches.pop(0)
            try:
                self.batches.append(super().__next__())
            except StopIteration:
                self.done = not self.batches
        elif not self.done:
            batch = super().__next__()
        else:
            raise StopIteration
        return batch

    @override
    def reset(self) -> None:
        if False:
            while True:
                i = 10
        super().reset()
        self.batches = []

class _DataLoaderIterDataFetcher(_DataFetcher):
    """This class is used to return directly the `dataloader_iter` to the ``LightningModule`` training_step for users
    to implement their own pre-fetching logic. This feature can be activated as follows:

    Example::

        Class MyModel(LightningModule):
            def training_step(self, dataloader_iter: Iterator) -> None:
                # it is the user responsibility to fetch and move the batch to the right device.
                batch, batch_idx, dataloader_idx = next(dataloader_iter)
                batch = batch.to(self.device)
                ...

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self._batch: Any = None
        self._batch_idx: int = 0
        self._dataloader_idx: int = 0

    @override
    def __iter__(self) -> '_DataLoaderIterDataFetcher':
        if False:
            i = 10
            return i + 15
        super().__iter__()
        self.iterator_wrapper = iter(_DataFetcherWrapper(self))
        return self

    @override
    def __next__(self) -> Iterator['_DataFetcherWrapper']:
        if False:
            while True:
                i = 10
        if self.done:
            raise StopIteration
        return self.iterator_wrapper

    @override
    def reset(self) -> None:
        if False:
            while True:
                i = 10
        super().reset()
        self._batch = None
        self._batch_idx = 0
        self._dataloader_idx = 0

class _DataFetcherWrapper(Iterator):

    def __init__(self, data_fetcher: _DataLoaderIterDataFetcher) -> None:
        if False:
            i = 10
            return i + 15
        self.data_fetcher = data_fetcher

    @property
    def done(self) -> bool:
        if False:
            while True:
                i = 10
        return self.data_fetcher.done

    @property
    def fetched(self) -> int:
        if False:
            print('Hello World!')
        return self.data_fetcher.fetched

    @property
    def length(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        return self.data_fetcher.length

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        if False:
            print('Hello World!')
        fetcher = self.data_fetcher
        if fetcher.done:
            raise StopIteration
        (batch, batch_idx, dataloader_idx) = super(_DataLoaderIterDataFetcher, fetcher).__next__()
        fetcher._batch = batch
        fetcher._batch_idx = batch_idx
        fetcher._dataloader_idx = dataloader_idx
        return (batch, batch_idx, dataloader_idx)