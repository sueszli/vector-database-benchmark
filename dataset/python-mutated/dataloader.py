from __future__ import annotations
import dataclasses
from abc import ABC, abstractmethod
from asyncio import create_task, gather, get_event_loop
from asyncio.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Generic, Hashable, Iterable, List, Mapping, Optional, Sequence, TypeVar, Union, overload
from .exceptions import WrongNumberOfResultsReturned
if TYPE_CHECKING:
    from asyncio.events import AbstractEventLoop
T = TypeVar('T')
K = TypeVar('K')

@dataclass
class LoaderTask(Generic[K, T]):
    key: K
    future: Future

@dataclass
class Batch(Generic[K, T]):
    tasks: List[LoaderTask] = dataclasses.field(default_factory=list)
    dispatched: bool = False

    def add_task(self, key: Any, future: Future) -> None:
        if False:
            i = 10
            return i + 15
        task = LoaderTask[K, T](key, future)
        self.tasks.append(task)

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.tasks)

class AbstractCache(Generic[K, T], ABC):

    @abstractmethod
    def get(self, key: K) -> Union[Future[T], None]:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def set(self, key: K, value: Future[T]) -> None:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def delete(self, key: K) -> None:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def clear(self) -> None:
        if False:
            print('Hello World!')
        pass

class DefaultCache(AbstractCache[K, T]):

    def __init__(self, cache_key_fn: Optional[Callable[[K], Hashable]]=None) -> None:
        if False:
            return 10
        self.cache_key_fn: Callable[[K], Hashable] = cache_key_fn if cache_key_fn is not None else lambda x: x
        self.cache_map: Dict[Hashable, Future[T]] = {}

    def get(self, key: K) -> Union[Future[T], None]:
        if False:
            return 10
        return self.cache_map.get(self.cache_key_fn(key))

    def set(self, key: K, value: Future[T]) -> None:
        if False:
            print('Hello World!')
        self.cache_map[self.cache_key_fn(key)] = value

    def delete(self, key: K) -> None:
        if False:
            return 10
        del self.cache_map[self.cache_key_fn(key)]

    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        self.cache_map.clear()

class DataLoader(Generic[K, T]):
    batch: Optional[Batch[K, T]] = None
    cache: bool = False
    cache_map: AbstractCache[K, T]

    @overload
    def __init__(self, load_fn: Callable[[List[K]], Awaitable[Sequence[Union[T, BaseException]]]], max_batch_size: Optional[int]=None, cache: bool=True, loop: Optional[AbstractEventLoop]=None, cache_map: Optional[AbstractCache[K, T]]=None, cache_key_fn: Optional[Callable[[K], Hashable]]=None) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __init__(self: DataLoader[K, Any], load_fn: Callable[[List[K]], Awaitable[List[Any]]], max_batch_size: Optional[int]=None, cache: bool=True, loop: Optional[AbstractEventLoop]=None, cache_map: Optional[AbstractCache[K, T]]=None, cache_key_fn: Optional[Callable[[K], Hashable]]=None) -> None:
        if False:
            while True:
                i = 10
        ...

    def __init__(self, load_fn: Callable[[List[K]], Awaitable[Sequence[Union[T, BaseException]]]], max_batch_size: Optional[int]=None, cache: bool=True, loop: Optional[AbstractEventLoop]=None, cache_map: Optional[AbstractCache[K, T]]=None, cache_key_fn: Optional[Callable[[K], Hashable]]=None):
        if False:
            while True:
                i = 10
        self.load_fn = load_fn
        self.max_batch_size = max_batch_size
        self._loop = loop
        self.cache = cache
        if self.cache:
            self.cache_map = DefaultCache(cache_key_fn) if cache_map is None else cache_map

    @property
    def loop(self) -> AbstractEventLoop:
        if False:
            print('Hello World!')
        if self._loop is None:
            self._loop = get_event_loop()
        return self._loop

    def load(self, key: K) -> Awaitable[T]:
        if False:
            return 10
        if self.cache:
            future = self.cache_map.get(key)
            if future and (not future.cancelled()):
                return future
        future = self.loop.create_future()
        if self.cache:
            self.cache_map.set(key, future)
        batch = get_current_batch(self)
        batch.add_task(key, future)
        return future

    def load_many(self, keys: Iterable[K]) -> Awaitable[List[T]]:
        if False:
            i = 10
            return i + 15
        return gather(*map(self.load, keys))

    def clear(self, key: K) -> None:
        if False:
            print('Hello World!')
        if self.cache:
            self.cache_map.delete(key)

    def clear_many(self, keys: Iterable[K]) -> None:
        if False:
            return 10
        if self.cache:
            for key in keys:
                self.cache_map.delete(key)

    def clear_all(self) -> None:
        if False:
            return 10
        if self.cache:
            self.cache_map.clear()

    def prime(self, key: K, value: T, force: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        self.prime_many({key: value}, force)

    def prime_many(self, data: Mapping[K, T], force: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.cache:
            for (key, value) in data.items():
                if not self.cache_map.get(key) or force:
                    future: Future = Future(loop=self.loop)
                    future.set_result(value)
                    self.cache_map.set(key, future)
        if self.batch is not None and (not self.batch.dispatched):
            batch_updated = False
            for task in self.batch.tasks:
                if task.key in data:
                    batch_updated = True
                    task.future.set_result(data[task.key])
            if batch_updated:
                self.batch.tasks = [task for task in self.batch.tasks if not task.future.done()]

def should_create_new_batch(loader: DataLoader, batch: Batch) -> bool:
    if False:
        i = 10
        return i + 15
    if batch.dispatched or (loader.max_batch_size and len(batch) >= loader.max_batch_size):
        return True
    return False

def get_current_batch(loader: DataLoader) -> Batch:
    if False:
        print('Hello World!')
    if loader.batch and (not should_create_new_batch(loader, loader.batch)):
        return loader.batch
    loader.batch = Batch()
    dispatch(loader, loader.batch)
    return loader.batch

def dispatch(loader: DataLoader, batch: Batch) -> None:
    if False:
        while True:
            i = 10
    loader.loop.call_soon(create_task, dispatch_batch(loader, batch))

async def dispatch_batch(loader: DataLoader, batch: Batch) -> None:
    batch.dispatched = True
    keys = [task.key for task in batch.tasks]
    if len(keys) == 0:
        return
    try:
        values = await loader.load_fn(keys)
        values = list(values)
        if len(values) != len(batch):
            raise WrongNumberOfResultsReturned(expected=len(batch), received=len(values))
        for (task, value) in zip(batch.tasks, values):
            if task.future.cancelled():
                continue
            if isinstance(value, BaseException):
                task.future.set_exception(value)
            else:
                task.future.set_result(value)
    except Exception as e:
        for task in batch.tasks:
            task.future.set_exception(e)