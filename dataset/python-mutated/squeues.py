"""
Scheduler queues
"""
import marshal
import pickle
from os import PathLike
from pathlib import Path
from typing import Union
from queuelib import queue
from scrapy.utils.request import request_from_dict

def _with_mkdir(queue_class):
    if False:
        print('Hello World!')

    class DirectoriesCreated(queue_class):

        def __init__(self, path: Union[str, PathLike], *args, **kwargs):
            if False:
                return 10
            dirname = Path(path).parent
            if not dirname.exists():
                dirname.mkdir(parents=True, exist_ok=True)
            super().__init__(path, *args, **kwargs)
    return DirectoriesCreated

def _serializable_queue(queue_class, serialize, deserialize):
    if False:
        i = 10
        return i + 15

    class SerializableQueue(queue_class):

        def push(self, obj):
            if False:
                print('Hello World!')
            s = serialize(obj)
            super().push(s)

        def pop(self):
            if False:
                while True:
                    i = 10
            s = super().pop()
            if s:
                return deserialize(s)

        def peek(self):
            if False:
                return 10
            'Returns the next object to be returned by :meth:`pop`,\n            but without removing it from the queue.\n\n            Raises :exc:`NotImplementedError` if the underlying queue class does\n            not implement a ``peek`` method, which is optional for queues.\n            '
            try:
                s = super().peek()
            except AttributeError as ex:
                raise NotImplementedError("The underlying queue class does not implement 'peek'") from ex
            if s:
                return deserialize(s)
    return SerializableQueue

def _scrapy_serialization_queue(queue_class):
    if False:
        print('Hello World!')

    class ScrapyRequestQueue(queue_class):

        def __init__(self, crawler, key):
            if False:
                for i in range(10):
                    print('nop')
            self.spider = crawler.spider
            super().__init__(key)

        @classmethod
        def from_crawler(cls, crawler, key, *args, **kwargs):
            if False:
                while True:
                    i = 10
            return cls(crawler, key)

        def push(self, request):
            if False:
                return 10
            request = request.to_dict(spider=self.spider)
            return super().push(request)

        def pop(self):
            if False:
                while True:
                    i = 10
            request = super().pop()
            if not request:
                return None
            return request_from_dict(request, spider=self.spider)

        def peek(self):
            if False:
                print('Hello World!')
            'Returns the next object to be returned by :meth:`pop`,\n            but without removing it from the queue.\n\n            Raises :exc:`NotImplementedError` if the underlying queue class does\n            not implement a ``peek`` method, which is optional for queues.\n            '
            request = super().peek()
            if not request:
                return None
            return request_from_dict(request, spider=self.spider)
    return ScrapyRequestQueue

def _scrapy_non_serialization_queue(queue_class):
    if False:
        return 10

    class ScrapyRequestQueue(queue_class):

        @classmethod
        def from_crawler(cls, crawler, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return cls()

        def peek(self):
            if False:
                print('Hello World!')
            'Returns the next object to be returned by :meth:`pop`,\n            but without removing it from the queue.\n\n            Raises :exc:`NotImplementedError` if the underlying queue class does\n            not implement a ``peek`` method, which is optional for queues.\n            '
            try:
                s = super().peek()
            except AttributeError as ex:
                raise NotImplementedError("The underlying queue class does not implement 'peek'") from ex
            return s
    return ScrapyRequestQueue

def _pickle_serialize(obj):
    if False:
        for i in range(10):
            print('nop')
    try:
        return pickle.dumps(obj, protocol=4)
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        raise ValueError(str(e)) from e
_PickleFifoSerializationDiskQueue = _serializable_queue(_with_mkdir(queue.FifoDiskQueue), _pickle_serialize, pickle.loads)
_PickleLifoSerializationDiskQueue = _serializable_queue(_with_mkdir(queue.LifoDiskQueue), _pickle_serialize, pickle.loads)
_MarshalFifoSerializationDiskQueue = _serializable_queue(_with_mkdir(queue.FifoDiskQueue), marshal.dumps, marshal.loads)
_MarshalLifoSerializationDiskQueue = _serializable_queue(_with_mkdir(queue.LifoDiskQueue), marshal.dumps, marshal.loads)
PickleFifoDiskQueue = _scrapy_serialization_queue(_PickleFifoSerializationDiskQueue)
PickleLifoDiskQueue = _scrapy_serialization_queue(_PickleLifoSerializationDiskQueue)
MarshalFifoDiskQueue = _scrapy_serialization_queue(_MarshalFifoSerializationDiskQueue)
MarshalLifoDiskQueue = _scrapy_serialization_queue(_MarshalLifoSerializationDiskQueue)
FifoMemoryQueue = _scrapy_non_serialization_queue(queue.FifoMemoryQueue)
LifoMemoryQueue = _scrapy_non_serialization_queue(queue.LifoMemoryQueue)