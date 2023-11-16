from abc import ABC, abstractmethod
import json
from typing import Any, Dict
from deeplake.util.json import HubJsonEncoder, HubJsonDecoder

class DeepLakeMemoryObject(ABC):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.is_dirty = True

    @property
    @abstractmethod
    def nbytes(self):
        if False:
            return 10
        'Returns the number of bytes in the object.'

    def __getstate__(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        if False:
            print('Hello World!')
        self.__dict__.update(state)

    def tobytes(self) -> bytes:
        if False:
            print('Hello World!')
        d = {str(k): v for (k, v) in self.__getstate__().items()}
        return bytes(json.dumps(d, sort_keys=True, indent=4, cls=HubJsonEncoder), 'utf-8')

    @classmethod
    def frombuffer(cls, buffer: bytes):
        if False:
            while True:
                i = 10
        instance = cls()
        if len(buffer) > 0:
            instance.__setstate__(json.loads(buffer, cls=HubJsonDecoder))
            instance.is_dirty = False
            return instance
        raise BufferError('Unable to instantiate the object as the buffer was empty.')