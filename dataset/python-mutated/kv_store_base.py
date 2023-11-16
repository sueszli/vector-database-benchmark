import abc
from abc import abstractmethod
from typing import Optional
from ray.util.annotations import DeveloperAPI

@DeveloperAPI
class KVStoreBase(metaclass=abc.ABCMeta):
    """Abstract class for KVStore defining APIs needed for ray serve
    use cases, currently (8/6/2021) controller state checkpointing.
    """

    @abstractmethod
    def get_storage_key(self, key: str) -> str:
        if False:
            i = 10
            return i + 15
        'Get internal key for storage.\n\n        Args:\n            key: User provided key\n\n        Returns:\n            storage_key: Formatted key for storage, usually by\n                prepending namespace.\n        '
        raise NotImplementedError('get_storage_key() has to be implemented')

    @abstractmethod
    def put(self, key: str, val: bytes) -> bool:
        if False:
            return 10
        'Put object into kv store, bytes only.\n\n        Args:\n            key: Key for object to be stored.\n            val: Byte value of object.\n        '
        raise NotImplementedError('put() has to be implemented')

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        if False:
            i = 10
            return i + 15
        'Get object from storage.\n\n        Args:\n            key: Key for object to be retrieved.\n\n        Returns:\n            val: Byte value of object from storage.\n        '
        raise NotImplementedError('get() has to be implemented')

    @abstractmethod
    def delete(self, key: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Delete an object.\n\n        Args:\n            key: Key for object to be deleted.\n        '
        raise NotImplementedError('delete() has to be implemented')