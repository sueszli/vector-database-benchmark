from __future__ import annotations
from typing import Any
from typing import Optional
from typing import Type
from ..node.credentials import SyftVerifyKey
from ..serde.serializable import serializable
from .document_store import DocumentStore
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition
from .locks import LockingConfig
from .locks import ThreadingLockingConfig

@serializable()
class DictBackingStore(dict, KeyValueBackingStore):
    """Dictionary-based Store core logic"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        super(dict).__init__()
        self._ddtype = kwargs.get('ddtype', None)

    def __getitem__(self, key: Any) -> Any:
        if False:
            return 10
        try:
            value = super().__getitem__(key)
            return value
        except KeyError as e:
            if self._ddtype:
                return self._ddtype()
            raise e

@serializable()
class DictStorePartition(KeyValueStorePartition):
    """Dictionary-based StorePartition

    Parameters:
        `settings`: PartitionSettings
            PySyft specific settings, used for indexing and partitioning
        `store_config`: DictStoreConfig
            DictStore specific configuration
    """

    def prune(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_store()

@serializable()
class DictDocumentStore(DocumentStore):
    """Dictionary-based Document Store

    Parameters:
        `store_config`: DictStoreConfig
            Dictionary Store specific configuration, containing the store type and the backing store type
    """
    partition_type = DictStorePartition

    def __init__(self, root_verify_key: Optional[SyftVerifyKey], store_config: Optional[DictStoreConfig]=None) -> None:
        if False:
            while True:
                i = 10
        if store_config is None:
            store_config = DictStoreConfig()
        super().__init__(root_verify_key=root_verify_key, store_config=store_config)

    def reset(self):
        if False:
            print('Hello World!')
        for (_, partition) in self.partitions.items():
            partition.prune()

@serializable()
class DictStoreConfig(StoreConfig):
    __canonical_name__ = 'DictStoreConfig'
    'Dictionary-based configuration\n\n    Parameters:\n        `store_type`: Type[DocumentStore]\n            The Document type used. Default: DictDocumentStore\n        `backing_store`: Type[KeyValueBackingStore]\n            The backend type used. Default: DictBackingStore\n        locking_config: LockingConfig\n            The config used for store locking. Available options:\n                * NoLockingConfig: no locking, ideal for single-thread stores.\n                * ThreadingLockingConfig: threading-based locking, ideal for same-process in-memory stores.\n                * FileLockingConfig: file based locking, ideal for same-device different-processes/threads stores.\n                * RedisLockingConfig: Redis-based locking, ideal for multi-device stores.\n            Defaults to ThreadingLockingConfig.\n    '
    store_type: Type[DocumentStore] = DictDocumentStore
    backing_store: Type[KeyValueBackingStore] = DictBackingStore
    locking_config: LockingConfig = ThreadingLockingConfig()