from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union
from pydantic import Field
from pydantic import validator
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self
from ..serde.deserialize import _deserialize
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
from ..types.uid import UID
from ..util.util import thread_ident
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .document_store import StoreClientConfig
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition
from .locks import FileLockingConfig
from .locks import LockingConfig

def _repr_debug_(value: Any) -> str:
    if False:
        return 10
    if hasattr(value, '_repr_debug_'):
        return str(value._repr_debug_())
    return repr(value)

@serializable(attrs=['index_name', 'settings', 'store_config'])
class SQLiteBackingStore(KeyValueBackingStore):
    """Core Store logic for the SQLite stores.

    Parameters:
        `index_name`: str
            Index name
        `settings`: PartitionSettings
            Syft specific settings
        `store_config`: SQLiteStoreConfig
            Connection Configuration
        `ddtype`: Type
            Class used as fallback on `get` errors
    """

    def __init__(self, index_name: str, settings: PartitionSettings, store_config: StoreConfig, ddtype: Optional[type]=None) -> None:
        if False:
            i = 10
            return i + 15
        self.index_name = index_name
        self.settings = settings
        self.store_config = store_config
        self._ddtype = ddtype
        self._db: Dict[int, sqlite3.Connection] = {}
        self._cur: Dict[int, sqlite3.Cursor] = {}
        self.create_table()

    @property
    def table_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{self.settings.name}_{self.index_name}'

    def _connect(self) -> None:
        if False:
            while True:
                i = 10
        self.file_path = self.store_config.client_config.file_path
        path = Path(self.file_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db[thread_ident()] = sqlite3.connect(self.file_path, timeout=self.store_config.client_config.timeout, check_same_thread=self.store_config.client_config.check_same_thread)

    def create_table(self) -> None:
        if False:
            i = 10
            return i + 15
        try:
            self.cur.execute(f'create table {self.table_name} (uid VARCHAR(32) NOT NULL PRIMARY KEY, ' + 'repr TEXT NOT NULL, value BLOB NOT NULL, ' + 'sqltime TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL)')
            self.db.commit()
        except sqlite3.OperationalError as e:
            if f'table {self.table_name} already exists' not in str(e):
                raise e

    @property
    def db(self) -> sqlite3.Connection:
        if False:
            i = 10
            return i + 15
        if thread_ident() not in self._db:
            self._connect()
        return self._db[thread_ident()]

    @property
    def cur(self) -> sqlite3.Cursor:
        if False:
            return 10
        if thread_ident() not in self._cur:
            self._cur[thread_ident()] = self.db.cursor()
        return self._cur[thread_ident()]

    def _close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._commit()
        self.db.close()

    def _commit(self) -> None:
        if False:
            return 10
        self.db.commit()

    def _execute(self, sql: str, *args: Optional[List[Any]]) -> Result[Ok[sqlite3.Cursor], Err[str]]:
        if False:
            return 10
        cursor: Optional[sqlite3.Cursor] = None
        err = None
        try:
            cursor = self.cur.execute(sql, *args)
        except BaseException as e:
            self.db.rollback()
            err = Err(str(e))
        else:
            self.db.commit()
        if err is not None:
            return err
        return Ok(cursor)

    def _set(self, key: UID, value: Any) -> None:
        if False:
            i = 10
            return i + 15
        if self._exists(key):
            self._update(key, value)
        else:
            insert_sql = f'insert into {self.table_name} (uid, repr, value) VALUES (?, ?, ?)'
            data = _serialize(value, to_bytes=True)
            res = self._execute(insert_sql, [str(key), _repr_debug_(value), data])
            if res.is_err():
                raise ValueError(res.err())

    def _update(self, key: UID, value: Any) -> None:
        if False:
            while True:
                i = 10
        insert_sql = f'update {self.table_name} set uid = ?, repr = ?, value = ? where uid = ?'
        data = _serialize(value, to_bytes=True)
        res = self._execute(insert_sql, [str(key), _repr_debug_(value), data, str(key)])
        if res.is_err():
            raise ValueError(res.err())

    def _get(self, key: UID) -> Any:
        if False:
            while True:
                i = 10
        select_sql = f'select * from {self.table_name} where uid = ? order by sqltime'
        res = self._execute(select_sql, [str(key)])
        if res.is_err():
            raise KeyError(f'Query {select_sql} failed')
        cursor = res.ok()
        row = cursor.fetchone()
        if row is None or len(row) == 0:
            raise KeyError(f'{key} not in {type(self)}')
        data = row[2]
        return _deserialize(data, from_bytes=True)

    def _exists(self, key: UID) -> bool:
        if False:
            print('Hello World!')
        select_sql = f'select uid from {self.table_name} where uid = ?'
        res = self._execute(select_sql, [str(key)])
        if res.is_err():
            return False
        cursor = res.ok()
        row = cursor.fetchone()
        if row is None:
            return False
        return bool(row)

    def _get_all(self) -> Any:
        if False:
            print('Hello World!')
        select_sql = f'select * from {self.table_name} order by sqltime'
        keys = []
        data = []
        res = self._execute(select_sql)
        if res.is_err():
            return {}
        cursor = res.ok()
        rows = cursor.fetchall()
        if rows is None:
            return {}
        for row in rows:
            keys.append(UID(row[0]))
            data.append(_deserialize(row[2], from_bytes=True))
        return dict(zip(keys, data))

    def _get_all_keys(self) -> Any:
        if False:
            return 10
        select_sql = f'select uid from {self.table_name} order by sqltime'
        keys = []
        res = self._execute(select_sql)
        if res.is_err():
            return []
        cursor = res.ok()
        rows = cursor.fetchall()
        if rows is None:
            return []
        for row in rows:
            keys.append(UID(row[0]))
        return keys

    def _delete(self, key: UID) -> None:
        if False:
            print('Hello World!')
        select_sql = f'delete from {self.table_name} where uid = ?'
        res = self._execute(select_sql, [str(key)])
        if res.is_err():
            raise ValueError(res.err())

    def _delete_all(self) -> None:
        if False:
            return 10
        select_sql = f'delete from {self.table_name}'
        res = self._execute(select_sql)
        if res.is_err():
            raise ValueError(res.err())

    def _len(self) -> int:
        if False:
            return 10
        select_sql = f'select count(uid) from {self.table_name}'
        res = self._execute(select_sql)
        if res.is_err():
            raise ValueError(res.err())
        cursor = res.ok()
        cnt = cursor.fetchone()[0]
        return cnt

    def __setitem__(self, key: Any, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._set(key, value)

    def __getitem__(self, key: Any) -> Self:
        if False:
            print('Hello World!')
        try:
            return self._get(key)
        except KeyError as e:
            if self._ddtype is not None:
                return self._ddtype()
            raise e

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return repr(self._get_all())

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._len()

    def __delitem__(self, key: str):
        if False:
            i = 10
            return i + 15
        self._delete(key)

    def clear(self) -> Self:
        if False:
            print('Hello World!')
        self._delete_all()

    def copy(self) -> Self:
        if False:
            i = 10
            return i + 15
        return deepcopy(self)

    def keys(self) -> Any:
        if False:
            i = 10
            return i + 15
        return self._get_all_keys()

    def values(self) -> Any:
        if False:
            while True:
                i = 10
        return self._get_all().values()

    def items(self) -> Any:
        if False:
            return 10
        return self._get_all().items()

    def pop(self, key: Any) -> Self:
        if False:
            while True:
                i = 10
        value = self._get(key)
        self._delete(key)
        return value

    def __contains__(self, key: Any) -> bool:
        if False:
            return 10
        return self._exists(key)

    def __iter__(self) -> Any:
        if False:
            return 10
        return iter(self.keys())

    def __del__(self):
        if False:
            return 10
        try:
            self._close()
        except BaseException:
            pass

@serializable()
class SQLiteStorePartition(KeyValueStorePartition):
    """SQLite StorePartition

    Parameters:
        `settings`: PartitionSettings
            PySyft specific settings, used for indexing and partitioning
        `store_config`: SQLiteStoreConfig
            SQLite specific configuration
    """

    def close(self) -> None:
        if False:
            return 10
        self.lock.acquire()
        try:
            self.data._close()
            self.unique_keys._close()
            self.searchable_keys._close()
        except BaseException:
            pass
        self.lock.release()

    def commit(self) -> None:
        if False:
            i = 10
            return i + 15
        self.lock.acquire()
        try:
            self.data._commit()
            self.unique_keys._commit()
            self.searchable_keys._commit()
        except BaseException:
            pass
        self.lock.release()

@serializable()
class SQLiteDocumentStore(DocumentStore):
    """SQLite Document Store

    Parameters:
        `store_config`: StoreConfig
            SQLite specific configuration, including connection details and client class type.
    """
    partition_type = SQLiteStorePartition

@serializable()
class SQLiteStoreClientConfig(StoreClientConfig):
    """SQLite connection config

    Parameters:
        `filename` : str
            Database name
        `path` : Path or str
            Database folder
        `check_same_thread`: bool
            If True (default), ProgrammingError will be raised if the database connection is used
            by a thread other than the one that created it. If False, the connection may be accessed
            in multiple threads; write operations may need to be serialized by the user to avoid
            data corruption.
        `timeout`: int
            How many seconds the connection should wait before raising an exception, if the database
            is locked by another connection. If another connection opens a transaction to modify the
            database, it will be locked until that transaction is committed. Default five seconds.
    """
    filename: Optional[str] = None
    path: Union[str, Path] = Field(default_factory=tempfile.gettempdir)
    check_same_thread: bool = True
    timeout: int = 5

    @validator('path', pre=True)
    def __default_path(cls, path: Optional[Union[str, Path]]) -> Union[str, Path]:
        if False:
            print('Hello World!')
        if path is None:
            return tempfile.gettempdir()
        return path

    @property
    def file_path(self) -> Optional[Path]:
        if False:
            return 10
        return Path(self.path) / self.filename if self.filename is not None else None

@serializable()
class SQLiteStoreConfig(StoreConfig):
    __canonical_name__ = 'SQLiteStoreConfig'
    'SQLite Store config, used by SQLiteStorePartition\n\n    Parameters:\n        `client_config`: SQLiteStoreClientConfig\n            SQLite connection configuration\n        `store_type`: DocumentStore\n            Class interacting with QueueStash. Default: SQLiteDocumentStore\n        `backing_store`: KeyValueBackingStore\n            The Store core logic. Default: SQLiteBackingStore\n        locking_config: LockingConfig\n            The config used for store locking. Available options:\n                * NoLockingConfig: no locking, ideal for single-thread stores.\n                * ThreadingLockingConfig: threading-based locking, ideal for same-process in-memory stores.\n                * FileLockingConfig: file based locking, ideal for same-device different-processes/threads stores.\n                * RedisLockingConfig: Redis-based locking, ideal for multi-device stores.\n            Defaults to FileLockingConfig.\n    '
    client_config: SQLiteStoreClientConfig
    store_type: Type[DocumentStore] = SQLiteDocumentStore
    backing_store: Type[KeyValueBackingStore] = SQLiteBackingStore
    locking_config: LockingConfig = FileLockingConfig()