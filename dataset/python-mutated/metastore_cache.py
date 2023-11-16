import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID, uuid3
from flask import current_app, Flask, has_app_context
from flask_caching import BaseCache
from superset.key_value.exceptions import KeyValueCreateFailedError
from superset.key_value.types import KeyValueCodec, KeyValueResource, PickleKeyValueCodec
from superset.key_value.utils import get_uuid_namespace
RESOURCE = KeyValueResource.METASTORE_CACHE
logger = logging.getLogger(__name__)

class SupersetMetastoreCache(BaseCache):

    def __init__(self, namespace: UUID, codec: KeyValueCodec, default_timeout: int=300) -> None:
        if False:
            print('Hello World!')
        super().__init__(default_timeout)
        self.namespace = namespace
        self.codec = codec

    @classmethod
    def factory(cls, app: Flask, config: dict[str, Any], args: list[Any], kwargs: dict[str, Any]) -> BaseCache:
        if False:
            i = 10
            return i + 15
        seed = config.get('CACHE_KEY_PREFIX', '')
        kwargs['namespace'] = get_uuid_namespace(seed)
        codec = config.get('CODEC') or PickleKeyValueCodec()
        if has_app_context() and (not current_app.debug) and isinstance(codec, PickleKeyValueCodec):
            logger.warning('Using PickleKeyValueCodec with SupersetMetastoreCache may be unsafe, use at your own risk.')
        kwargs['codec'] = codec
        return cls(*args, **kwargs)

    def get_key(self, key: str) -> UUID:
        if False:
            return 10
        return uuid3(self.namespace, key)

    @staticmethod
    def _prune() -> None:
        if False:
            i = 10
            return i + 15
        from superset.key_value.commands.delete_expired import DeleteExpiredKeyValueCommand
        DeleteExpiredKeyValueCommand(resource=RESOURCE).run()

    def _get_expiry(self, timeout: Optional[int]) -> Optional[datetime]:
        if False:
            print('Hello World!')
        timeout = self._normalize_timeout(timeout)
        if timeout is not None and timeout > 0:
            return datetime.now() + timedelta(seconds=timeout)
        return None

    def set(self, key: str, value: Any, timeout: Optional[int]=None) -> bool:
        if False:
            for i in range(10):
                print('nop')
        from superset.key_value.commands.upsert import UpsertKeyValueCommand
        UpsertKeyValueCommand(resource=RESOURCE, key=self.get_key(key), value=value, codec=self.codec, expires_on=self._get_expiry(timeout)).run()
        return True

    def add(self, key: str, value: Any, timeout: Optional[int]=None) -> bool:
        if False:
            return 10
        from superset.key_value.commands.create import CreateKeyValueCommand
        try:
            CreateKeyValueCommand(resource=RESOURCE, value=value, codec=self.codec, key=self.get_key(key), expires_on=self._get_expiry(timeout)).run()
            self._prune()
            return True
        except KeyValueCreateFailedError:
            return False

    def get(self, key: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        from superset.key_value.commands.get import GetKeyValueCommand
        return GetKeyValueCommand(resource=RESOURCE, key=self.get_key(key), codec=self.codec).run()

    def has(self, key: str) -> bool:
        if False:
            print('Hello World!')
        entry = self.get(key)
        if entry:
            return True
        return False

    def delete(self, key: str) -> Any:
        if False:
            print('Hello World!')
        from superset.key_value.commands.delete import DeleteKeyValueCommand
        return DeleteKeyValueCommand(resource=RESOURCE, key=self.get_key(key)).run()