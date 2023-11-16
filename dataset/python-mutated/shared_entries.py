from typing import Any, Optional
from uuid import uuid3
from superset.key_value.types import JsonKeyValueCodec, KeyValueResource, SharedKey
from superset.key_value.utils import get_uuid_namespace, random_key
RESOURCE = KeyValueResource.APP
NAMESPACE = get_uuid_namespace('')
CODEC = JsonKeyValueCodec()

def get_shared_value(key: SharedKey) -> Optional[Any]:
    if False:
        while True:
            i = 10
    from superset.key_value.commands.get import GetKeyValueCommand
    uuid_key = uuid3(NAMESPACE, key)
    return GetKeyValueCommand(RESOURCE, key=uuid_key, codec=CODEC).run()

def set_shared_value(key: SharedKey, value: Any) -> None:
    if False:
        for i in range(10):
            print('nop')
    from superset.key_value.commands.create import CreateKeyValueCommand
    uuid_key = uuid3(NAMESPACE, key)
    CreateKeyValueCommand(resource=RESOURCE, value=value, key=uuid_key, codec=CODEC).run()

def get_permalink_salt(key: SharedKey) -> str:
    if False:
        for i in range(10):
            print('nop')
    salt = get_shared_value(key)
    if salt is None:
        salt = random_key()
        set_shared_value(key, value=salt)
    return salt