from typing import Any
from kombu import serialization
import syft as sy
from syft.logger import error

def loads(data: bytes) -> Any:
    if False:
        i = 10
        return i + 15
    org_payload = sy.deserialize(data, from_bytes=True)
    if len(org_payload) > 0 and len(org_payload[0]) > 0 and isinstance(org_payload[0][0], bytes):
        try:
            nested_data = org_payload[0][0]
            org_obj = sy.deserialize(nested_data, from_bytes=True)
            org_payload[0][0] = org_obj
        except Exception as e:
            error(f'Unable to deserialize nested payload. {e}')
            raise e
    return org_payload

def dumps(obj: Any) -> bytes:
    if False:
        print('Hello World!')
    return sy.serialize(obj, to_bytes=True)
serialization.register('syft', dumps, loads, content_type='application/syft', content_encoding='binary')