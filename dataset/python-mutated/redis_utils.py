import re
import secrets
from typing import Any, Dict, Mapping, Optional
import orjson
import redis
from django.conf import settings
MAX_KEY_LENGTH = 1024

class ZulipRedisError(Exception):
    pass

class ZulipRedisKeyTooLongError(ZulipRedisError):
    pass

class ZulipRedisKeyOfWrongFormatError(ZulipRedisError):
    pass

def get_redis_client() -> 'redis.StrictRedis[bytes]':
    if False:
        while True:
            i = 10
    return redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, password=settings.REDIS_PASSWORD, db=0, decode_responses=False)

def put_dict_in_redis(redis_client: 'redis.StrictRedis[bytes]', key_format: str, data_to_store: Mapping[str, Any], expiration_seconds: int, token_length: int=64, token: Optional[str]=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    key_length = len(key_format) - len('{token}') + token_length
    if key_length > MAX_KEY_LENGTH:
        raise ZulipRedisKeyTooLongError(f'Requested key too long in put_dict_in_redis. Key format: {key_format}, token length: {token_length}')
    if token is None:
        token = secrets.token_hex(token_length // 2)
    key = key_format.format(token=token)
    with redis_client.pipeline() as pipeline:
        pipeline.set(key, orjson.dumps(data_to_store))
        pipeline.expire(key, expiration_seconds)
        pipeline.execute()
    return key

def get_dict_from_redis(redis_client: 'redis.StrictRedis[bytes]', key_format: str, key: str) -> Optional[Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    if len(key) > MAX_KEY_LENGTH:
        raise ZulipRedisKeyTooLongError(f'Requested key too long in get_dict_from_redis: {key}')
    validate_key_fits_format(key, key_format)
    data = redis_client.get(key)
    if data is None:
        return None
    return orjson.loads(data)

def validate_key_fits_format(key: str, key_format: str) -> None:
    if False:
        print('Hello World!')
    assert '{token}' in key_format
    regex = key_format.format(token='[a-zA-Z0-9]+')
    if not re.fullmatch(regex, key):
        raise ZulipRedisKeyOfWrongFormatError(f'{key} does not match format {key_format}')