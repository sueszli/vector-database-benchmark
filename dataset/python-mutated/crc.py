from binascii import crc_hqx
from redis.typing import EncodedT
REDIS_CLUSTER_HASH_SLOTS = 16384
__all__ = ['key_slot', 'REDIS_CLUSTER_HASH_SLOTS']

def key_slot(key: EncodedT, bucket: int=REDIS_CLUSTER_HASH_SLOTS) -> int:
    if False:
        return 10
    'Calculate key slot for a given key.\n    See Keys distribution model in https://redis.io/topics/cluster-spec\n    :param key - bytes\n    :param bucket - int\n    '
    start = key.find(b'{')
    if start > -1:
        end = key.find(b'}', start + 1)
        if end > -1 and end != start + 1:
            key = key[start + 1:end]
    return crc_hqx(key, 0) % bucket