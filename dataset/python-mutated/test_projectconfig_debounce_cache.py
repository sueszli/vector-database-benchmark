from sentry.relay.projectconfig_debounce_cache.redis import RedisProjectConfigDebounceCache

def test_key_lifecycle():
    if False:
        for i in range(10):
            print('nop')
    cache = RedisProjectConfigDebounceCache()
    kwargs = {'public_key': 'abc', 'project_id': None, 'organization_id': None}
    assert not cache.is_debounced(**kwargs)
    cache.debounce(**kwargs)
    assert cache.is_debounced(**kwargs)
    cache.mark_task_done(**kwargs)
    assert not cache.is_debounced(**kwargs)

def test_split_debounce_lifecycle():
    if False:
        for i in range(10):
            print('nop')
    cache = RedisProjectConfigDebounceCache()
    kwargs = {'public_key': 'abc', 'project_id': None, 'organization_id': None}
    assert not cache.is_debounced(**kwargs)
    cache.debounce(**kwargs)
    assert cache.is_debounced(**kwargs)
    cache.debounce(**kwargs)
    assert cache.is_debounced(**kwargs)
    cache.mark_task_done(**kwargs)
    assert not cache.is_debounced(**kwargs)
    cache.mark_task_done(**kwargs)
    assert not cache.is_debounced(**kwargs)

def test_default_prefix():
    if False:
        for i in range(10):
            print('nop')
    cache = RedisProjectConfigDebounceCache()
    kwargs = {'public_key': 'abc', 'project_id': None, 'organization_id': None}
    cache.debounce(**kwargs)
    expected_key = 'relayconfig-debounce:k:abc'
    redis = cache._get_redis_client(expected_key)
    assert redis.get(expected_key) == b'1'

def test_custom_prefix():
    if False:
        print('Hello World!')
    cache = RedisProjectConfigDebounceCache(key_prefix='hello:world')
    kwargs = {'public_key': 'abc', 'project_id': None, 'organization_id': None}
    cache.debounce(**kwargs)
    expected_key = 'hello:world:k:abc'
    redis = cache._get_redis_client(expected_key)
    assert redis.get(expected_key) == b'1'