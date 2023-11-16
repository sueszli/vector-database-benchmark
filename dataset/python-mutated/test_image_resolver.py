import json
from localstack.aws.api.lambda_ import Runtime
from localstack.services.lambda_.invocation.docker_runtime_executor import RuntimeImageResolver

def test_custom_pattern_mapping():
    if False:
        return 10
    resolver = RuntimeImageResolver()
    resolved_image = resolver._resolve(Runtime.python3_9, custom_image_mapping='custom/_<runtime>_:new')
    assert resolved_image == 'custom/_python3.9_:new'

def test_custom_json_mapping():
    if False:
        return 10
    resolver = RuntimeImageResolver()
    resolved_image = resolver._resolve(Runtime.python3_9, custom_image_mapping=json.dumps({Runtime.python3_9: 'custom/py.thon.3:9'}))
    assert resolved_image == 'custom/py.thon.3:9'

def test_custom_json_mapping_fallback():
    if False:
        while True:
            i = 10
    resolver = RuntimeImageResolver()
    resolved_image = resolver._resolve(Runtime.python3_8, custom_image_mapping=json.dumps({Runtime.python3_9: 'custom/py.thon.3:9'}))
    assert resolved_image is not None
    assert resolved_image != 'custom/py.thon.3:9'
    assert 'custom' not in resolved_image

def test_default_mapping():
    if False:
        return 10
    resolver = RuntimeImageResolver()
    resolved_image = resolver._resolve(Runtime.python3_9)
    assert 'custom' not in resolved_image

def test_custom_default_mapping():
    if False:
        print('Hello World!')

    def custom_default(a):
        if False:
            for i in range(10):
                print('nop')
        return f'custom-{a}'
    resolver = RuntimeImageResolver(default_resolve_fn=custom_default)
    resolved_image = resolver._resolve(Runtime.python3_9)
    assert resolved_image == 'custom-python3.9'