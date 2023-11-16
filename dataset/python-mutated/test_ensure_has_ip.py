from sentry.event_manager import EventManager

def validate_and_normalize(report, client_ip=None):
    if False:
        return 10
    manager = EventManager(report, client_ip=client_ip)
    manager.normalize()
    return manager.get_data()

def test_with_remote_addr():
    if False:
        return 10
    inp = {'request': {'url': 'http://example.com/', 'env': {'REMOTE_ADDR': '192.168.0.1'}}}
    out = validate_and_normalize(inp, client_ip='127.0.0.1')
    assert out['request']['env']['REMOTE_ADDR'] == '192.168.0.1'

def test_with_user_ip():
    if False:
        print('Hello World!')
    inp = {'user': {'ip_address': '192.168.0.1'}}
    out = validate_and_normalize(inp, client_ip='127.0.0.1')
    assert out['user']['ip_address'] == '192.168.0.1'

def test_with_user_auto_ip():
    if False:
        while True:
            i = 10
    inp = {'user': {'ip_address': '{{auto}}'}}
    out = validate_and_normalize(inp, client_ip='127.0.0.1')
    assert out['user']['ip_address'] == '127.0.0.1'
    inp = {'user': {'ip_address': '{{auto}}'}}
    out = validate_and_normalize(inp, client_ip='127.0.0.1')
    assert out['user']['ip_address'] == '127.0.0.1'

def test_without_ip_values():
    if False:
        while True:
            i = 10
    inp = {'platform': 'javascript', 'user': {}, 'request': {'url': 'http://example.com/', 'env': {}}}
    out = validate_and_normalize(inp, client_ip='127.0.0.1')
    assert out['user']['ip_address'] == '127.0.0.1'

def test_without_any_values():
    if False:
        while True:
            i = 10
    inp = {'platform': 'javascript'}
    out = validate_and_normalize(inp, client_ip='127.0.0.1')
    assert out['user']['ip_address'] == '127.0.0.1'

def test_with_http_auto_ip():
    if False:
        while True:
            i = 10
    inp = {'request': {'url': 'http://example.com/', 'env': {'REMOTE_ADDR': '{{auto}}'}}}
    out = validate_and_normalize(inp, client_ip='127.0.0.1')
    assert out['request']['env']['REMOTE_ADDR'] == '127.0.0.1'

def test_with_all_auto_ip():
    if False:
        for i in range(10):
            print('nop')
    inp = {'user': {'ip_address': '{{auto}}'}, 'request': {'url': 'http://example.com/', 'env': {'REMOTE_ADDR': '{{auto}}'}}}
    out = validate_and_normalize(inp, client_ip='127.0.0.1')
    assert out['request']['env']['REMOTE_ADDR'] == '127.0.0.1'
    assert out['user']['ip_address'] == '127.0.0.1'