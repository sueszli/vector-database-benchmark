from mitmproxy.proxy import server_hooks

def test_noop():
    if False:
        for i in range(10):
            print('nop')
    assert server_hooks