import falcon

def test_asgi():
    if False:
        for i in range(10):
            print('nop')
    assert falcon.ASGI_SUPPORTED