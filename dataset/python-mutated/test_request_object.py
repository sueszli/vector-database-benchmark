from robyn.robyn import Request, Url

def test_request_object():
    if False:
        while True:
            i = 10
    url = Url(scheme='https', host='localhost', path='/user')
    request = Request(queries={}, headers={'Content-Type': 'application/json'}, path_params={}, body='', method='GET', url=url, ip_addr=None, identity=None)
    assert request.url.scheme == 'https'
    assert request.url.host == 'localhost'
    assert request.headers['Content-Type'] == 'application/json'
    assert request.method == 'GET'