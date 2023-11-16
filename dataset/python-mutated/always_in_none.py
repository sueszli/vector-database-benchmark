from builtins import _test_sink, _test_source
from django.http import HttpRequest, HttpResponse

class ComplicatedService:

    def serve_tainted_request(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Valid'

def test(complicated_service: ComplicatedService):
    if False:
        i = 10
        return i + 15
    exception = False
    result = None
    try:
        result = complicated_service.serve_tainted_request()
    except:
        exception = True
    if exception:
        try:
            result = complicated_service.serve_tainted_request()
        except:
            raise
    _test_sink(result)

def test_none_clears_taint():
    if False:
        while True:
            i = 10
    x = _test_source()
    x = None
    _test_sink(x)