from builtins import _test_sink, _test_source
from django.http.request import HttpRequest

def test_untainted_assign(request: HttpRequest):
    if False:
        return 10
    request.GET = {}
    _test_sink(request.GET)

def test_trace_has_no_tito(request: HttpRequest):
    if False:
        for i in range(10):
            print('nop')
    request.GET = _test_source()
    _test_sink(request.GET)

def request_get_flows_to_sink(request: HttpRequest):
    if False:
        while True:
            i = 10
    _test_sink(request.GET)

def test_hop_is_cut_off(request: HttpRequest):
    if False:
        while True:
            i = 10
    request.GET = _test_source()
    request_get_flows_to_sink(request)