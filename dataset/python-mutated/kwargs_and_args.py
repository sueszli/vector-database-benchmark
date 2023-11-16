from builtins import _test_source, _test_sink
from typing import Dict, Any

def async_render_to_response(request: str, context: Dict[str, Any]):
    if False:
        while True:
            i = 10
    _test_sink(context)

def async_distillery_render(request, **kwargs: Any):
    if False:
        while True:
            i = 10
    kwargs['request'] = _test_source()
    kwargs['context'] = _test_source()
    kwargs.pop('context')
    async_render_to_response(**kwargs)
    return kwargs

def args_sink(*args):
    if False:
        i = 10
        return i + 15
    _test_sink(args[1])