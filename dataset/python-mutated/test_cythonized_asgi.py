import sys
import time
import pytest
import falcon
from falcon import testing
import falcon.asgi
from falcon.util import is_python_func
try:
    import pyximport
    pyximport.install()
except ImportError:
    pyximport = None
if pyximport:
    from . import _cythonized
    _CYTHON_FUNC_TEST_TYPES = [_cythonized.nop_method, _cythonized.nop_method_async, _cythonized.NOPClass.nop_method, _cythonized.NOPClass.nop_method_async, _cythonized.NOPClass().nop_method, _cythonized.NOPClass().nop_method_async]
else:
    _CYTHON_FUNC_TEST_TYPES = []
from _util import disable_asgi_non_coroutine_wrapping
CYTHON_COROUTINE_HINT = sys.version_info >= (3, 10)

@pytest.fixture
def client():
    if False:
        i = 10
        return i + 15
    return testing.TestClient(falcon.asgi.App())

def nop_method(self):
    if False:
        for i in range(10):
            print('nop')
    pass

async def nop_method_async(self):
    pass

class NOPClass:

    def nop_method(self):
        if False:
            i = 10
            return i + 15
        pass

    async def nop_method_async(self):
        pass

@pytest.mark.skipif(not pyximport, reason='Cython not installed')
@pytest.mark.parametrize('func', _CYTHON_FUNC_TEST_TYPES)
def test_is_cython_func(func):
    if False:
        i = 10
        return i + 15
    assert not is_python_func(func)

@pytest.mark.parametrize('func', [nop_method, nop_method_async, NOPClass.nop_method, NOPClass.nop_method_async, NOPClass().nop_method, NOPClass().nop_method_async])
def test_not_cython_func(func):
    if False:
        while True:
            i = 10
    assert is_python_func(func)

@pytest.mark.skipif(not pyximport, reason='Cython not installed')
def test_jsonchema_validator(client):
    if False:
        for i in range(10):
            print('nop')
    with disable_asgi_non_coroutine_wrapping():
        if CYTHON_COROUTINE_HINT:
            client.app.add_route('/', _cythonized.TestResourceWithValidationNoHint())
        else:
            with pytest.raises(TypeError):
                client.app.add_route('/wowsuchfail', _cythonized.TestResourceWithValidationNoHint())
            client.app.add_route('/', _cythonized.TestResourceWithValidation())
    client.simulate_get()

@pytest.mark.skipif(not pyximport, reason='Cython not installed')
def test_scheduled_jobs(client):
    if False:
        for i in range(10):
            print('nop')
    resource = _cythonized.TestResourceWithScheduledJobs()
    client.app.add_route('/', resource)
    client.simulate_get()
    time.sleep(0.5)
    assert resource.counter['backround:on_get:async'] == 2
    assert resource.counter['backround:on_get:sync'] == 40

@pytest.mark.skipif(not pyximport, reason='Cython not installed')
def test_scheduled_jobs_type_error(client):
    if False:
        print('Hello World!')
    client.app.add_route('/wowsuchfail', _cythonized.TestResourceWithScheduledJobsAsyncRequired())
    with pytest.raises(TypeError):
        client.simulate_get('/wowsuchfail')

@pytest.mark.skipif(not pyximport, reason='Cython not installed')
def test_hooks(client):
    if False:
        while True:
            i = 10
    with disable_asgi_non_coroutine_wrapping():
        if CYTHON_COROUTINE_HINT:
            client.app.add_route('/', _cythonized.TestResourceWithHooksNoHint())
        else:
            with pytest.raises(TypeError):
                client.app.add_route('/', _cythonized.TestResourceWithHooksNoHint())
            client.app.add_route('/', _cythonized.TestResourceWithHooks())
    result = client.simulate_get()
    assert result.headers['x-answer'] == '42'
    assert result.json == {'answer': 42}