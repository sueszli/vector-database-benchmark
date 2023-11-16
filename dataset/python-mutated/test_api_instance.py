from unittest import mock
import pytest
from ninja import NinjaAPI, Router
from ninja.errors import ConfigError
api = NinjaAPI()
router = Router()

@api.get('/global')
def global_op(request):
    if False:
        print('Hello World!')
    pass

@router.get('/router')
def router_op(request):
    if False:
        i = 10
        return i + 15
    pass
api.add_router('/', router)

def test_api_instance():
    if False:
        print('Hello World!')
    assert len(api._routers) == 2
    for (_path, rtr) in api._routers:
        for path_ops in rtr.path_operations.values():
            for op in path_ops.operations:
                assert op.api is api

def test_reuse_router_error():
    if False:
        for i in range(10):
            print('nop')
    test_api = NinjaAPI()
    test_router = Router()
    test_api.add_router('/', test_router)
    match = "Router@'/another-path' has already been attached to API NinjaAPI:1.0.0"
    with pytest.raises(ConfigError, match=match):
        with mock.patch('ninja.main._imported_while_running_in_debug_server', False):
            test_api.add_router('/another-path', test_router)
    with mock.patch('ninja.main._imported_while_running_in_debug_server', True):
        test_api.add_router('/another-path', test_router)