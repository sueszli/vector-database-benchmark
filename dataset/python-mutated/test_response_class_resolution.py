from typing import Optional
import pytest
from litestar import Controller, HttpMethod, Litestar, Response, Router, get
from litestar.types import Empty
router_response = type('router_response', (Response,), {})
controller_response = type('controller_response', (Response,), {})
app_response = type('app_response', (Response,), {})
handler_response = type('local_response', (Response,), {})
test_path = '/test'

@pytest.mark.parametrize('layer, expected', [[0, handler_response], [1, controller_response], [2, router_response], [3, app_response], [None, Response]])
def test_response_class_resolution_of_layers(layer: Optional[int], expected: Response) -> None:
    if False:
        while True:
            i = 10

    class MyController(Controller):
        path = test_path

        @get(path='/{path_param:str}')
        def test_method(self) -> None:
            if False:
                return 10
            pass
    MyController.test_method._resolved_response_class = Empty if layer != 0 else expected
    MyController.response_class = None if layer != 1 else expected
    router = Router(path='/users', route_handlers=[MyController], response_class=None if layer != 2 else expected)
    app = Litestar(route_handlers=[router], response_class=None if layer != 3 else expected)
    (route_handler, _) = app.routes[0].route_handler_map[HttpMethod.GET]
    layer_map = {0: route_handler, 1: MyController, 2: router, 3: app}
    component = layer_map.get(layer)
    if component:
        component.response_class = expected
        assert component.response_class is expected
    response_class = route_handler.resolve_response_class()
    assert response_class is expected
    if component:
        component.response_class = None
        assert component.response_class is None

def test_response_class_resolution_overrides() -> None:
    if False:
        print('Hello World!')

    class MyController(Controller):
        path = '/path'
        response_class = controller_response

        @get('/', response_class=handler_response)
        def handler(self) -> None:
            if False:
                print('Hello World!')
            return
    assert MyController.handler.resolve_response_class() is handler_response

def test_response_class_resolution_defaults() -> None:
    if False:
        return 10

    class MyController(Controller):
        path = '/path'

        @get('/')
        def handler(self) -> None:
            if False:
                print('Hello World!')
            return
    assert MyController.handler.resolve_response_class() is Response