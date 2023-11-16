from random import randint
from litestar import Controller, MediaType, get
from litestar.di import Provide
from litestar.status_codes import HTTP_200_OK
from litestar.testing import create_test_client

def test_inter_dependencies() -> None:
    if False:
        print('Hello World!')

    async def top_dependency(query_param: int) -> int:
        return query_param

    async def mid_level_dependency() -> int:
        return 5

    async def local_dependency(path_param: int, mid_level: int, top_level: int) -> int:
        return path_param + mid_level + top_level

    class MyController(Controller):
        path = '/test'
        dependencies = {'mid_level': Provide(mid_level_dependency)}

        @get(path='/{path_param:int}', dependencies={'summed': Provide(local_dependency)}, media_type=MediaType.TEXT)
        def test_function(self, summed: int) -> str:
            if False:
                while True:
                    i = 10
            return str(summed)
    with create_test_client(MyController, dependencies={'top_level': Provide(top_dependency)}) as client:
        response = client.get('/test/5?query_param=5')
        assert response.text == '15'

def test_inter_dependencies_on_same_app_level() -> None:
    if False:
        return 10

    async def first_dependency() -> int:
        return randint(1, 10)

    async def second_dependency(injected_integer: int) -> bool:
        return injected_integer % 2 == 0

    @get('/true-or-false')
    def true_or_false_handler(injected_bool: bool) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'its true!' if injected_bool else 'nope, its false...'
    with create_test_client(true_or_false_handler, dependencies={'injected_integer': Provide(first_dependency), 'injected_bool': Provide(second_dependency)}) as client:
        response = client.get('/true-or-false')
        assert response.status_code == HTTP_200_OK