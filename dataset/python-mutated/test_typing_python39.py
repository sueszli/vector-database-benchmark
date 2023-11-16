from fastapi import FastAPI
from fastapi.testclient import TestClient
from .utils import needs_py310

@needs_py310
def test_typing():
    if False:
        for i in range(10):
            print('nop')
    types = {list[int]: [1, 2, 3], dict[str, list[int]]: {'a': [1, 2, 3], 'b': [4, 5, 6]}, set[int]: [1, 2, 3], tuple[int, ...]: [1, 2, 3]}
    for (test_type, expect) in types.items():
        app = FastAPI()

        @app.post('/', response_model=test_type)
        def post_endpoint(input: test_type):
            if False:
                print('Hello World!')
            return input
        res = TestClient(app).post('/', json=expect)
        assert res.status_code == 200, res.json()
        assert res.json() == expect