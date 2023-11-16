from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pytest
from fastapi import FastAPI
from pydantic import BaseModel
import ray
from ray._private.pydantic_compat import IS_PYDANTIC_2

@pytest.fixture(scope='session')
def start_ray():
    if False:
        return 10
    ray.init(ignore_reinit_error=True)

def test_serialize_cls(start_ray):
    if False:
        i = 10
        return i + 15

    class User(BaseModel):
        name: str
    ray.get(ray.put(User))

def test_serialize_instance(start_ray):
    if False:
        return 10

    class User(BaseModel):
        name: str
    ray.get(ray.put(User(name='a')))

def test_serialize_imported_cls(start_ray):
    if False:
        return 10
    from pydantic_module import User
    ray.get(ray.put(User))

def test_serialize_imported_instance(start_ray):
    if False:
        for i in range(10):
            print('nop')
    from pydantic_module import user
    ray.get(ray.put(user))

def test_serialize_app_no_route(start_ray):
    if False:
        return 10
    app = FastAPI()
    ray.get(ray.put(app))

def test_serialize_app_no_validation(start_ray):
    if False:
        i = 10
        return i + 15
    app = FastAPI()

    @app.get('/')
    def hello() -> str:
        if False:
            while True:
                i = 10
        return 'hi'
    ray.get(ray.put(app))

def test_serialize_app_primitive_type(start_ray):
    if False:
        return 10
    app = FastAPI()

    @app.get('/')
    def hello(v: str) -> str:
        if False:
            while True:
                i = 10
        return 'hi'
    ray.get(ray.put(app))

def test_serialize_app_pydantic_type_imported(start_ray):
    if False:
        for i in range(10):
            print('nop')
    from pydantic_module import User
    app = FastAPI()

    @app.get('/')
    def hello(v: str, u: User) -> str:
        if False:
            while True:
                i = 10
        return 'hi'
    ray.get(ray.put(app))

def test_serialize_app_pydantic_type_inline(start_ray):
    if False:
        while True:
            i = 10

    class User(BaseModel):
        name: str
    app = FastAPI()

    @app.get('/')
    def hello(v: str, u: User) -> str:
        if False:
            while True:
                i = 10
        return 'hi'
    ray.get(ray.put(app))

def test_serialize_app_imported(start_ray):
    if False:
        for i in range(10):
            print('nop')
    from pydantic_module import app
    ray.get(ray.put(app))

def test_serialize_app_pydantic_type_closure_ref(start_ray):
    if False:
        i = 10
        return i + 15

    class User(BaseModel):
        name: str

    def make():
        if False:
            print('Hello World!')
        app = FastAPI()

        @app.get('/')
        def hello(v: str, u: User) -> str:
            if False:
                return 10
            return 'hi'
        return app
    ray.get(ray.put(make))

def test_serialize_app_pydantic_type_closure_ref_import(start_ray):
    if False:
        return 10
    from pydantic_module import User

    def make():
        if False:
            for i in range(10):
                print('nop')
        app = FastAPI()

        @app.get('/')
        def hello(v: str, u: User) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return 'hi'
        return app
    ray.get(ray.put(make))

def test_serialize_app_pydantic_type_closure(start_ray):
    if False:
        i = 10
        return i + 15

    def make():
        if False:
            while True:
                i = 10

        class User(BaseModel):
            name: str
        app = FastAPI()

        @app.get('/')
        def hello(v: str, u: User) -> str:
            if False:
                while True:
                    i = 10
            return 'hi'
        return app
    ray.get(ray.put(make))

def test_serialize_app_imported_closure(start_ray):
    if False:
        i = 10
        return i + 15
    from pydantic_module import closure
    ray.get(ray.put(closure))

@pytest.mark.skipif(not IS_PYDANTIC_2, reason='Test fails with Pydantic 1.10.12, but succeeds with Pydantic 1.9.2.')
def test_serialize_serve_dataclass(start_ray):
    if False:
        return 10

    @dataclass
    class BackendMetadata:
        is_blocking: bool = True
        autoscaling_config: Optional[Dict[str, Any]] = None

    class BackendConfig(BaseModel):
        internal_metadata: BackendMetadata = BackendMetadata()
    ray.get(ray.put(BackendConfig()))

    @ray.remote
    def consume(f):
        if False:
            print('Hello World!')
        pass
    ray.get(consume.remote(BackendConfig()))

def test_serialize_nested_field(start_ray):
    if False:
        i = 10
        return i + 15

    class B(BaseModel):
        v: List[int]
    B(v=[1])

    @ray.remote
    def func():
        if False:
            for i in range(10):
                print('nop')
        return B(v=[1])
    ray.get(func.remote())
if __name__ == '__main__':
    import os
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))