import starlette.requests
import requests
from ray import serve

@serve.deployment
class Counter:

    def __call__(self, request: starlette.requests.Request):
        if False:
            while True:
                i = 10
        return request.query_params
serve.run(Counter.bind())
resp = requests.get('http://localhost:8000?a=b&c=d')
assert resp.json() == {'a': 'b', 'c': 'd'}
import ray
import requests
from fastapi import FastAPI
from ray import serve
app = FastAPI()

@serve.deployment
@serve.ingress(app)
class MyFastAPIDeployment:

    @app.get('/')
    def root(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Hello, world!'
serve.run(MyFastAPIDeployment.bind(), route_prefix='/hello')
resp = requests.get('http://localhost:8000/hello')
assert resp.json() == 'Hello, world!'
import ray
import requests
from fastapi import FastAPI
from ray import serve
app = FastAPI()

@serve.deployment
@serve.ingress(app)
class MyFastAPIDeployment:

    @app.get('/')
    def root(self):
        if False:
            while True:
                i = 10
        return 'Hello, world!'

    @app.post('/{subpath}')
    def root(self, subpath: str):
        if False:
            for i in range(10):
                print('nop')
        return f'Hello from {subpath}!'
serve.run(MyFastAPIDeployment.bind(), route_prefix='/hello')
resp = requests.post('http://localhost:8000/hello/Serve')
assert resp.json() == 'Hello from Serve!'
import ray
import requests
from fastapi import FastAPI
from ray import serve
app = FastAPI()

@app.get('/')
def f():
    if False:
        for i in range(10):
            print('nop')
    return 'Hello from the root!'

@serve.deployment
@serve.ingress(app)
class FastAPIWrapper:
    pass
serve.run(FastAPIWrapper.bind(), route_prefix='/')
resp = requests.get('http://localhost:8000/')
assert resp.json() == 'Hello from the root!'