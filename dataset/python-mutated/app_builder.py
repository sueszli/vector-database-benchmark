from typing import Dict
from ray import serve
from ray.serve import Application

@serve.deployment
class HelloWorld:

    def __init__(self, message: str):
        if False:
            for i in range(10):
                print('nop')
        self._message = message
        print('Message:', self._message)

    def __call__(self, request):
        if False:
            for i in range(10):
                print('nop')
        return self._message

def app_builder(args: Dict[str, str]) -> Application:
    if False:
        for i in range(10):
            print('nop')
    return HelloWorld.bind(args['message'])
import requests
serve.run(app_builder({'message': 'Hello bar'}))
resp = requests.get('http://localhost:8000')
assert resp.text == 'Hello bar'
from pydantic import BaseModel
from ray import serve
from ray.serve import Application

class HelloWorldArgs(BaseModel):
    message: str

@serve.deployment
class HelloWorld:

    def __init__(self, message: str):
        if False:
            i = 10
            return i + 15
        self._message = message
        print('Message:', self._message)

    def __call__(self, request):
        if False:
            while True:
                i = 10
        return self._message

def typed_app_builder(args: HelloWorldArgs) -> Application:
    if False:
        print('Hello World!')
    return HelloWorld.bind(args.message)
serve.run(typed_app_builder(HelloWorldArgs(message='Hello baz')))
resp = requests.get('http://localhost:8000')
assert resp.text == 'Hello baz'
from pydantic import BaseModel
from ray.serve import Application

class ComposedArgs(BaseModel):
    model1_uri: str
    model2_uri: str

def composed_app_builder(args: ComposedArgs) -> Application:
    if False:
        return 10
    return IngressDeployment.bind(Model1.bind(args.model1_uri), Model2.bind(args.model2_uri))