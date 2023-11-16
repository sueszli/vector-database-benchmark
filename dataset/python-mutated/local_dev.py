from starlette.requests import Request
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse

@serve.deployment
class Doubler:

    def double(self, s: str):
        if False:
            while True:
                i = 10
        return s + ' ' + s

@serve.deployment
class HelloDeployment:

    def __init__(self, doubler: DeploymentHandle):
        if False:
            while True:
                i = 10
        self.doubler = doubler

    async def say_hello_twice(self, name: str):
        return await self.doubler.double.remote(f'Hello, {name}!')

    async def __call__(self, request: Request):
        return await self.say_hello_twice(request.query_params['name'])
app = HelloDeployment.bind(Doubler.bind())
handle: DeploymentHandle = serve.run(app)
response: DeploymentResponse = handle.say_hello_twice.remote(name='Ray')
assert response.result() == 'Hello, Ray! Hello, Ray!'