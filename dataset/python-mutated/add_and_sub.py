from enum import Enum
from typing import List
import starlette.requests
from ray import serve
from ray.serve.deployment_graph import InputNode
from ray.serve.drivers import DAGDriver
from ray.serve.handle import DeploymentHandle

class Operation(str, Enum):
    ADD = 'ADD'
    SUBTRACT = 'SUB'

@serve.deployment(ray_actor_options={'num_cpus': 0.1})
class Add:

    def add(self, input: int) -> int:
        if False:
            print('Hello World!')
        from dir2.library import add_one
        return add_one(input)

@serve.deployment(ray_actor_options={'num_cpus': 0.1})
class Subtract:

    def subtract(self, input: int) -> int:
        if False:
            return 10
        from test_module.test import one
        return input - one()

@serve.deployment(ray_actor_options={'num_cpus': 0.1})
class Router:

    def __init__(self, adder: DeploymentHandle, subtractor: DeploymentHandle):
        if False:
            while True:
                i = 10
        self.adder = adder
        self.subtractor = subtractor

    async def route(self, op: Operation, input: int) -> int:
        if op == Operation.ADD:
            return await self.adder.add.remote(input)
        elif op == Operation.SUBTRACT:
            return await self.subtractor.subtract.remote(input)

async def json_resolver(request: starlette.requests.Request) -> List:
    return await request.json()
with InputNode() as inp:
    (operation, amount_input) = (inp[0], inp[1])
    adder = Add.bind()
    subtractor = Subtract.bind()
    router = Router.bind(adder, subtractor)
    amount = router.route.bind(operation, amount_input)
serve_dag = DAGDriver.bind(amount, http_adapter=json_resolver)