from enum import Enum
from typing import Dict, List
import starlette.requests
from ray import serve
from ray.serve.handle import DeploymentHandle

class Operation(str, Enum):
    ADDITION = 'ADD'
    MULTIPLICATION = 'MUL'

@serve.deployment(ray_actor_options={'num_cpus': 0.15})
class Router:

    def __init__(self, multiplier: DeploymentHandle, adder: DeploymentHandle):
        if False:
            print('Hello World!')
        self.adder = adder
        self.multiplier = multiplier

    async def route(self, op: Operation, input: int) -> str:
        if op == Operation.ADDITION:
            amount = await self.adder.add.remote(input)
        elif op == Operation.MULTIPLICATION:
            amount = await self.multiplier.multiply.remote(input)
        return f'{amount} pizzas please!'

    async def __call__(self, request: starlette.requests.Request) -> str:
        (op, input) = await request.json()
        return await self.route(op, input)

@serve.deployment(user_config={'factor': 3}, ray_actor_options={'num_cpus': 0.15})
class Multiplier:

    def __init__(self, factor: int):
        if False:
            return 10
        self.factor = factor

    def reconfigure(self, config: Dict):
        if False:
            print('Hello World!')
        self.factor = config.get('factor', -1)

    def multiply(self, input_factor: int) -> int:
        if False:
            print('Hello World!')
        return input_factor * self.factor

@serve.deployment(user_config={'increment': 2}, ray_actor_options={'num_cpus': 0.15})
class Adder:

    def __init__(self, increment: int):
        if False:
            while True:
                i = 10
        self.increment = increment

    def reconfigure(self, config: Dict):
        if False:
            i = 10
            return i + 15
        self.increment = config.get('increment', -1)

    def add(self, input: int) -> int:
        if False:
            while True:
                i = 10
        return input + self.increment

async def json_resolver(request: starlette.requests.Request) -> List:
    return await request.json()
ORIGINAL_INCREMENT = 1
ORIGINAL_FACTOR = 1
multiplier = Multiplier.bind(ORIGINAL_FACTOR)
adder = Adder.bind(ORIGINAL_INCREMENT)
serve_dag = Router.bind(multiplier, adder)