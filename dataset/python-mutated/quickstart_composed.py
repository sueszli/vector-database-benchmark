import requests
import starlette
from typing import Dict
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment
class Adder:

    def __init__(self, increment: int):
        if False:
            return 10
        self.increment = increment

    def add(self, inp: int):
        if False:
            for i in range(10):
                print('nop')
        return self.increment + inp

@serve.deployment
class Combiner:

    def average(self, *inputs) -> float:
        if False:
            i = 10
            return i + 15
        return sum(inputs) / len(inputs)

@serve.deployment
class Ingress:

    def __init__(self, adder1: DeploymentHandle, adder2: DeploymentHandle, combiner: DeploymentHandle):
        if False:
            while True:
                i = 10
        self._adder1 = adder1
        self._adder2 = adder2
        self._combiner = combiner

    async def __call__(self, request: starlette.requests.Request) -> Dict[str, float]:
        input_json = await request.json()
        final_result = await self._combiner.average.remote(self._adder1.add.remote(input_json['val']), self._adder2.add.remote(input_json['val']))
        return {'result': final_result}
app = Ingress.bind(Adder.bind(increment=1), Adder.bind(increment=2), Combiner.bind())
serve.run(app)
print(requests.post('http://localhost:8000/', json={'val': 100.0}).json())