from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse

@serve.deployment
class Adder:

    def __init__(self, increment: int):
        if False:
            while True:
                i = 10
        self._increment = increment

    def __call__(self, val: int) -> int:
        if False:
            print('Hello World!')
        return val + self._increment

@serve.deployment
class Multiplier:

    def __init__(self, multiple: int):
        if False:
            return 10
        self._multiple = multiple

    def __call__(self, val: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        return val * self._multiple

@serve.deployment
class Ingress:

    def __init__(self, adder: DeploymentHandle, multiplier: DeploymentHandle):
        if False:
            return 10
        self._adder = adder
        self._multiplier = multiplier

    async def __call__(self, input: int) -> int:
        adder_response: DeploymentResponse = self._adder.remote(input)
        multiplier_response: DeploymentResponse = self._multiplier.remote(adder_response)
        return await multiplier_response
app = Ingress.bind(Adder.bind(increment=1), Multiplier.bind(multiple=2))
handle: DeploymentHandle = serve.run(app)
response = handle.remote(5)
assert response.result() == 12, '(5 + 1) * 2 = 12'