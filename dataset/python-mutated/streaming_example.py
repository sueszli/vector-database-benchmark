from typing import AsyncGenerator, Generator
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator

@serve.deployment
class Streamer:

    def __call__(self, limit: int) -> Generator[int, None, None]:
        if False:
            while True:
                i = 10
        for i in range(limit):
            yield i

@serve.deployment
class Caller:

    def __init__(self, streamer: DeploymentHandle):
        if False:
            while True:
                i = 10
        self._streamer = streamer.options(stream=True)

    async def __call__(self, limit: int) -> AsyncGenerator[int, None]:
        r: DeploymentResponseGenerator = self._streamer.remote(limit)
        async for i in r:
            yield i
app = Caller.bind(Streamer.bind())
handle: DeploymentHandle = serve.run(app).options(stream=True)
r: DeploymentResponseGenerator = handle.remote(10)
assert list(r) == list(range(10))