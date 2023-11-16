from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment(ray_actor_options={'num_cpus': 0.1})
def f(*args):
    if False:
        return 10
    return 'wonderful world'

@serve.deployment(ray_actor_options={'num_cpus': 0.1})
class BasicDriver:

    def __init__(self, h: DeploymentHandle):
        if False:
            while True:
                i = 10
        self._h = h

    async def __call__(self):
        return await self._h.remote()
FNode = f.bind()
DagNode = BasicDriver.bind(FNode)