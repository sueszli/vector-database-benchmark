import ray
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse

@ray.remote
def say_hi_task(inp: str):
    if False:
        for i in range(10):
            print('nop')
    return f"Ray task got message: '{inp}'"

@serve.deployment
class SayHi:

    def __call__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'Hi from Serve deployment'

@serve.deployment
class Ingress:

    def __init__(self, say_hi: DeploymentHandle):
        if False:
            print('Hello World!')
        self._say_hi = say_hi

    async def __call__(self):
        response: DeploymentResponse = self._say_hi.remote()
        response_obj_ref: ray.ObjectRef = await response._to_object_ref()
        final_obj_ref: ray.ObjectRef = say_hi_task.remote(response_obj_ref)
        return await final_obj_ref
app = Ingress.bind(SayHi.bind())
handle: DeploymentHandle = serve.run(app)
assert handle.remote().result() == "Ray task got message: 'Hi from Serve deployment'"