from ray import serve
from ray._private.pydantic_compat import BaseModel

class TypedArgs(BaseModel):
    message: str = 'DEFAULT'

@serve.deployment(ray_actor_options={'num_cpus': 0})
class Echo:

    def __init__(self, message: str):
        if False:
            for i in range(10):
                print('nop')
        print('Echo message:', message)
        self._message = message

    def __call__(self, *args):
        if False:
            print('Hello World!')
        return self._message

def build_echo_app(args):
    if False:
        for i in range(10):
            print('nop')
    return Echo.bind(args.get('message', 'DEFAULT'))

def build_echo_app_typed(args: TypedArgs):
    if False:
        while True:
            i = 10
    return Echo.bind(args.message)