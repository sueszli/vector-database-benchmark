import ray
from ray import serve
ray.init(address='auto')

@serve.deployment
def f():
    if False:
        i = 10
        return i + 15
    return 'foobar'
app = f.bind()