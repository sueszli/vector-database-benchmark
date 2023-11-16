"""
A dummy ray driver script that executes in subprocess. Prints namespace
from ray's runtime context for job submission API testing.
"""
import ray

def run():
    if False:
        while True:
            i = 10
    ray.init()

    @ray.remote
    def foo():
        if False:
            return 10
        return 'bar'
    ray.get(foo.remote())
    print(ray.get_runtime_context().namespace)
if __name__ == '__main__':
    run()