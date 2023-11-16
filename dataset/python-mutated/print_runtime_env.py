"""
A dummy ray driver script that executes in subprocess. Prints runtime_env
from ray's runtime context for job submission API testing.
"""
import ray

def run():
    if False:
        i = 10
        return i + 15
    ray.init()

    @ray.remote
    def foo():
        if False:
            while True:
                i = 10
        return 'bar'
    ray.get(foo.remote())
    print(ray.get_runtime_context().runtime_env)
if __name__ == '__main__':
    run()