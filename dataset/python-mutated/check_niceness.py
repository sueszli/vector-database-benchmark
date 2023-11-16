"""
A dummy ray driver script that executes in subprocess.
Checks that job manager's environment variable is different.
"""
import ray
import os

def run():
    if False:
        while True:
            i = 10
    ray.init()

    @ray.remote
    def foo():
        if False:
            return 10
        print('worker', os.nice(0))
    ray.get(foo.remote())
if __name__ == '__main__':
    print('driver', os.nice(0))
    run()