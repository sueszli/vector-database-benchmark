import sys
import time
import ray
num_exceptions = 0

def interceptor(e):
    if False:
        return 10
    global num_exceptions
    num_exceptions += 1

@ray.remote
def f():
    if False:
        i = 10
        return i + 15
    raise ValueError()
if __name__ == '__main__':
    setattr(sys, 'ps1', 'dummy')
    ray.init(num_cpus=1)
    ray._private.worker._unhandled_error_handler = interceptor
    x1 = f.remote()
    start = time.time()
    while time.time() - start < 10:
        if num_exceptions == 1:
            sys.exit(0)
        time.sleep(0.5)
        print('wait for exception', num_exceptions)
    assert False