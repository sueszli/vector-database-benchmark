import os
import sys
import time
import psutil
import pytest
import ray
from ray._private import ray_constants
from ray._private.test_utils import RayTestTimeoutException, get_error_message, init_error_pubsub, object_memory_usage, run_string_as_driver, run_string_as_driver_nonblocking, wait_for_condition

@pytest.mark.parametrize('call_ray_start', ['ray start --head --num-cpus=1 --min-worker-port=0 --max-worker-port=0 --port 0'], indirect=True)
def test_cleanup_on_driver_exit(call_ray_start):
    if False:
        i = 10
        return i + 15
    address = call_ray_start
    ray.init(address=address)
    driver_script = '\nimport time\nimport ray\nimport numpy as np\nfrom ray._private.test_utils import object_memory_usage\nimport os\n\n\nray.init(address="{}")\nobject_refs = [ray.put(np.zeros(200 * 1024, dtype=np.uint8))\n              for i in range(1000)]\nstart_time = time.time()\nwhile time.time() - start_time < 30:\n    if object_memory_usage() > 0:\n        break\nelse:\n    raise Exception("Objects did not appear in object table.")\n\n@ray.remote\ndef f():\n    time.sleep(1)\n\nprint("success")\n# Submit some tasks without waiting for them to finish. Their workers should\n# still get cleaned up eventually, even if they get started after the driver\n# exits.\n[f.remote() for _ in range(10)]\n'.format(address)
    out = run_string_as_driver(driver_script)
    assert 'success' in out
    start_time = time.time()
    while time.time() - start_time < 30:
        if object_memory_usage() == 0:
            break
    else:
        raise Exception('Objects were not all removed from object table.')

    def all_workers_exited():
        if False:
            for i in range(10):
                print('nop')
        result = True
        print('list of idle workers:')
        for proc in psutil.process_iter():
            if ray_constants.WORKER_PROCESS_TYPE_IDLE_WORKER in proc.name():
                print(f'{proc}')
                result = False
        return result
    wait_for_condition(all_workers_exited, timeout=15, retry_interval_ms=1000)

def test_error_isolation(call_ray_start):
    if False:
        for i in range(10):
            print('nop')
    address = call_ray_start
    ray.init(address=address)
    subscribers = [init_error_pubsub() for _ in range(3)]
    errors = get_error_message(subscribers[0], 1, timeout=2)
    assert len(errors) == 0
    error_string1 = 'error_string1'
    error_string2 = 'error_string2'

    @ray.remote
    def f():
        if False:
            print('Hello World!')
        raise Exception(error_string1)
    with pytest.raises(Exception):
        ray.get(f.remote())
    errors = get_error_message(subscribers[1], 1)
    assert len(errors) == 1
    assert error_string1 in errors[0]['error_message']
    driver_script = '\nimport ray\nimport time\nfrom ray._private.test_utils import init_error_pubsub, get_error_message\n\nray.init(address="{}")\nsubscribers = [init_error_pubsub() for _ in range(2)]\ntime.sleep(1)\nerrors = get_error_message(subscribers[0], 1, timeout=2)\nassert len(errors) == 0\n\n@ray.remote\ndef f():\n    raise Exception("{}")\n\ntry:\n    ray.get(f.remote())\nexcept Exception as e:\n    pass\n\nerrors = get_error_message(subscribers[1], 1)\nassert len(errors) == 1\n\nassert "{}" in errors[0]["error_message"]\n\nprint("success")\n'.format(address, error_string2, error_string2)
    out = run_string_as_driver(driver_script)
    assert 'success' in out
    errors = get_error_message(subscribers[2], 1)
    assert len(errors) == 1

def test_remote_function_isolation(call_ray_start):
    if False:
        return 10
    address = call_ray_start
    ray.init(address=address)
    driver_script = '\nimport ray\nimport time\nray.init(address="{}")\n@ray.remote\ndef f():\n    return 3\n@ray.remote\ndef g(x, y):\n    return 4\nfor _ in range(10000):\n    result = ray.get([f.remote(), g.remote(0, 0)])\n    assert result == [3, 4]\nprint("success")\n'.format(address)
    out = run_string_as_driver(driver_script)

    @ray.remote
    def f():
        if False:
            i = 10
            return i + 15
        return 1

    @ray.remote
    def g(x):
        if False:
            for i in range(10):
                print('nop')
        return 2
    for _ in range(10000):
        result = ray.get([f.remote(), g.remote(0)])
        assert result == [1, 2]
    assert 'success' in out

def test_driver_exiting_quickly(call_ray_start):
    if False:
        while True:
            i = 10
    address = call_ray_start
    ray.init(address=address)
    driver_script1 = '\nimport ray\nray.init(address="{}")\n@ray.remote\nclass Foo:\n    def __init__(self):\n        pass\nFoo.remote()\nprint("success")\n'.format(address)
    driver_script2 = '\nimport ray\nray.init(address="{}")\n@ray.remote\ndef f():\n    return 1\nf.remote()\nprint("success")\n'.format(address)
    for _ in range(3):
        out = run_string_as_driver(driver_script1)
        assert 'success' in out
        out = run_string_as_driver(driver_script2)
        assert 'success' in out

def test_drivers_named_actors(call_ray_start):
    if False:
        while True:
            i = 10
    address = call_ray_start
    ray.init(address=address, namespace='test')
    driver_script1 = '\nimport ray\nimport time\nray.init(address="{}", namespace="test")\n@ray.remote\nclass Counter:\n    def __init__(self):\n        self.count = 0\n    def increment(self):\n        self.count += 1\n        return self.count\ncounter = Counter.options(name="Counter").remote()\ntime.sleep(100)\n'.format(address)
    driver_script2 = '\nimport ray\nimport time\nray.init(address="{}", namespace="test")\nwhile True:\n    try:\n        counter = ray.get_actor("Counter")\n        break\n    except ValueError:\n        time.sleep(1)\nassert ray.get(counter.increment.remote()) == {}\nprint("success")\n'.format(address, '{}')
    process_handle = run_string_as_driver_nonblocking(driver_script1)
    for i in range(3):
        driver_script = driver_script2.format(i + 1)
        out = run_string_as_driver(driver_script)
        assert 'success' in out
    process_handle.kill()

def test_receive_late_worker_logs():
    if False:
        while True:
            i = 10
    log_message = 'some helpful debugging message' + ray_constants.TESTING_NEVER_DEDUP_TOKEN
    driver_script = '\nimport ray\nimport random\nimport time\n\nlog_message = "{}"\n\n@ray.remote\nclass Actor:\n    def log(self):\n        print(log_message)\n\n@ray.remote\ndef f():\n    print(log_message)\n\nray.init(num_cpus=2)\n\na = Actor.remote()\nray.get([a.log.remote(), f.remote()])\nray.get([a.log.remote(), f.remote()])\n'.format(log_message)
    for _ in range(2):
        out = run_string_as_driver(driver_script)
        assert out.count(log_message) == 4

@pytest.mark.parametrize('call_ray_start', ['ray start --head --num-cpus=1 --num-gpus=1 ' + '--min-worker-port=0 --max-worker-port=0 --port 0'], indirect=True)
def test_drivers_release_resources(call_ray_start):
    if False:
        while True:
            i = 10
    address = call_ray_start
    driver_script1 = '\nimport time\nimport ray\n\nray.init(address="{}")\n\n@ray.remote\ndef f(duration):\n    time.sleep(duration)\n\n@ray.remote(num_gpus=1)\ndef g(duration):\n    time.sleep(duration)\n\n@ray.remote(num_gpus=1)\nclass Foo:\n    def __init__(self):\n        pass\n\n# Make sure some resources are available for us to run tasks.\nray.get(f.remote(0))\nray.get(g.remote(0))\n\n# Start a bunch of actors and tasks that use resources. These should all be\n# cleaned up when this driver exits.\nfoos = [Foo.remote() for _ in range(100)]\n[f.remote(10 ** 6) for _ in range(100)]\n\nprint("success")\n'.format(address)
    driver_script2 = driver_script1 + 'import sys\nsys.stdout.flush()\ntime.sleep(10 ** 6)\n'

    def wait_for_success_output(process_handle, timeout=10):
        if False:
            while True:
                i = 10
        start_time = time.time()
        while time.time() - start_time < timeout:
            output_line = ray._private.utils.decode(process_handle.stdout.readline()).strip()
            print(output_line)
            if output_line == 'success':
                return
            time.sleep(1)
        raise RayTestTimeoutException('Timed out waiting for process to print success.')
    for _ in range(5):
        out = run_string_as_driver(driver_script1)
        assert 'success' in out
        process_handle = run_string_as_driver_nonblocking(driver_script2)
        wait_for_success_output(process_handle)
        process_handle.kill()
if __name__ == '__main__':
    import pytest
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))