import os
import signal
import sys
import time
import numpy as np
import pytest
import ray
from ray._private.test_utils import SignalActor, run_string_as_driver_nonblocking
SIGKILL = signal.SIGKILL if sys.platform != 'win32' else signal.SIGTERM

def test_dying_worker_get(ray_start_2_cpus):
    if False:
        i = 10
        return i + 15

    @ray.remote
    def sleep_forever(signal):
        if False:
            print('Hello World!')
        ray.get(signal.send.remote())
        time.sleep(10 ** 6)

    @ray.remote
    def get_worker_pid():
        if False:
            for i in range(10):
                print('nop')
        return os.getpid()
    signal = SignalActor.remote()
    x_id = sleep_forever.remote(signal)
    ray.get(signal.wait.remote())
    worker_pid = ray.get(get_worker_pid.remote())

    @ray.remote
    def f(id_in_a_list):
        if False:
            while True:
                i = 10
        ray.get(id_in_a_list[0])
    result_id = f.remote([x_id])
    time.sleep(1)
    (ready_ids, _) = ray.wait([result_id], timeout=0)
    assert len(ready_ids) == 0
    os.kill(worker_pid, SIGKILL)
    time.sleep(0.1)
    (ready_ids, _) = ray.wait([x_id], timeout=0)
    assert len(ready_ids) == 0
    obj = np.ones(200 * 1024, dtype=np.uint8)
    ray._private.worker.global_worker.put_object(obj, x_id)
    time.sleep(0.1)
    assert ray._private.services.remaining_processes_alive()

def test_dying_driver_get(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')
    address_info = ray_start_regular

    @ray.remote
    def sleep_forever():
        if False:
            return 10
        time.sleep(10 ** 6)
    x_id = sleep_forever.remote()
    driver = '\nimport ray\nray.init("{}")\nray.get(ray.ObjectRef(ray._private.utils.hex_to_binary("{}")))\n'.format(address_info['address'], x_id.hex())
    p = run_string_as_driver_nonblocking(driver)
    time.sleep(1)
    assert p.poll() is None
    p.kill()
    p.wait()
    time.sleep(0.1)
    (ready_ids, _) = ray.wait([x_id], timeout=0)
    assert len(ready_ids) == 0
    obj = np.ones(200 * 1024, dtype=np.uint8)
    ray._private.worker.global_worker.put_object(obj, x_id)
    time.sleep(0.1)
    assert ray._private.services.remaining_processes_alive()

def test_dying_worker_wait(ray_start_2_cpus):
    if False:
        while True:
            i = 10

    @ray.remote
    def sleep_forever():
        if False:
            print('Hello World!')
        time.sleep(10 ** 6)

    @ray.remote
    def get_pid():
        if False:
            for i in range(10):
                print('nop')
        return os.getpid()
    x_id = sleep_forever.remote()
    time.sleep(0.1)
    worker_pid = ray.get(get_pid.remote())

    @ray.remote
    def block_in_wait(object_ref_in_list):
        if False:
            i = 10
            return i + 15
        ray.wait(object_ref_in_list)
    block_in_wait.remote([x_id])
    time.sleep(0.1)
    os.kill(worker_pid, SIGKILL)
    time.sleep(0.1)
    obj = np.ones(200 * 1024, dtype=np.uint8)
    ray._private.worker.global_worker.put_object(obj, x_id)
    time.sleep(0.1)
    assert ray._private.services.remaining_processes_alive()

def test_dying_driver_wait(ray_start_regular):
    if False:
        while True:
            i = 10
    address_info = ray_start_regular

    @ray.remote
    def sleep_forever():
        if False:
            while True:
                i = 10
        time.sleep(10 ** 6)
    x_id = sleep_forever.remote()
    driver = '\nimport ray\nray.init("{}")\nray.wait([ray.ObjectRef(ray._private.utils.hex_to_binary("{}"))])\n'.format(address_info['address'], x_id.hex())
    p = run_string_as_driver_nonblocking(driver)
    time.sleep(1)
    assert p.poll() is None
    p.kill()
    p.wait()
    time.sleep(0.1)
    (ready_ids, _) = ray.wait([x_id], timeout=0)
    assert len(ready_ids) == 0
    obj = np.ones(200 * 1024, dtype=np.uint8)
    ray._private.worker.global_worker.put_object(obj, x_id)
    time.sleep(0.1)
    assert ray._private.services.remaining_processes_alive()
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))