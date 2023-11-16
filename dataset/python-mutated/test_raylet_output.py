import glob
import os
import sys
import pytest
import ray
from ray._private.test_utils import wait_for_condition

def enable_export_loglevel(func):
    if False:
        while True:
            i = 10
    os.environ['RAY_BACKEND_LOG_LEVEL'] = 'info'
    return func

@enable_export_loglevel
def test_ray_log_redirected(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')
    session_dir = ray._private.worker._global_node.get_session_dir_path()
    assert os.path.exists(session_dir), 'Session dir not found.'
    raylet_out_path = '{}/logs/raylet.out'.format(session_dir)
    raylet_err_path = '{}/logs/raylet.err'.format(session_dir)

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                print('Hello World!')
            pass

        def get_pid(self):
            if False:
                print('Hello World!')
            return os.getpid()

    def file_exists_and_not_empty(filename):
        if False:
            while True:
                i = 10
        return os.path.exists(filename) and os.path.getsize(filename) > 0
    actor = Actor.remote()
    remote_pid = ray.get(actor.get_pid.remote())
    local_pid = os.getpid()
    wait_for_condition(lambda : all(map(file_exists_and_not_empty, [raylet_out_path, raylet_err_path])))
    core_worker_logs = glob.glob('{}/logs/python-core-worker*{}.log'.format(session_dir, remote_pid))
    driver_log = glob.glob('{}/logs/python-core-driver*{}.log'.format(session_dir, local_pid))
    assert len(core_worker_logs) > 0 and len(driver_log) > 0
    all_worker_logs = core_worker_logs + driver_log
    wait_for_condition(lambda : all(map(file_exists_and_not_empty, all_worker_logs)))
if __name__ == '__main__':
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))