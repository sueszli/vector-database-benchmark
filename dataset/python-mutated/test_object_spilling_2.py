import os
import platform
import random
import subprocess
import sys
import tempfile
import numpy as np
import pytest
import ray
from ray._private.test_utils import run_string_as_driver, wait_for_condition
from ray.tests.test_object_spilling import assert_no_thrashing, is_dir_empty
from ray._private.external_storage import FileSystemStorage, ExternalStorageRayStorageImpl

def test_delete_objects(object_spilling_config, shutdown_only):
    if False:
        print('Hello World!')
    (object_spilling_config, temp_folder) = object_spilling_config
    address = ray.init(object_store_memory=75 * 1024 * 1024, _system_config={'max_io_workers': 1, 'min_spilling_size': 0, 'automatic_object_spilling_enabled': True, 'object_store_full_delay_ms': 100, 'object_spilling_config': object_spilling_config})
    arr = np.random.rand(1024 * 1024)
    replay_buffer = []
    for _ in range(80):
        ref = None
        while ref is None:
            ref = ray.put(arr)
            replay_buffer.append(ref)
    print('-----------------------------------')
    del replay_buffer
    del ref
    wait_for_condition(lambda : is_dir_empty(temp_folder))
    assert_no_thrashing(address['address'])

def test_delete_objects_delete_while_creating(object_spilling_config, shutdown_only):
    if False:
        return 10
    (object_spilling_config, temp_folder) = object_spilling_config
    address = ray.init(object_store_memory=75 * 1024 * 1024, _system_config={'max_io_workers': 4, 'min_spilling_size': 0, 'automatic_object_spilling_enabled': True, 'object_store_full_delay_ms': 100, 'object_spilling_config': object_spilling_config})
    arr = np.random.rand(1024 * 1024)
    replay_buffer = []
    for _ in range(80):
        ref = None
        while ref is None:
            ref = ray.put(arr)
            replay_buffer.append(ref)
        if random.randint(0, 9) < 6:
            replay_buffer.pop()
    for _ in range(200):
        ref = random.choice(replay_buffer)
        sample = ray.get(ref, timeout=None)
        assert np.array_equal(sample, arr)
    del replay_buffer
    del ref
    wait_for_condition(lambda : is_dir_empty(temp_folder))
    assert_no_thrashing(address['address'])

@pytest.mark.skipif(platform.system() in ['Windows'], reason='Failing on Windows.')
def test_delete_objects_on_worker_failure(object_spilling_config, shutdown_only):
    if False:
        while True:
            i = 10
    (object_spilling_config, temp_folder) = object_spilling_config
    address = ray.init(object_store_memory=75 * 1024 * 1024, _system_config={'max_io_workers': 4, 'automatic_object_spilling_enabled': True, 'object_store_full_delay_ms': 100, 'object_spilling_config': object_spilling_config, 'min_spilling_size': 0})
    arr = np.random.rand(1024 * 1024)

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                return 10
            self.replay_buffer = []

        def get_pid(self):
            if False:
                for i in range(10):
                    print('nop')
            return os.getpid()

        def create_objects(self):
            if False:
                for i in range(10):
                    print('nop')
            for _ in range(80):
                ref = None
                while ref is None:
                    ref = ray.put(arr)
                    self.replay_buffer.append(ref)
                if random.randint(0, 9) < 6:
                    self.replay_buffer.pop()
            for _ in range(200):
                ref = random.choice(self.replay_buffer)
                sample = ray.get(ref, timeout=None)
                assert np.array_equal(sample, arr)
    a = Actor.remote()
    actor_pid = ray.get(a.get_pid.remote())
    ray.get(a.create_objects.remote())
    os.kill(actor_pid, 9)

    def wait_until_actor_dead():
        if False:
            return 10
        try:
            ray.get(a.get_pid.remote())
        except ray.exceptions.RayActorError:
            return True
        return False
    wait_for_condition(wait_until_actor_dead)
    wait_for_condition(lambda : is_dir_empty(temp_folder))
    assert_no_thrashing(address['address'])

@pytest.mark.skipif(platform.system() in ['Windows'], reason='Failing on Windows.')
def test_delete_file_non_exists(shutdown_only, tmp_path):
    if False:
        i = 10
        return i + 15
    ray.init(storage=str(tmp_path))

    def create_spilled_files(num_files):
        if False:
            i = 10
            return i + 15
        spilled_files = []
        uris = []
        for _ in range(3):
            (fd, path) = tempfile.mkstemp()
            with os.fdopen(fd, 'w') as tmp:
                tmp.write('stuff')
            spilled_files.append(path)
            uris.append((path + '?offset=0&size=10').encode('ascii'))
        return (spilled_files, uris)
    for storage in [ExternalStorageRayStorageImpl('session'), FileSystemStorage('/tmp')]:
        (spilled_files, uris) = create_spilled_files(3)
        storage.delete_spilled_objects(uris)
        for file in spilled_files:
            assert not os.path.exists(file)
        (spilled_files1, uris1) = create_spilled_files(3)
        spilled_files += spilled_files1
        uris += uris1
        storage.delete_spilled_objects(uris)
        for file in spilled_files:
            assert not os.path.exists(file)

@pytest.mark.skipif(platform.system() in ['Windows'], reason='Failing on Windows and MacOS.')
def test_delete_objects_multi_node(multi_node_object_spilling_config, ray_start_cluster):
    if False:
        while True:
            i = 10
    (object_spilling_config, temp_folder) = multi_node_object_spilling_config
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1, object_store_memory=75 * 1024 * 1024, _system_config={'max_io_workers': 2, 'min_spilling_size': 20 * 1024 * 1024, 'automatic_object_spilling_enabled': True, 'object_store_full_delay_ms': 100, 'object_spilling_config': object_spilling_config})
    ray.init(address=cluster.address)
    for _ in range(2):
        cluster.add_node(num_cpus=1, object_store_memory=75 * 1024 * 1024)
    cluster.wait_for_nodes()
    arr = np.random.rand(1024 * 1024)

    @ray.remote(num_cpus=1)
    class Actor:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.replay_buffer = []

        def ping(self):
            if False:
                return 10
            return

        def create_objects(self):
            if False:
                while True:
                    i = 10
            for _ in range(80):
                ref = None
                while ref is None:
                    ref = ray.put(arr)
                    self.replay_buffer.append(ref)
                if random.randint(0, 9) < 6:
                    self.replay_buffer.pop()
            for _ in range(50):
                ref = random.choice(self.replay_buffer)
                sample = ray.get(ref, timeout=10)
                assert np.array_equal(sample, arr)
    actors = [Actor.remote() for _ in range(3)]
    ray.get([actor.create_objects.remote() for actor in actors])

    def wait_until_actor_dead(actor):
        if False:
            i = 10
            return i + 15
        try:
            ray.get(actor.ping.remote())
        except ray.exceptions.RayActorError:
            return True
        return False
    for actor in actors:
        ray.kill(actor)
        wait_for_condition(lambda : wait_until_actor_dead(actor))
    wait_for_condition(lambda : is_dir_empty(temp_folder))
    assert_no_thrashing(cluster.address)

def test_fusion_objects(fs_only_object_spilling_config, shutdown_only):
    if False:
        print('Hello World!')
    (object_spilling_config, temp_folder) = fs_only_object_spilling_config
    min_spilling_size = 10 * 1024 * 1024
    address = ray.init(object_store_memory=75 * 1024 * 1024, _system_config={'max_io_workers': 3, 'automatic_object_spilling_enabled': True, 'object_store_full_delay_ms': 100, 'object_spilling_config': object_spilling_config, 'min_spilling_size': min_spilling_size})
    replay_buffer = []
    solution_buffer = []
    buffer_length = 100
    for _ in range(buffer_length):
        ref = None
        while ref is None:
            multiplier = random.choice([1, 2, 3])
            arr = np.random.rand(multiplier * 1024 * 1024)
            ref = ray.put(arr)
            replay_buffer.append(ref)
            solution_buffer.append(arr)
    print('-----------------------------------')
    for _ in range(1000):
        index = random.choice(list(range(buffer_length)))
        ref = replay_buffer[index]
        solution = solution_buffer[index]
        sample = ray.get(ref, timeout=None)
        assert np.array_equal(sample, solution)
    is_test_passing = False
    temp_folder = temp_folder / ray._private.ray_constants.DEFAULT_OBJECT_PREFIX
    for path in temp_folder.iterdir():
        file_size = path.stat().st_size
        if file_size >= min_spilling_size:
            is_test_passing = True
    assert is_test_passing
    assert_no_thrashing(address['address'])

def test_release_resource(object_spilling_config, shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    (object_spilling_config, temp_folder) = object_spilling_config
    address = ray.init(num_cpus=1, object_store_memory=75 * 1024 * 1024, _system_config={'max_io_workers': 1, 'automatic_object_spilling_enabled': True, 'object_spilling_config': object_spilling_config})
    plasma_obj = ray.put(np.ones(50 * 1024 * 1024, dtype=np.uint8))
    for _ in range(5):
        ray.put(np.ones(50 * 1024 * 1024, dtype=np.uint8))

    @ray.remote
    def sneaky_task_tries_to_steal_released_resources():
        if False:
            i = 10
            return i + 15
        print('resources were released!')

    @ray.remote
    def f(dep):
        if False:
            return 10
        while True:
            try:
                ray.get(dep[0], timeout=0.001)
            except ray.exceptions.GetTimeoutError:
                pass
    done = f.remote([plasma_obj])
    canary = sneaky_task_tries_to_steal_released_resources.remote()
    (ready, _) = ray.wait([canary], timeout=2)
    assert not ready
    assert_no_thrashing(address['address'])

def test_spill_objects_on_object_transfer(object_spilling_config, ray_start_cluster_enabled):
    if False:
        i = 10
        return i + 15
    (object_spilling_config, _) = object_spilling_config
    cluster = ray_start_cluster_enabled
    object_size = int(10000000.0)
    num_objects = 10
    num_tasks = 10
    cluster.add_node(num_cpus=0, object_store_memory=2 * num_tasks * num_objects * object_size, _system_config={'max_io_workers': 1, 'automatic_object_spilling_enabled': True, 'object_store_full_delay_ms': 100, 'object_spilling_config': object_spilling_config, 'min_spilling_size': 0})
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    cluster.add_node(num_cpus=1, object_store_memory=1.5 * num_objects * object_size)
    cluster.wait_for_nodes()

    @ray.remote
    def foo(*args):
        if False:
            while True:
                i = 10
        return

    @ray.remote
    def allocate(*args):
        if False:
            while True:
                i = 10
        return np.zeros(object_size, dtype=np.uint8)
    allocated = [allocate.remote() for _ in range(num_objects)]
    ray.get(allocated)
    print('done allocating')
    args = []
    for _ in range(num_tasks):
        task_args = [ray.put(np.zeros(object_size, dtype=np.uint8)) for _ in range(num_objects)]
        args.append(task_args)
    tasks = [foo.remote(*task_args) for task_args in args]
    ray.get(tasks)
    assert_no_thrashing(cluster.address)

@pytest.mark.skipif(platform.system() in ['Windows'], reason='Failing on Windows and Mac.')
def test_file_deleted_when_driver_exits(tmp_path, shutdown_only):
    if False:
        while True:
            i = 10
    temp_folder = tmp_path / 'spill'
    temp_folder.mkdir()
    driver = '\nimport json\nimport os\nimport signal\nimport numpy as np\nimport ray\nray.init(\n    object_store_memory=75 * 1024 * 1024,\n    _system_config={{\n        "max_io_workers": 2,\n        "min_spilling_size": 0,\n        "automatic_object_spilling_enabled": True,\n        "object_store_full_delay_ms": 100,\n        "object_spilling_config": json.dumps({{\n            "type": "filesystem",\n            "params": {{\n                "directory_path": "{temp_dir}"\n            }}\n        }}),\n    }})\narr = np.random.rand(1024 * 1024)  # 8 MB data\nreplay_buffer = []\n# Spill lots of objects\nfor _ in range(30):\n    ref = None\n    while ref is None:\n        ref = ray.put(arr)\n        replay_buffer.append(ref)\n# Send sigterm to itself.\nsignum = {signum}\nsig = None\nif signum == 2:\n    sig = signal.SIGINT\nelif signum == 15:\n    sig = signal.SIGTERM\nos.kill(os.getpid(), sig)\n'
    print('Sending sigint...')
    with pytest.raises(subprocess.CalledProcessError):
        print(run_string_as_driver(driver.format(temp_dir=str(temp_folder), signum=2)))
    wait_for_condition(lambda : is_dir_empty(temp_folder, append_path=''))
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))