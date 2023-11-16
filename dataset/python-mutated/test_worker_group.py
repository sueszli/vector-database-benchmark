import time
import pytest
import ray
import ray._private.ray_constants as ray_constants
from ray.train._internal.worker_group import Worker, WorkerGroup, WorkerMetadata

@pytest.fixture
def ray_start_2_cpus():
    if False:
        for i in range(10):
            print('nop')
    address_info = ray.init(num_cpus=2)
    yield address_info
    ray.shutdown()

@pytest.fixture
def ray_start_2_cpus_and_gpus():
    if False:
        while True:
            i = 10
    address_info = ray.init(num_cpus=2, num_gpus=2)
    yield address_info
    ray.shutdown()

@pytest.fixture
def ray_start_2_cpus_and_neuron_core_accelerator():
    if False:
        i = 10
        return i + 15
    address_info = ray.init(num_cpus=2, resources={ray_constants.NEURON_CORES: 2})
    yield address_info
    ray.shutdown()

def test_worker_creation(ray_start_2_cpus):
    if False:
        for i in range(10):
            print('nop')
    assert ray.available_resources()['CPU'] == 2
    wg = WorkerGroup(num_workers=2)
    assert len(wg.workers) == 2
    time.sleep(1)
    assert 'CPU' not in ray.available_resources()
    wg.shutdown()

def test_worker_creation_num_cpus(ray_start_2_cpus):
    if False:
        while True:
            i = 10
    assert ray.available_resources()['CPU'] == 2
    wg = WorkerGroup(num_cpus_per_worker=2)
    time.sleep(1)
    assert len(wg.workers) == 1
    assert 'CPU' not in ray.available_resources()
    wg.shutdown()

def test_worker_shutdown(ray_start_2_cpus):
    if False:
        i = 10
        return i + 15
    assert ray.available_resources()['CPU'] == 2
    wg = WorkerGroup(num_workers=2)
    time.sleep(1)
    assert 'CPU' not in ray.available_resources()
    assert len(ray._private.state.actors()) == 2
    wg.shutdown()
    time.sleep(1)
    assert ray.available_resources()['CPU'] == 2
    with pytest.raises(RuntimeError):
        wg.execute(lambda : 1)

def test_worker_restart(ray_start_2_cpus):
    if False:
        while True:
            i = 10
    wg = WorkerGroup(num_workers=2)
    with pytest.raises(RuntimeError):
        wg.start()
    time.sleep(1)
    wg.shutdown(0)
    wg.start()
    wg.execute(lambda : 1)

def test_worker_with_gpu_ids(ray_start_2_cpus_and_gpus):
    if False:
        while True:
            i = 10
    num_gpus = 2
    wg = WorkerGroup(num_workers=2, num_gpus_per_worker=1)
    assert len(wg.workers) == 2
    time.sleep(1)
    assert ray_constants.GPU not in ray.available_resources()
    wg.execute(lambda : 1)
    assert len(wg.workers) == 2
    for w in wg.workers:
        resource_ids = w.metadata.resource_ids
        gpu_ids = resource_ids[ray_constants.GPU]
        for gpu_id in gpu_ids:
            assert gpu_id in [str(i) for i in range(num_gpus)]
        assert len(resource_ids[ray_constants.NEURON_CORES]) == 0

def test_worker_with_neuron_core_accelerator_ids(ray_start_2_cpus_and_neuron_core_accelerator):
    if False:
        return 10
    num_nc = 2
    wg = WorkerGroup(num_workers=2, additional_resources_per_worker={ray_constants.NEURON_CORES: 1})
    assert len(wg.workers) == 2
    time.sleep(1)
    assert ray_constants.NEURON_CORES not in ray.available_resources()
    wg.execute(lambda : 1)
    assert len(wg.workers) == 2
    for w in wg.workers:
        resource_ids = w.metadata.resource_ids
        assert len(resource_ids[ray_constants.GPU]) == 0
        neuron_core_ids = resource_ids[ray_constants.NEURON_CORES]
        for neuron_core_id in neuron_core_ids:
            assert neuron_core_id in [str(i) for i in range(num_nc)]

def test_execute_async(ray_start_2_cpus):
    if False:
        print('Hello World!')
    wg = WorkerGroup(num_workers=2)
    futures = wg.execute_async(lambda : 1)
    assert len(futures) == 2
    outputs = ray.get(futures)
    assert all((o == 1 for o in outputs))

def test_execute(ray_start_2_cpus):
    if False:
        for i in range(10):
            print('nop')
    wg = WorkerGroup(num_workers=2)
    outputs = wg.execute(lambda : 1)
    assert len(outputs) == 2
    assert all((o == 1 for o in outputs))

def test_execute_args(ray_start_2_cpus):
    if False:
        print('Hello World!')
    wg = WorkerGroup(num_workers=2)
    outputs = wg.execute(lambda x: x, 1)
    assert len(outputs) == 2
    assert all((o == 1 for o in outputs))

def test_group_workers_by_ip(ray_start_2_cpus):
    if False:
        print('Hello World!')

    def create_worker_group(ips):
        if False:
            for i in range(10):
                print('nop')
        wg = WorkerGroup(num_workers=2)
        wg.workers = [Worker(actor=None, metadata=WorkerMetadata(node_id='dummy', node_ip=ip, hostname='dummy', resource_ids=None, pid=0)) for ip in ips]
        return wg
    wg = create_worker_group(['2', '3', '1', '4', '2', '1', '3', '3', '4', '2'])
    wg.group_workers_by_ip()
    expected = ['2', '2', '2', '3', '3', '3', '1', '1', '4', '4']
    ips = [w.metadata.node_ip for w in wg.workers]
    assert ips == expected, 'Workers should be grouped by IP and follow the same original order of IPs encountered (2, 3, 1, 4).'
    wg = create_worker_group(['2', '3', '1', '4', '2', '1', '3', '3', '4', '2'])
    wg.group_workers_by_ip(_first_ip='1')
    expected = ['1', '1', '2', '2', '2', '3', '3', '3', '4', '4']
    ips = [w.metadata.node_ip for w in wg.workers]
    assert ips == expected, 'Workers should be grouped by IP, with the first IP being 1.'

def test_execute_single(ray_start_2_cpus):
    if False:
        return 10
    wg = WorkerGroup(num_workers=2)

    def f():
        if False:
            print('Hello World!')
        import os
        os.environ['TEST'] = '1'
    wg.execute_single(1, f)

    def check():
        if False:
            i = 10
            return i + 15
        import os
        return os.environ.get('TEST', '0')
    assert wg.execute(check) == ['0', '1']

def test_bad_resources(ray_start_2_cpus):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        WorkerGroup(num_workers=-1)
    with pytest.raises(ValueError):
        WorkerGroup(num_cpus_per_worker=-1)
    with pytest.raises(ValueError):
        WorkerGroup(num_gpus_per_worker=-1)

def test_placement_group(ray_start_2_cpus):
    if False:
        while True:
            i = 10
    'Tests that workers can be removed and added to a placement group.'
    num_workers = 2
    bundle = {'CPU': 1}
    bundles = [bundle.copy() for _ in range(num_workers)]
    placement_group = ray.util.placement_group(bundles)
    wg = WorkerGroup(num_workers=num_workers, placement_group=placement_group)
    wg.remove_workers([0])
    wg.add_workers(1)
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', '-x', __file__]))