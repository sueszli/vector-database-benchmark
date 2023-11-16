import sys
import threading
import pytest
import ray
from ray.tests.test_memory_pressure import allocate_memory, Leaker, get_additional_bytes_to_reach_memory_usage_pct, memory_usage_threshold, memory_monitor_refresh_ms

@pytest.fixture
def ray_with_memory_monitor(shutdown_only):
    if False:
        i = 10
        return i + 15
    with ray.init(num_cpus=1, object_store_memory=100 * 1024 * 1024, _system_config={'memory_usage_threshold': memory_usage_threshold, 'memory_monitor_refresh_ms': memory_monitor_refresh_ms, 'metrics_report_interval_ms': 100, 'task_failure_entry_ttl_ms': 2 * 60 * 1000, 'task_oom_retries': 0, 'min_memory_free_bytes': -1}) as addr:
        yield addr

def alloc_mem(bytes):
    if False:
        print('Hello World!')
    chunks = 10
    mem = []
    bytes_per_chunk = bytes // 8 // chunks
    for _ in range(chunks):
        mem.append([0] * bytes_per_chunk)
    return mem

@ray.remote
def task_with_nested_actor(first_fraction, second_fraction, leaker, actor_allocation_first=True):
    if False:
        print('Hello World!')
    first_bytes = get_additional_bytes_to_reach_memory_usage_pct(first_fraction)
    second_bytes = get_additional_bytes_to_reach_memory_usage_pct(second_fraction) - first_bytes
    if actor_allocation_first:
        ray.get(leaker.allocate.remote(first_bytes))
        dummy = alloc_mem(second_bytes)
    else:
        dummy = alloc_mem(first_bytes)
        ray.get(leaker.allocate.remote(second_bytes))
    return dummy[0]

@pytest.mark.skipif(sys.platform != 'linux' and sys.platform != 'linux2', reason='memory monitor only on linux currently')
def test_churn_long_running(ray_with_memory_monitor):
    if False:
        while True:
            i = 10
    long_running_bytes = get_additional_bytes_to_reach_memory_usage_pct(memory_usage_threshold - 0.1)
    ray.get(allocate_memory.options(max_retries=0).remote(long_running_bytes, post_allocate_sleep_s=30))
    small_bytes = get_additional_bytes_to_reach_memory_usage_pct(memory_usage_threshold + 0.2)
    with pytest.raises(ray.exceptions.OutOfMemoryError) as _:
        ray.get(allocate_memory.options(max_retries=0).remote(small_bytes))

@pytest.mark.skipif(sys.platform != 'linux' and sys.platform != 'linux2', reason='memory monitor only on linux currently')
def test_deadlock_task_with_nested_task(ray_with_memory_monitor):
    if False:
        for i in range(10):
            print('nop')
    task_bytes = get_additional_bytes_to_reach_memory_usage_pct(memory_usage_threshold - 0.1)
    nested_task_bytes = get_additional_bytes_to_reach_memory_usage_pct(memory_usage_threshold + 0.2) - task_bytes
    with pytest.raises(ray.exceptions.RayTaskError) as _:
        ray.get(task_with_nested_task.options(max_retries=0).remote(task_bytes=task_bytes, nested_task_bytes=nested_task_bytes, barrier=None))

@pytest.mark.skipif(sys.platform != 'linux' and sys.platform != 'linux2', reason='memory monitor only on linux currently')
def test_deadlock_task_with_nested_actor_with_actor_first(ray_with_memory_monitor):
    if False:
        i = 10
        return i + 15
    leaker = Leaker.options(max_restarts=0, max_task_retries=0).remote()
    with pytest.raises(ray.exceptions.OutOfMemoryError) as _:
        ray.get(task_with_nested_actor.remote(first_fraction=memory_usage_threshold - 0.1, second_fraction=memory_usage_threshold + 0.25, leaker=leaker, actor_allocation_first=True))

@pytest.mark.skipif(sys.platform != 'linux' and sys.platform != 'linux2', reason='memory monitor only on linux currently')
def test_deadlock_task_with_nested_actor_with_actor_last(ray_with_memory_monitor):
    if False:
        i = 10
        return i + 15
    leaker = Leaker.options(max_restarts=0, max_task_retries=0).remote()
    with pytest.raises(ray.exceptions.OutOfMemoryError) as _:
        ray.get(task_with_nested_actor.remote(first_fraction=memory_usage_threshold - 0.1, second_fraction=memory_usage_threshold + 0.25, leaker=leaker, actor_allocation_first=False))

@ray.remote
class BarrierActor:

    def __init__(self, num_objects):
        if False:
            i = 10
            return i + 15
        self.barrier = threading.Barrier(num_objects, timeout=30)

    def wait_all_done(self):
        if False:
            print('Hello World!')
        self.barrier.wait()

@ray.remote
class ActorWithNestedTask:

    def __init__(self, barrier=None):
        if False:
            for i in range(10):
                print('nop')
        self.mem = []
        self.barrier = barrier

    def perform_allocations(self, actor_bytes, nested_task_bytes):
        if False:
            print('Hello World!')
        self.mem = alloc_mem(actor_bytes)
        if self.barrier:
            ray.get(self.barrier.wait_all_done.remote())
        ray.get(allocate_memory.options(max_retries=0).remote(nested_task_bytes))

@pytest.mark.skipif(sys.platform != 'linux' and sys.platform != 'linux2', reason='memory monitor only on linux currently')
def test_deadlock_actor_with_nested_task(ray_with_memory_monitor):
    if False:
        for i in range(10):
            print('nop')
    leaker = ActorWithNestedTask.options(max_restarts=0, max_task_retries=0).remote()
    actor_bytes = get_additional_bytes_to_reach_memory_usage_pct(memory_usage_threshold - 0.1)
    nested_task_bytes = get_additional_bytes_to_reach_memory_usage_pct(memory_usage_threshold + 0.1) - actor_bytes
    with pytest.raises(ray.exceptions.RayTaskError) as _:
        ray.get(leaker.perform_allocations.remote(actor_bytes, nested_task_bytes))

@pytest.mark.skipif(sys.platform != 'linux' and sys.platform != 'linux2', reason='memory monitor only on linux currently')
def test_deadlock_two_sets_of_actor_with_nested_task(ray_with_memory_monitor):
    if False:
        print('Hello World!')
    barrier = BarrierActor.options(max_restarts=0, max_task_retries=0, max_concurrency=2).remote(2)
    leaker1 = ActorWithNestedTask.options(max_restarts=0, max_task_retries=0).remote(barrier)
    leaker2 = ActorWithNestedTask.options(max_restarts=0, max_task_retries=0).remote(barrier)
    parent_bytes = get_additional_bytes_to_reach_memory_usage_pct((memory_usage_threshold - 0.05) / 2)
    nested_bytes = get_additional_bytes_to_reach_memory_usage_pct(memory_usage_threshold + 0.1) - 2 * parent_bytes
    ref1 = leaker1.perform_allocations.remote(parent_bytes, nested_bytes)
    ref2 = leaker2.perform_allocations.remote(parent_bytes, nested_bytes)
    with pytest.raises(ray.exceptions.RayTaskError) as _:
        ray.get(ref1)
    with pytest.raises(ray.exceptions.RayTaskError) as _:
        ray.get(ref2)

@ray.remote
def task_with_nested_task(task_bytes, nested_task_bytes, barrier=None):
    if False:
        while True:
            i = 10
    dummy = alloc_mem(task_bytes)
    if barrier:
        ray.get(barrier.wait_all_done.remote())
    ray.get(allocate_memory.options(max_retries=0).remote(nested_task_bytes, post_allocate_sleep_s=0.1))
    return dummy[0]

@pytest.mark.skipif(sys.platform != 'linux' and sys.platform != 'linux2', reason='memory monitor only on linux currently')
def test_deadlock_two_sets_of_task_with_nested_task(ray_with_memory_monitor):
    if False:
        i = 10
        return i + 15
    'task_with_nested_task allocates a block of memory, then runs\n    a nested task which also allocates a block memory.\n    This test runs two instances of task_with_nested_task.\n    We expect the second one to fail.'
    parent_bytes = get_additional_bytes_to_reach_memory_usage_pct((memory_usage_threshold - 0.05) / 2)
    nested_bytes = get_additional_bytes_to_reach_memory_usage_pct(memory_usage_threshold + 0.1) - 2 * parent_bytes
    barrier = BarrierActor.options(max_restarts=0, max_task_retries=0, max_concurrency=2).remote(2)
    ref1 = task_with_nested_task.options(max_retries=0).remote(parent_bytes, nested_bytes, barrier)
    ref2 = task_with_nested_task.options(max_retries=0).remote(parent_bytes, nested_bytes, barrier)
    with pytest.raises(ray.exceptions.RayTaskError) as _:
        ray.get(ref1)
    with pytest.raises(ray.exceptions.RayTaskError) as _:
        ray.get(ref2)
if __name__ == '__main__':
    pass