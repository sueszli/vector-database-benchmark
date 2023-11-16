import sys
import threading
import time
import numpy as np
import pytest
import ray
import ray._private.test_utils as test_utils
from ray._private.state import available_resources

def ensure_cpu_returned(expected_cpus):
    if False:
        i = 10
        return i + 15
    test_utils.wait_for_condition(lambda : available_resources().get('CPU', 0) == expected_cpus)

def test_threaded_actor_basic(shutdown_only):
    if False:
        i = 10
        return i + 15
    'Test the basic threaded actor.'
    ray.init(num_cpus=1)

    @ray.remote(num_cpus=1)
    class ThreadedActor:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.received = []
            self.lock = threading.Lock()

        def add(self, seqno):
            if False:
                while True:
                    i = 10
            with self.lock:
                self.received.append(seqno)

        def get_all(self):
            if False:
                for i in range(10):
                    print('nop')
            with self.lock:
                return self.received
    a = ThreadedActor.options(max_concurrency=10).remote()
    max_seq = 50
    ray.get([a.add.remote(seqno) for seqno in range(max_seq)])
    seqnos = ray.get(a.get_all.remote())
    assert sorted(seqnos) == list(range(max_seq))
    ray.kill(a)
    ensure_cpu_returned(1)

def test_threaded_actor_api_thread_safe(shutdown_only):
    if False:
        while True:
            i = 10
    'Test if Ray APIs are thread safe\n    when they are used within threaded actor.\n    '
    ray.init(num_cpus=8, _system_config={'max_direct_call_object_size': 1024})

    @ray.remote
    def in_memory_return(i):
        if False:
            for i in range(10):
                print('nop')
        return i

    @ray.remote
    def plasma_return(i):
        if False:
            return 10
        arr = np.zeros(8 * 1024 * i, dtype=np.uint8)
        return arr

    @ray.remote(num_cpus=1)
    class ThreadedActor:

        def __init__(self):
            if False:
                print('Hello World!')
            self.received = []
            self.lock = threading.Lock()

        def in_memory_return_test(self, i):
            if False:
                return 10
            self._add(i)
            return ray.get(in_memory_return.remote(i))

        def plasma_return_test(self, i):
            if False:
                for i in range(10):
                    print('nop')
            self._add(i)
            return ray.get(plasma_return.remote(i))

        def _add(self, seqno):
            if False:
                i = 10
                return i + 15
            with self.lock:
                self.received.append(seqno)

        def get_all(self):
            if False:
                i = 10
                return i + 15
            with self.lock:
                return self.received
    a = ThreadedActor.options(max_concurrency=10).remote()
    max_seq = 50
    seqnos = ray.get([a.in_memory_return_test.remote(seqno) for seqno in range(max_seq)])
    assert sorted(seqnos) == list(range(max_seq))
    real = ray.get([a.plasma_return_test.remote(seqno) for seqno in range(max_seq)])
    expected = [np.zeros(8 * 1024 * i, dtype=np.uint8) for i in range(max_seq)]
    for (r, e) in zip(real, expected):
        assert np.array_equal(r, e)
    ray.kill(a)
    ensure_cpu_returned(8)

def test_threaded_actor_creation_and_kill(ray_start_cluster):
    if False:
        return 10
    'Test the scenario where the threaded actors are created and killed.'
    cluster = ray_start_cluster
    NUM_CPUS_PER_NODE = 3
    NUM_NODES = 2
    for _ in range(NUM_NODES):
        cluster.add_node(num_cpus=NUM_CPUS_PER_NODE)
    ray.init(address=cluster.address)

    @ray.remote(num_cpus=0)
    class ThreadedActor:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.received = []
            self.lock = threading.Lock()

        def add(self, seqno):
            if False:
                print('Hello World!')
            time.sleep(1)
            with self.lock:
                self.received.append(seqno)

        def get_all(self):
            if False:
                while True:
                    i = 10
            with self.lock:
                return self.received

        def ready(self):
            if False:
                print('Hello World!')
            pass

        def terminate(self):
            if False:
                while True:
                    i = 10
            ray.actor.exit_actor()
    for _ in range(10):
        actors = [ThreadedActor.options(max_concurrency=10).remote() for _ in range(NUM_NODES * NUM_CPUS_PER_NODE)]
        ray.get([actor.ready.remote() for actor in actors])
        for _ in range(10):
            for actor in actors:
                actor.add.remote(1)
        time.sleep(0.5)
        for actor in actors:
            ray.kill(actor)
    ensure_cpu_returned(NUM_NODES * NUM_CPUS_PER_NODE)
    for _ in range(10):
        actors = [ThreadedActor.options(max_concurrency=10).remote() for _ in range(NUM_NODES * NUM_CPUS_PER_NODE)]
        ray.get([actor.ready.remote() for actor in actors])
        for _ in range(10):
            for actor in actors:
                actor.add.remote(1)
        time.sleep(0.5)
        for actor in actors:
            actor.terminate.remote()
    ensure_cpu_returned(NUM_NODES * NUM_CPUS_PER_NODE)

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
@pytest.mark.parametrize('ray_start_cluster_head', [{'num_cpus': 2}], indirect=True)
def test_threaded_actor_integration_test_stress(ray_start_cluster_head, log_pubsub, error_pubsub):
    if False:
        for i in range(10):
            print('nop')
    'This is a sanity test that checks threaded actors are\n    working with the nightly stress test.\n    '
    cluster = ray_start_cluster_head
    p = log_pubsub
    e = error_pubsub
    num_remote_nodes = 4
    num_parents = 6
    num_children = 6
    death_probability = 0.95
    max_concurrency = 10
    for _ in range(num_remote_nodes):
        cluster.add_node(num_cpus=2)

    @ray.remote
    class Child(object):

        def __init__(self, death_probability):
            if False:
                for i in range(10):
                    print('nop')
            self.death_probability = death_probability

        def ping(self):
            if False:
                i = 10
                return i + 15
            exit_chance = np.random.rand()
            if exit_chance > self.death_probability:
                sys.exit(-1)

    @ray.remote
    class Parent(object):

        def __init__(self, num_children, death_probability=0.95):
            if False:
                print('Hello World!')
            self.death_probability = death_probability
            self.children = [Child.options(max_concurrency=max_concurrency).remote(death_probability) for _ in range(num_children)]

        def ping(self, num_pings):
            if False:
                while True:
                    i = 10
            children_outputs = []
            for _ in range(num_pings):
                children_outputs += [child.ping.remote() for child in self.children]
            try:
                ray.get(children_outputs)
            except Exception:
                self.__init__(len(self.children), self.death_probability)

        def kill(self):
            if False:
                for i in range(10):
                    print('nop')
            ray.get([child.__ray_terminate__.remote() for child in self.children])
    parents = [Parent.options(max_concurrency=max_concurrency).remote(num_children, death_probability) for _ in range(num_parents)]
    start = time.time()
    loop_times = []
    for _ in range(10):
        loop_start = time.time()
        ray.get([parent.ping.remote(10) for parent in parents])
        exit_chance = np.random.rand()
        if exit_chance > death_probability:
            parent_index = np.random.randint(len(parents))
            parents[parent_index].kill.remote()
            parents[parent_index] = Parent.options(max_concurrency=max_concurrency).remote(num_children, death_probability)
        loop_times.append(time.time() - loop_start)
    result = {}
    print('Finished in: {}s'.format(time.time() - start))
    print('Average iteration time: {}s'.format(np.mean(loop_times)))
    print('Max iteration time: {}s'.format(max(loop_times)))
    print('Min iteration time: {}s'.format(min(loop_times)))
    result['total_time'] = time.time() - start
    result['avg_iteration_time'] = np.mean(loop_times)
    result['max_iteration_time'] = max(loop_times)
    result['min_iteration_time'] = min(loop_times)
    result['success'] = 1
    print(result)
    ensure_cpu_returned(10)
    del parents
    parents = [Parent.options(max_concurrency=max_concurrency).remote(num_children, death_probability) for _ in range(num_parents)]
    ray.get([parent.ping.remote(10) for parent in parents])
    '\n    Make sure there are not SIGSEGV, SIGBART, or other odd check failures.\n    '
    logs = test_utils.get_log_message(p, timeout=20)
    for log in logs:
        assert 'SIG' not in log, "There's the segfault or SIGBART reported."
        assert 'Check failed' not in log, "There's the check failure reported."
    errors = test_utils.get_error_message(e, timeout=10)
    for error in errors:
        print(error)
        assert 'You can ignore this message if' not in error['error_message'], "Resource deadlock warning shouldn't be printed, but it did."
if __name__ == '__main__':
    import os
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))