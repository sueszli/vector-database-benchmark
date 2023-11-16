import pytest
import platform
import ray
from ray.cluster_utils import AutoscalingCluster

@pytest.mark.skipif(platform.system() == 'Windows', reason='Failing on Windows.')
def test_fake_autoscaler_basic_e2e(shutdown_only):
    if False:
        print('Hello World!')
    cluster = AutoscalingCluster(head_resources={'CPU': 2}, worker_node_types={'cpu_node': {'resources': {'CPU': 4, 'object_store_memory': 1024 * 1024 * 1024}, 'node_config': {}, 'min_workers': 0, 'max_workers': 2}, 'gpu_node': {'resources': {'CPU': 2, 'GPU': 1, 'object_store_memory': 1024 * 1024 * 1024}, 'node_config': {}, 'min_workers': 0, 'max_workers': 2}, 'tpu_node': {'resources': {'CPU': 2, 'TPU': 4, 'object_store_memory': 1024 * 1024 * 1024}, 'node_config': {}, 'min_workers': 0, 'max_workers': 2}})
    try:
        cluster.start()
        ray.init('auto')

        @ray.remote(num_gpus=1)
        def f():
            if False:
                i = 10
                return i + 15
            print('gpu ok')

        @ray.remote(num_cpus=3)
        def g():
            if False:
                print('Hello World!')
            print('cpu ok')

        @ray.remote(resources={'TPU': 4})
        def h():
            if False:
                for i in range(10):
                    print('nop')
            print('tpu ok')
        ray.get(f.remote())
        ray.get(g.remote())
        ray.get(h.remote())
        ray.shutdown()
    finally:
        cluster.shutdown()

def test_zero_cpu_default_actor():
    if False:
        print('Hello World!')
    cluster = AutoscalingCluster(head_resources={'CPU': 0}, worker_node_types={'cpu_node': {'resources': {'CPU': 1}, 'node_config': {}, 'min_workers': 0, 'max_workers': 1}})
    try:
        cluster.start()
        ray.init('auto')

        @ray.remote
        class Actor:

            def ping(self):
                if False:
                    print('Hello World!')
                pass
        actor = Actor.remote()
        ray.get(actor.ping.remote())
        ray.shutdown()
    finally:
        cluster.shutdown()

def test_autoscaler_cpu_task_gpu_node_up():
    if False:
        print('Hello World!')
    'Validates that CPU tasks can trigger GPU upscaling.\n    See https://github.com/ray-project/ray/pull/31202.\n    '
    cluster = AutoscalingCluster(head_resources={'CPU': 0}, worker_node_types={'gpu_node_type': {'resources': {'CPU': 1, 'GPU': 1}, 'node_config': {}, 'min_workers': 0, 'max_workers': 1}})
    try:
        cluster.start()
        ray.init('auto')

        @ray.remote(num_cpus=1)
        def task():
            if False:
                i = 10
                return i + 15
            return True
        ray.get(task.remote(), timeout=30)
        ray.shutdown()
    finally:
        cluster.shutdown()
if __name__ == '__main__':
    import os
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))