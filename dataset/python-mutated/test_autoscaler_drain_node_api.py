import logging
import platform
import time
import pytest
import ray
import ray._private.ray_constants as ray_constants
from ray._private.test_utils import get_error_message, init_error_pubsub, wait_for_condition
from ray.autoscaler._private.fake_multi_node.node_provider import FakeMultiNodeProvider
from ray.cluster_utils import AutoscalingCluster
logger = logging.getLogger(__name__)

class MockFakeProvider(FakeMultiNodeProvider):
    """FakeMultiNodeProvider, with Ray node process termination mocked out.

    Used to check that a Ray node can be terminated by DrainNode API call
    from the autoscaler.
    """

    def _kill_ray_processes(self, node):
        if False:
            i = 10
            return i + 15
        logger.info('Leaving Raylet termination to autoscaler Drain API!')

class MockAutoscalingCluster(AutoscalingCluster):
    """AutoscalingCluster modified to used the above MockFakeProvider."""

    def _generate_config(self, head_resources, worker_node_types):
        if False:
            return 10
        config = super()._generate_config(head_resources, worker_node_types)
        config['provider']['type'] = 'external'
        config['provider']['module'] = 'ray.tests.test_autoscaler_drain_node_api.MockFakeProvider'
        return config

@pytest.mark.skipif(platform.system() == 'Windows', reason='Failing on Windows.')
def test_drain_api(shutdown_only):
    if False:
        i = 10
        return i + 15
    "E2E test of the autoscaler's use of the DrainNode API.\n\n    Adapted from test_autoscaler_fake_multinode.py.\n\n    The strategy is to mock out Ray node process termination in\n    FakeMultiNodeProvider, leaving node termination to the DrainNode API.\n\n    Scale-down is verified by `ray.cluster_resources`. It is verified that\n    no removed_node errors are issued adter scale-down.\n\n    Validity of this test depends on the current implementation of DrainNode.\n    DrainNode currently works by asking the GCS to de-register and shut down\n    Ray nodes.\n    "
    cluster = MockAutoscalingCluster(head_resources={'CPU': 1}, worker_node_types={'gpu_node': {'resources': {'CPU': 1, 'GPU': 1, 'object_store_memory': 1024 * 1024 * 1024}, 'node_config': {}, 'min_workers': 0, 'max_workers': 2}})
    try:
        cluster.start()
        ray.init('auto')

        @ray.remote(num_gpus=1)
        def f():
            if False:
                for i in range(10):
                    print('nop')
            print('gpu ok')
        ray.get(f.remote())
        wait_for_condition(lambda : ray.cluster_resources().get('GPU', 0) == 1)
        time.sleep(12)
        wait_for_condition(lambda : ray.cluster_resources().get('GPU', 0) == 0)
        try:
            p = init_error_pubsub()
            errors = get_error_message(p, 1, ray_constants.REMOVED_NODE_ERROR, timeout=5)
            assert len(errors) == 0
        finally:
            p.close()
    finally:
        cluster.shutdown()
if __name__ == '__main__':
    import os
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))