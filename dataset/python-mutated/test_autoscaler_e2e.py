import subprocess
from ray.autoscaler._private.constants import AUTOSCALER_METRIC_PORT
import pytest
import ray
import sys
from ray._private.test_utils import reset_autoscaler_v2_enabled_cache, wait_for_condition, get_metric_check_condition, MetricSamplePattern
from ray.cluster_utils import AutoscalingCluster
from ray.autoscaler.node_launch_exception import NodeLaunchException

@pytest.mark.parametrize('enable_v2', [True, False])
def test_ray_status_activity(shutdown_only, enable_v2):
    if False:
        print('Hello World!')
    reset_autoscaler_v2_enabled_cache()
    cluster = AutoscalingCluster(head_resources={'CPU': 0}, worker_node_types={'type-i': {'resources': {'CPU': 4, 'fun': 1}, 'node_config': {}, 'min_workers': 1, 'max_workers': 1}, 'type-ii': {'resources': {'CPU': 3, 'fun': 100}, 'node_config': {}, 'min_workers': 1, 'max_workers': 1}})
    try:
        cluster.start(_system_config={'enable_autoscaler_v2': enable_v2})
        ray.init(address='auto')
        if enable_v2:
            assert subprocess.check_output('ray status --verbose', shell=True).decode().count('Idle: ') > 0

        @ray.remote(num_cpus=2, resources={'fun': 2})
        class Actor:

            def ping(self):
                if False:
                    print('Hello World!')
                return None
        actor = Actor.remote()
        ray.get(actor.ping.remote())
        occurrences = 1 if enable_v2 else 0
        assert subprocess.check_output('ray status --verbose', shell=True).decode().count('Resource: CPU currently in use.') == occurrences
        from ray.util.placement_group import placement_group
        pg = placement_group([{'CPU': 2}], strategy='STRICT_SPREAD')
        ray.get(pg.ready())
        occurrences = 2 if enable_v2 else 0
        assert subprocess.check_output('ray status --verbose', shell=True).decode().count('Resource: CPU currently in use.') == occurrences
        assert subprocess.check_output('ray status --verbose', shell=True).decode().count('Resource: bundle_group_') == 0
    finally:
        cluster.shutdown()

@pytest.mark.parametrize('enable_v2', [True, False])
def test_ray_status_e2e(shutdown_only, enable_v2):
    if False:
        i = 10
        return i + 15
    reset_autoscaler_v2_enabled_cache()
    cluster = AutoscalingCluster(head_resources={'CPU': 0}, worker_node_types={'type-i': {'resources': {'CPU': 1, 'fun': 1}, 'node_config': {}, 'min_workers': 1, 'max_workers': 1}, 'type-ii': {'resources': {'CPU': 1, 'fun': 100}, 'node_config': {}, 'min_workers': 1, 'max_workers': 1}})
    try:
        cluster.start(_system_config={'enable_autoscaler_v2': enable_v2})
        ray.init(address='auto')

        @ray.remote(num_cpus=0, resources={'fun': 2})
        class Actor:

            def ping(self):
                if False:
                    while True:
                        i = 10
                return None
        actor = Actor.remote()
        ray.get(actor.ping.remote())
        assert 'Demands' in subprocess.check_output('ray status', shell=True).decode()
        assert 'Total Demands' not in subprocess.check_output('ray status', shell=True).decode()
        assert 'Total Demands' in subprocess.check_output('ray status -v', shell=True).decode()
        assert 'Total Demands' in subprocess.check_output('ray status --verbose', shell=True).decode()
    finally:
        cluster.shutdown()

def test_metrics(shutdown_only):
    if False:
        while True:
            i = 10
    cluster = AutoscalingCluster(head_resources={'CPU': 0}, worker_node_types={'type-i': {'resources': {'CPU': 1}, 'node_config': {}, 'min_workers': 0, 'max_workers': 1}, 'type-ii': {'resources': {'CPU': 1}, 'node_config': {}, 'min_workers': 0, 'max_workers': 1}})
    try:
        cluster.start()
        info = ray.init(address='auto')
        autoscaler_export_addr = '{}:{}'.format(info.address_info['node_ip_address'], AUTOSCALER_METRIC_PORT)

        @ray.remote(num_cpus=1)
        class Foo:

            def ping(self):
                if False:
                    for i in range(10):
                        print('nop')
                return True
        zero_reported_condition = get_metric_check_condition([MetricSamplePattern(name='autoscaler_cluster_resources', value=0, partial_label_match={'resource': 'CPU'}), MetricSamplePattern(name='autoscaler_pending_resources', value=0), MetricSamplePattern(name='autoscaler_pending_nodes', value=0), MetricSamplePattern(name='autoscaler_active_nodes', value=0, partial_label_match={'NodeType': 'type-i'}), MetricSamplePattern(name='autoscaler_active_nodes', value=0, partial_label_match={'NodeType': 'type-ii'}), MetricSamplePattern(name='autoscaler_active_nodes', value=1, partial_label_match={'NodeType': 'ray.head.default'})], export_addr=autoscaler_export_addr)
        wait_for_condition(zero_reported_condition)
        actors = [Foo.remote() for _ in range(2)]
        ray.get([actor.ping.remote() for actor in actors])
        two_cpu_no_pending_condition = get_metric_check_condition([MetricSamplePattern(name='autoscaler_cluster_resources', value=2, partial_label_match={'resource': 'CPU'}), MetricSamplePattern(name='autoscaler_pending_nodes', value=0, partial_label_match={'NodeType': 'type-i'}), MetricSamplePattern(name='autoscaler_pending_nodes', value=0, partial_label_match={'NodeType': 'type-ii'}), MetricSamplePattern(name='autoscaler_active_nodes', value=1, partial_label_match={'NodeType': 'type-i'}), MetricSamplePattern(name='autoscaler_active_nodes', value=1, partial_label_match={'NodeType': 'type-ii'}), MetricSamplePattern(name='autoscaler_active_nodes', value=1, partial_label_match={'NodeType': 'ray.head.default'})], export_addr=autoscaler_export_addr)
        wait_for_condition(two_cpu_no_pending_condition)
    finally:
        cluster.shutdown()

def test_node_launch_exception_serialization(shutdown_only):
    if False:
        while True:
            i = 10
    ray.init(num_cpus=1)
    exc_info = None
    try:
        raise Exception('Test exception.')
    except Exception:
        exc_info = sys.exc_info()
    assert exc_info is not None
    exc = NodeLaunchException('cat', 'desc', exc_info)
    after_serialization = ray.get(ray.put(exc))
    assert after_serialization.category == exc.category
    assert after_serialization.description == exc.description
    assert after_serialization.src_exc_info is None
if __name__ == '__main__':
    import os
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))