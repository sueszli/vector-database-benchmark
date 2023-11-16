import signal
import json
import os
import pathlib
import sys
import requests
from pprint import pformat
from unittest.mock import MagicMock
import numpy as np
import pytest
import ray
from ray.util.state import list_nodes
from ray._private.metrics_agent import PrometheusServiceDiscoveryWriter
from ray._private.ray_constants import PROMETHEUS_SERVICE_DISCOVERY_FILE
from ray._private.test_utils import SignalActor, fetch_prometheus, fetch_prometheus_metrics, get_log_batch, wait_for_condition, raw_metrics
from ray.autoscaler._private.constants import AUTOSCALER_METRIC_PORT
from ray.dashboard.consts import DASHBOARD_METRIC_PORT
from ray.util.metrics import Counter, Gauge, Histogram
os.environ['RAY_event_stats'] = '1'
try:
    import prometheus_client
except ImportError:
    prometheus_client = None
_METRICS = ['ray_node_disk_usage', 'ray_node_mem_used', 'ray_node_mem_total', 'ray_node_cpu_utilization', 'ray_object_store_available_memory', 'ray_object_store_used_memory', 'ray_object_store_num_local_objects', 'ray_object_store_memory', 'ray_object_manager_num_pull_requests', 'ray_object_directory_subscriptions', 'ray_object_directory_updates', 'ray_object_directory_lookups', 'ray_object_directory_added_locations', 'ray_object_directory_removed_locations', 'ray_internal_num_processes_started', 'ray_internal_num_spilled_tasks', 'ray_grpc_server_req_process_time_ms', 'ray_grpc_server_req_new_total', 'ray_grpc_server_req_handling_total', 'ray_grpc_server_req_finished_total', 'ray_object_manager_received_chunks', 'ray_pull_manager_usage_bytes', 'ray_pull_manager_requested_bundles', 'ray_pull_manager_requests', 'ray_pull_manager_active_bundles', 'ray_pull_manager_retries_total', 'ray_push_manager_in_flight_pushes', 'ray_push_manager_chunks', 'ray_scheduler_failed_worker_startup_total', 'ray_scheduler_tasks', 'ray_scheduler_unscheduleable_tasks', 'ray_spill_manager_objects', 'ray_spill_manager_objects_bytes', 'ray_spill_manager_request_total', 'ray_gcs_placement_group_creation_latency_ms_sum', 'ray_gcs_placement_group_scheduling_latency_ms_sum', 'ray_gcs_placement_group_count', 'ray_gcs_actors_count']
_AUTOSCALER_METRICS = ['autoscaler_config_validation_exceptions', 'autoscaler_node_launch_exceptions', 'autoscaler_pending_nodes', 'autoscaler_reset_exceptions', 'autoscaler_running_workers', 'autoscaler_started_nodes', 'autoscaler_stopped_nodes', 'autoscaler_update_loop_exceptions', 'autoscaler_worker_create_node_time', 'autoscaler_worker_update_time', 'autoscaler_updating_nodes', 'autoscaler_successful_updates', 'autoscaler_failed_updates', 'autoscaler_failed_create_nodes', 'autoscaler_recovering_nodes', 'autoscaler_successful_recoveries', 'autoscaler_failed_recoveries', 'autoscaler_drain_node_exceptions', 'autoscaler_update_time', 'autoscaler_cluster_resources', 'autoscaler_pending_resources']
_DASHBOARD_METRICS = ['ray_dashboard_api_requests_duration_seconds_bucket', 'ray_dashboard_api_requests_duration_seconds_created', 'ray_dashboard_api_requests_count_requests_total', 'ray_dashboard_api_requests_count_requests_created', 'ray_component_cpu_percentage', 'ray_component_uss_mb']
_NODE_METRICS = ['ray_node_cpu_utilization', 'ray_node_cpu_count', 'ray_node_mem_used', 'ray_node_mem_available', 'ray_node_mem_total', 'ray_node_disk_io_read', 'ray_node_disk_io_write', 'ray_node_disk_io_read_count', 'ray_node_disk_io_write_count', 'ray_node_disk_io_read_speed', 'ray_node_disk_io_write_speed', 'ray_node_disk_read_iops', 'ray_node_disk_write_iops', 'ray_node_disk_usage', 'ray_node_disk_free', 'ray_node_disk_utilization_percentage', 'ray_node_network_sent', 'ray_node_network_received', 'ray_node_network_send_speed', 'ray_node_network_receive_speed']
if sys.platform == 'linux' or sys.platform == 'linux2':
    _NODE_METRICS.append('ray_node_mem_shared_bytes')
_NODE_COMPONENT_METRICS = ['ray_component_cpu_percentage', 'ray_component_rss_mb', 'ray_component_uss_mb', 'ray_component_num_fds']
_METRICS.append('ray_health_check_rpc_latency_ms_sum')

@pytest.fixture
def _setup_cluster_for_test(request, ray_start_cluster):
    if False:
        for i in range(10):
            print('nop')
    enable_metrics_collection = request.param
    NUM_NODES = 2
    cluster = ray_start_cluster
    cluster.add_node(_system_config={'metrics_report_interval_ms': 1000, 'event_stats_print_interval_ms': 500, 'event_stats': True, 'enable_metrics_collection': enable_metrics_collection})
    [cluster.add_node() for _ in range(NUM_NODES - 1)]
    cluster.wait_for_nodes()
    ray_context = ray.init(address=cluster.address)
    worker_should_exit = SignalActor.remote()
    counter = Counter('test_driver_counter', description='desc')
    counter.inc()

    @ray.remote
    def f():
        if False:
            for i in range(10):
                print('nop')
        counter = Counter('test_counter', description='desc')
        counter.inc()
        counter = ray.get(ray.put(counter))
        counter.inc()
        counter.inc(2)
        ray.get(worker_should_exit.wait.remote())
    pg = ray.util.placement_group(bundles=[{'CPU': 1}])
    ray.get(pg.ready())
    ray.util.remove_placement_group(pg)

    @ray.remote
    class A:

        async def ping(self):
            histogram = Histogram('test_histogram', description='desc', boundaries=[0.1, 1.6])
            histogram = ray.get(ray.put(histogram))
            histogram.observe(1.5)
            ray.get(worker_should_exit.wait.remote())
    a = A.remote()
    obj_refs = [f.remote(), a.ping.remote()]
    b = f.options(resources={'a': 1})
    requests.get(f'http://{ray_context.dashboard_url}/nodes')
    node_info_list = ray.nodes()
    prom_addresses = []
    for node_info in node_info_list:
        metrics_export_port = node_info['MetricsExportPort']
        addr = node_info['NodeManagerAddress']
        prom_addresses.append(f'{addr}:{metrics_export_port}')
    autoscaler_export_addr = '{}:{}'.format(cluster.head_node.node_ip_address, AUTOSCALER_METRIC_PORT)
    dashboard_export_addr = '{}:{}'.format(cluster.head_node.node_ip_address, DASHBOARD_METRIC_PORT)
    yield (prom_addresses, autoscaler_export_addr, dashboard_export_addr)
    ray.get(worker_should_exit.send.remote())
    ray.get(obj_refs)
    ray.shutdown()
    cluster.shutdown()

@pytest.mark.skipif(prometheus_client is None, reason='Prometheus not installed')
@pytest.mark.parametrize('_setup_cluster_for_test', [True], indirect=True)
def test_metrics_export_end_to_end(_setup_cluster_for_test):
    if False:
        return 10
    TEST_TIMEOUT_S = 30
    (prom_addresses, autoscaler_export_addr, dashboard_export_addr) = _setup_cluster_for_test

    def test_cases():
        if False:
            print('Hello World!')
        (components_dict, metric_names, metric_samples) = fetch_prometheus(prom_addresses)
        session_name = ray._private.worker.global_worker.node.session_name
        assert all(('raylet' in components for components in components_dict.values()))
        assert any(('gcs_server' in components for components in components_dict.values()))
        assert any(('core_worker' in components for components in components_dict.values()))
        for metric_name in ['test_counter', 'test_histogram', 'test_driver_counter']:
            assert any((metric_name in full_name for full_name in metric_names))
        for metric in _METRICS:
            assert metric in metric_names, f'metric {metric} not in {metric_names}'
        for sample in metric_samples:
            if sample.name in _METRICS:
                assert sample.labels['SessionName'] == session_name
            if sample.name in _DASHBOARD_METRICS:
                assert sample.labels['SessionName'] == session_name
        test_counter_sample = [m for m in metric_samples if 'test_counter' in m.name][0]
        assert test_counter_sample.value == 4.0
        test_driver_counter_sample = [m for m in metric_samples if 'test_driver_counter' in m.name][0]
        assert test_driver_counter_sample.value == 1.0
        test_histogram_samples = [m for m in metric_samples if 'test_histogram' in m.name]
        buckets = {m.labels['le']: m.value for m in test_histogram_samples if '_bucket' in m.name}
        assert buckets == {'0.1': 0.0, '1.6': 1.0, '+Inf': 1.0}
        hist_count = [m for m in test_histogram_samples if '_count' in m.name][0].value
        hist_sum = [m for m in test_histogram_samples if '_sum' in m.name][0].value
        assert hist_count == 1
        assert hist_sum == 1.5
        grpc_metrics = ['ray_grpc_server_req_process_time_ms', 'ray_grpc_server_req_new_total', 'ray_grpc_server_req_handling_total', 'ray_grpc_server_req_finished_total']
        for grpc_metric in grpc_metrics:
            grpc_samples = [m for m in metric_samples if grpc_metric in m.name]
            for grpc_sample in grpc_samples:
                assert grpc_sample.labels['Component'] != 'core_worker'
        (_, autoscaler_metric_names, autoscaler_samples) = fetch_prometheus([autoscaler_export_addr])
        for metric in _AUTOSCALER_METRICS:
            assert any((name.startswith(metric) for name in autoscaler_metric_names)), f'{metric} not in {autoscaler_metric_names}'
            for sample in autoscaler_samples:
                assert sample.labels['SessionName'] == session_name
        (_, dashboard_metric_names, _) = fetch_prometheus([dashboard_export_addr])
        for metric in _DASHBOARD_METRICS:
            assert any((name.startswith(metric) for name in dashboard_metric_names)), f'{metric} not in {dashboard_metric_names}'

    def wrap_test_case_for_retry():
        if False:
            while True:
                i = 10
        try:
            test_cases()
            return True
        except AssertionError:
            return False
    try:
        wait_for_condition(wrap_test_case_for_retry, timeout=TEST_TIMEOUT_S, retry_interval_ms=1000)
    except RuntimeError:
        print(f'The components are {pformat(fetch_prometheus(prom_addresses))}')
        test_cases()

@pytest.mark.skipif(sys.platform == 'win32', reason='Not working in Windows.')
@pytest.mark.skipif(prometheus_client is None, reason='Prometheus not installed')
def test_metrics_export_node_metrics(shutdown_only):
    if False:
        return 10
    addr = ray.init()
    dashboard_export_addr = '{}:{}'.format(addr['raylet_ip_address'], DASHBOARD_METRIC_PORT)

    def verify_node_metrics():
        if False:
            for i in range(10):
                print('nop')
        avail_metrics = raw_metrics(addr)
        components = set()
        for metric in _NODE_COMPONENT_METRICS:
            samples = avail_metrics[metric]
            for sample in samples:
                components.add(sample.labels['Component'])
        assert components == {'raylet', 'agent', 'ray::IDLE'}
        avail_metrics = set(avail_metrics)
        for node_metric in _NODE_METRICS:
            assert node_metric in avail_metrics
        for node_metric in _NODE_COMPONENT_METRICS:
            assert node_metric in avail_metrics
        return True

    def verify_dashboard_metrics():
        if False:
            while True:
                i = 10
        avail_metrics = fetch_prometheus_metrics([dashboard_export_addr])
        list_nodes()
        avail_metrics = avail_metrics
        for metric in _DASHBOARD_METRICS:
            assert len(avail_metrics[metric]) > 0
            samples = avail_metrics[metric]
            for sample in samples:
                assert sample.labels['Component'] == 'dashboard'
        return True
    wait_for_condition(verify_node_metrics)
    wait_for_condition(verify_dashboard_metrics)

def test_operation_stats(monkeypatch, shutdown_only):
    if False:
        print('Hello World!')
    operation_metrics = ['ray_operation_count', 'ray_operation_run_time_ms', 'ray_operation_queue_time_ms', 'ray_operation_active_count']
    with monkeypatch.context() as m:
        m.setenv('RAY_event_stats_metrics', '1')
        addr = ray.init()
        signal = SignalActor.remote()

        @ray.remote
        class Actor:

            def __init__(self, signal):
                if False:
                    while True:
                        i = 10
                self.signal = signal

            def get_worker_id(self):
                if False:
                    print('Hello World!')
                return ray.get_runtime_context().get_worker_id()

            def wait(self):
                if False:
                    i = 10
                    return i + 15
                ray.get(self.signal.wait.remote())
        actor = Actor.remote(signal)
        worker_id = ray.get(actor.get_worker_id.remote())
        obj_ref = actor.wait.remote()

        def verify():
            if False:
                for i in range(10):
                    print('nop')
            metrics = raw_metrics(addr)
            samples = metrics['ray_operation_count']
            found = False
            for sample in samples:
                if sample.labels['Method'] == 'CoreWorkerService.grpc_client.PushTask' and sample.labels['Component'] == 'core_worker' and (sample.labels['WorkerId'] == worker_id):
                    found = True
                    assert sample.value == 1
            if not found:
                return False
            samples = metrics['ray_operation_active_count']
            found = False
            for sample in samples:
                if sample.labels['Method'] == 'CoreWorkerService.grpc_client.PushTask' and sample.labels['Component'] == 'core_worker' and (sample.labels['WorkerId'] == worker_id):
                    found = True
                    assert sample.value == 1
            if not found:
                return False
            return True
        wait_for_condition(verify, timeout=60)
        ray.get(signal.send.remote())
        ray.get(obj_ref)

        def verify():
            if False:
                print('Hello World!')
            metrics = raw_metrics(addr)
            samples = metrics['ray_operation_count']
            found = False
            for sample in samples:
                if sample.labels['Method'] == 'CoreWorkerService.grpc_client.PushTask' and sample.labels['Component'] == 'core_worker' and (sample.labels['WorkerId'] == worker_id):
                    found = True
                    assert sample.value == 1
            if not found:
                return False
            found = False
            for sample in samples:
                if sample.labels['Method'] == 'CoreWorkerService.grpc_client.PushTask.OnReplyReceived' and sample.labels['Component'] == 'core_worker' and (sample.labels['WorkerId'] == worker_id):
                    found = True
                    assert sample.value == 1
            if not found:
                return False
            samples = metrics['ray_operation_active_count']
            found = False
            for sample in samples:
                if sample.labels['Method'] == 'CoreWorkerService.grpc_client.PushTask' and sample.labels['Component'] == 'core_worker' and (sample.labels['WorkerId'] == worker_id):
                    found = True
                    assert sample.value == 0
            if not found:
                return False
            found = False
            for sample in samples:
                if sample.labels['Method'] == 'CoreWorkerService.grpc_client.PushTask.OnReplyReceived' and sample.labels['Component'] == 'core_worker' and (sample.labels['WorkerId'] == worker_id):
                    found = True
                    assert sample.value == 0
            if not found:
                return False
            metric_names = set(metrics.keys())
            for op_metric in operation_metrics:
                assert op_metric in metric_names
                samples = metrics[op_metric]
                components = set()
                for sample in samples:
                    components.add(sample.labels['Component'])
            assert {'raylet', 'gcs_server', 'core_worker'} == components
            return True
        wait_for_condition(verify, timeout=60)

@pytest.mark.skipif(sys.platform == 'win32', reason='Not working in Windows.')
def test_per_func_name_stats(shutdown_only):
    if False:
        while True:
            i = 10
    comp_metrics = ['ray_component_cpu_percentage', 'ray_component_rss_mb', 'ray_component_num_fds']
    if sys.platform == 'linux' or sys.platform == 'linux2':
        comp_metrics.append('ray_component_uss_mb')
        comp_metrics.append('ray_component_mem_shared_bytes')
    addr = ray.init(num_cpus=2)

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.arr = np.random.rand(5 * 1024 * 1024)
            self.shared_arr = ray.put(np.random.rand(5 * 1024 * 1024))

        def pid(self):
            if False:
                for i in range(10):
                    print('nop')
            return os.getpid()

    @ray.remote
    class ActorB:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.arr = np.random.rand(5 * 1024 * 1024)
            self.shared_arr = ray.put(np.random.rand(5 * 1024 * 1024))
    a = Actor.remote()
    b = ActorB.remote()

    def verify_components():
        if False:
            while True:
                i = 10
        metrics = raw_metrics(addr)
        metric_names = set(metrics.keys())
        for metric in comp_metrics:
            assert metric in metric_names
            samples = metrics[metric]
            components = set()
            for sample in samples:
                components.add(sample.labels['Component'])
        assert {'raylet', 'agent', 'ray::Actor', 'ray::ActorB', 'ray::IDLE'} == components
        return True
    wait_for_condition(verify_components, timeout=30)

    def verify_mem_usage():
        if False:
            while True:
                i = 10
        metrics = raw_metrics(addr)
        for metric in comp_metrics:
            samples = metrics[metric]
            for sample in samples:
                if sample.labels['Component'] == 'ray::ActorB':
                    assert sample.value > 0.0
                    print(sample)
                    print(sample.value)
                if sample.labels['Component'] == 'ray::Actor':
                    assert sample.value > 0.0
                    print(sample)
                    print(sample.value)
        return True
    wait_for_condition(verify_mem_usage, timeout=30)
    ray.kill(b)
    pid = ray.get(a.pid.remote())
    os.kill(pid, signal.SIGKILL)

    def verify_mem_cleaned():
        if False:
            while True:
                i = 10
        metrics = raw_metrics(addr)
        for metric in comp_metrics:
            samples = metrics[metric]
            for sample in samples:
                if sample.labels['Component'] == 'ray::ActorB':
                    assert sample.value == 0.0
                if sample.labels['Component'] == 'ray::Actor':
                    assert sample.value == 0.0
        return True
    wait_for_condition(verify_mem_cleaned, timeout=30)

def test_prometheus_file_based_service_discovery(ray_start_cluster):
    if False:
        print('Hello World!')
    NUM_NODES = 5
    cluster = ray_start_cluster
    nodes = [cluster.add_node() for _ in range(NUM_NODES)]
    cluster.wait_for_nodes()
    addr = ray.init(address=cluster.address)
    writer = PrometheusServiceDiscoveryWriter(addr['gcs_address'], '/tmp/ray')

    def get_metrics_export_address_from_node(nodes):
        if False:
            return 10
        node_export_addrs = ['{}:{}'.format(node.node_ip_address, node.metrics_export_port) for node in nodes]
        autoscaler_export_addr = '{}:{}'.format(cluster.head_node.node_ip_address, AUTOSCALER_METRIC_PORT)
        dashboard_export_addr = '{}:{}'.format(cluster.head_node.node_ip_address, DASHBOARD_METRIC_PORT)
        return node_export_addrs + [autoscaler_export_addr, dashboard_export_addr]
    loaded_json_data = json.loads(writer.get_file_discovery_content())[0]
    assert set(get_metrics_export_address_from_node(nodes)) == set(loaded_json_data['targets'])
    for _ in range(3):
        nodes.append(cluster.add_node())
    loaded_json_data = json.loads(writer.get_file_discovery_content())[0]
    assert set(get_metrics_export_address_from_node(nodes)) == set(loaded_json_data['targets'])

def test_prome_file_discovery_run_by_dashboard(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=0)
    global_node = ray._private.worker._global_node
    temp_dir = global_node.get_temp_dir_path()

    def is_service_discovery_exist():
        if False:
            i = 10
            return i + 15
        for path in pathlib.Path(temp_dir).iterdir():
            if PROMETHEUS_SERVICE_DISCOVERY_FILE in str(path):
                return True
        return False
    wait_for_condition(is_service_discovery_exist)

@pytest.fixture
def metric_mock():
    if False:
        print('Hello World!')
    mock = MagicMock()
    mock.record.return_value = 'haha'
    yield mock
'\nUnit test custom metrics.\n'

def test_basic_custom_metrics(metric_mock):
    if False:
        return 10
    count = Counter('count', tag_keys=('a',))
    with pytest.raises(TypeError):
        count.inc('hi')
    with pytest.raises(ValueError):
        count.inc(0)
    with pytest.raises(ValueError):
        count.inc(-1)
    count._metric = metric_mock
    count.inc(1, {'a': '1'})
    metric_mock.record.assert_called_with(1, tags={'a': '1'})
    gauge = Gauge('gauge', description='gauge')
    gauge._metric = metric_mock
    gauge.record(4)
    metric_mock.record.assert_called_with(4, tags={})
    histogram = Histogram('hist', description='hist', boundaries=[1.0, 3.0], tag_keys=('a', 'b'))
    histogram._metric = metric_mock
    tags = {'a': '10', 'b': 'b'}
    histogram.observe(8, tags=tags)
    metric_mock.record.assert_called_with(8, tags=tags)

def test_custom_metrics_info(metric_mock):
    if False:
        print('Hello World!')
    histogram = Histogram('hist', description='hist', boundaries=[1.0, 2.0], tag_keys=('a', 'b'))
    assert histogram.info['name'] == 'hist'
    assert histogram.info['description'] == 'hist'
    assert histogram.info['boundaries'] == [1.0, 2.0]
    assert histogram.info['tag_keys'] == ('a', 'b')
    assert histogram.info['default_tags'] == {}
    histogram.set_default_tags({'a': 'a'})
    assert histogram.info['default_tags'] == {'a': 'a'}

def test_custom_metrics_default_tags(metric_mock):
    if False:
        i = 10
        return i + 15
    histogram = Histogram('hist', description='hist', boundaries=[1.0, 2.0], tag_keys=('a', 'b')).set_default_tags({'b': 'b'})
    histogram._metric = metric_mock
    histogram.observe(10, tags={'a': 'a'})
    metric_mock.record.assert_called_with(10, tags={'a': 'a', 'b': 'b'})
    tags = {'a': '10', 'b': 'c'}
    histogram.observe(8, tags=tags)
    metric_mock.record.assert_called_with(8, tags=tags)

def test_custom_metrics_edge_cases(metric_mock):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):
        Histogram('hist')
    with pytest.raises(ValueError):
        Histogram('hist', boundaries=[])
    with pytest.raises(ValueError):
        Counter('')
    with pytest.raises(TypeError):
        Counter('name', tag_keys='a')
    with pytest.raises(ValueError):
        Histogram('hist', boundaries=[-1, 1, 2])
    with pytest.raises(ValueError):
        Histogram('hist', boundaries=[0, 1, 2])
    with pytest.raises(ValueError):
        Histogram('hist', boundaries=[-1, -0.5, -0.1])

def test_metrics_override_shouldnt_warn(ray_start_regular, log_pubsub):
    if False:
        i = 10
        return i + 15

    @ray.remote
    def override():
        if False:
            for i in range(10):
                print('nop')
        a = Counter('num_count', description='')
        b = Counter('num_count', description='')
        a.inc(1)
        b.inc(1)
    ray.get(override.remote())

    def matcher(log_batch):
        if False:
            while True:
                i = 10
        return any(('Attempt to register measure' in line for line in log_batch['lines']))
    match = get_log_batch(log_pubsub, 1, timeout=5, matcher=matcher)
    assert len(match) == 0, match

def test_custom_metrics_validation(shutdown_only):
    if False:
        print('Hello World!')
    ray.init()
    metric = Counter('name', tag_keys=('a', 'b'))
    metric.set_default_tags({'a': '1'})
    metric.inc(1.0, {'b': '2'})
    metric.inc(1.0, {'a': '1', 'b': '2'})
    with pytest.raises(ValueError):
        metric.inc(1.0)
    with pytest.raises(ValueError):
        metric.inc(1.0, {'a': '2'})
    metric = Counter('name', tag_keys=('a',))
    with pytest.raises(ValueError):
        metric.inc(1.0, {'a': '1', 'b': '2'})
    with pytest.raises(TypeError):
        Counter('name', tag_keys='a')
    with pytest.raises(TypeError):
        Counter('name', tag_keys=(1,))
    metric = Counter('name', tag_keys=('a',))
    with pytest.raises(ValueError):
        metric.set_default_tags({'a': '1', 'c': '2'})
    with pytest.raises(TypeError):
        metric.set_default_tags({'a': 1})
    with pytest.raises(TypeError):
        metric.inc(1.0, {'a': 1})

@pytest.mark.parametrize('_setup_cluster_for_test', [False], indirect=True)
def test_metrics_disablement(_setup_cluster_for_test):
    if False:
        for i in range(10):
            print('nop')
    'Make sure the metrics are not exported when it is disabled.'
    (prom_addresses, autoscaler_export_addr, _) = _setup_cluster_for_test

    def verify_metrics_not_collected():
        if False:
            i = 10
            return i + 15
        (components_dict, metric_names, _) = fetch_prometheus(prom_addresses)
        for (_, comp) in components_dict.items():
            if len(comp) > 0:
                print(f'metrics from a component {comp} exists although it should not.')
                return False
        for metric in _METRICS + _AUTOSCALER_METRICS + _DASHBOARD_METRICS:
            if metric in metric_names:
                print('f{metric} exists although it should not.')
                return False
        return True
    for _ in range(10):
        assert verify_metrics_not_collected()
        import time
        time.sleep(1)
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))