from collections import defaultdict
import os
import pytest
import ray
from ray._private.test_utils import fetch_prometheus_metrics, run_string_as_driver_nonblocking, wait_for_condition
METRIC_CONFIG = {'_system_config': {'metrics_report_interval_ms': 100}}

def raw_metrics(info):
    if False:
        print('Hello World!')
    metrics_page = 'localhost:{}'.format(info['metrics_export_port'])
    print('Fetch metrics from', metrics_page)
    res = fetch_prometheus_metrics([metrics_page])
    return res

def resources_by_state(info) -> dict:
    if False:
        return 10
    res = raw_metrics(info)
    if 'ray_resources' in res:
        states = defaultdict(int)
        for sample in res['ray_resources']:
            state = sample.labels['State']
            name = sample.labels['Name']
            states[name, state] += sample.value
            if states[name, state] == 0:
                del states[name, state]
        print('Resources by state: {}'.format(states))
        return states
    else:
        return {}

def test_resources_metrics(shutdown_only):
    if False:
        while True:
            i = 10
    info = ray.init(num_cpus=4, num_gpus=2, resources={'a': 3}, **METRIC_CONFIG)
    driver = '\nimport ray\nimport time\n\nray.init("auto")\n\n@ray.remote(num_gpus=1, num_cpus=1, resources={"a": 1})\ndef f():\n    time.sleep(999)\n\npg = ray.util.placement_group(bundles=[{"CPU": 1}])\n\nray.get([f.remote() for _ in range(2)])'
    proc = run_string_as_driver_nonblocking(driver)

    def verify():
        if False:
            while True:
                i = 10
        state = resources_by_state(info)
        assert ('object_store_memory', 'AVAILABLE') in state
        assert ('memory', 'AVAILABLE') in state
        node_ip_resource = None
        pg_resources = []
        for resource in ray.available_resources():
            if resource.startswith('node:'):
                node_ip_resource = resource
            elif '_group_' in resource:
                pg_resources.append(resource)
        assert (node_ip_resource, 'AVAILABLE') not in state
        assert (node_ip_resource, 'USED') not in state
        for pg_resource in pg_resources:
            assert (pg_resource, 'AVAILABLE') not in state
            assert (pg_resource, 'USED') not in state
        assert state['CPU', 'AVAILABLE'] == 1.0
        assert state['CPU', 'USED'] == 3.0
        assert ('GPU', 'AVAILABLE') not in state
        assert state['GPU', 'USED'] == 2.0
        assert state['a', 'AVAILABLE'] == 1.0
        assert state['a', 'USED'] == 2.0
        return True
    wait_for_condition(verify, timeout=20, retry_interval_ms=500)
    proc.kill()
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))