import asyncio
import time
from typing import Dict, Optional, List
from pprint import pprint
import requests
import ray
import logging
from collections import defaultdict
from ray.util.state import list_nodes
from ray._private.test_utils import fetch_prometheus_metrics
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from pydantic import BaseModel
from ray.dashboard.consts import DASHBOARD_METRIC_PORT
from ray.dashboard.utils import get_address_for_submission_client
logger = logging.getLogger(__name__)

def calc_p(latencies, percent):
    if False:
        print('Hello World!')
    if len(latencies) == 0:
        return 0
    return round(sorted(latencies)[int(len(latencies) / 100 * percent)] * 1000, 3)

class Result(BaseModel):
    success: bool
    result: Dict[str, List[float]]
    memory_mb: Optional[float]
endpoints = ['/logical/actors', '/nodes?view=summary', '/', '/api/cluster_status', '/events', '/api/jobs/', '/log_index', '/api/prometheus_health']

@ray.remote(num_cpus=0)
class DashboardTester:

    def __init__(self, interval_s: int=1):
        if False:
            for i in range(10):
                print('nop')
        self.dashboard_url = get_address_for_submission_client(None)
        self.interval_s = interval_s
        self.result = defaultdict(list)

    async def run(self):
        await asyncio.gather(*[self.ping(endpoint) for endpoint in endpoints])

    async def ping(self, endpoint):
        """Synchronously call an endpoint."""
        while True:
            start = time.monotonic()
            resp = requests.get(self.dashboard_url + endpoint, timeout=30)
            elapsed = time.monotonic() - start
            if resp.status_code == 200:
                self.result[endpoint].append(time.monotonic() - start)
            else:
                try:
                    resp.raise_for_status()
                except Exception as e:
                    logger.exception(e)
            await asyncio.sleep(max(0, self.interval_s, elapsed))

    def get_result(self):
        if False:
            i = 10
            return i + 15
        return self.result

class DashboardTestAtScale:
    """This is piggybacked into existing scalability tests."""

    def __init__(self, addr: ray._private.worker.RayContext):
        if False:
            return 10
        self.addr = addr
        current_node_ip = ray._private.worker.global_worker.node_ip_address
        nodes = list_nodes(filters=[('node_ip', '=', current_node_ip)])
        assert len(nodes) > 0, f'{current_node_ip} not found in the cluster'
        node = nodes[0]
        self.tester = DashboardTester.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node['node_id'], soft=False)).remote()
        self.tester.run.remote()

    def get_result(self):
        if False:
            print('Hello World!')
        'Get the result from the test.\n\n        Returns:\n            A tuple of success, and the result (Result object).\n        '
        try:
            result = ray.get(self.tester.get_result.remote(), timeout=60)
        except ray.exceptions.GetTimeoutError:
            return Result(success=False)
        dashboard_export_addr = '{}:{}'.format(self.addr['raylet_ip_address'], DASHBOARD_METRIC_PORT)
        metrics = fetch_prometheus_metrics([dashboard_export_addr])
        memories = []
        for (name, samples) in metrics.items():
            if name == 'ray_component_uss_mb':
                for sample in samples:
                    if sample.labels['Component'] == 'dashboard':
                        memories.append(sample.value)
        return Result(success=True, result=result, memory_mb=max(memories) if memories else None)

    def update_release_test_result(self, release_result: dict):
        if False:
            print('Hello World!')
        test_result = self.get_result()

        def calc_endpoints_p(result, percent):
            if False:
                return 10
            return {endpoint: calc_p(latencies, percent) for (endpoint, latencies) in result.items()}
        print('======Print per dashboard endpoint latencies======')
        print('=====================P50==========================')
        pprint(calc_endpoints_p(test_result.result, 50))
        print('=====================P95==========================')
        pprint(calc_endpoints_p(test_result.result, 95))
        print('=====================P99==========================')
        pprint(calc_endpoints_p(test_result.result, 99))
        latencies = []
        for per_endpoint_latencies in test_result.result.values():
            latencies.extend(per_endpoint_latencies)
        aggregated_metrics = {'p50': calc_p(latencies, 50), 'p95': calc_p(latencies, 95), 'p99': calc_p(latencies, 99)}
        print('=====================Aggregated====================')
        pprint(aggregated_metrics)
        release_result['_dashboard_test_success'] = test_result.success
        if test_result.success:
            if 'perf_metrics' not in release_result:
                release_result['perf_metrics'] = []
            release_result['perf_metrics'].extend([{'perf_metric_name': f'dashboard_{p}_latency_ms', 'perf_metric_value': value, 'perf_metric_type': 'LATENCY'} for (p, value) in aggregated_metrics.items()])
            release_result['_dashboard_memory_usage_mb'] = test_result.memory_mb