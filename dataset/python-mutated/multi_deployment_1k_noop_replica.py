"""
Benchmark test for multi deployment at 1k no-op replica scale.

1) Start with a single head node.
2) Start 1000 deployments each with 10 no-op replicas
3) Launch wrk in each running node to simulate load balanced request
4) Recursively send queries to random deployments, up to depth=5
5) Run a 10-minute wrk trial on each node, aggregate results.

Report:
    per_thread_latency_avg_ms
    per_thread_latency_max_ms
    per_thread_avg_tps
    per_thread_max_tps

    per_node_avg_tps
    per_node_avg_transfer_per_sec_KB

    cluster_total_thoughput
    cluster_total_transfer_KB
    cluster_max_P50_latency_ms
    cluster_max_P75_latency_ms
    cluster_max_P90_latency_ms
    cluster_max_P99_latency_ms
"""
import click
import logging
import math
import random
from ray import serve
from serve_test_utils import aggregate_all_metrics, run_wrk_on_all_nodes, save_test_results, is_smoke_test
from serve_test_cluster_utils import setup_local_single_node_cluster, setup_anyscale_cluster, NUM_CPU_PER_NODE, NUM_CONNECTIONS
from typing import List, Optional
logger = logging.getLogger(__file__)
DEFAULT_SMOKE_TEST_NUM_REPLICA = 4
DEFAULT_SMOKE_TEST_NUM_DEPLOYMENTS = 4
DEFAULT_FULL_TEST_NUM_REPLICA = 1000
DEFAULT_FULL_TEST_NUM_DEPLOYMENTS = 10
DEFAULT_SMOKE_TEST_TRIAL_LENGTH = '5s'
DEFAULT_FULL_TEST_TRIAL_LENGTH = '10m'

def setup_multi_deployment_replicas(num_replicas, num_deployments) -> List[str]:
    if False:
        return 10
    num_replica_per_deployment = num_replicas // num_deployments
    all_deployment_names = [f'Echo_{i + 1}' for i in range(num_deployments)]

    @serve.deployment(num_replicas=num_replica_per_deployment)
    class Echo:

        def __init__(self):
            if False:
                return 10
            self.all_app_async_handles = []

        async def get_random_async_handle(self):
            if len(self.all_app_async_handles) < len(all_deployment_names):
                applications = list(serve.status().applications.keys())
                self.all_app_async_handles = [serve.get_app_handle(app) for app in applications]
            return random.choice(self.all_app_async_handles)

        async def handle_request(self, request, depth: int):
            if depth > 4:
                return 'hi'
            next_async_handle = await self.get_random_async_handle()
            fut = next_async_handle.handle_request.remote(request, depth + 1)
            return await fut

        async def __call__(self, request):
            return await self.handle_request(request, 0)
    for name in all_deployment_names:
        serve.run(Echo.bind(), name=name, route_prefix=f'/{name}')
    return all_deployment_names

@click.command()
@click.option('--num-replicas', type=int)
@click.option('--num-deployments', type=int)
@click.option('--trial-length', type=str)
def main(num_replicas: Optional[int], num_deployments: Optional[int], trial_length: Optional[str]):
    if False:
        return 10
    if is_smoke_test():
        num_replicas = num_replicas or DEFAULT_SMOKE_TEST_NUM_REPLICA
        num_deployments = num_deployments or DEFAULT_SMOKE_TEST_NUM_DEPLOYMENTS
        trial_length = trial_length or DEFAULT_SMOKE_TEST_TRIAL_LENGTH
        logger.info(f'Running smoke test with {num_replicas} replicas, {num_deployments} deployments .. \n')
        num_nodes = int(math.ceil(num_replicas / NUM_CPU_PER_NODE))
        logger.info(f'Setting up local ray cluster with {num_nodes} nodes .. \n')
        serve_client = setup_local_single_node_cluster(num_nodes)[0]
    else:
        num_replicas = num_replicas or DEFAULT_FULL_TEST_NUM_REPLICA
        num_deployments = num_deployments or DEFAULT_FULL_TEST_NUM_DEPLOYMENTS
        trial_length = trial_length or DEFAULT_FULL_TEST_TRIAL_LENGTH
        logger.info(f'Running full test with {num_replicas} replicas, {num_deployments} deployments .. \n')
        logger.info('Setting up anyscale ray cluster .. \n')
        serve_client = setup_anyscale_cluster()
    http_host = str(serve_client._http_config.host)
    http_port = str(serve_client._http_config.port)
    logger.info(f'Ray serve http_host: {http_host}, http_port: {http_port}')
    logger.info(f'Deploying with {num_replicas} target replicas ....\n')
    all_endpoints = setup_multi_deployment_replicas(num_replicas, num_deployments)
    logger.info('Warming up cluster...\n')
    run_wrk_on_all_nodes(DEFAULT_SMOKE_TEST_TRIAL_LENGTH, NUM_CONNECTIONS, http_host, http_port, all_endpoints=all_endpoints, ignore_output=True)
    logger.info(f'Starting wrk trial on all nodes for {trial_length} ....\n')
    (all_metrics, all_wrk_stdout) = run_wrk_on_all_nodes(trial_length, NUM_CONNECTIONS, http_host, http_port, all_endpoints=all_endpoints)
    aggregated_metrics = aggregate_all_metrics(all_metrics)
    logger.info('Wrk stdout on each node: ')
    for wrk_stdout in all_wrk_stdout:
        logger.info(wrk_stdout)
    logger.info('Final aggregated metrics: ')
    for (key, val) in aggregated_metrics.items():
        logger.info(f'{key}: {val}')
    save_test_results(aggregated_metrics, default_output_file='/tmp/multi_deployment_1k_noop_replica.json')
if __name__ == '__main__':
    main()
    import pytest
    import sys
    sys.exit(pytest.main(['-v', '-s', __file__]))