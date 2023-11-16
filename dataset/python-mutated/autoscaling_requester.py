import math
import threading
import time
from typing import Dict, List
import ray
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
RESOURCE_REQUEST_TIMEOUT = 60
PURGE_INTERVAL = RESOURCE_REQUEST_TIMEOUT * 2
ARTIFICIAL_CPU_SCALING_FACTOR = 1.2

@ray.remote(num_cpus=0, max_restarts=-1, max_task_retries=-1)
class AutoscalingRequester:
    """Actor to make resource requests to autoscaler for the datasets.

    The resource requests are set to timeout after RESOURCE_REQUEST_TIMEOUT seconds.
    For those live requests, we keep track of the last request made for each execution,
    which overrides all previous requests it made; then sum the requested amounts
    across all executions as the final request to the autoscaler.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._resource_requests = {}
        self._timeout = RESOURCE_REQUEST_TIMEOUT
        self._self_handle = ray.get_runtime_context().current_actor

        def purge_thread_run():
            if False:
                return 10
            while True:
                time.sleep(PURGE_INTERVAL)
                ray.get(self._self_handle.purge_expired_requests.remote())
        self._purge_thread = threading.Thread(target=purge_thread_run, daemon=True)
        self._purge_thread.start()

    def purge_expired_requests(self):
        if False:
            return 10
        self._purge()
        ray.autoscaler.sdk.request_resources(bundles=self._aggregate_requests())

    def request_resources(self, req: List[Dict], execution_id: str):
        if False:
            for i in range(10):
                print('nop')
        self._purge()
        self._resource_requests[execution_id] = (req, time.time() + self._timeout)
        ray.autoscaler.sdk.request_resources(bundles=self._aggregate_requests())

    def _purge(self):
        if False:
            return 10
        now = time.time()
        for (k, (_, t)) in list(self._resource_requests.items()):
            if t < now:
                self._resource_requests.pop(k)

    def _aggregate_requests(self) -> List[Dict]:
        if False:
            for i in range(10):
                print('nop')
        req = []
        for (_, (r, _)) in self._resource_requests.items():
            req.extend(r)

        def get_cpus(req):
            if False:
                for i in range(10):
                    print('nop')
            num_cpus = 0
            for r in req:
                if 'CPU' in r:
                    num_cpus += r['CPU']
            return num_cpus
        num_cpus = get_cpus(req)
        if num_cpus > 0:
            total = ray.cluster_resources()
            if 'CPU' in total and num_cpus <= total['CPU']:
                delta = math.ceil(ARTIFICIAL_CPU_SCALING_FACTOR * total['CPU']) - num_cpus
                req.extend([{'CPU': 1}] * delta)
        return req

    def _test_set_timeout(self, ttl):
        if False:
            print('Hello World!')
        'Set the timeout. This is for test only'
        self._timeout = ttl

def get_or_create_autoscaling_requester_actor():
    if False:
        for i in range(10):
            print('nop')
    ctx = DataContext.get_current()
    scheduling_strategy = ctx.scheduling_strategy
    scheduling_strategy = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=True, _spill_on_unavailable=True)
    return AutoscalingRequester.options(name='AutoscalingRequester', namespace='AutoscalingRequester', get_if_exists=True, lifetime='detached', scheduling_strategy=scheduling_strategy).remote()