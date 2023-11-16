import logging
import ray
import requests
from ray._private.test_utils import monitor_memory_usage
from ray.cluster_utils import Cluster
from ray import serve
from ray.serve.config import DeploymentMode
from ray.serve.context import _get_global_client
logger = logging.getLogger(__file__)
NUM_CPU_PER_NODE = 10
NUM_CONNECTIONS = 10

def setup_local_single_node_cluster(num_nodes: int, num_cpu_per_node=NUM_CPU_PER_NODE, namespace='serve'):
    if False:
        i = 10
        return i + 15
    'Setup ray cluster locally via ray.init() and Cluster()\n\n    Each actor is simulated in local process on single node,\n    thus smaller scale by default.\n    '
    cluster = Cluster()
    for i in range(num_nodes):
        cluster.add_node(redis_port=6380 if i == 0 else None, num_cpus=num_cpu_per_node, num_gpus=0, resources={str(i): 2, 'proxy': 1})
    ray.init(address=cluster.address, dashboard_host='0.0.0.0', namespace=namespace)
    serve.start(detached=True, proxy_location=DeploymentMode.EveryNode)
    return (_get_global_client(), cluster)

def setup_anyscale_cluster():
    if False:
        print('Hello World!')
    'Setup ray cluster at anyscale via ray.client()\n\n    Note this is by default large scale and should be kicked off\n    less frequently.\n    '
    ray.init(address='auto', runtime_env={'env_vars': {'SERVE_ENABLE_SCALING_LOG': '0'}})
    serve.start(proxy_location=DeploymentMode.EveryNode)
    monitor_memory_usage()
    return _get_global_client()

@ray.remote
def warm_up_one_cluster(num_warmup_iterations: int, http_host: str, http_port: str, endpoint: str, nonblocking: bool=False) -> None:
    if False:
        print('Hello World!')
    timeout = 0.0001 if nonblocking else None
    logger.info(f'Warming up {endpoint} ..')
    for _ in range(num_warmup_iterations):
        try:
            resp = requests.get(f'http://{http_host}:{http_port}/{endpoint}', timeout=timeout).text
            logger.info(resp)
        except requests.exceptions.ReadTimeout:
            logger.info('Issued nonblocking HTTP request.')
    return endpoint