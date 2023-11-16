import decimal
import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from ray.autoscaler._private.constants import DISABLE_LAUNCH_CONFIG_CHECK_KEY, DISABLE_NODE_UPDATERS_KEY, FOREGROUND_NODE_LAUNCH_KEY, WORKER_LIVENESS_CHECK_KEY, WORKER_RPC_DRAIN_KEY
from ray.autoscaler._private.kuberay import node_provider, utils
from ray.autoscaler._private.util import validate_config
logger = logging.getLogger(__name__)
AUTOSCALER_OPTIONS_KEY = 'autoscalerOptions'
IDLE_SECONDS_KEY = 'idleTimeoutSeconds'
UPSCALING_KEY = 'upscalingMode'
UPSCALING_VALUE_AGGRESSIVE = 'Aggressive'
UPSCALING_VALUE_DEFAULT = 'Default'
UPSCALING_VALUE_CONSERVATIVE = 'Conservative'
MAX_RAYCLUSTER_FETCH_TRIES = 5
RAYCLUSTER_FETCH_RETRY_S = 5
_HEAD_GROUP_NAME = 'head-group'

class AutoscalingConfigProducer:
    """Produces an autoscaling config by reading data from the RayCluster CR.

    Used to fetch the autoscaling config at the beginning of each autoscaler iteration.

    In the context of Ray deployment on Kubernetes, the autoscaling config is an
    internal interface.

    The autoscaling config carries the strict subset of RayCluster CR data required by
    the autoscaler to make scaling decisions; in particular, the autoscaling config does
    not carry pod configuration data.

    This class is the only public object in this file.
    """

    def __init__(self, ray_cluster_name, ray_cluster_namespace):
        if False:
            return 10
        (self._headers, self._verify) = node_provider.load_k8s_secrets()
        self._ray_cr_url = node_provider.url_from_resource(namespace=ray_cluster_namespace, path=f'rayclusters/{ray_cluster_name}')

    def __call__(self):
        if False:
            return 10
        ray_cr = self._fetch_ray_cr_from_k8s_with_retries()
        autoscaling_config = _derive_autoscaling_config_from_ray_cr(ray_cr)
        return autoscaling_config

    def _fetch_ray_cr_from_k8s_with_retries(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Fetch the RayCluster CR by querying the K8s API server.\n\n        Retry on HTTPError for robustness, in particular to protect autoscaler\n        initialization.\n        '
        for i in range(1, MAX_RAYCLUSTER_FETCH_TRIES + 1):
            try:
                return self._fetch_ray_cr_from_k8s()
            except requests.HTTPError as e:
                if i < MAX_RAYCLUSTER_FETCH_TRIES:
                    logger.exception('Failed to fetch RayCluster CR from K8s. Retrying.')
                    time.sleep(RAYCLUSTER_FETCH_RETRY_S)
                else:
                    raise e from None
        raise AssertionError

    def _fetch_ray_cr_from_k8s(self) -> Dict[str, Any]:
        if False:
            return 10
        result = requests.get(self._ray_cr_url, headers=self._headers, verify=self._verify)
        if not result.status_code == 200:
            result.raise_for_status()
        ray_cr = result.json()
        return ray_cr

def _derive_autoscaling_config_from_ray_cr(ray_cr: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    provider_config = _generate_provider_config(ray_cr['metadata']['namespace'])
    available_node_types = _generate_available_node_types_from_ray_cr_spec(ray_cr['spec'])
    global_max_workers = sum((node_type['max_workers'] for node_type in available_node_types.values()))
    legacy_autoscaling_fields = _generate_legacy_autoscaling_config_fields()
    autoscaler_options = ray_cr['spec'].get(AUTOSCALER_OPTIONS_KEY, {})
    if IDLE_SECONDS_KEY in autoscaler_options:
        idle_timeout_minutes = autoscaler_options[IDLE_SECONDS_KEY] / 60.0
    else:
        idle_timeout_minutes = 1.0
    if autoscaler_options.get(UPSCALING_KEY) == UPSCALING_VALUE_CONSERVATIVE:
        upscaling_speed = 1
    elif autoscaler_options.get(UPSCALING_KEY) == UPSCALING_VALUE_DEFAULT:
        upscaling_speed = 1000
    elif autoscaler_options.get(UPSCALING_KEY) == UPSCALING_VALUE_AGGRESSIVE:
        upscaling_speed = 1000
    else:
        upscaling_speed = 1000
    autoscaling_config = {'provider': provider_config, 'cluster_name': ray_cr['metadata']['name'], 'head_node_type': _HEAD_GROUP_NAME, 'available_node_types': available_node_types, 'max_workers': global_max_workers, 'idle_timeout_minutes': idle_timeout_minutes, 'upscaling_speed': upscaling_speed, **legacy_autoscaling_fields}
    validate_config(autoscaling_config)
    return autoscaling_config

def _generate_provider_config(ray_cluster_namespace: str) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    'Generates the `provider` field of the autoscaling config, which carries data\n    required to instantiate the KubeRay node provider.\n    '
    return {'type': 'kuberay', 'namespace': ray_cluster_namespace, DISABLE_NODE_UPDATERS_KEY: True, DISABLE_LAUNCH_CONFIG_CHECK_KEY: True, FOREGROUND_NODE_LAUNCH_KEY: True, WORKER_LIVENESS_CHECK_KEY: False, WORKER_RPC_DRAIN_KEY: True}

def _generate_legacy_autoscaling_config_fields() -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    'Generates legacy autoscaling config fields required for compatibiliy.'
    return {'file_mounts': {}, 'cluster_synced_files': [], 'file_mounts_sync_continuously': False, 'initialization_commands': [], 'setup_commands': [], 'head_setup_commands': [], 'worker_setup_commands': [], 'head_start_ray_commands': [], 'worker_start_ray_commands': [], 'auth': {}}

def _generate_available_node_types_from_ray_cr_spec(ray_cr_spec: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    'Formats autoscaler "available_node_types" field based on the Ray CR\'s group\n    specs.\n    '
    headGroupSpec = ray_cr_spec['headGroupSpec']
    return {_HEAD_GROUP_NAME: _node_type_from_group_spec(headGroupSpec, is_head=True), **{worker_group_spec['groupName']: _node_type_from_group_spec(worker_group_spec, is_head=False) for worker_group_spec in ray_cr_spec['workerGroupSpecs']}}

def _node_type_from_group_spec(group_spec: Dict[str, Any], is_head: bool) -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    'Converts CR group spec to autoscaler node type.'
    if is_head:
        min_workers = max_workers = 0
    else:
        min_workers = group_spec['minReplicas']
        max_workers = group_spec['maxReplicas']
    resources = _get_ray_resources_from_group_spec(group_spec, is_head)
    return {'min_workers': min_workers, 'max_workers': max_workers, 'node_config': {}, 'resources': resources}

def _get_ray_resources_from_group_spec(group_spec: Dict[str, Any], is_head: bool) -> Dict[str, int]:
    if False:
        while True:
            i = 10
    '\n    Infers Ray resources from rayStartCommands and K8s limits.\n    The resources extracted are used in autoscaling calculations.\n\n    TODO: Expose a better interface in the RayCluster CRD for Ray resource annotations.\n    For now, we take the rayStartParams as the primary source of truth.\n    '
    ray_start_params = group_spec['rayStartParams']
    k8s_resource_limits = group_spec['template']['spec']['containers'][0].get('resources', {}).get('limits', {})
    group_name = _HEAD_GROUP_NAME if is_head else group_spec['groupName']
    num_cpus = _get_num_cpus(ray_start_params, k8s_resource_limits, group_name)
    num_gpus = _get_num_gpus(ray_start_params, k8s_resource_limits, group_name)
    custom_resource_dict = _get_custom_resources(ray_start_params, group_name)
    memory = _get_memory(ray_start_params, k8s_resource_limits)
    resources = {}
    assert isinstance(num_cpus, int)
    resources['CPU'] = num_cpus
    if num_gpus is not None:
        resources['GPU'] = num_gpus
    if memory is not None:
        resources['memory'] = memory
    resources.update(custom_resource_dict)
    return resources

def _get_num_cpus(ray_start_params: Dict[str, str], k8s_resource_limits: Dict[str, str], group_name: str) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Get CPU annotation from ray_start_params or k8s_resource_limits,\n    with priority for ray_start_params.\n    '
    if 'num-cpus' in ray_start_params:
        return int(ray_start_params['num-cpus'])
    elif 'cpu' in k8s_resource_limits:
        cpu_quantity: str = k8s_resource_limits['cpu']
        return _round_up_k8s_quantity(cpu_quantity)
    else:
        raise ValueError(f'Autoscaler failed to detect `CPU` resources for group {group_name}.\nSet the `--num-cpus` rayStartParam and/or the CPU resource limit for the Ray container.')

def _get_memory(ray_start_params: Dict[str, str], k8s_resource_limits: Dict[str, Any]) -> Optional[int]:
    if False:
        for i in range(10):
            print('nop')
    'Get memory resource annotation from ray_start_params or k8s_resource_limits,\n    with priority for ray_start_params.\n    '
    if 'memory' in ray_start_params:
        return int(ray_start_params['memory'])
    elif 'memory' in k8s_resource_limits:
        memory_quantity: str = k8s_resource_limits['memory']
        return _round_up_k8s_quantity(memory_quantity)
    return None

def _get_num_gpus(ray_start_params: Dict[str, str], k8s_resource_limits: Dict[str, Any], group_name: str) -> Optional[int]:
    if False:
        print('Hello World!')
    'Get memory resource annotation from ray_start_params or k8s_resource_limits,\n    with priority for ray_start_params.\n    '
    if 'num-gpus' in ray_start_params:
        return int(ray_start_params['num-gpus'])
    else:
        for key in k8s_resource_limits:
            if key.endswith('gpu'):
                gpu_resource_quantity = k8s_resource_limits[key]
                num_gpus = _round_up_k8s_quantity(gpu_resource_quantity)
                if num_gpus > 0:
                    return num_gpus
    return None

def _round_up_k8s_quantity(quantity: str) -> int:
    if False:
        i = 10
        return i + 15
    'Rounds a Kubernetes resource quantity up to the nearest integer.\n\n    Args:\n        quantity: Resource quantity as a string in the canonical K8s form.\n\n    Returns:\n        The quantity, rounded up, as an integer.\n    '
    resource_decimal: decimal.Decimal = utils.parse_quantity(quantity)
    rounded = resource_decimal.to_integral_value(rounding=decimal.ROUND_UP)
    return int(rounded)

def _get_custom_resources(ray_start_params: Dict[str, Any], group_name: str) -> Dict[str, int]:
    if False:
        print('Hello World!')
    'Format custom resources based on the `resources` Ray start param.\n\n    Currently, the value of the `resources` field must\n    be formatted as follows:\n    \'"{"Custom1": 1, "Custom2": 5}"\'.\n\n    This method first converts the input to a correctly formatted\n    json string and then loads that json string to a dict.\n    '
    if 'resources' not in ray_start_params:
        return {}
    resources_string = ray_start_params['resources']
    try:
        resources_json = resources_string[1:-1].replace('\\', '')
        resources = json.loads(resources_json)
        assert isinstance(resources, dict)
        for (key, value) in resources.items():
            assert isinstance(key, str)
            assert isinstance(value, int)
    except Exception as e:
        logger.error(f'Error reading `resource` rayStartParam for group {group_name}. For the correct format, refer to example configuration at https://github.com/ray-project/ray/blob/master/python/ray/autoscaler/kuberay/ray-cluster.complete.yaml.')
        raise e
    return resources