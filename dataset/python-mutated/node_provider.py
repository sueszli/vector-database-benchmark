import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import requests
from ray.autoscaler._private.constants import WORKER_LIVENESS_CHECK_KEY, WORKER_RPC_DRAIN_KEY
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.batching_node_provider import BatchingNodeProvider, NodeData, ScaleRequest
from ray.autoscaler.tags import NODE_KIND_HEAD, NODE_KIND_WORKER, STATUS_UP_TO_DATE, STATUS_UPDATE_FAILED, TAG_RAY_USER_NODE_TYPE
KUBERAY_LABEL_KEY_KIND = 'ray.io/node-type'
KUBERAY_LABEL_KEY_TYPE = 'ray.io/group'
KUBERAY_KIND_HEAD = 'head'
KUBERAY_TYPE_HEAD = 'head-group'
KUBERAY_CRD_VER = os.getenv('KUBERAY_CRD_VER', 'v1alpha1')
RAY_HEAD_POD_NAME = os.getenv('RAY_HEAD_POD_NAME')
logger = logging.getLogger(__name__)
provider_exists = False

def node_data_from_pod(pod: Dict[str, Any]) -> NodeData:
    if False:
        for i in range(10):
            print('nop')
    'Converts a Ray pod extracted from K8s into Ray NodeData.\n    NodeData is processed by BatchingNodeProvider.\n    '
    (kind, type) = kind_and_type(pod)
    status = status_tag(pod)
    ip = pod_ip(pod)
    return NodeData(kind=kind, type=type, status=status, ip=ip)

def kind_and_type(pod: Dict[str, Any]) -> Tuple[NodeKind, NodeType]:
    if False:
        i = 10
        return i + 15
    "Determine Ray node kind (head or workers) and node type (worker group name)\n    from a Ray pod's labels.\n    "
    labels = pod['metadata']['labels']
    if labels[KUBERAY_LABEL_KEY_KIND] == KUBERAY_KIND_HEAD:
        kind = NODE_KIND_HEAD
        type = KUBERAY_TYPE_HEAD
    else:
        kind = NODE_KIND_WORKER
        type = labels[KUBERAY_LABEL_KEY_TYPE]
    return (kind, type)

def pod_ip(pod: Dict[str, Any]) -> NodeIP:
    if False:
        for i in range(10):
            print('nop')
    return pod['status'].get('podIP', 'IP not yet assigned')

def status_tag(pod: Dict[str, Any]) -> NodeStatus:
    if False:
        print('Hello World!')
    'Convert pod state to Ray autoscaler node status.\n\n    See the doc string of the class\n    batching_node_provider.NodeData for the semantics of node status.\n    '
    if 'containerStatuses' not in pod['status'] or not pod['status']['containerStatuses']:
        return 'pending'
    state = pod['status']['containerStatuses'][0]['state']
    if 'pending' in state:
        return 'pending'
    if 'running' in state:
        return STATUS_UP_TO_DATE
    if 'waiting' in state:
        return 'waiting'
    if 'terminated' in state:
        return STATUS_UPDATE_FAILED
    raise ValueError('Unexpected container state.')

def worker_delete_patch(group_index: str, workers_to_delete: List[NodeID]):
    if False:
        for i in range(10):
            print('nop')
    path = f'/spec/workerGroupSpecs/{group_index}/scaleStrategy'
    value = {'workersToDelete': workers_to_delete}
    return replace_patch(path, value)

def worker_replica_patch(group_index: str, target_replicas: int):
    if False:
        while True:
            i = 10
    path = f'/spec/workerGroupSpecs/{group_index}/replicas'
    value = target_replicas
    return replace_patch(path, value)

def replace_patch(path: str, value: Any) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    return {'op': 'replace', 'path': path, 'value': value}

def load_k8s_secrets() -> Tuple[Dict[str, str], str]:
    if False:
        i = 10
        return i + 15
    '\n    Loads secrets needed to access K8s resources.\n\n    Returns:\n        headers: Headers with K8s access token\n        verify: Path to certificate\n    '
    with open('/var/run/secrets/kubernetes.io/serviceaccount/token') as secret:
        token = secret.read()
    headers = {'Authorization': 'Bearer ' + token}
    verify = '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt'
    return (headers, verify)

def url_from_resource(namespace: str, path: str) -> str:
    if False:
        print('Hello World!')
    'Convert resource path to REST URL for Kubernetes API server.\n\n    Args:\n        namespace: The K8s namespace of the resource\n        path: The part of the resource path that starts with the resource type.\n            Supported resource types are "pods" and "rayclusters".\n    '
    if path.startswith('pods'):
        api_group = '/api/v1'
    elif path.startswith('rayclusters'):
        api_group = '/apis/ray.io/' + KUBERAY_CRD_VER
    else:
        raise NotImplementedError('Tried to access unknown entity at {}'.format(path))
    return 'https://kubernetes.default:443' + api_group + '/namespaces/' + namespace + '/' + path

def _worker_group_index(raycluster: Dict[str, Any], group_name: str) -> int:
    if False:
        print('Hello World!')
    'Extract worker group index from RayCluster.'
    group_names = [spec['groupName'] for spec in raycluster['spec'].get('workerGroupSpecs', [])]
    return group_names.index(group_name)

def _worker_group_max_replicas(raycluster: Dict[str, Any], group_index: int) -> Optional[int]:
    if False:
        for i in range(10):
            print('nop')
    'Extract the maxReplicas of a worker group.\n\n    If maxReplicas is unset, return None, to be interpreted as "no constraint".\n    At time of writing, it should be impossible for maxReplicas to be unset, but it\'s\n    better to handle this anyway.\n    '
    return raycluster['spec']['workerGroupSpecs'][group_index].get('maxReplicas')

def _worker_group_replicas(raycluster: Dict[str, Any], group_index: int):
    if False:
        while True:
            i = 10
    return raycluster['spec']['workerGroupSpecs'][group_index].get('replicas', 1)

class KuberayNodeProvider(BatchingNodeProvider):

    def __init__(self, provider_config: Dict[str, Any], cluster_name: str, _allow_multiple: bool=False):
        if False:
            for i in range(10):
                print('nop')
        logger.info('Creating KuberayNodeProvider.')
        self.namespace = provider_config['namespace']
        self.cluster_name = cluster_name
        (self.headers, self.verify) = load_k8s_secrets()
        assert provider_config.get(WORKER_LIVENESS_CHECK_KEY, True) is False, f'To use KuberayNodeProvider, must set `{WORKER_LIVENESS_CHECK_KEY}:False`.'
        assert provider_config.get(WORKER_RPC_DRAIN_KEY, False) is True, f'To use KuberayNodeProvider, must set `{WORKER_RPC_DRAIN_KEY}:True`.'
        BatchingNodeProvider.__init__(self, provider_config, cluster_name, _allow_multiple)

    def get_node_data(self) -> Dict[NodeID, NodeData]:
        if False:
            for i in range(10):
                print('nop')
        'Queries K8s for pods in the RayCluster. Converts that pod data into a\n        map of pod name to Ray NodeData, as required by BatchingNodeProvider.\n        '
        self._raycluster = self._get(f'rayclusters/{self.cluster_name}')
        resource_version = self._get_pods_resource_version()
        if resource_version:
            logger.info(f'Listing pods for RayCluster {self.cluster_name} in namespace {self.namespace} at pods resource version >= {resource_version}.')
        label_selector = requests.utils.quote(f'ray.io/cluster={self.cluster_name}')
        resource_path = f'pods?labelSelector={label_selector}'
        if resource_version:
            resource_path += f'&resourceVersion={resource_version}' + '&resourceVersionMatch=NotOlderThan'
        pod_list = self._get(resource_path)
        fetched_resource_version = pod_list['metadata']['resourceVersion']
        logger.info(f'Fetched pod data at resource version {fetched_resource_version}.')
        node_data_dict = {}
        for pod in pod_list['items']:
            if 'deletionTimestamp' in pod['metadata']:
                continue
            pod_name = pod['metadata']['name']
            node_data_dict[pod_name] = node_data_from_pod(pod)
        return node_data_dict

    def submit_scale_request(self, scale_request: ScaleRequest):
        if False:
            print('Hello World!')
        "Converts the scale request generated by BatchingNodeProvider into\n        a patch that modifies the RayCluster CR's replicas and/or workersToDelete\n        fields. Then submits the patch to the K8s API server.\n        "
        patch_payload = self._scale_request_to_patch_payload(scale_request, self._raycluster)
        logger.info(f'Autoscaler is submitting the following patch to RayCluster {self.cluster_name} in namespace {self.namespace}.')
        logger.info(patch_payload)
        self._submit_raycluster_patch(patch_payload)

    def safe_to_scale(self) -> bool:
        if False:
            while True:
                i = 10
        "Returns False iff non_terminated_nodes contains any pods in the RayCluster's\n        workersToDelete lists.\n\n        Explanation:\n        If there are any workersToDelete which are non-terminated,\n        we should wait for the operator to do its job and delete those\n        pods. Therefore, we back off the autoscaler update.\n\n        If, on the other hand, all of the workersToDelete have already been cleaned up,\n        then we patch away the workersToDelete lists and return True.\n        In the future, we may consider having the operator clean up workersToDelete\n        on it own:\n        https://github.com/ray-project/kuberay/issues/733\n\n        Note (Dmitri):\n        It is stylistically bad that this function has a side effect.\n        "
        node_set = set(self.node_data_dict.keys())
        worker_groups = self._raycluster['spec'].get('workerGroupSpecs', [])
        non_empty_worker_group_indices = []
        for (group_index, worker_group) in enumerate(worker_groups):
            workersToDelete = worker_group.get('scaleStrategy', {}).get('workersToDelete', [])
            if workersToDelete:
                non_empty_worker_group_indices.append(group_index)
            for worker in workersToDelete:
                if worker in node_set:
                    logger.warning(f'Waiting for operator to remove worker {worker}.')
                    return False
        patch_payload = []
        for group_index in non_empty_worker_group_indices:
            patch = worker_delete_patch(group_index, workers_to_delete=[])
            patch_payload.append(patch)
        if patch_payload:
            logger.info('Cleaning up workers to delete.')
            logger.info(f'Submitting patch {patch_payload}.')
            self._submit_raycluster_patch(patch_payload)
        return True

    def _get_pods_resource_version(self) -> str:
        if False:
            return 10
        "\n        Extract a recent pods resource version by reading the head pod's\n        metadata.resourceVersion of the response.\n        "
        if not RAY_HEAD_POD_NAME:
            return None
        pod_resp = self._get(f'pods/{RAY_HEAD_POD_NAME}')
        return pod_resp['metadata']['resourceVersion']

    def _scale_request_to_patch_payload(self, scale_request: ScaleRequest, raycluster: Dict[str, Any]) -> List[Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        'Converts autoscaler scale request into a RayCluster CR patch payload.'
        patch_payload = []
        for (node_type, target_replicas) in scale_request.desired_num_workers.items():
            group_index = _worker_group_index(raycluster, node_type)
            group_max_replicas = _worker_group_max_replicas(raycluster, group_index)
            if group_max_replicas is not None and group_max_replicas < target_replicas:
                logger.warning('Autoscaler attempted to create ' + 'more than maxReplicas pods of type {}.'.format(node_type))
                target_replicas = group_max_replicas
            if target_replicas == _worker_group_replicas(raycluster, group_index):
                continue
            patch = worker_replica_patch(group_index, target_replicas)
            patch_payload.append(patch)
        deletion_groups = defaultdict(list)
        for worker in scale_request.workers_to_delete:
            node_type = self.node_tags(worker)[TAG_RAY_USER_NODE_TYPE]
            deletion_groups[node_type].append(worker)
        for (node_type, workers_to_delete) in deletion_groups.items():
            group_index = _worker_group_index(raycluster, node_type)
            patch = worker_delete_patch(group_index, workers_to_delete)
            patch_payload.append(patch)
        return patch_payload

    def _submit_raycluster_patch(self, patch_payload: List[Dict[str, Any]]):
        if False:
            while True:
                i = 10
        'Submits a patch to modify a RayCluster CR.'
        path = 'rayclusters/{}'.format(self.cluster_name)
        self._patch(path, patch_payload)

    def _url(self, path: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Convert resource path to REST URL for Kubernetes API server.'
        if path.startswith('pods'):
            api_group = '/api/v1'
        elif path.startswith('rayclusters'):
            api_group = '/apis/ray.io/' + KUBERAY_CRD_VER
        else:
            raise NotImplementedError('Tried to access unknown entity at {}'.format(path))
        return 'https://kubernetes.default:443' + api_group + '/namespaces/' + self.namespace + '/' + path

    def _get(self, path: str) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Wrapper for REST GET of resource with proper headers.'
        url = url_from_resource(namespace=self.namespace, path=path)
        result = requests.get(url, headers=self.headers, verify=self.verify)
        if not result.status_code == 200:
            result.raise_for_status()
        return result.json()

    def _patch(self, path: str, payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Wrapper for REST PATCH of resource with proper headers.'
        url = url_from_resource(namespace=self.namespace, path=path)
        result = requests.patch(url, json.dumps(payload), headers={**self.headers, 'Content-type': 'application/json-patch+json'}, verify=self.verify)
        if not result.status_code == 200:
            result.raise_for_status()
        return result.json()