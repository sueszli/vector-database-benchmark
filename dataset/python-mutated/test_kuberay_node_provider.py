import copy
import mock
import sys
import jsonpatch
import pytest
from ray.autoscaler.batching_node_provider import NodeData
from ray.autoscaler._private.kuberay.node_provider import _worker_group_index, _worker_group_max_replicas, _worker_group_replicas, KuberayNodeProvider, ScaleRequest
from ray.autoscaler._private.util import NodeID
from pathlib import Path
import yaml
from ray.tests.kuberay.test_autoscaling_config import get_basic_ray_cr
from typing import Set, List

def _get_basic_ray_cr_workers_to_delete(cpu_workers_to_delete: List[NodeID], gpu_workers_to_delete: List[NodeID]):
    if False:
        i = 10
        return i + 15
    'Generate a Ray cluster with non-empty workersToDelete field.'
    raycluster = get_basic_ray_cr()
    raycluster['spec']['workerGroupSpecs'][0]['scaleStrategy'] = {'workersToDelete': cpu_workers_to_delete}
    raycluster['spec']['workerGroupSpecs'][1]['scaleStrategy'] = {'workersToDelete': gpu_workers_to_delete}
    return raycluster

def _get_test_yaml(file_name):
    if False:
        for i in range(10):
            print('nop')
    file_path = str(Path(__file__).resolve().parent / 'test_files' / file_name)
    return yaml.safe_load(open(file_path).read())

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Not relevant on Windows.')
@pytest.mark.parametrize('group_name,expected_index', [('small-group', 0), ('gpu-group', 1)])
def test_worker_group_index(group_name, expected_index):
    if False:
        for i in range(10):
            print('nop')
    'Basic unit test for _worker_group_index.\n\n    Uses a RayCluster CR with worker groups "small-group" and "gpu-group",\n    listed in that order.\n    '
    raycluster_cr = get_basic_ray_cr()
    assert _worker_group_index(raycluster_cr, group_name) == expected_index

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Not relevant on Windows.')
@pytest.mark.parametrize('group_index,expected_max_replicas,expected_replicas', [(0, 300, 1), (1, 200, 1), (2, None, 0)])
def test_worker_group_replicas(group_index, expected_max_replicas, expected_replicas):
    if False:
        while True:
            i = 10
    'Basic unit test for _worker_group_max_replicas and _worker_group_replicas\n\n    Uses a RayCluster CR with worker groups with 300 maxReplicas, 200 maxReplicas,\n    and unspecified maxReplicas, in that order.\n    '
    raycluster = get_basic_ray_cr()
    no_max_replicas_group = copy.deepcopy(raycluster['spec']['workerGroupSpecs'][0])
    no_max_replicas_group['groupName'] = 'no-max-replicas'
    del no_max_replicas_group['maxReplicas']
    no_max_replicas_group['replicas'] = 0
    raycluster['spec']['workerGroupSpecs'].append(no_max_replicas_group)
    assert _worker_group_max_replicas(raycluster, group_index) == expected_max_replicas
    assert _worker_group_replicas(raycluster, group_index) == expected_replicas

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Not relevant on Windows.')
@pytest.mark.parametrize('attempted_target_replica_count,expected_target_replica_count', [(200, 200), (250, 250), (300, 300), (400, 300), (1000, 300)])
def test_create_node_cap_at_max(attempted_target_replica_count, expected_target_replica_count):
    if False:
        print('Hello World!')
    'Validates that KuberayNodeProvider does not attempt to create more nodes than\n    allowed by maxReplicas. For the config in this test, maxReplicas is fixed at 300.\n\n    Args:\n        attempted_target_replica_count: The mocked desired replica count for a given\n            worker group.\n        expected_target_replica_count: The actual requested replicaCount. Should be\n            capped at maxReplicas (300, for the config in this test.)\n    '
    raycluster = get_basic_ray_cr()
    with mock.patch.object(KuberayNodeProvider, '__init__', return_value=None):
        kr_node_provider = KuberayNodeProvider(provider_config={}, cluster_name='fake')
        scale_request = ScaleRequest(workers_to_delete=set(), desired_num_workers={'small-group': attempted_target_replica_count})
        patch = kr_node_provider._scale_request_to_patch_payload(scale_request=scale_request, raycluster=raycluster)
        assert patch[0]['value'] == expected_target_replica_count

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Not relevant on Windows.')
@pytest.mark.parametrize('podlist_file,expected_node_data', [('podlist1.yaml', {'raycluster-autoscaler-head-8zsc8': NodeData(kind='head', type='head-group', ip='10.4.2.6', status='up-to-date'), 'raycluster-autoscaler-worker-small-group-dkz2r': NodeData(kind='worker', type='small-group', ip='10.4.1.8', status='waiting')}), ('podlist2.yaml', {'raycluster-autoscaler-head-8zsc8': NodeData(kind='head', type='head-group', ip='10.4.2.6', status='up-to-date'), 'raycluster-autoscaler-worker-fake-gpu-group-2qnhv': NodeData(kind='worker', type='fake-gpu-group', ip='10.4.0.6', status='up-to-date'), 'raycluster-autoscaler-worker-small-group-dkz2r': NodeData(kind='worker', type='small-group', ip='10.4.1.8', status='up-to-date'), 'raycluster-autoscaler-worker-small-group-lbfm4': NodeData(kind='worker', type='small-group', ip='10.4.0.5', status='up-to-date')})])
def test_get_node_data(podlist_file: str, expected_node_data):
    if False:
        for i in range(10):
            print('nop')
    'Test translation of a K8s pod list into autoscaler node data.'
    pod_list = _get_test_yaml(podlist_file)

    def mock_get(node_provider, path):
        if False:
            i = 10
            return i + 15
        if 'pods' in path:
            return pod_list
        elif 'raycluster' in path:
            return get_basic_ray_cr()
        else:
            raise ValueError('Invalid path.')
    with mock.patch.object(KuberayNodeProvider, '__init__', return_value=None), mock.patch.object(KuberayNodeProvider, '_get', mock_get):
        kr_node_provider = KuberayNodeProvider(provider_config={}, cluster_name='fake')
        kr_node_provider.cluster_name = 'fake'
        nodes = kr_node_provider.non_terminated_nodes({})
        assert kr_node_provider.node_data_dict == expected_node_data
        assert set(nodes) == set(expected_node_data.keys())

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Not relevant on Windows.')
@pytest.mark.parametrize('node_data_dict,scale_request,expected_patch_payload', [({'raycluster-autoscaler-head-8zsc8': NodeData(kind='head', type='head-group', ip='10.4.2.6', status='up-to-date'), 'raycluster-autoscaler-worker-fake-gpu-group-2qnhv': NodeData(kind='worker', type='fake-gpu-group', ip='10.4.0.6', status='up-to-date'), 'raycluster-autoscaler-worker-small-group-dkz2r': NodeData(kind='worker', type='small-group', ip='10.4.1.8', status='up-to-date'), 'raycluster-autoscaler-worker-small-group-lbfm4': NodeData(kind='worker', type='small-group', ip='10.4.0.5', status='up-to-date')}, ScaleRequest(desired_num_workers={'small-group': 1, 'gpu-group': 1, 'blah-group': 5}, workers_to_delete={'raycluster-autoscaler-worker-small-group-dkz2r'}), [{'op': 'replace', 'path': '/spec/workerGroupSpecs/2/replicas', 'value': 5}, {'op': 'replace', 'path': '/spec/workerGroupSpecs/0/scaleStrategy', 'value': {'workersToDelete': ['raycluster-autoscaler-worker-small-group-dkz2r']}}])])
def test_submit_scale_request(node_data_dict, scale_request, expected_patch_payload):
    if False:
        while True:
            i = 10
    "Test the KuberayNodeProvider's RayCluster patch payload given a dict\n    of current node counts and a scale request.\n    "
    raycluster = get_basic_ray_cr()
    blah_group = copy.deepcopy(raycluster['spec']['workerGroupSpecs'][1])
    blah_group['groupName'] = 'blah-group'
    raycluster['spec']['workerGroupSpecs'].append(blah_group)
    with mock.patch.object(KuberayNodeProvider, '__init__', return_value=None):
        kr_node_provider = KuberayNodeProvider(provider_config={}, cluster_name='fake')
        kr_node_provider.node_data_dict = node_data_dict
        patch_payload = kr_node_provider._scale_request_to_patch_payload(scale_request=scale_request, raycluster=raycluster)
        assert patch_payload == expected_patch_payload

@pytest.mark.parametrize('node_set', [{'A', 'B', 'C', 'D', 'E'}])
@pytest.mark.parametrize('cpu_workers_to_delete', ['A', 'Z'])
@pytest.mark.parametrize('gpu_workers_to_delete', ['B', 'Y'])
@pytest.mark.skipif(sys.platform.startswith('win'), reason='Not relevant on Windows.')
def test_safe_to_scale(node_set: Set[NodeID], cpu_workers_to_delete: List[NodeID], gpu_workers_to_delete: List[NodeID]):
    if False:
        return 10
    mock_node_data = NodeData('-', '-', '-', '-')
    node_data_dict = {node_id: mock_node_data for node_id in node_set}
    raycluster = _get_basic_ray_cr_workers_to_delete(cpu_workers_to_delete, gpu_workers_to_delete)

    def mock_patch(kuberay_provider, path, patch_payload):
        if False:
            return 10
        patch = jsonpatch.JsonPatch(patch_payload)
        kuberay_provider._patched_raycluster = patch.apply(kuberay_provider._raycluster)
    with mock.patch.object(KuberayNodeProvider, '__init__', return_value=None), mock.patch.object(KuberayNodeProvider, '_patch', mock_patch):
        kr_node_provider = KuberayNodeProvider(provider_config={}, cluster_name='fake')
        kr_node_provider.cluster_name = 'fake'
        kr_node_provider._patched_raycluster = raycluster
        kr_node_provider._raycluster = raycluster
        kr_node_provider.node_data_dict = node_data_dict
        actual_safe = kr_node_provider.safe_to_scale()
    expected_safe = not any((cpu_worker_to_delete in node_set for cpu_worker_to_delete in cpu_workers_to_delete)) and (not any((gpu_worker_to_delete in node_set for gpu_worker_to_delete in gpu_workers_to_delete)))
    assert expected_safe is actual_safe
    patched_cpu_workers_to_delete = kr_node_provider._patched_raycluster['spec']['workerGroupSpecs'][0]['scaleStrategy']['workersToDelete']
    patched_gpu_workers_to_delete = kr_node_provider._patched_raycluster['spec']['workerGroupSpecs'][1]['scaleStrategy']['workersToDelete']
    if expected_safe:
        assert patched_cpu_workers_to_delete == []
        assert patched_gpu_workers_to_delete == []
    else:
        assert patched_cpu_workers_to_delete == cpu_workers_to_delete
        assert patched_gpu_workers_to_delete == gpu_workers_to_delete
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', __file__]))