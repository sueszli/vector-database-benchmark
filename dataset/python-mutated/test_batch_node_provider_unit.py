"""Unit test for BatchingNodeProvider.
Validates BatchingNodeProvider's book-keeping logic.
"""
from copy import copy
from uuid import uuid4
import os
import random
import sys
from typing import Any, Dict
from collections import defaultdict
import pytest
from ray.autoscaler.batching_node_provider import BatchingNodeProvider, NodeData, ScaleRequest
from ray.autoscaler._private.util import NodeID, NodeType
from ray.autoscaler.tags import STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE, TAG_RAY_NODE_KIND, TAG_RAY_NODE_STATUS, NODE_KIND_HEAD, NODE_KIND_WORKER
from ray.autoscaler._private.constants import DISABLE_LAUNCH_CONFIG_CHECK_KEY, DISABLE_NODE_UPDATERS_KEY, FOREGROUND_NODE_LAUNCH_KEY

class MockBatchingNodeProvider(BatchingNodeProvider):
    """Mock implementation of a BatchingNodeProvider."""

    def __init__(self, provider_config: Dict[str, Any], cluster_name: str, _allow_multiple: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        BatchingNodeProvider.__init__(self, provider_config, cluster_name, _allow_multiple)
        self._node_data_dict: Dict[NodeID, NodeData] = {}
        self._add_node(node_type='head', node_kind=NODE_KIND_HEAD)
        self._safe_to_scale_flag = True
        self._scale_request_submitted_count = 0
        self.num_non_terminated_nodes_calls = 0

    def get_node_data(self) -> Dict[NodeID, NodeData]:
        if False:
            for i in range(10):
                print('nop')
        self.num_non_terminated_nodes_calls += 1
        return self._node_data_dict

    def submit_scale_request(self, scale_request: ScaleRequest) -> None:
        if False:
            return 10
        'Simulate modification of cluster state by an external cluster manager.'
        self._scale_request_submitted_count += 1
        for node_id in self.scale_request.workers_to_delete:
            del self._node_data_dict[node_id]
        cur_num_workers = self._cur_num_workers(self._node_data_dict)
        for node_type in self.scale_request.desired_num_workers:
            diff = self.scale_request.desired_num_workers[node_type] - cur_num_workers[node_type]
            assert diff >= 0, diff
            for _ in range(diff):
                self._add_node(node_type, NODE_KIND_WORKER)

    def _add_node(self, node_type, node_kind):
        if False:
            return 10
        new_node_id = str(uuid4())
        self._node_data_dict[new_node_id] = NodeData(kind=node_kind, ip=str(uuid4()), status=STATUS_UP_TO_DATE, type=node_type)

    def non_terminated_node_ips(self, tag_filters):
        if False:
            print('Hello World!')
        'This method is used in test_autoscaler.py.'
        return [node_data.ip for (node_id, node_data) in self._node_data_dict.items() if tag_filters.items() <= self.node_tags(node_id).items()]

    def safe_to_scale(self) -> bool:
        if False:
            return 10
        return self.safe_to_scale_flag

    def _assert_worker_counts(self, expected_worker_counts: Dict[NodeType, int]) -> None:
        if False:
            print('Hello World!')
        assert self._cur_num_workers(self._node_data_dict) == expected_worker_counts

class BatchingNodeProviderTester:
    """Utility to test BatchingNodeProvider."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.node_provider = MockBatchingNodeProvider(provider_config={DISABLE_LAUNCH_CONFIG_CHECK_KEY: True, DISABLE_NODE_UPDATERS_KEY: True, FOREGROUND_NODE_LAUNCH_KEY: True}, cluster_name='test-cluster', _allow_multiple=True)
        self.expected_node_counts = defaultdict(int)
        self.expected_node_counts['head'] = 1
        self.expected_scale_request_submitted_count = 0

    def update(self, create_node_requests, terminate_nodes_requests, safe_to_scale_flag):
        if False:
            while True:
                i = 10
        'Simulates an autoscaler update with multiple terminate and create calls.\n\n        Calls non_terminated_nodes, then create/terminate nodes, then post_process.\n\n        Args:\n            create_node_requests (List[Tuple(str, int)]): List of pairs\n                (node type, count). Each pair is used in a create_node call that\n                creates count nodes of the node type.\n            terminate_nodes_requests (List[Tuple(str, int)]): List of pairs\n                (node type, count). Each pair is used in a terminate_nodes call\n                that terminates up to count nodes of the node type.\n            safe_to_scale_flag (bool): Passed to the node provider to determine  # noqa\n                where provider.safe_to_scale() evaluates to True or False.\n        '
        self.node_provider.safe_to_scale_flag = safe_to_scale_flag
        self.validate_non_terminated_nodes()
        if not self.node_provider.safe_to_scale():
            return
        to_terminate_this_update = set()
        for (node_type, count) in terminate_nodes_requests:
            to_terminate_this_request = []
            for node in self.node_provider._node_data_dict:
                if len(to_terminate_this_request) >= count:
                    break
                if self.node_provider.node_tags(node)[TAG_RAY_USER_NODE_TYPE] != node_type:
                    continue
                if node in to_terminate_this_update:
                    continue
                to_terminate_this_update.add(node)
                to_terminate_this_request.append(node)
            self.node_provider.terminate_nodes(to_terminate_this_request)
            self.expected_node_counts[node_type] -= len(to_terminate_this_request)
        for (node_type, count) in create_node_requests:
            self.node_provider.create_node(node_config={}, tags={TAG_RAY_USER_NODE_TYPE: node_type}, count=count)
            self.expected_node_counts[node_type] += count
        assert self.node_provider.scale_change_needed is bool(create_node_requests or terminate_nodes_requests)
        self.node_provider.post_process()
        if create_node_requests or terminate_nodes_requests:
            self.expected_scale_request_submitted_count += 1

    def validate_non_terminated_nodes(self):
        if False:
            for i in range(10):
                print('nop')
        "Calls non_terminated_nodes and validates output against this test classes's\n        accumulated expected state.\n\n        Tests methods internal_ip, node_tags, non_terminated_nodes of\n        BatchingNodeProvider.\n        "
        nodes = self.node_provider.non_terminated_nodes({})
        actual_node_counts = defaultdict(int)
        for node in nodes:
            assert isinstance(self.node_provider.internal_ip(node), str)
            tags = self.node_provider.node_tags(node)
            assert set(tags.keys()) == {TAG_RAY_USER_NODE_TYPE, TAG_RAY_NODE_STATUS, TAG_RAY_NODE_KIND}
            node_type = tags[TAG_RAY_USER_NODE_TYPE]
            node_kind = tags[TAG_RAY_NODE_KIND]
            node_status = tags[TAG_RAY_NODE_STATUS]
            if node_type == 'head':
                assert node_kind == NODE_KIND_HEAD
            else:
                assert node_kind == NODE_KIND_WORKER
            assert node_status == STATUS_UP_TO_DATE
            actual_node_counts[node_type] += 1
        for (k, v) in copy(self.expected_node_counts).items():
            if v == 0:
                del self.expected_node_counts[k]
        assert actual_node_counts == self.expected_node_counts
        actual_node_counts_again = {}
        for node_type in actual_node_counts:
            actual_node_counts_again[node_type] = len(self.node_provider.non_terminated_nodes(tag_filters={TAG_RAY_USER_NODE_TYPE: node_type}))
        assert actual_node_counts_again == self.expected_node_counts
        workers = self.node_provider.non_terminated_nodes(tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        heads = self.node_provider.non_terminated_nodes(tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_HEAD})
        assert len(heads) == 1
        assert set(nodes) == set(workers) | set(heads)
        up_to_date_nodes = self.node_provider.non_terminated_nodes(tag_filters={TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE})
        assert set(up_to_date_nodes) == set(nodes)
        expected_node_counts_without_head = copy(self.expected_node_counts)
        del expected_node_counts_without_head['head']
        assert self.node_provider.scale_request.desired_num_workers == expected_node_counts_without_head
        assert self.node_provider.scale_change_needed is False
        assert self.node_provider._scale_request_submitted_count == self.expected_scale_request_submitted_count

    def update_with_random_requests(self):
        if False:
            while True:
                i = 10
        random_requests = self.generate_random_requests()
        self.update(*random_requests)

    def generate_random_requests(self):
        if False:
            while True:
                i = 10
        'Generates random sequences of create_node and terminate_nodes requests\n        for the node provider. Generates random safe_to_scale_flag.\n        '
        num_creates = random.choice(range(100))
        num_terminates = random.choice(range(100))
        create_node_requests = []
        for _ in range(num_creates):
            node_type = random.choice([f'type-{x}' for x in range(5)])
            count = random.choice(range(10))
            create_node_requests.append((node_type, count))
        terminate_nodes_requests = []
        for _ in range(num_terminates):
            node_type = random.choice([f'type-{x}' for x in range(5)])
            count = random.choice(range(10))
            terminate_nodes_requests.append((node_type, count))
        safe_to_scale_flag = random.choice([True, False])
        return (create_node_requests, terminate_nodes_requests, safe_to_scale_flag)

    def assert_worker_counts(self, expected_worker_counts):
        if False:
            i = 10
            return i + 15
        'Validates worker counts against internal node provider state.'
        self.node_provider._assert_worker_counts(expected_worker_counts)

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Not relevant on Windows.')
def test_batching_node_provider_basic():
    if False:
        print('Hello World!')
    tester = BatchingNodeProviderTester()
    tester.update(create_node_requests=[('type-1', 5)], terminate_nodes_requests=[], safe_to_scale_flag=True)
    tester.assert_worker_counts({'type-1': 5})
    assert tester.node_provider._scale_request_submitted_count == 1
    tester.update(create_node_requests=[('type-2', 5), ('type-2', 5)], terminate_nodes_requests=[('type-1', 2)], safe_to_scale_flag=True)
    tester.assert_worker_counts({'type-1': 3, 'type-2': 10})
    assert tester.node_provider._scale_request_submitted_count == 2
    tester.update(create_node_requests=[], terminate_nodes_requests=[('type-1', 2), ('type-2', 1), ('type-2', 1)], safe_to_scale_flag=True)
    tester.assert_worker_counts({'type-1': 1, 'type-2': 8})
    assert tester.node_provider._scale_request_submitted_count == 3
    tester.update(create_node_requests=[], terminate_nodes_requests=[], safe_to_scale_flag=True)
    tester.assert_worker_counts({'type-1': 1, 'type-2': 8})
    assert tester.node_provider._scale_request_submitted_count == 3

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Not relevant on Windows.')
def test_batching_node_provider_many_requests():
    if False:
        while True:
            i = 10
    'Simulate 10 autoscaler updates with randomly generated create/terminate\n    requests.\n    '
    tester = BatchingNodeProviderTester()
    for _ in range(2):
        tester.update_with_random_requests()
    tester.validate_non_terminated_nodes()

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Not relevant on Windows.')
def test_terminate_safeguards():
    if False:
        for i in range(10):
            print('nop')
    'Tests the following behaviors:\n    - the node provider ignores requests to terminate a node twice.\n    - the node provider ignores requests to terminate an unknown node.\n    '
    node_provider = MockBatchingNodeProvider(provider_config={DISABLE_LAUNCH_CONFIG_CHECK_KEY: True, DISABLE_NODE_UPDATERS_KEY: True, FOREGROUND_NODE_LAUNCH_KEY: True}, cluster_name='test-cluster', _allow_multiple=True)
    nodes = node_provider.non_terminated_nodes({})
    assert len(nodes) == 1
    head_node = nodes[0]
    node_provider.create_node(node_config={}, tags={TAG_RAY_USER_NODE_TYPE: 'type'}, count=1)
    node_provider.post_process()
    nodes = node_provider.non_terminated_nodes({})
    assert len(nodes) == 2
    worker_node = ''
    for node in nodes:
        if node == head_node:
            continue
        else:
            worker_node = node
    unknown_node = node + worker_node
    node_provider.terminate_node(unknown_node)
    node_provider.post_process()
    nodes = node_provider.non_terminated_nodes({})
    assert len(nodes) == 2
    node_provider.terminate_node(worker_node)
    node_provider.terminate_node(worker_node)
    node_provider.post_process()
    nodes = node_provider.non_terminated_nodes({})
    assert len(nodes) == 1
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))