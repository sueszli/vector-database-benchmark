import copy
import os
import json
import shutil
import tempfile
import time
import unittest
from dataclasses import asdict
from datetime import datetime
from time import sleep
from unittest import mock
import pytest
import yaml
import ray
import ray._private.ray_constants
from ray._private.gcs_utils import PlacementGroupTableData
from ray._private.test_utils import same_elements
from ray.autoscaler._private.node_provider_availability_tracker import NodeAvailabilityRecord, NodeAvailabilitySummary, UnavailableNodeInformation
from ray.autoscaler._private.autoscaler import AutoscalerSummary
from ray.autoscaler._private.commands import get_or_create_head_node
from ray.autoscaler._private.constants import AUTOSCALER_MAX_RESOURCE_DEMAND_VECTOR_SIZE, AUTOSCALER_UTILIZATION_SCORER_KEY
from ray.autoscaler._private.load_metrics import LoadMetrics
from ray.autoscaler._private.providers import _NODE_PROVIDERS, _clear_provider_cache
from ray.autoscaler._private.resource_demand_scheduler import ResourceDemandScheduler, _add_min_workers_nodes, _resource_based_utilization_scorer, _default_utilization_scorer, get_bin_pack_residual
from ray.autoscaler._private.resource_demand_scheduler import get_nodes_for as _get
from ray.autoscaler._private.util import LoadMetricsSummary, format_info_string, is_placement_group_resource
from ray.autoscaler.tags import NODE_KIND_HEAD, NODE_KIND_WORKER, STATUS_UNINITIALIZED, STATUS_UP_TO_DATE, STATUS_WAITING_FOR_SSH, TAG_RAY_NODE_KIND, TAG_RAY_NODE_STATUS, TAG_RAY_USER_NODE_TYPE
from ray.core.generated.common_pb2 import Bundle, PlacementStrategy
from ray.tests.test_autoscaler import MULTI_WORKER_CLUSTER, TYPES_A, MockAutoscaler, MockGcsClient, MockProcessRunner, MockProvider, fill_in_raylet_ids, mock_raylet_id
from functools import partial
GET_DEFAULT_METHOD = 'ray.autoscaler._private.util._get_default_config'
EMPTY_AVAILABILITY_SUMMARY = NodeAvailabilitySummary({})
utilization_scorer = partial(_default_utilization_scorer, node_availability_summary=EMPTY_AVAILABILITY_SUMMARY)

def get_nodes_for(*a, **kw):
    if False:
        for i in range(10):
            print('nop')
    return _get(*a, utilization_scorer=utilization_scorer, **kw)[0]

def test_util_score():
    if False:
        return 10
    assert _resource_based_utilization_scorer({'CPU': 64}, [{'TPU': 16}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) is None
    assert _resource_based_utilization_scorer({'GPU': 4}, [{'GPU': 2}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) == (True, 1, 0.5, 0.5)
    assert _resource_based_utilization_scorer({'GPU': 4}, [{'GPU': 1}, {'GPU': 1}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) == (True, 1, 0.5, 0.5)
    assert _resource_based_utilization_scorer({'GPU': 2}, [{'GPU': 2}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) == (True, 1, 2, 2)
    assert _resource_based_utilization_scorer({'GPU': 2}, [{'GPU': 1}, {'GPU': 1}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) == (True, 1, 2, 2)
    assert _resource_based_utilization_scorer({'GPU': 1}, [{'GPU': 1, 'CPU': 1}, {'GPU': 1}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) == (True, 1, 1, 1)
    assert _resource_based_utilization_scorer({'GPU': 1, 'CPU': 1}, [{'GPU': 1, 'CPU': 1}, {'GPU': 1}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) == (True, 2, 1, 1)
    assert _resource_based_utilization_scorer({'GPU': 2, 'TPU': 1}, [{'GPU': 2}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) == (True, 1, 0, 1)
    assert _resource_based_utilization_scorer({'CPU': 64}, [{'CPU': 64}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) == (True, 1, 64, 64)
    assert _resource_based_utilization_scorer({'CPU': 64}, [{'CPU': 32}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) == (True, 1, 8, 8)
    assert _resource_based_utilization_scorer({'CPU': 64}, [{'CPU': 16}, {'CPU': 16}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) == (True, 1, 8, 8)

def test_gpu_node_util_score():
    if False:
        while True:
            i = 10
    utilization_score = _resource_based_utilization_scorer({'GPU': 1, 'CPU': 1}, [{'CPU': 1}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY)
    gpu_ok = utilization_score[0]
    assert gpu_ok is False
    assert _resource_based_utilization_scorer({'GPU': 1, 'CPU': 1}, [{'CPU': 1, 'GPU': 1}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) == (True, 2, 1.0, 1.0)
    assert _resource_based_utilization_scorer({'GPU': 1, 'CPU': 1}, [{'GPU': 1}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) == (True, 1, 0.0, 0.5)

def test_zero_resource():
    if False:
        for i in range(10):
            print('nop')
    assert _resource_based_utilization_scorer({'CPU': 0, 'custom': 0}, [{'custom': 1}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY) is None
    _resource_based_utilization_scorer({'CPU': 0, 'custom': 1}, [{'custom': 1}], node_availability_summary=EMPTY_AVAILABILITY_SUMMARY)

def test_bin_pack():
    if False:
        return 10
    assert get_bin_pack_residual([], [{'GPU': 2}, {'GPU': 2}])[0] == [{'GPU': 2}, {'GPU': 2}]
    assert get_bin_pack_residual([{'GPU': 2}], [{'GPU': 2}, {'GPU': 2}])[0] == [{'GPU': 2}]
    assert get_bin_pack_residual([{'GPU': 4}], [{'GPU': 2}, {'GPU': 2}])[0] == []
    arg = [{'GPU': 2}, {'GPU': 2, 'CPU': 2}]
    assert get_bin_pack_residual(arg, [{'GPU': 2}, {'GPU': 2}])[0] == []
    arg = [{'CPU': 2}, {'GPU': 2}]
    assert get_bin_pack_residual(arg, [{'GPU': 2}, {'GPU': 2}])[0] == [{'GPU': 2}]
    arg = [{'GPU': 3}]
    assert get_bin_pack_residual(arg, [{'GPU': 1}, {'GPU': 1}], strict_spread=False)[0] == []
    assert get_bin_pack_residual(arg, [{'GPU': 1}, {'GPU': 1}], strict_spread=True) == ([{'GPU': 1}], [{'GPU': 2}])
    implicit_resource = ray._raylet.IMPLICIT_RESOURCE_PREFIX + 'a'
    assert get_bin_pack_residual([{'CPU': 1}], [{implicit_resource: 0.5}, {implicit_resource: 0.5}])[0] == []
    assert get_bin_pack_residual([{'CPU': 1}], [{implicit_resource: 1}, {implicit_resource: 0.5}]) == ([{implicit_resource: 0.5}], [{'CPU': 1, implicit_resource: 0}])

def test_get_nodes_packing_heuristic():
    if False:
        while True:
            i = 10
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'GPU': 8}]) == {'p2.8xlarge': 1}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'GPU': 1}] * 6) == {'p2.8xlarge': 1}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'GPU': 1}] * 4) == {'p2.xlarge': 4}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'CPU': 32, 'GPU': 1}] * 3) == {'p2.8xlarge': 3}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'CPU': 64, 'GPU': 1}] * 3) == {}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'CPU': 64}] * 3) == {'m4.16xlarge': 3}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'CPU': 64}, {'CPU': 1}]) == {'m4.16xlarge': 1, 'm4.large': 1}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'CPU': 64}, {'CPU': 9}, {'CPU': 9}]) == {'m4.16xlarge': 1, 'm4.4xlarge': 2}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'CPU': 16}] * 5) == {'m4.16xlarge': 1, 'm4.4xlarge': 1}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'CPU': 8}] * 10) == {'m4.16xlarge': 1, 'm4.4xlarge': 1}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'CPU': 1}] * 100) == {'m4.16xlarge': 1, 'm4.4xlarge': 2, 'm4.large': 2}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'GPU': 1}] + [{'CPU': 1}] * 64) == {'m4.16xlarge': 1, 'p2.xlarge': 1}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'GPU': 1}] * 8 + [{'CPU': 1}] * 64) == {'m4.4xlarge': 2, 'p2.8xlarge': 1}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'GPU': 1}] * 8, strict_spread=False) == {'p2.8xlarge': 1}
    assert get_nodes_for(TYPES_A, {}, 'empty_node', 9999, [{'GPU': 1}] * 8, strict_spread=True) == {'p2.xlarge': 8}

def test_node_packing_gpu_cpu_bundles():
    if False:
        for i in range(10):
            print('nop')
    TYPES = {'cpu': {'resources': {'CPU': 16}, 'max_workers': 10}, 'gpu': {'resources': {'CPU': 16, 'GPU': 1}, 'max_workers': 10}}
    nodes = get_nodes_for(TYPES, {}, 'cpu', 9999, [{'CPU': 1}] * 30 + [{'GPU': 1, 'CPU': 1}])
    assert nodes == {'gpu': 1, 'cpu': 1}
    nodes = get_nodes_for(TYPES, {}, 'cpu', 9999, [{'GPU': 1, 'CPU': 1}] + [{'CPU': 1}] * 30)
    assert nodes == {'gpu': 1, 'cpu': 1}
    nodes = get_nodes_for(TYPES, {}, 'cpu', 9999, [{'GPU': 1, 'CPU': 1}] + [{'CPU': 1}] * 15)
    assert nodes == {'gpu': 1}

def test_gpu_node_avoid_cpu_task():
    if False:
        while True:
            i = 10
    types = {'cpu': {'resources': {'CPU': 1}, 'max_workers': 10}, 'gpu': {'resources': {'GPU': 1, 'CPU': 100}, 'max_workers': 10}}
    r1 = [{'CPU': 1}] * 100
    assert get_nodes_for(types, {}, 'empty_node', 10, r1) == {'cpu': 10}
    assert get_nodes_for(types, {}, 'empty_node', 11, r1) == {'cpu': 10, 'gpu': 1}
    r2 = [{'GPU': 1}] + [{'CPU': 1}] * 100
    assert get_nodes_for(types, {}, 'empty_node', 100, r2) == {'gpu': 1}
    r3 = [{'GPU': 1}] * 4 + [{'CPU': 1}] * 404
    assert get_nodes_for(types, {}, 'empty_node', 100, r3) == {'gpu': 4, 'cpu': 4}

def test_get_nodes_respects_max_limit():
    if False:
        return 10
    types = {'m4.large': {'resources': {'CPU': 2}, 'max_workers': 10}, 'gpu': {'resources': {'GPU': 1}, 'max_workers': 99999}}
    assert get_nodes_for(types, {}, 'empty_node', 2, [{'CPU': 1}] * 10) == {'m4.large': 2}
    assert get_nodes_for(types, {'m4.large': 9999}, 'empty_node', 9999, [{'CPU': 1}] * 10) == {}
    assert get_nodes_for(types, {'m4.large': 0}, 'empty_node', 9999, [{'CPU': 1}] * 10) == {'m4.large': 5}
    assert get_nodes_for(types, {'m4.large': 7}, 'm4.large', 4, [{'CPU': 1}] * 10) == {'m4.large': 4}
    assert get_nodes_for(types, {'m4.large': 7}, 'm4.large', 2, [{'CPU': 1}] * 10) == {'m4.large': 2}

def test_add_min_workers_nodes():
    if False:
        return 10
    types = {'m2.large': {'resources': {'CPU': 2}, 'min_workers': 50, 'max_workers': 100}, 'm4.large': {'resources': {'CPU': 2}, 'min_workers': 0, 'max_workers': 10}, 'gpu': {'resources': {'GPU': 1}, 'min_workers': 99999, 'max_workers': 99999}, 'gpubla': {'resources': {'GPU': 1}, 'min_workers': 10, 'max_workers': 0}}
    assert _add_min_workers_nodes([], {}, types, None, None, None, utilization_scorer=utilization_scorer) == ([{'CPU': 2}] * 50 + [{'GPU': 1}] * 99999, {'m2.large': 50, 'gpu': 99999}, {'m2.large': 50, 'gpu': 99999})
    assert _add_min_workers_nodes([{'CPU': 2}] * 5, {'m2.large': 5}, types, None, None, None, utilization_scorer=utilization_scorer) == ([{'CPU': 2}] * 50 + [{'GPU': 1}] * 99999, {'m2.large': 50, 'gpu': 99999}, {'m2.large': 45, 'gpu': 99999})
    assert _add_min_workers_nodes([{'CPU': 2}] * 60, {'m2.large': 60}, types, None, None, None, utilization_scorer=utilization_scorer) == ([{'CPU': 2}] * 60 + [{'GPU': 1}] * 99999, {'m2.large': 60, 'gpu': 99999}, {'gpu': 99999})
    assert _add_min_workers_nodes([{'CPU': 2}] * 50 + [{'GPU': 1}] * 99999, {'m2.large': 50, 'gpu': 99999}, types, None, None, None, utilization_scorer=utilization_scorer) == ([{'CPU': 2}] * 50 + [{'GPU': 1}] * 99999, {'m2.large': 50, 'gpu': 99999}, {})
    assert _add_min_workers_nodes([], {}, {'gpubla': types['gpubla']}, None, None, None, utilization_scorer=utilization_scorer) == ([], {}, {})
    types['gpubla']['max_workers'] = 10
    assert _add_min_workers_nodes([], {}, {'gpubla': types['gpubla']}, None, None, None, utilization_scorer=utilization_scorer) == ([{'GPU': 1}] * 10, {'gpubla': 10}, {'gpubla': 10})

def test_get_nodes_to_launch_with_min_workers():
    if False:
        while True:
            i = 10
    provider = MockProvider()
    new_types = copy.deepcopy(TYPES_A)
    new_types['p2.8xlarge']['min_workers'] = 2
    scheduler = ResourceDemandScheduler(provider, new_types, 3, head_node_type='p2.8xlarge', upscaling_speed=1)
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_NODE_KIND: NODE_KIND_HEAD}, 1)
    nodes = provider.non_terminated_nodes({})
    ips = provider.non_terminated_node_ips({})
    utilizations = {ip: {'GPU': 8} for ip in ips}
    (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, {}, [{'GPU': 8}], utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.8xlarge': 2}
    assert not rem
    (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, {}, [{'GPU': 8}] * 6, utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.8xlarge': 3}
    assert rem == [{'GPU': 8}, {'GPU': 8}]

def test_get_nodes_to_launch_with_min_workers_and_bin_packing():
    if False:
        i = 10
        return i + 15
    provider = MockProvider()
    new_types = copy.deepcopy(TYPES_A)
    new_types['p2.8xlarge']['min_workers'] = 2
    scheduler = ResourceDemandScheduler(provider, new_types, 10, head_node_type='p2.8xlarge', upscaling_speed=1)
    provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge'}, 1)
    provider.create_node({}, {TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge'}, 1)
    nodes = provider.non_terminated_nodes({})
    ips = provider.non_terminated_node_ips({})
    utilizations = {ip: {'GPU': 8} for ip in ips}
    pending_nodes = {'p2.8xlarge': 1}
    demands = [{'GPU': 8}] * (len(utilizations) + 1) + [{'GPU': 1}]
    (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, pending_nodes, demands, utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.xlarge': 1}
    assert not rem
    new_types['p2.8xlarge']['min_workers'] = 3
    scheduler = ResourceDemandScheduler(provider, new_types, 10, head_node_type='p2.8xlarge', upscaling_speed=1)
    (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, pending_nodes, demands, utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.8xlarge': 1}
    assert not rem

def test_get_nodes_to_launch_limits():
    if False:
        print('Hello World!')
    provider = MockProvider()
    scheduler = ResourceDemandScheduler(provider, TYPES_A, 3, head_node_type='p2.8xlarge', upscaling_speed=1)
    provider.create_node({}, {TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge'}, 2)
    nodes = provider.non_terminated_nodes({})
    ips = provider.non_terminated_node_ips({})
    utilizations = {ip: {'GPU': 8} for ip in ips}
    (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, {'p2.8xlarge': 1}, [{'GPU': 8}] * 2, utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {}
    assert not rem
    (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, {'p2.8xlarge': 1}, [{'GPU': 8}] * 20, utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.8xlarge': 1}
    assert rem == [{'GPU': 8}] * 16

def test_calculate_node_resources():
    if False:
        print('Hello World!')
    provider = MockProvider()
    scheduler = ResourceDemandScheduler(provider, TYPES_A, 10, head_node_type='p2.8xlarge', upscaling_speed=1)
    provider.create_node({}, {TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge'}, 2)
    nodes = provider.non_terminated_nodes({})
    ips = provider.non_terminated_node_ips({})
    utilizations = {ip: {'GPU': 8} for ip in ips}
    pending_nodes = {'p2.8xlarge': 1}
    demands = [{'GPU': 8}] * (len(utilizations) + 2)
    (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, pending_nodes, demands, utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.8xlarge': 1}
    assert not rem

def test_request_resources_gpu_no_gpu_nodes():
    if False:
        for i in range(10):
            print('nop')
    provider = MockProvider()
    TYPES = {'m5.8xlarge': {'node_config': {}, 'resources': {'CPU': 32}, 'max_workers': 40}}
    scheduler = ResourceDemandScheduler(provider, TYPES, max_workers=100, head_node_type='empty_node', upscaling_speed=1)
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'm5.8xlarge', TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}, 1)
    all_nodes = provider.non_terminated_nodes({})
    node_ips = provider.non_terminated_node_ips({})
    assert len(node_ips) == 1, node_ips
    avail_by_ip = {ip: {} for ip in node_ips}
    max_by_ip = {ip: {'CPU': 32} for ip in node_ips}
    demands = [{'CPU': 1, 'GPU': 1}] * 1
    (to_launch, rem) = scheduler.get_nodes_to_launch(all_nodes, {}, [], avail_by_ip, [], max_by_ip, demands, EMPTY_AVAILABILITY_SUMMARY)
    assert len(to_launch) == 0, to_launch
    assert not rem
    demands = [{'CPU': 1, 'GPU': 0}] * 33
    (to_launch, rem) = scheduler.get_nodes_to_launch(all_nodes, {}, [], avail_by_ip, [], max_by_ip, demands, EMPTY_AVAILABILITY_SUMMARY)
    assert len(to_launch) == 1, to_launch
    assert not rem

def test_request_resources_existing_usage():
    if False:
        i = 10
        return i + 15
    provider = MockProvider()
    TYPES = {'p2.8xlarge': {'node_config': {}, 'resources': {'CPU': 32, 'GPU': 8}, 'max_workers': 40}}
    scheduler = ResourceDemandScheduler(provider, TYPES, max_workers=100, head_node_type='empty_node', upscaling_speed=1)
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_KIND: NODE_KIND_WORKER, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}, 2)
    all_nodes = provider.non_terminated_nodes({})
    node_ips = provider.non_terminated_node_ips({})
    assert len(node_ips) == 2, node_ips
    avail_by_ip = {ip: {} for ip in node_ips}
    max_by_ip = {ip: {'GPU': 8, 'CPU': 32} for ip in node_ips}
    demands = []
    (to_launch, rem) = scheduler.get_nodes_to_launch(all_nodes, {}, [], avail_by_ip, [], max_by_ip, demands, EMPTY_AVAILABILITY_SUMMARY)
    assert len(to_launch) == 0, to_launch
    assert not rem
    avail_by_ip = {ip: {} for ip in node_ips}
    demands = [{'GPU': 4}] * 4
    (to_launch, rem) = scheduler.get_nodes_to_launch(all_nodes, {}, [], avail_by_ip, [], max_by_ip, demands, EMPTY_AVAILABILITY_SUMMARY)
    assert len(to_launch) == 0, to_launch
    assert not rem
    avail_by_ip = {ip: {} for ip in node_ips}
    demands = [{'GPU': 4}] * 7
    (to_launch, rem) = scheduler.get_nodes_to_launch(all_nodes, {}, [], avail_by_ip, [], max_by_ip, demands, EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch.get('p2.8xlarge') == 2, to_launch
    assert not rem
    avail_by_ip = {ip: {'GPU': 4, 'CPU': 32} for ip in node_ips}
    demands = []
    (to_launch, rem) = scheduler.get_nodes_to_launch(all_nodes, {}, [], avail_by_ip, [], max_by_ip, demands, EMPTY_AVAILABILITY_SUMMARY)
    assert len(to_launch) == 0, to_launch
    assert not rem
    avail_by_ip = {ip: {'GPU': 4, 'CPU': 32} for ip in node_ips}
    demands = [{'GPU': 4}] * 4
    (to_launch, rem) = scheduler.get_nodes_to_launch(all_nodes, {}, [], avail_by_ip, [], max_by_ip, demands, EMPTY_AVAILABILITY_SUMMARY)
    assert len(to_launch) == 0, to_launch
    assert not rem
    avail_by_ip = {ip: {'GPU': 4, 'CPU': 32} for ip in node_ips}
    demands = [{'GPU': 4}] * 7
    (to_launch, rem) = scheduler.get_nodes_to_launch(all_nodes, {}, [], avail_by_ip, [], max_by_ip, demands, EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch.get('p2.8xlarge') == 2, to_launch
    assert not rem
    avail_by_ip = {ip: {'GPU': 4, 'CPU': 32} for ip in node_ips}
    demands = [{'GPU': 4}] * 70
    (to_launch, rem) = scheduler.get_nodes_to_launch(all_nodes, {}, [], avail_by_ip, [], max_by_ip, demands, EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch.get('p2.8xlarge') == 33, to_launch
    assert not rem

def test_backlog_queue_impact_on_binpacking_time():
    if False:
        return 10
    new_types = copy.deepcopy(TYPES_A)
    new_types['p2.8xlarge']['max_workers'] = 1000
    new_types['m4.16xlarge']['max_workers'] = 1000

    def test_backlog_queue_impact_on_binpacking_time_aux(num_available_nodes, time_to_assert, demand_request_shape):
        if False:
            for i in range(10):
                print('nop')
        provider = MockProvider()
        scheduler = ResourceDemandScheduler(provider, new_types, max_workers=10000, head_node_type='m4.16xlarge', upscaling_speed=1)
        provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'm4.16xlarge', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}, num_available_nodes)
        cpu_ips = provider.non_terminated_node_ips({})
        provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}, num_available_nodes)
        all_nodes = provider.non_terminated_nodes({})
        all_ips = provider.non_terminated_node_ips({})
        gpu_ips = [ip for ip in all_ips if ip not in cpu_ips]
        usage_by_ip = {}
        for i in range(num_available_nodes):
            usage_by_ip[cpu_ips[i]] = {'CPU': 64}
            usage_by_ip[gpu_ips[i]] = {'GPU': 8, 'CPU': 32}
        demands = demand_request_shape * AUTOSCALER_MAX_RESOURCE_DEMAND_VECTOR_SIZE
        t1 = time.time()
        (to_launch, rem) = scheduler.get_nodes_to_launch(all_nodes, {}, demands, usage_by_ip, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
        t2 = time.time()
        assert t2 - t1 < time_to_assert
        print('The time took to launch', to_launch, 'with number of available nodes set to', num_available_nodes, 'is:', t2 - t1)
        return to_launch
    to_launch = test_backlog_queue_impact_on_binpacking_time_aux(num_available_nodes=0, time_to_assert=10, demand_request_shape=[{'GPU': 1}, {'CPU': 1}])
    assert to_launch == {'m4.16xlarge': 1, 'p2.8xlarge': 5, 'p2.xlarge': 1}
    to_launch = test_backlog_queue_impact_on_binpacking_time_aux(num_available_nodes=50, time_to_assert=10, demand_request_shape=[{'GPU': 1}, {'CPU': 2}])
    assert to_launch == {'p2.8xlarge': 50}
    to_launch = test_backlog_queue_impact_on_binpacking_time_aux(num_available_nodes=125, time_to_assert=10, demand_request_shape=[{'GPU': 1}, {'CPU': 1}])
    assert to_launch == {}
    to_launch = test_backlog_queue_impact_on_binpacking_time_aux(num_available_nodes=500, time_to_assert=10, demand_request_shape=[{'GPU': 8}, {'CPU': 64}])
    assert to_launch == {'m4.16xlarge': 500, 'p2.8xlarge': 500}

class TestPlacementGroupScaling:

    def test_strategies(self):
        if False:
            print('Hello World!')
        provider = MockProvider()
        scheduler = ResourceDemandScheduler(provider, TYPES_A, 10, head_node_type='p2.8xlarge', upscaling_speed=1)
        provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge'}, 2)
        nodes = provider.non_terminated_nodes({})
        resource_demands = [{'GPU': 4}] * 2
        pending_placement_groups = [PlacementGroupTableData(state=PlacementGroupTableData.PENDING, strategy=PlacementStrategy.STRICT_SPREAD, bundles=[Bundle(unit_resources={'GPU': 2}), Bundle(unit_resources={'GPU': 2}), Bundle(unit_resources={'GPU': 2})]), PlacementGroupTableData(state=PlacementGroupTableData.PENDING, strategy=PlacementStrategy.STRICT_PACK, bundles=[Bundle(unit_resources={'GPU': 2})] * 4), PlacementGroupTableData(state=PlacementGroupTableData.PENDING, strategy=PlacementStrategy.PACK, bundles=[Bundle(unit_resources={'GPU': 2})] * 2), PlacementGroupTableData(state=PlacementGroupTableData.PENDING, strategy=PlacementStrategy.SPREAD, bundles=[Bundle(unit_resources={'GPU': 2})] * 2)]
        (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, {}, resource_demands, {}, pending_placement_groups, {}, [], EMPTY_AVAILABILITY_SUMMARY)
        assert to_launch == {'p2.8xlarge': 2}
        assert not rem

    def test_many_strict_spreads(self):
        if False:
            while True:
                i = 10
        provider = MockProvider()
        scheduler = ResourceDemandScheduler(provider, TYPES_A, 10, head_node_type='p2.8xlarge', upscaling_speed=1)
        provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge'}, 2)
        nodes = provider.non_terminated_nodes({})
        resource_demands = [{'GPU': 1}] * 6
        pending_placement_groups = [PlacementGroupTableData(state=PlacementGroupTableData.PENDING, strategy=PlacementStrategy.STRICT_SPREAD, bundles=[Bundle(unit_resources={'GPU': 2})] * 3)]
        pending_placement_groups = pending_placement_groups * 3
        (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, {}, resource_demands, {}, pending_placement_groups, {}, [], EMPTY_AVAILABILITY_SUMMARY)
        assert to_launch == {'p2.8xlarge': 1}
        assert not rem

    def test_packing(self):
        if False:
            while True:
                i = 10
        provider = MockProvider()
        scheduler = ResourceDemandScheduler(provider, TYPES_A, 10, head_node_type='p2.8xlarge', upscaling_speed=1)
        provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge'}, 1)
        nodes = provider.non_terminated_nodes({})
        resource_demands = [{'GPU': 1}] * 2
        pending_placement_groups = [PlacementGroupTableData(state=PlacementGroupTableData.PENDING, strategy=PlacementStrategy.STRICT_PACK, bundles=[Bundle(unit_resources={'GPU': 2})] * 3)]
        (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, {}, resource_demands, {}, pending_placement_groups, {}, [], EMPTY_AVAILABILITY_SUMMARY)
        assert to_launch == {}
        assert not rem

def test_get_concurrent_resource_demand_to_launch():
    if False:
        i = 10
        return i + 15
    node_types = copy.deepcopy(TYPES_A)
    node_types['p2.8xlarge']['min_workers'] = 1
    node_types['p2.8xlarge']['max_workers'] = 10
    node_types['m4.large']['min_workers'] = 2
    node_types['m4.large']['max_workers'] = 100
    provider = MockProvider()
    scheduler = ResourceDemandScheduler(provider, node_types, 200, head_node_type='empty_node', upscaling_speed=1)
    assert len(provider.non_terminated_nodes({})) == 0
    updated_to_launch = scheduler._get_concurrent_resource_demand_to_launch({}, [], [], {}, {}, {})
    assert updated_to_launch == {}
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_KIND: NODE_KIND_WORKER}, 1)
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'm4.large', TAG_RAY_NODE_KIND: NODE_KIND_WORKER}, 2)
    to_launch = {'p2.8xlarge': 4, 'm4.large': 40}
    non_terminated_nodes = provider.non_terminated_nodes({})
    pending_launches_nodes = {'p2.8xlarge': 1, 'm4.large': 1}
    connected_nodes = []
    updated_to_launch = scheduler._get_concurrent_resource_demand_to_launch(to_launch, connected_nodes, non_terminated_nodes, pending_launches_nodes, {}, {})
    assert updated_to_launch == {'p2.8xlarge': 3, 'm4.large': 2}
    updated_to_launch = scheduler._get_concurrent_resource_demand_to_launch(to_launch, connected_nodes, non_terminated_nodes, pending_launches_nodes, adjusted_min_workers={'m4.large': 40}, placement_group_nodes={})
    assert updated_to_launch == {'p2.8xlarge': 3, 'm4.large': 40}
    updated_to_launch = scheduler._get_concurrent_resource_demand_to_launch(to_launch, connected_nodes, non_terminated_nodes, pending_launches_nodes, {}, placement_group_nodes={'m4.large': 40})
    assert updated_to_launch == {'p2.8xlarge': 3, 'm4.large': 40}
    updated_to_launch = scheduler._get_concurrent_resource_demand_to_launch(to_launch, connected_nodes, non_terminated_nodes, pending_launches_nodes, adjusted_min_workers={'m4.large': 25}, placement_group_nodes={'m4.large': 15})
    assert updated_to_launch == {'p2.8xlarge': 3, 'm4.large': 40}
    connected_nodes = [provider.internal_ip(node_id) for node_id in non_terminated_nodes]
    updated_to_launch = scheduler._get_concurrent_resource_demand_to_launch(to_launch, connected_nodes, non_terminated_nodes, pending_launches_nodes, {}, {})
    assert updated_to_launch == {'p2.8xlarge': 4, 'm4.large': 4}
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_KIND: NODE_KIND_WORKER}, 5)
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'm4.large', TAG_RAY_NODE_KIND: NODE_KIND_WORKER}, 5)
    non_terminated_nodes = provider.non_terminated_nodes({})
    to_launch = {'m4.large': 36}
    pending_launches_nodes = {}
    updated_to_launch = scheduler._get_concurrent_resource_demand_to_launch(to_launch, connected_nodes, non_terminated_nodes, pending_launches_nodes, {}, {})
    assert updated_to_launch == {}
    connected_nodes = [provider.internal_ip(node_id) for node_id in non_terminated_nodes]
    updated_to_launch = scheduler._get_concurrent_resource_demand_to_launch(to_launch, connected_nodes, non_terminated_nodes, pending_launches_nodes, {}, {})
    assert updated_to_launch == {'m4.large': 7}
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'm4.large', TAG_RAY_NODE_KIND: NODE_KIND_WORKER}, 7)
    non_terminated_nodes = provider.non_terminated_nodes({})
    to_launch = {'m4.large': 29}
    pending_launches_nodes = {'m4.large': 1}
    updated_to_launch = scheduler._get_concurrent_resource_demand_to_launch(to_launch, connected_nodes, non_terminated_nodes, pending_launches_nodes, {}, {})
    assert updated_to_launch == {}
    connected_nodes = [provider.internal_ip(node_id) for node_id in non_terminated_nodes]
    updated_to_launch = scheduler._get_concurrent_resource_demand_to_launch(to_launch, connected_nodes, non_terminated_nodes, pending_launches_nodes, {}, {})
    assert updated_to_launch == {'m4.large': 13}

def test_get_concurrent_resource_demand_to_launch_with_upscaling_speed():
    if False:
        i = 10
        return i + 15
    node_types = copy.deepcopy(TYPES_A)
    node_types['p2.8xlarge']['min_workers'] = 1
    node_types['p2.8xlarge']['max_workers'] = 10
    node_types['m4.large']['min_workers'] = 2
    node_types['m4.large']['max_workers'] = 100

    def create_provider():
        if False:
            print('Hello World!')
        provider = MockProvider()
        provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_KIND: NODE_KIND_WORKER}, 0)
        provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'm4.large', TAG_RAY_NODE_KIND: NODE_KIND_WORKER}, 0)
        return provider
    slow_scheduler = ResourceDemandScheduler(create_provider(), node_types, 200, head_node_type='empty_node', upscaling_speed=1)
    to_launch = slow_scheduler._get_concurrent_resource_demand_to_launch({'m4.large': 50}, [], slow_scheduler.provider.non_terminated_nodes({}), {}, {}, {})
    assert to_launch == {'m4.large': 5}
    mid_scheduler = ResourceDemandScheduler(create_provider(), node_types, 200, head_node_type='empty_node', upscaling_speed=25)
    to_launch = mid_scheduler._get_concurrent_resource_demand_to_launch({'m4.large': 50}, [], mid_scheduler.provider.non_terminated_nodes({}), {}, {}, {})
    assert to_launch == {'m4.large': 25}
    fast_scheduler = ResourceDemandScheduler(create_provider(), node_types, 200, head_node_type='empty_node', upscaling_speed=9999)
    to_launch = fast_scheduler._get_concurrent_resource_demand_to_launch({'m4.large': 50}, [], fast_scheduler.provider.non_terminated_nodes({}), {}, {}, {})
    assert to_launch == {'m4.large': 50}

def test_get_nodes_to_launch_max_launch_concurrency_placement_groups():
    if False:
        i = 10
        return i + 15
    provider = MockProvider()
    new_types = copy.deepcopy(TYPES_A)
    new_types['p2.8xlarge']['min_workers'] = 10
    new_types['p2.8xlarge']['max_workers'] = 40
    scheduler = ResourceDemandScheduler(provider, new_types, 50, head_node_type=None, upscaling_speed=1)
    pending_placement_groups = [PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.PACK, bundles=[Bundle(unit_resources={'GPU': 8})] * 25)]
    (to_launch, rem) = scheduler.get_nodes_to_launch([], {}, [], {}, pending_placement_groups, {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.8xlarge': 25}
    assert not rem
    pending_placement_groups = [PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.STRICT_SPREAD, bundles=[Bundle(unit_resources={'GPU': 2})] * 25), PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.PACK, bundles=[Bundle(unit_resources={'GPU': 6})] * 30)]
    (to_launch, rem) = scheduler.get_nodes_to_launch([], {}, [], {}, pending_placement_groups, {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.8xlarge': 30}
    assert not rem
    pending_placement_groups = [PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.STRICT_SPREAD, bundles=[Bundle(unit_resources={'GPU': 2})] * 25), PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.PACK, bundles=[Bundle(unit_resources={'GPU': 6})] * 60)]
    (to_launch, rem) = scheduler.get_nodes_to_launch([], {}, [], {}, pending_placement_groups, {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.8xlarge': 40}
    assert rem == [{'GPU': 6.0}] * 20
    scheduler.node_types['p2.8xlarge']['max_workers'] = 60
    (to_launch, rem) = scheduler.get_nodes_to_launch([], {}, [], {}, pending_placement_groups, {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.8xlarge': 51}
    assert rem == [{'GPU': 6.0}] * 9

def test_get_nodes_to_launch_max_launch_concurrency():
    if False:
        return 10
    provider = MockProvider()
    new_types = copy.deepcopy(TYPES_A)
    new_types['p2.8xlarge']['min_workers'] = 10
    new_types['p2.8xlarge']['max_workers'] = 40
    scheduler = ResourceDemandScheduler(provider, new_types, 30, head_node_type=None, upscaling_speed=1)
    (to_launch, rem) = scheduler.get_nodes_to_launch([], {}, [], {}, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.8xlarge': 10}
    assert not rem
    scheduler.node_types['p2.8xlarge']['min_workers'] = 4
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_STATUS: STATUS_UNINITIALIZED}, 1)
    nodes = provider.non_terminated_nodes({})
    ips = provider.non_terminated_node_ips({TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE})
    utilizations = {ip: {'GPU': 8} for ip in ips}
    launching_nodes = {'p2.8xlarge': 1}
    demands = [{'GPU': 8}] * (len(utilizations) + 40)
    (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, launching_nodes, demands, utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.8xlarge': 3}
    assert rem == [{'GPU': 8}] * 9
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}, 8)
    nodes = provider.non_terminated_nodes({})
    ips = provider.non_terminated_node_ips({TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE})
    utilizations = {ip: {'GPU': 8} for ip in ips}
    launching_nodes = {'p2.8xlarge': 1}
    demands = [{'GPU': 8}] * (len(utilizations) + 15)
    (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, launching_nodes, demands, utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'p2.8xlarge': 6}
    assert not rem

class LoadMetricsTest(unittest.TestCase):

    def testResourceDemandVector(self):
        if False:
            for i in range(10):
                print('nop')
        lm = LoadMetrics()
        lm.update('1.1.1.1', mock_raylet_id(), {'CPU': 2}, {'CPU': 1}, waiting_bundles=[{'GPU': 1}], infeasible_bundles=[{'CPU': 16}])
        assert same_elements(lm.get_resource_demand_vector(), [{'CPU': 16}, {'GPU': 1}])

    def testPlacementGroupLoad(self):
        if False:
            while True:
                i = 10
        lm = LoadMetrics()
        pending_placement_groups = [PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.PACK, bundles=[Bundle(unit_resources={'GPU': 2})] * 2), PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.SPREAD, bundles=[Bundle(unit_resources={'GPU': 2})] * 2)]
        lm.update('1.1.1.1', mock_raylet_id(), {}, {}, pending_placement_groups=pending_placement_groups)
        assert lm.get_pending_placement_groups() == pending_placement_groups

    def testSummary(self):
        if False:
            return 10
        lm = LoadMetrics()
        assert lm.summary() is not None
        pending_placement_groups = [PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.PACK, bundles=[Bundle(unit_resources={'GPU': 2})] * 2), PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.PACK, bundles=[Bundle(unit_resources={'GPU': 2})] * 2)]
        lm.update('1.1.1.1', mock_raylet_id(), {'CPU': 64, 'memory': 1000 * 1024 * 1024, 'object_store_memory': 2000 * 1024 * 1024}, {'CPU': 2, 'memory': 500 * 1024 * 1024, 'object_store_memory': 1000 * 1024 * 1024})
        lm.update('1.1.1.2', mock_raylet_id(), {'CPU': 64, 'GPU': 8, 'accelerator_type:V100': 1}, {'CPU': 0, 'GPU': 1, 'accelerator_type:V100': 1})
        lm.update('1.1.1.3', mock_raylet_id(), {'CPU': 64, 'GPU': 8, 'accelerator_type:V100': 1}, {'CPU': 0, 'GPU': 0, 'accelerator_type:V100': 0.92})
        lm.update('1.1.1.4', mock_raylet_id(), {'CPU': 2}, {'CPU': 2}, waiting_bundles=[{'GPU': 2}] * 10, infeasible_bundles=[{'CPU': 16}, {'GPU': 2}, {'CPU': 16, 'GPU': 2}], pending_placement_groups=pending_placement_groups)
        lm.set_resource_requests([{'CPU': 64}, {'GPU': 8}, {'GPU': 8}])
        summary = lm.summary()
        assert summary.usage['CPU'] == (190, 194)
        assert summary.usage['GPU'] == (15, 16)
        assert summary.usage['memory'] == (500 * 2 ** 20, 1000 * 2 ** 20)
        assert summary.usage['object_store_memory'] == (1000 * 2 ** 20, 2000 * 2 ** 20)
        assert summary.usage['accelerator_type:V100'][1] == 2, 'Not comparing the usage value due to floating point error.'
        assert ({'GPU': 2}, 11) in summary.resource_demand
        assert ({'CPU': 16}, 1) in summary.resource_demand
        assert ({'CPU': 16, 'GPU': 2}, 1) in summary.resource_demand
        assert len(summary.resource_demand) == 3
        assert ({'bundles': [({'GPU': 2}, 2)], 'strategy': 'PACK'}, 2) in summary.pg_demand
        assert len(summary.pg_demand) == 1
        assert ({'GPU': 8}, 2) in summary.request_demand
        assert ({'CPU': 64}, 1) in summary.request_demand
        assert len(summary.request_demand) == 2
        assert len(summary.node_types) == 3, summary.node_types
        summary_dict = asdict(summary)
        assert summary_dict['usage']['CPU'] == (190, 194)
        assert summary_dict['usage']['GPU'] == (15, 16)
        assert summary_dict['usage']['memory'] == (500 * 2 ** 20, 1000 * 2 ** 20)
        assert summary_dict['usage']['object_store_memory'] == (1000 * 2 ** 20, 2000 * 2 ** 20)
        assert summary_dict['usage']['accelerator_type:V100'][1] == 2, 'Not comparing the usage value due to floating point error.'
        assert ({'GPU': 2}, 11) in summary_dict['resource_demand']
        assert ({'CPU': 16}, 1) in summary_dict['resource_demand']
        assert ({'CPU': 16, 'GPU': 2}, 1) in summary_dict['resource_demand']
        assert len(summary_dict['resource_demand']) == 3
        assert ({'bundles': [({'GPU': 2}, 2)], 'strategy': 'PACK'}, 2) in summary_dict['pg_demand']
        assert len(summary_dict['pg_demand']) == 1
        assert ({'GPU': 8}, 2) in summary_dict['request_demand']
        assert ({'CPU': 64}, 1) in summary_dict['request_demand']
        assert len(summary_dict['request_demand']) == 2
        assert len(summary_dict['node_types']) == 3, summary_dict['node_types']
        json.dumps(summary_dict)
        summary_dict['head_ip'] = '1.1.1.1'
        LoadMetricsSummary(**summary_dict)

class AutoscalingTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        _NODE_PROVIDERS['mock'] = lambda config: self.create_provider
        self.provider = None
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.provider = None
        del _NODE_PROVIDERS['mock']
        _clear_provider_cache()
        shutil.rmtree(self.tmpdir)
        ray.shutdown()

    def waitForNodes(self, expected, comparison=None, tag_filters=None):
        if False:
            while True:
                i = 10
        if tag_filters is None:
            tag_filters = {}
        MAX_ITER = 50
        for i in range(MAX_ITER):
            n = len(self.provider.non_terminated_nodes(tag_filters))
            if comparison is None:
                comparison = self.assertEqual
            try:
                comparison(n, expected)
                return
            except Exception:
                if i == MAX_ITER - 1:
                    raise
            time.sleep(0.1)

    def create_provider(self, config, cluster_name):
        if False:
            for i in range(10):
                print('nop')
        assert self.provider
        return self.provider

    def write_config(self, config):
        if False:
            for i in range(10):
                print('nop')
        path = self.tmpdir + '/simple.yaml'
        with open(path, 'w') as f:
            f.write(yaml.dump(config))
        return path

    def testGetOrCreateMultiNodeType(self):
        if False:
            i = 10
            return i + 15
        config_path = self.write_config(MULTI_WORKER_CLUSTER)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        runner.respond_to_call('json .Config.Env', ['[]'])
        get_or_create_head_node(MULTI_WORKER_CLUSTER, printable_config_file=config_path, no_restart=False, restart_only=False, yes=True, override_cluster_name=None, _provider=self.provider, _runner=runner)
        self.waitForNodes(1)
        runner.assert_has_call('1.2.3.4', 'init_cmd')
        runner.assert_has_call('1.2.3.4', 'setup_cmd')
        runner.assert_has_call('1.2.3.4', 'start_ray_head')
        self.assertEqual(self.provider.mock_nodes['0'].node_type, 'empty_node')
        self.assertEqual(self.provider.mock_nodes['0'].node_config.get('FooProperty'), 42)
        self.assertEqual(self.provider.mock_nodes['0'].node_config.get('TestProp'), 1)
        self.assertEqual(self.provider.mock_nodes['0'].tags.get(TAG_RAY_USER_NODE_TYPE), 'empty_node')

    def testGetOrCreateMultiNodeTypeCustomHeadResources(self):
        if False:
            while True:
                i = 10
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['available_node_types']['empty_node']['resources'] = {'empty_resource_name': 1000}
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        runner.respond_to_call('json .Config.Env', ['[]'])
        get_or_create_head_node(config, printable_config_file=config_path, no_restart=False, restart_only=False, yes=True, override_cluster_name=None, _provider=self.provider, _runner=runner)
        self.waitForNodes(1)
        runner.assert_has_call('1.2.3.4', 'init_cmd')
        runner.assert_has_call('1.2.3.4', 'setup_cmd')
        runner.assert_has_call('1.2.3.4', 'start_ray_head')
        runner.assert_has_call('1.2.3.4', 'empty_resource_name')
        self.assertEqual(self.provider.mock_nodes['0'].node_type, 'empty_node')
        self.assertEqual(self.provider.mock_nodes['0'].node_config.get('FooProperty'), 42)
        self.assertEqual(self.provider.mock_nodes['0'].node_config.get('TestProp'), 1)
        self.assertEqual(self.provider.mock_nodes['0'].tags.get(TAG_RAY_USER_NODE_TYPE), 'empty_node')

    def testSummary(self):
        if False:
            return 10
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['available_node_types']['m4.large']['min_workers'] = 2
        config['max_workers'] = 10
        config['docker'] = {}
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_USER_NODE_TYPE: 'empty_node', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}, 1)
        head_ip = self.provider.non_terminated_node_ips({})[0]
        lm = LoadMetrics()
        autoscaler = MockAutoscaler(config_path, lm, MockGcsClient(), max_failures=0, max_launch_batch=1, max_concurrent_launches=10, process_runner=runner, update_interval_s=0)
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(3)
        for ip in self.provider.non_terminated_node_ips({}):
            lm.update(ip, mock_raylet_id(), {'CPU': 2}, {'CPU': 0})
        lm.update(head_ip, mock_raylet_id(), {'CPU': 16}, {'CPU': 1})
        autoscaler.update()
        while True:
            if len(self.provider.non_terminated_nodes({TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE})) == 3:
                break
        runner.ready_to_run.clear()
        lm.update(head_ip, mock_raylet_id(), {'CPU': 16}, {'CPU': 1}, waiting_bundles=[{'GPU': 1}])
        autoscaler.update()
        self.waitForNodes(4)
        self.provider.ready_to_create.clear()
        lm.set_resource_requests([{'CPU': 64}] * 2)
        autoscaler.update()
        obj = ('172.0.0.4', 'm4.4xlarge')
        autoscaler.node_tracker._add_node_mapping(4, obj)
        print(f'Head ip: {head_ip}')
        summary = autoscaler.summary()
        assert summary.active_nodes['m4.large'] == 2
        assert summary.active_nodes['empty_node'] == 1
        assert len(summary.active_nodes) == 2, summary.active_nodes
        assert summary.pending_nodes == [('172.0.0.3', 'p2.xlarge', STATUS_WAITING_FOR_SSH)]
        assert summary.pending_launches == {'m4.16xlarge': 2}
        assert summary.failed_nodes == [('172.0.0.4', 'm4.4xlarge')]
        assert summary.pending_resources == {'GPU': 1, 'CPU': 144}, summary.pending_resources
        summary_dict = asdict(summary)
        assert summary_dict['active_nodes']['m4.large'] == 2
        assert summary_dict['active_nodes']['empty_node'] == 1
        assert len(summary_dict['active_nodes']) == 2, summary_dict['active_nodes']
        assert summary_dict['pending_nodes'] == [('172.0.0.3', 'p2.xlarge', STATUS_WAITING_FOR_SSH)]
        assert summary_dict['pending_launches'] == {'m4.16xlarge': 2}
        assert summary_dict['failed_nodes'] == [('172.0.0.4', 'm4.4xlarge')]
        assert summary.node_type_mapping == {'172.0.0.0': 'empty_node', '172.0.0.1': 'm4.large', '172.0.0.2': 'm4.large', '172.0.0.3': 'p2.xlarge'}
        json.dumps(summary_dict)
        assert len(autoscaler.info_string()) > 1

    def testScaleUpMinSanity(self):
        if False:
            for i in range(10):
                print('nop')
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['available_node_types']['m4.large']['min_workers'] = 2
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        autoscaler = MockAutoscaler(config_path, LoadMetrics(), MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(3)
        autoscaler.update()
        self.waitForNodes(3)

    def testScaleUpMinSanityWithHeadNode(self):
        if False:
            while True:
                i = 10
        'Make sure when min_workers is used with head node it does not count\n        head_node in min_workers.'
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['available_node_types']['empty_node']['min_workers'] = 2
        config['available_node_types']['empty_node']['max_workers'] = 2
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        autoscaler = MockAutoscaler(config_path, LoadMetrics(), MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(3)
        autoscaler.update()
        self.waitForNodes(3)

    def testPlacementGroup(self):
        if False:
            return 10
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['min_workers'] = 0
        config['max_workers'] = 999
        config['head_node_type'] = 'm4.4xlarge'
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: 'head', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'm4.4xlarge'}, 1)
        head_ip = self.provider.non_terminated_node_ips({})[0]
        lm = LoadMetrics()
        autoscaler = MockAutoscaler(config_path, lm, MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        head_ip = self.provider.non_terminated_node_ips({})[0]
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(1)
        pending_placement_groups = [PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.STRICT_SPREAD, bundles=[Bundle(unit_resources={'GPU': 2})] * 3), PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.PACK, bundles=[Bundle(unit_resources={'GPU': 2})] * 5)]
        placement_group_resource_demands = [{'GPU_group_0_6c2506ac733bc37496295b02c4fad446': 0.0101, 'GPU_group_6c2506ac733bc37496295b02c4fad446': 0.0101}]
        lm.update(head_ip, mock_raylet_id(), {'CPU': 16}, {'CPU': 16}, infeasible_bundles=placement_group_resource_demands, waiting_bundles=[{'GPU': 8}], pending_placement_groups=pending_placement_groups)
        autoscaler.update()
        self.waitForNodes(5)
        for i in range(1, 5):
            assert self.provider.mock_nodes[str(i)].node_type == 'p2.8xlarge'
        pending_placement_groups = [PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.STRICT_PACK, bundles=[Bundle(unit_resources={'GPU': 2})] * 4), PlacementGroupTableData(state=PlacementGroupTableData.RESCHEDULING, strategy=PlacementStrategy.SPREAD, bundles=[Bundle(unit_resources={'GPU': 2})] * 2)]

    def testScaleUpMinWorkers(self):
        if False:
            return 10
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['max_workers'] = 50
        config['idle_timeout_minutes'] = 1
        config['available_node_types']['m4.large']['min_workers'] = 1
        config['available_node_types']['p2.8xlarge']['min_workers'] = 1
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        lm = LoadMetrics()
        autoscaler = MockAutoscaler(config_path, lm, MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(3)
        assert len(self.provider.mock_nodes) == 3
        assert {self.provider.mock_nodes['1'].node_type, self.provider.mock_nodes['2'].node_type} == {'p2.8xlarge', 'm4.large'}
        self.provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_NODE_KIND: NODE_KIND_WORKER, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}, 2)
        self.provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'm4.16xlarge', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_NODE_KIND: NODE_KIND_WORKER, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}, 2)
        assert len(self.provider.non_terminated_nodes({})) == 7
        for node_id in self.provider.non_terminated_nodes({}):
            lm.last_used_time_by_ip[self.provider.internal_ip(node_id)] = -60
        fill_in_raylet_ids(self.provider, lm)
        autoscaler.update()
        self.waitForNodes(3)
        cnt = 0
        for id in list(self.provider.mock_nodes.keys())[1:]:
            if self.provider.mock_nodes[id].state == 'running' or self.provider.mock_nodes[id].state == 'pending':
                assert self.provider.mock_nodes[id].node_type in {'p2.8xlarge', 'm4.large'}
                cnt += 1
        assert cnt == 2

    def testScaleUpIgnoreUsed(self):
        if False:
            while True:
                i = 10
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['min_workers'] = 0
        config['target_utilization_fraction'] = 1.0
        config['head_node_type'] = 'p2.xlarge'
        config_path = self.write_config(config)
        self.provider = MockProvider()
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: 'head', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'p2.xlarge', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}, 1)
        head_ip = self.provider.non_terminated_node_ips({})[0]
        self.provider.finish_starting_nodes()
        runner = MockProcessRunner()
        lm = LoadMetrics()
        autoscaler = MockAutoscaler(config_path, lm, MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        autoscaler.update()
        self.waitForNodes(1)
        lm.update(head_ip, mock_raylet_id(), {'CPU': 4, 'GPU': 1}, {})
        self.waitForNodes(1)
        lm.update(head_ip, mock_raylet_id(), {'CPU': 4, 'GPU': 1}, {'GPU': 0}, waiting_bundles=[{'GPU': 1}])
        autoscaler.update()
        self.waitForNodes(2)
        assert self.provider.mock_nodes['1'].node_type == 'p2.xlarge'

    def testRequestBundlesAccountsForHeadNode(self):
        if False:
            for i in range(10):
                print('nop')
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['head_node_type'] = 'p2.8xlarge'
        config['min_workers'] = 0
        config['max_workers'] = 50
        config_path = self.write_config(config)
        self.provider = MockProvider()
        self.provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_NODE_KIND: 'head', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}, 1)
        runner = MockProcessRunner()
        autoscaler = MockAutoscaler(config_path, LoadMetrics(), MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(1)
        autoscaler.load_metrics.set_resource_requests([{'CPU': 1}])
        autoscaler.update()
        self.waitForNodes(1)
        assert len(self.provider.mock_nodes) == 1
        autoscaler.load_metrics.set_resource_requests([{'GPU': 8}])
        autoscaler.update()
        self.waitForNodes(1)
        autoscaler.load_metrics.set_resource_requests([{'GPU': 8}] * 2)
        autoscaler.update()
        self.waitForNodes(2)
        assert self.provider.mock_nodes['1'].node_type == 'p2.8xlarge'

    def testRequestBundles(self):
        if False:
            return 10
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['min_workers'] = 0
        config['max_workers'] = 50
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        runner.respond_to_call('json .Config.Env', ['[]' for i in range(6)])
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        autoscaler = MockAutoscaler(config_path, LoadMetrics(), MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(1)
        autoscaler.load_metrics.set_resource_requests([{'CPU': 1}])
        autoscaler.update()
        self.waitForNodes(2)
        assert self.provider.mock_nodes['1'].node_type == 'm4.large'
        autoscaler.load_metrics.set_resource_requests([{'GPU': 8}])
        autoscaler.update()
        self.waitForNodes(3)
        assert self.provider.mock_nodes['2'].node_type == 'p2.8xlarge'
        autoscaler.load_metrics.set_resource_requests([{'CPU': 32}] * 4)
        autoscaler.update()
        self.waitForNodes(5)
        assert self.provider.mock_nodes['3'].node_type == 'm4.16xlarge'
        assert self.provider.mock_nodes['4'].node_type == 'm4.16xlarge'

    def testResourcePassing(self):
        if False:
            print('Hello World!')
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['min_workers'] = 0
        config['max_workers'] = 50
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        runner.respond_to_call('json .Config.Env', ['[]' for i in range(2)])
        self.provider.create_node({}, {TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        autoscaler = MockAutoscaler(config_path, LoadMetrics(), MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(0, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        autoscaler.load_metrics.set_resource_requests([{'CPU': 1}])
        autoscaler.update()
        self.waitForNodes(1, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        assert self.provider.mock_nodes['1'].node_type == 'm4.large'
        autoscaler.load_metrics.set_resource_requests([{'GPU': 8}])
        autoscaler.update()
        self.waitForNodes(2, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        assert self.provider.mock_nodes['2'].node_type == 'p2.8xlarge'
        autoscaler.update()
        sleep(0.1)
        runner.assert_has_call('172.0.0.1', 'RAY_OVERRIDE_RESOURCES=')
        runner.assert_has_call('172.0.0.1', '"CPU":2')
        runner.assert_has_call('172.0.0.2', 'RAY_OVERRIDE_RESOURCES=')
        runner.assert_has_call('172.0.0.2', '"CPU":32')
        runner.assert_has_call('172.0.0.2', '"GPU":8')

    def testScaleUpLoadMetrics(self):
        if False:
            while True:
                i = 10
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['min_workers'] = 0
        config['max_workers'] = 50
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        lm = LoadMetrics()
        autoscaler = MockAutoscaler(config_path, lm, MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(0, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        autoscaler.update()
        lm.update('1.2.3.4', mock_raylet_id(), {}, {}, waiting_bundles=[{'GPU': 1}], infeasible_bundles=[{'CPU': 16}])
        autoscaler.update()
        self.waitForNodes(1, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        nodes = {self.provider.mock_nodes['1'].node_type}
        assert nodes == {'p2.xlarge'}

    def testCommandPassing(self):
        if False:
            i = 10
            return i + 15
        t = 'custom'
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['available_node_types']['p2.8xlarge']['worker_setup_commands'] = ['new_worker_setup_command']
        config['available_node_types']['p2.xlarge']['initialization_commands'] = ['new_worker_initialization_cmd']
        config['available_node_types']['p2.xlarge']['resources'][t] = 1
        config['min_workers'] = 0
        config['max_workers'] = 10
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        runner.respond_to_call('json .Config.Env', ['[]' for i in range(4)])
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        lm = LoadMetrics()
        lm.update('172.0.0.0', mock_raylet_id(), {'CPU': 0}, {'CPU': 0})
        autoscaler = MockAutoscaler(config_path, lm, MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(1)
        autoscaler.load_metrics.set_resource_requests([{'CPU': 1}])
        autoscaler.update()
        self.waitForNodes(2)
        assert self.provider.mock_nodes['1'].node_type == 'm4.large'
        autoscaler.load_metrics.set_resource_requests([{'GPU': 8}])
        autoscaler.update()
        self.waitForNodes(3)
        assert self.provider.mock_nodes['2'].node_type == 'p2.8xlarge'
        autoscaler.load_metrics.set_resource_requests([{'GPU': 1}] * 9)
        autoscaler.update()
        self.waitForNodes(4)
        assert self.provider.mock_nodes['3'].node_type == 'p2.xlarge'
        autoscaler.update()
        sleep(0.1)
        runner.assert_has_call(self.provider.mock_nodes['2'].internal_ip, 'new_worker_setup_command')
        runner.assert_not_has_call(self.provider.mock_nodes['2'].internal_ip, 'setup_cmd')
        runner.assert_not_has_call(self.provider.mock_nodes['2'].internal_ip, 'worker_setup_cmd')
        runner.assert_has_call(self.provider.mock_nodes['3'].internal_ip, 'new_worker_initialization_cmd')
        runner.assert_not_has_call(self.provider.mock_nodes['3'].internal_ip, 'init_cmd')

    def testDockerWorkers(self):
        if False:
            for i in range(10):
                print('nop')
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['available_node_types']['p2.8xlarge']['docker'] = {'worker_image': 'p2.8x_image:latest', 'worker_run_options': ['p2.8x-run-options']}
        config['available_node_types']['p2.xlarge']['docker'] = {'worker_image': 'p2x_image:nightly'}
        config['docker']['run_options'] = ['head-and-worker-run-options']
        config['docker']['worker_run_options'] = ['standard-run-options']
        config['docker']['image'] = 'default-image:nightly'
        config['docker']['worker_image'] = 'default-image:nightly'
        config['min_workers'] = 0
        config['max_workers'] = 10
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        runner.respond_to_call('json .Config.Env', ['[]' for i in range(5)])
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        autoscaler = MockAutoscaler(config_path, LoadMetrics(), MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(1)
        autoscaler.load_metrics.set_resource_requests([{'CPU': 1}])
        autoscaler.update()
        self.waitForNodes(2)
        assert self.provider.mock_nodes['1'].node_type == 'm4.large'
        autoscaler.load_metrics.set_resource_requests([{'GPU': 8}])
        autoscaler.update()
        self.waitForNodes(3)
        assert self.provider.mock_nodes['2'].node_type == 'p2.8xlarge'
        autoscaler.load_metrics.set_resource_requests([{'GPU': 1}] * 9)
        autoscaler.update()
        self.waitForNodes(4)
        assert self.provider.mock_nodes['3'].node_type == 'p2.xlarge'
        autoscaler.update()
        autoscaler.load_metrics.set_resource_requests([{'CPU': 2}, {'CPU': 16}, {'CPU': 32}, {'CPU': 2}])
        autoscaler.update()
        self.waitForNodes(5)
        assert self.provider.mock_nodes['4'].node_type == 'm4.large'
        autoscaler.update()
        sleep(0.1)
        runner.assert_has_call(self.provider.mock_nodes['2'].internal_ip, 'p2.8x-run-options')
        runner.assert_has_call(self.provider.mock_nodes['2'].internal_ip, 'head-and-worker-run-options')
        runner.assert_has_call(self.provider.mock_nodes['2'].internal_ip, 'p2.8x_image:latest')
        runner.assert_not_has_call(self.provider.mock_nodes['2'].internal_ip, 'default-image:nightly')
        runner.assert_not_has_call(self.provider.mock_nodes['2'].internal_ip, 'standard-run-options')
        runner.assert_has_call(self.provider.mock_nodes['3'].internal_ip, 'p2x_image:nightly')
        runner.assert_has_call(self.provider.mock_nodes['3'].internal_ip, 'standard-run-options')
        runner.assert_has_call(self.provider.mock_nodes['3'].internal_ip, 'head-and-worker-run-options')
        runner.assert_not_has_call(self.provider.mock_nodes['3'].internal_ip, 'p2.8x-run-options')
        runner.assert_has_call(self.provider.mock_nodes['4'].internal_ip, 'default-image:nightly')
        runner.assert_has_call(self.provider.mock_nodes['4'].internal_ip, 'standard-run-options')
        runner.assert_has_call(self.provider.mock_nodes['4'].internal_ip, 'head-and-worker-run-options')
        runner.assert_not_has_call(self.provider.mock_nodes['4'].internal_ip, 'p2.8x-run-options')
        runner.assert_not_has_call(self.provider.mock_nodes['4'].internal_ip, 'p2x_image:nightly')

    def testUpdateConfig(self):
        if False:
            for i in range(10):
                print('nop')
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['available_node_types']['m4.large']['min_workers'] = 2
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        lm = LoadMetrics()
        autoscaler = MockAutoscaler(config_path, lm, MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(2, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        config['available_node_types']['m4.large']['min_workers'] = 0
        config['available_node_types']['m4.large']['node_config']['field_changed'] = 1
        config_path = self.write_config(config)
        fill_in_raylet_ids(self.provider, lm)
        autoscaler.update()
        self.waitForNodes(0, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})

    def testEmptyDocker(self):
        if False:
            return 10
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        del config['docker']
        config['min_workers'] = 0
        config['max_workers'] = 10
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        autoscaler = MockAutoscaler(config_path, LoadMetrics(), MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        assert len(self.provider.non_terminated_nodes({})) == 1
        autoscaler.update()
        self.waitForNodes(1)
        autoscaler.load_metrics.set_resource_requests([{'CPU': 1}])
        autoscaler.update()
        self.waitForNodes(2)
        assert self.provider.mock_nodes['1'].node_type == 'm4.large'
        autoscaler.load_metrics.set_resource_requests([{'GPU': 8}])
        autoscaler.update()
        self.waitForNodes(3)
        assert self.provider.mock_nodes['2'].node_type == 'p2.8xlarge'

    def testRequestResourcesIdleTimeout(self):
        if False:
            for i in range(10):
                print('nop')
        'Test request_resources() with and without idle timeout.'
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['max_workers'] = 4
        config['idle_timeout_minutes'] = 0
        config['available_node_types'] = {'empty_node': {'node_config': {}, 'resources': {'CPU': 2}, 'max_workers': 1}, 'def_worker': {'node_config': {}, 'resources': {'CPU': 2, 'WORKER': 1}, 'max_workers': 3}}
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        lm = LoadMetrics()
        runner.respond_to_call('json .Config.Env', ['[]' for i in range(3)])
        autoscaler = MockAutoscaler(config_path, lm, MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        autoscaler.update()
        self.waitForNodes(0, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        autoscaler.load_metrics.set_resource_requests([{'CPU': 0.2, 'WORKER': 1.0}])
        autoscaler.update()
        self.waitForNodes(1, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        non_terminated_nodes = autoscaler.provider.non_terminated_nodes({})
        assert len(non_terminated_nodes) == 2
        node_id = non_terminated_nodes[1]
        node_ip = autoscaler.provider.non_terminated_node_ips({})[1]
        autoscaler.provider.mock_nodes[node_id].state = 'unterminatable'
        lm.update(node_ip, mock_raylet_id(), config['available_node_types']['def_worker']['resources'], config['available_node_types']['def_worker']['resources'], waiting_bundles=[{'CPU': 0.2, 'WORKER': 1.0}])
        autoscaler.update()
        self.waitForNodes(1, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        autoscaler.load_metrics.set_resource_requests([{'CPU': 0.2, 'WORKER': 1.0}] * 2)
        autoscaler.update()
        self.waitForNodes(2, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        autoscaler.load_metrics.set_resource_requests([{'CPU': 0.2, 'WORKER': 1.0}])
        lm.update(node_ip, mock_raylet_id(), config['available_node_types']['def_worker']['resources'], {}, waiting_bundles=[{'CPU': 0.2, 'WORKER': 1.0}])
        autoscaler.update()
        self.waitForNodes(2, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        lm.update(node_ip, mock_raylet_id(), config['available_node_types']['def_worker']['resources'], config['available_node_types']['def_worker']['resources'], waiting_bundles=[{'CPU': 0.2, 'WORKER': 1.0}])
        autoscaler.update()
        self.waitForNodes(2, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        assert autoscaler.provider.mock_nodes[node_id].state == 'unterminatable'
        lm.update('172.0.0.2', mock_raylet_id(), config['available_node_types']['def_worker']['resources'], config['available_node_types']['def_worker']['resources'], waiting_bundles=[{'CPU': 0.2, 'WORKER': 1.0}])
        autoscaler.update()
        self.waitForNodes(1, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})

    def testRequestResourcesRaceConditionsLong(self):
        if False:
            print('Hello World!')
        'Test request_resources(), race conditions & demands/min_workers.\n\n        Tests when request_resources() is called simultaneously with resource\n        demands and min_workers constraint in multiple orders upscaling and\n        downscaling.\n        '
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['max_workers'] = 4
        config['idle_timeout_minutes'] = 0
        config['available_node_types'] = {'empty_node': {'node_config': {}, 'resources': {'CPU': 2}, 'max_workers': 1}, 'def_worker': {'node_config': {}, 'resources': {'CPU': 2, 'WORKER': 1}, 'max_workers': 3, 'min_workers': 1}}
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        runner.respond_to_call('json .Config.Env', ['[]' for i in range(3)])
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        lm = LoadMetrics()
        autoscaler = MockAutoscaler(config_path, lm, MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        autoscaler.load_metrics.set_resource_requests([{'CPU': 0.2, 'WORKER': 1.0}])
        autoscaler.update()
        self.waitForNodes(1, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        non_terminated_nodes = autoscaler.provider.non_terminated_nodes({})
        assert len(non_terminated_nodes) == 2
        node_id = non_terminated_nodes[1]
        node_ip = autoscaler.provider.non_terminated_node_ips({})[1]
        autoscaler.provider.mock_nodes[node_id].state = 'unterminatable'
        lm.update(node_ip, mock_raylet_id(), config['available_node_types']['def_worker']['resources'], config['available_node_types']['def_worker']['resources'], waiting_bundles=[{'CPU': 0.2, 'WORKER': 1.0}])
        autoscaler.load_metrics.set_resource_requests([{'CPU': 0.2, 'WORKER': 1.0}] * 2)
        autoscaler.update()
        self.waitForNodes(2, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        autoscaler.load_metrics.set_resource_requests([{'CPU': 0.2, 'WORKER': 1.0}])
        autoscaler.update()
        self.waitForNodes(2, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        autoscaler.load_metrics.set_resource_requests([{'CPU': 0.2, 'WORKER': 1.0}] * 3)
        lm.update(node_ip, mock_raylet_id(), config['available_node_types']['def_worker']['resources'], {}, waiting_bundles=[{'CPU': 0.2, 'WORKER': 1.0}] * 3)
        autoscaler.update()
        self.waitForNodes(3, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        autoscaler.load_metrics.set_resource_requests([])
        lm.update('172.0.0.2', mock_raylet_id(), config['available_node_types']['def_worker']['resources'], config['available_node_types']['def_worker']['resources'])
        lm.update('172.0.0.3', mock_raylet_id(), config['available_node_types']['def_worker']['resources'], config['available_node_types']['def_worker']['resources'])
        lm.update(node_ip, mock_raylet_id(), config['available_node_types']['def_worker']['resources'], {})
        print('============ Should scale down from here =============', node_id)
        autoscaler.update()
        self.waitForNodes(1, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        assert autoscaler.provider.mock_nodes[node_id].state == 'unterminatable'

    def testRequestResourcesRaceConditionWithMinWorker(self):
        if False:
            for i in range(10):
                print('nop')
        'Test request_resources() with min_workers.\n\n        Tests when request_resources() is called simultaneously with adding\n        min_workers constraint.\n        '
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['available_node_types'] = {'empty_node': {'node_config': {}, 'resources': {'CPU': 2}, 'max_workers': 1}, 'def_worker': {'node_config': {}, 'resources': {'CPU': 2, 'WORKER': 1}, 'max_workers': 3, 'min_workers': 1}}
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        runner.respond_to_call('json .Config.Env', ['[]' for i in range(2)])
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        lm = LoadMetrics()
        autoscaler = MockAutoscaler(config_path, lm, MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        autoscaler.load_metrics.set_resource_requests([{'CPU': 2, 'WORKER': 1.0}] * 2)
        autoscaler.update()
        self.waitForNodes(2, tag_filters={TAG_RAY_NODE_KIND: NODE_KIND_WORKER})

    def testRequestResourcesRaceConditionWithResourceDemands(self):
        if False:
            while True:
                i = 10
        'Test request_resources() with resource_demands.\n\n        Tests when request_resources() is called simultaneously with resource\n        demands in multiple orders.\n        '
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config['available_node_types'].update({'empty_node': {'node_config': {}, 'resources': {'CPU': 2, 'GPU': 1}, 'max_workers': 1}, 'def_worker': {'node_config': {}, 'resources': {'CPU': 2, 'GPU': 1, 'WORKER': 1}, 'max_workers': 3}})
        config['idle_timeout_minutes'] = 0
        config_path = self.write_config(config)
        self.provider = MockProvider()
        self.provider.create_node({}, {TAG_RAY_NODE_KIND: 'head', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_USER_NODE_TYPE: 'empty_node'}, 1)
        runner = MockProcessRunner()
        runner.respond_to_call('json .Config.Env', ['[]' for i in range(2)])
        lm = LoadMetrics()
        autoscaler = MockAutoscaler(config_path, lm, MockGcsClient(), max_failures=0, process_runner=runner, update_interval_s=0)
        lm.update('127.0.0.0', mock_raylet_id(), {'CPU': 2, 'GPU': 1}, {'CPU': 2}, waiting_bundles=[{'CPU': 2}])
        autoscaler.load_metrics.set_resource_requests([{'CPU': 2, 'GPU': 1}] * 2)
        autoscaler.update()
        self.waitForNodes(2)
        lm.update('127.0.0.0', mock_raylet_id(), {'CPU': 2, 'GPU': 1}, {'CPU': 2}, waiting_bundles=[{'CPU': 2}])
        for _ in range(10):
            autoscaler.update()
        self.waitForNodes(2)

def format_pg(pg):
    if False:
        while True:
            i = 10
    strategy = pg['strategy']
    bundles = pg['bundles']
    shape_strs = []
    for (bundle, count) in bundles:
        shape_strs.append(f'{bundle} * {count}')
    bundles_str = ', '.join(shape_strs)
    return f'{bundles_str} ({strategy})'

def test_memory_string_formatting():
    if False:
        for i in range(10):
            print('nop')
    assert ray.autoscaler._private.util.format_memory(0) == '0B'
    assert ray.autoscaler._private.util.format_memory(0.0) == '0B', "Bytes aren't decimals"
    assert ray.autoscaler._private.util.format_memory(1) == '1B'
    assert ray.autoscaler._private.util.format_memory(1023) == '1023B'
    assert ray.autoscaler._private.util.format_memory(1024) == '1.00KiB'
    assert ray.autoscaler._private.util.format_memory(1025) == '1.00KiB'
    assert ray.autoscaler._private.util.format_memory(1037) == '1.01KiB'
    assert ray.autoscaler._private.util.format_memory(1200) == '1.17KiB'
    assert ray.autoscaler._private.util.format_memory(2 ** 20 - 10) == '1023.99KiB'
    assert ray.autoscaler._private.util.format_memory(2 ** 20 - 1) == '1024.00KiB'
    assert ray.autoscaler._private.util.format_memory(2 ** 20) == '1.00MiB'
    assert ray.autoscaler._private.util.format_memory(2 ** 30) == '1.00GiB'
    assert ray.autoscaler._private.util.format_memory(5.001 * 2 ** 30) == '5.00GiB'
    assert ray.autoscaler._private.util.format_memory(5.004 * 2 ** 30) == '5.00GiB', 'rounds down'
    assert ray.autoscaler._private.util.format_memory(5.005 * 2 ** 30) == '5.00GiB', 'rounds down'
    assert ray.autoscaler._private.util.format_memory(2 ** 40) == '1.00TiB'

def test_info_string():
    if False:
        print('Hello World!')
    lm_summary = LoadMetricsSummary(usage={'CPU': (530.0, 544.0), 'GPU': (2, 2), 'AcceleratorType:V100': (0, 2), 'memory': (2 * 2 ** 30, 2 ** 33), 'object_store_memory': (3.14 * 2 ** 30, 2 ** 34), 'accelerator_type:T4': (1, 1)}, resource_demand=[({'CPU': 1}, 150)], pg_demand=[({'bundles': [({'CPU': 4}, 5)], 'strategy': 'PACK'}, 420)], request_demand=[({'CPU': 16}, 100)], node_types=[])
    autoscaler_summary = AutoscalerSummary(active_nodes={'p3.2xlarge': 2, 'm4.4xlarge': 20}, pending_nodes=[('1.2.3.4', 'm4.4xlarge', STATUS_WAITING_FOR_SSH), ('1.2.3.5', 'm4.4xlarge', STATUS_WAITING_FOR_SSH)], idle_nodes=[], pending_launches={'m4.4xlarge': 2}, failed_nodes=[('1.2.3.6', 'p3.2xlarge')])
    expected = "\n======== Autoscaler status: 2020-12-28 01:02:03 ========\nNode status\n--------------------------------------------------------\nActive:\n 2 p3.2xlarge\n 20 m4.4xlarge\nIdle:\n (no idle nodes)\nPending:\n m4.4xlarge, 2 launching\n 1.2.3.4: m4.4xlarge, waiting-for-ssh\n 1.2.3.5: m4.4xlarge, waiting-for-ssh\nRecent failures:\n p3.2xlarge: NodeTerminated (ip: 1.2.3.6)\n\nResources\n--------------------------------------------------------\nUsage:\n 0/2 AcceleratorType:V100\n 530.0/544.0 CPU\n 2/2 GPU\n 2.00GiB/8.00GiB memory\n 3.14GiB/16.00GiB object_store_memory\n\nDemands:\n {'CPU': 1}: 150+ pending tasks/actors\n {'CPU': 4} * 5 (PACK): 420+ pending placement groups\n {'CPU': 16}: 100+ from request_resources()\n".strip()
    actual = format_info_string(lm_summary, autoscaler_summary, time=datetime(year=2020, month=12, day=28, hour=1, minute=2, second=3))
    print(actual)
    assert expected == actual

def test_info_string_verbose():
    if False:
        return 10
    lm_summary = LoadMetricsSummary(usage={'CPU': (530.0, 544.0), 'GPU': (2, 2), 'accelerator_type:V100': (1, 2), 'memory': (2 * 2 ** 30, 2 ** 33), 'object_store_memory': (3.14 * 2 ** 30, 2 ** 34)}, resource_demand=[({'CPU': 1}, 150)], pg_demand=[({'bundles': [({'CPU': 4}, 5)], 'strategy': 'PACK'}, 420)], request_demand=[({'CPU': 16}, 100)], node_types=[], usage_by_node={'192.168.1.1': {'CPU': (5.0, 20.0), 'GPU': (0.7, 1), 'accelerator_type:V100': (0.1, 1), 'memory': (2 ** 30, 2 ** 32), 'object_store_memory': (3.14 * 2 ** 30, 2 ** 32)}, '192.168.1.2': {'CPU': (15.0, 20.0), 'GPU': (0.3, 1), 'accelerator_type:V100': (0.9, 1), 'memory': (2 ** 30, 1.5 * 2 ** 33), 'object_store_memory': (0, 2 ** 32)}})
    autoscaler_summary = AutoscalerSummary(active_nodes=[], idle_nodes={'p3.2xlarge': 2, 'm4.4xlarge': 20}, pending_nodes=[('1.2.3.4', 'm4.4xlarge', STATUS_WAITING_FOR_SSH), ('1.2.3.5', 'm4.4xlarge', STATUS_WAITING_FOR_SSH)], pending_launches={'m4.4xlarge': 2}, failed_nodes=[('1.2.3.6', 'p3.2xlarge')], node_activities={'192.168.1.1': ('m4.4xlarge', ['CPU in use.', 'GPU in use.', 'Active workers.']), '192.168.1.2': ('m4.4xlarge', ['GPU in use.', 'Active workers.'])})
    expected = "\n======== Autoscaler status: 2020-12-28 01:02:03 ========\nGCS request time: 3.141500s\nNode Provider non_terminated_nodes time: 1.618000s\n\nNode status\n--------------------------------------------------------\nActive:\n (no active nodes)\nIdle:\n 2 p3.2xlarge\n 20 m4.4xlarge\nPending:\n m4.4xlarge, 2 launching\n 1.2.3.4: m4.4xlarge, waiting-for-ssh\n 1.2.3.5: m4.4xlarge, waiting-for-ssh\nRecent failures:\n p3.2xlarge: NodeTerminated (ip: 1.2.3.6)\n\nResources\n--------------------------------------------------------\nTotal Usage:\n 530.0/544.0 CPU\n 2/2 GPU\n 1/2 accelerator_type:V100\n 2.00GiB/8.00GiB memory\n 3.14GiB/16.00GiB object_store_memory\n\nTotal Demands:\n {'CPU': 1}: 150+ pending tasks/actors\n {'CPU': 4} * 5 (PACK): 420+ pending placement groups\n {'CPU': 16}: 100+ from request_resources()\n\nNode: 192.168.1.1\n Usage:\n  5.0/20.0 CPU\n  0.7/1 GPU\n  0.1/1 accelerator_type:V100\n  1.00GiB/4.00GiB memory\n  3.14GiB/4.00GiB object_store_memory\n Activity:\n  CPU in use.\n  GPU in use.\n  Active workers.\n\nNode: 192.168.1.2\n Usage:\n  15.0/20.0 CPU\n  0.3/1 GPU\n  0.9/1 accelerator_type:V100\n  1.00GiB/12.00GiB memory\n  0B/4.00GiB object_store_memory\n Activity:\n  GPU in use.\n  Active workers.\n".strip()
    actual = format_info_string(lm_summary, autoscaler_summary, time=datetime(year=2020, month=12, day=28, hour=1, minute=2, second=3), gcs_request_time=3.1415, non_terminated_nodes_time=1.618, verbose=True)
    print(actual)
    assert expected == actual

def test_info_string_verbose_node_types():
    if False:
        while True:
            i = 10
    lm_summary = LoadMetricsSummary(usage={'CPU': (530.0, 544.0), 'GPU': (2, 2), 'accelerator_type:V100': (1, 2), 'memory': (2 * 2 ** 30, 2 ** 33), 'object_store_memory': (3.14 * 2 ** 30, 2 ** 34)}, resource_demand=[({'CPU': 1}, 150)], pg_demand=[({'bundles': [({'CPU': 4}, 5)], 'strategy': 'PACK'}, 420)], request_demand=[({'CPU': 16}, 100)], node_types=[], usage_by_node={'192.168.1.1': {'CPU': (5.0, 20.0), 'GPU': (0.7, 1), 'accelerator_type:V100': (0.1, 1), 'memory': (2 ** 30, 2 ** 32), 'object_store_memory': (3.14 * 2 ** 30, 2 ** 32)}, '192.168.1.2': {'CPU': (15.0, 20.0), 'GPU': (0.3, 1), 'accelerator_type:V100': (0.9, 1), 'memory': (2 ** 30, 1.5 * 2 ** 33), 'object_store_memory': (0, 2 ** 32)}})
    autoscaler_summary = AutoscalerSummary(active_nodes={'p3.2xlarge': 2, 'm4.4xlarge': 20}, pending_nodes=[('1.2.3.4', 'm4.4xlarge', STATUS_WAITING_FOR_SSH), ('1.2.3.5', 'm4.4xlarge', STATUS_WAITING_FOR_SSH)], idle_nodes=[], pending_launches={'m4.4xlarge': 2}, failed_nodes=[('1.2.3.6', 'p3.2xlarge')], node_type_mapping={'192.168.1.1': 'head-node', '192.168.1.2': 'gpu-worker'})
    expected = "\n======== Autoscaler status: 2020-12-28 01:02:03 ========\nGCS request time: 3.141500s\nNode Provider non_terminated_nodes time: 1.618000s\nAutoscaler iteration time: 3.141500s\n\nNode status\n--------------------------------------------------------\nActive:\n 2 p3.2xlarge\n 20 m4.4xlarge\nIdle:\n (no idle nodes)\nPending:\n m4.4xlarge, 2 launching\n 1.2.3.4: m4.4xlarge, waiting-for-ssh\n 1.2.3.5: m4.4xlarge, waiting-for-ssh\nRecent failures:\n p3.2xlarge: NodeTerminated (ip: 1.2.3.6)\n\nResources\n--------------------------------------------------------\nTotal Usage:\n 530.0/544.0 CPU\n 2/2 GPU\n 1/2 accelerator_type:V100\n 2.00GiB/8.00GiB memory\n 3.14GiB/16.00GiB object_store_memory\n\nTotal Demands:\n {'CPU': 1}: 150+ pending tasks/actors\n {'CPU': 4} * 5 (PACK): 420+ pending placement groups\n {'CPU': 16}: 100+ from request_resources()\n\nNode: 192.168.1.1 (head-node)\n Usage:\n  5.0/20.0 CPU\n  0.7/1 GPU\n  0.1/1 accelerator_type:V100\n  1.00GiB/4.00GiB memory\n  3.14GiB/4.00GiB object_store_memory\n\nNode: 192.168.1.2 (gpu-worker)\n Usage:\n  15.0/20.0 CPU\n  0.3/1 GPU\n  0.9/1 accelerator_type:V100\n  1.00GiB/12.00GiB memory\n  0B/4.00GiB object_store_memory\n".strip()
    actual = format_info_string(lm_summary, autoscaler_summary, time=datetime(year=2020, month=12, day=28, hour=1, minute=2, second=3), gcs_request_time=3.1415, non_terminated_nodes_time=1.618, autoscaler_update_time=3.1415, verbose=True)
    print(actual)
    assert expected == actual

def test_info_string_verbose_no_breakdown():
    if False:
        print('Hello World!')
    '\n    Test the verbose string but with node reporting feature flagged off.\n    '
    lm_summary = LoadMetricsSummary(usage={'CPU': (530.0, 544.0), 'GPU': (2, 2), 'AcceleratorType:V100': (1, 2), 'memory': (2 * 2 ** 30, 2 ** 33), 'object_store_memory': (3.14 * 2 ** 30, 2 ** 34)}, resource_demand=[({'CPU': 1}, 150)], pg_demand=[({'bundles': [({'CPU': 4}, 5)], 'strategy': 'PACK'}, 420)], request_demand=[({'CPU': 16}, 100)], node_types=[], usage_by_node=None)
    autoscaler_summary = AutoscalerSummary(active_nodes=[], idle_nodes={'p3.2xlarge': 2, 'm4.4xlarge': 20}, pending_nodes=[('1.2.3.4', 'm4.4xlarge', STATUS_WAITING_FOR_SSH), ('1.2.3.5', 'm4.4xlarge', STATUS_WAITING_FOR_SSH)], pending_launches={'m4.4xlarge': 2}, failed_nodes=[('1.2.3.6', 'p3.2xlarge')])
    expected = "\n======== Autoscaler status: 2020-12-28 01:02:03 ========\nGCS request time: 3.141500s\nNode Provider non_terminated_nodes time: 1.618000s\n\nNode status\n--------------------------------------------------------\nActive:\n (no active nodes)\nIdle:\n 2 p3.2xlarge\n 20 m4.4xlarge\nPending:\n m4.4xlarge, 2 launching\n 1.2.3.4: m4.4xlarge, waiting-for-ssh\n 1.2.3.5: m4.4xlarge, waiting-for-ssh\nRecent failures:\n p3.2xlarge: NodeTerminated (ip: 1.2.3.6)\n\nResources\n--------------------------------------------------------\nTotal Usage:\n 1/2 AcceleratorType:V100\n 530.0/544.0 CPU\n 2/2 GPU\n 2.00GiB/8.00GiB memory\n 3.14GiB/16.00GiB object_store_memory\n\nTotal Demands:\n {'CPU': 1}: 150+ pending tasks/actors\n {'CPU': 4} * 5 (PACK): 420+ pending placement groups\n {'CPU': 16}: 100+ from request_resources()\n".strip()
    actual = format_info_string(lm_summary, autoscaler_summary, time=datetime(year=2020, month=12, day=28, hour=1, minute=2, second=3), gcs_request_time=3.1415, non_terminated_nodes_time=1.618, verbose=True)
    print(actual)
    assert expected == actual

def test_info_string_with_launch_failures():
    if False:
        print('Hello World!')
    lm_summary = LoadMetricsSummary(usage={'CPU': (530.0, 544.0), 'GPU': (2, 2), 'AcceleratorType:V100': (0, 2), 'memory': (2 * 2 ** 30, 2 ** 33), 'object_store_memory': (3.14 * 2 ** 30, 2 ** 34)}, resource_demand=[({'CPU': 1}, 150)], pg_demand=[({'bundles': [({'CPU': 4}, 5)], 'strategy': 'PACK'}, 420)], request_demand=[({'CPU': 16}, 100)], node_types=[])
    base_timestamp = datetime(year=2012, month=12, day=21, hour=13, minute=3, second=1).timestamp()
    autoscaler_summary = AutoscalerSummary(active_nodes={'p3.2xlarge': 2, 'm4.4xlarge': 20}, pending_nodes=[('1.2.3.4', 'm4.4xlarge', STATUS_WAITING_FOR_SSH), ('1.2.3.5', 'm4.4xlarge', STATUS_WAITING_FOR_SSH)], idle_nodes=[], pending_launches={'m4.4xlarge': 2}, failed_nodes=[('1.2.3.6', 'p3.2xlarge')], node_availability_summary=NodeAvailabilitySummary(node_availabilities={'A100': NodeAvailabilityRecord(node_type='A100', is_available=False, last_checked_timestamp=base_timestamp + 1, unavailable_node_information=UnavailableNodeInformation(category='InstanceLimitExceeded', description=':)')), 'Inferentia-Spot': NodeAvailabilityRecord(node_type='Inferentia-Spot', is_available=False, last_checked_timestamp=base_timestamp, unavailable_node_information=UnavailableNodeInformation(category='InsufficientInstanceCapacity', description='mo nodes mo problems'))}))
    expected = "\n======== Autoscaler status: 2020-12-28 01:02:03 ========\nNode status\n--------------------------------------------------------\nActive:\n 2 p3.2xlarge\n 20 m4.4xlarge\nIdle:\n (no idle nodes)\nPending:\n m4.4xlarge, 2 launching\n 1.2.3.4: m4.4xlarge, waiting-for-ssh\n 1.2.3.5: m4.4xlarge, waiting-for-ssh\nRecent failures:\n A100: InstanceLimitExceeded (latest_attempt: 13:03:02)\n Inferentia-Spot: InsufficientInstanceCapacity (latest_attempt: 13:03:01)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.6)\n\nResources\n--------------------------------------------------------\nUsage:\n 0/2 AcceleratorType:V100\n 530.0/544.0 CPU\n 2/2 GPU\n 2.00GiB/8.00GiB memory\n 3.14GiB/16.00GiB object_store_memory\n\nDemands:\n {'CPU': 1}: 150+ pending tasks/actors\n {'CPU': 4} * 5 (PACK): 420+ pending placement groups\n {'CPU': 16}: 100+ from request_resources()\n".strip()
    actual = format_info_string(lm_summary, autoscaler_summary, time=datetime(year=2020, month=12, day=28, hour=1, minute=2, second=3))
    print(actual)
    assert expected == actual

def test_info_string_with_launch_failures_verbose():
    if False:
        i = 10
        return i + 15
    lm_summary = LoadMetricsSummary(usage={'CPU': (530.0, 544.0), 'GPU': (2, 2), 'AcceleratorType:V100': (0, 2), 'memory': (2 * 2 ** 30, 2 ** 33), 'object_store_memory': (3.14 * 2 ** 30, 2 ** 34)}, resource_demand=[({'CPU': 1}, 150)], pg_demand=[({'bundles': [({'CPU': 4}, 5)], 'strategy': 'PACK'}, 420)], request_demand=[({'CPU': 16}, 100)], node_types=[])
    base_timestamp = datetime(year=2012, month=12, day=21, hour=13, minute=3, second=1).timestamp()
    autoscaler_summary = AutoscalerSummary(active_nodes=[], idle_nodes={'p3.2xlarge': 2, 'm4.4xlarge': 20}, pending_nodes=[('1.2.3.4', 'm4.4xlarge', STATUS_WAITING_FOR_SSH), ('1.2.3.5', 'm4.4xlarge', STATUS_WAITING_FOR_SSH)], pending_launches={'m4.4xlarge': 2}, failed_nodes=[('1.2.3.6', 'p3.2xlarge')], node_availability_summary=NodeAvailabilitySummary(node_availabilities={'A100': NodeAvailabilityRecord(node_type='A100', is_available=False, last_checked_timestamp=base_timestamp + 1, unavailable_node_information=UnavailableNodeInformation(category='InstanceLimitExceeded', description='you should fix it')), 'Inferentia-Spot': NodeAvailabilityRecord(node_type='Inferentia-Spot', is_available=False, last_checked_timestamp=base_timestamp, unavailable_node_information=UnavailableNodeInformation(category='InsufficientInstanceCapacity', description='desc'))}))
    expected = "\n======== Autoscaler status: 2020-12-28 01:02:03 ========\n\nNode status\n--------------------------------------------------------\nActive:\n (no active nodes)\nIdle:\n 2 p3.2xlarge\n 20 m4.4xlarge\nPending:\n m4.4xlarge, 2 launching\n 1.2.3.4: m4.4xlarge, waiting-for-ssh\n 1.2.3.5: m4.4xlarge, waiting-for-ssh\nRecent failures:\n A100: InstanceLimitExceeded (latest_attempt: 13:03:02) - you should fix it\n Inferentia-Spot: InsufficientInstanceCapacity (latest_attempt: 13:03:01) - desc\n p3.2xlarge: NodeTerminated (ip: 1.2.3.6)\n\nResources\n--------------------------------------------------------\nTotal Usage:\n 0/2 AcceleratorType:V100\n 530.0/544.0 CPU\n 2/2 GPU\n 2.00GiB/8.00GiB memory\n 3.14GiB/16.00GiB object_store_memory\n\nTotal Demands:\n {'CPU': 1}: 150+ pending tasks/actors\n {'CPU': 4} * 5 (PACK): 420+ pending placement groups\n {'CPU': 16}: 100+ from request_resources()\n".strip()
    actual = format_info_string(lm_summary, autoscaler_summary, time=datetime(year=2020, month=12, day=28, hour=1, minute=2, second=3), verbose=True)
    print(actual)
    assert expected == actual

def test_info_string_failed_node_cap():
    if False:
        for i in range(10):
            print('nop')
    lm_summary = LoadMetricsSummary(usage={'CPU': (530.0, 544.0), 'GPU': (2, 2), 'AcceleratorType:V100': (0, 2), 'memory': (2 * 2 ** 30, 2 ** 33), 'object_store_memory': (3.14 * 2 ** 30, 2 ** 34), 'CPU_group_4a82a217aadd8326a3a49f02700ac5c2': (2.0, 2.0)}, resource_demand=[({'CPU': 2.0}, 150), ({'CPU_group_4a82a217aadd8326a3a49f02700ac5c2': 2.0}, 3), ({'GPU_group_0_4a82a2add8326a3a49f02700ac5c2': 0.5}, 100)], pg_demand=[({'bundles': [({'CPU': 4}, 5)], 'strategy': 'PACK'}, 420)], request_demand=[({'CPU': 16}, 100)], node_types=[])
    autoscaler_summary = AutoscalerSummary(active_nodes=[], idle_nodes={'p3.2xlarge': 2, 'm4.4xlarge': 20}, pending_nodes=[('1.2.3.4', 'm4.4xlarge', STATUS_WAITING_FOR_SSH), ('1.2.3.5', 'm4.4xlarge', STATUS_WAITING_FOR_SSH)], pending_launches={'m4.4xlarge': 2}, failed_nodes=[(f'1.2.3.{i}', 'p3.2xlarge') for i in range(100)])
    expected = "\n======== Autoscaler status: 2020-12-28 01:02:03 ========\nNode status\n--------------------------------------------------------\nActive:\n (no active nodes)\nIdle:\n 2 p3.2xlarge\n 20 m4.4xlarge\nPending:\n m4.4xlarge, 2 launching\n 1.2.3.4: m4.4xlarge, waiting-for-ssh\n 1.2.3.5: m4.4xlarge, waiting-for-ssh\nRecent failures:\n p3.2xlarge: NodeTerminated (ip: 1.2.3.99)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.98)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.97)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.96)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.95)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.94)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.93)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.92)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.91)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.90)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.89)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.88)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.87)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.86)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.85)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.84)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.83)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.82)\n p3.2xlarge: NodeTerminated (ip: 1.2.3.81)\n\nResources\n--------------------------------------------------------\nUsage:\n 0/2 AcceleratorType:V100\n 530.0/544.0 CPU (2.0 used of 2.0 reserved in placement groups)\n 2/2 GPU\n 2.00GiB/8.00GiB memory\n 3.14GiB/16.00GiB object_store_memory\n\nDemands:\n {'CPU': 2.0}: 153+ pending tasks/actors (3+ using placement groups)\n {'GPU': 0.5}: 100+ pending tasks/actors (100+ using placement groups)\n {'CPU': 4} * 5 (PACK): 420+ pending placement groups\n {'CPU': 16}: 100+ from request_resources()\n"
    actual = format_info_string(lm_summary, autoscaler_summary, time=datetime(year=2020, month=12, day=28, hour=1, minute=2, second=3))
    print(actual)
    assert expected.strip() == actual

def test_placement_group_match_string():
    if False:
        print('Hello World!')
    assert is_placement_group_resource('bundle_group_ffe7d420752c6e8658638d19ecf2b68c') is True
    assert is_placement_group_resource('CPU_group_0_625ace126f848864c46f50dced5e0ef7') is True
    assert is_placement_group_resource('CPU_group_625ace126f848864c46f50dced5e0ef7') is True
    assert is_placement_group_resource('CPU') is False
    assert is_placement_group_resource('GPU') is False
    assert is_placement_group_resource('custom_resource') is False
    assert is_placement_group_resource('ip:192.168.1.1') is False
    provider = MockProvider()
    new_types = copy.deepcopy(TYPES_A)
    scheduler = ResourceDemandScheduler(provider, new_types, 3, head_node_type='p2.8xlarge', upscaling_speed=1)
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_NODE_KIND: NODE_KIND_HEAD}, 1)
    nodes = provider.non_terminated_nodes({})
    ips = provider.non_terminated_node_ips({})
    utilizations = {ip: {'GPU': 8} for ip in ips}
    with mock.patch('ray.autoscaler._private.resource_demand_scheduler.logger') as logger_mock:
        (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, {}, [{'CPU_group_0_625ace126f848864c46f50dced5e0ef7': 8}], utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
        logger_mock.warning.assert_not_called()
    assert to_launch == {}
    assert rem == [{'CPU_group_0_625ace126f848864c46f50dced5e0ef7': 8}]
    with mock.patch('ray.autoscaler._private.resource_demand_scheduler.logger') as logger_mock:
        (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, {}, [{'non-existent-custom': 8}], utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
        logger_mock.warning.assert_called()
    assert to_launch == {}
    assert rem == [{'non-existent-custom': 8}]

def _launch_nothing_utilization_scorer_plugin(node_resources, resources, node_type, *, node_availability_summary):
    if False:
        for i in range(10):
            print('nop')
    assert node_availability_summary is not None
    return None

@pytest.fixture
def launch_nothing_utilization_score_plugin():
    if False:
        print('Hello World!')
    os.environ[AUTOSCALER_UTILIZATION_SCORER_KEY] = 'ray.tests.test_resource_demand_scheduler._launch_nothing_utilization_scorer_plugin'
    try:
        yield None
    finally:
        del os.environ[AUTOSCALER_UTILIZATION_SCORER_KEY]

def test_utilization_score_plugin_1(launch_nothing_utilization_score_plugin):
    if False:
        i = 10
        return i + 15
    assert launch_nothing_utilization_score_plugin is None, 'Keep mypy happy.'
    provider = MockProvider()
    new_types = copy.deepcopy(TYPES_A)
    scheduler = ResourceDemandScheduler(provider, new_types, 3, head_node_type='p2.8xlarge', upscaling_speed=1)
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_NODE_KIND: NODE_KIND_HEAD}, 1)
    nodes = provider.non_terminated_nodes({})
    ips = provider.non_terminated_node_ips({})
    utilizations = {ip: {'GPU': 8} for ip in ips}
    (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, {}, [{'GPU': 8}] * 2, utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {}

def _lexical_scorer_plugin(node_resources, resources, node_type, *, node_availability_summary):
    if False:
        return 10
    assert node_availability_summary is not None
    if _resource_based_utilization_scorer(node_resources, resources, node_availability_summary=node_availability_summary) is not None:
        return node_type
    else:
        return None

@pytest.fixture
def lexical_score_plugin():
    if False:
        print('Hello World!')
    os.environ[AUTOSCALER_UTILIZATION_SCORER_KEY] = 'ray.tests.test_resource_demand_scheduler._lexical_scorer_plugin'
    try:
        yield None
    finally:
        del os.environ[AUTOSCALER_UTILIZATION_SCORER_KEY]

def test_utilization_score_plugin_2(lexical_score_plugin):
    if False:
        i = 10
        return i + 15
    assert lexical_score_plugin is None, 'Keep mypy happy.'
    provider = MockProvider()
    new_types = copy.deepcopy(TYPES_A)
    new_types['z2.8xlarge'] = new_types['p2.8xlarge']
    scheduler = ResourceDemandScheduler(provider, new_types, 3, head_node_type='p2.8xlarge', upscaling_speed=1)
    provider.create_node({}, {TAG_RAY_USER_NODE_TYPE: 'p2.8xlarge', TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE, TAG_RAY_NODE_KIND: NODE_KIND_HEAD}, 1)
    nodes = provider.non_terminated_nodes({})
    ips = provider.non_terminated_node_ips({})
    utilizations = {ip: {'GPU': 8} for ip in ips}
    (to_launch, rem) = scheduler.get_nodes_to_launch(nodes, {}, [{'GPU': 8}] * 2, utilizations, [], {}, [], EMPTY_AVAILABILITY_SUMMARY)
    assert to_launch == {'z2.8xlarge': 1}
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))