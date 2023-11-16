"""Implements multi-node-type autoscaling.

This file implements an autoscaling algorithm that is aware of multiple node
types (e.g., example-multi-node-type.yaml). The Ray autoscaler will pass in
a vector of resource shape demands, and the resource demand scheduler will
return a list of node types that can satisfy the demands given constraints
(i.e., reverse bin packing).
"""
import collections
import copy
import logging
import os
from abc import abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import AUTOSCALER_CONSERVE_GPU_NODES, AUTOSCALER_UTILIZATION_SCORER_KEY
from ray.autoscaler._private.loader import load_function_or_class
from ray.autoscaler._private.node_provider_availability_tracker import NodeAvailabilitySummary
from ray.autoscaler._private.util import NodeID, NodeIP, NodeType, NodeTypeConfigDict, ResourceDict, is_placement_group_resource
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import NODE_KIND_HEAD, NODE_KIND_UNMANAGED, NODE_KIND_WORKER, TAG_RAY_NODE_KIND, TAG_RAY_USER_NODE_TYPE
from ray.core.generated.common_pb2 import PlacementStrategy
logger = logging.getLogger(__name__)
UPSCALING_INITIAL_NUM_NODES = 5
NodeResources = ResourceDict
ResourceDemands = List[ResourceDict]

class UtilizationScore:
    """This fancy class just defines the `UtilizationScore` protocol to be
    some type that is a "totally ordered set" (i.e. things that can be sorted).

    What we're really trying to express is

    ```
    UtilizationScore = TypeVar("UtilizationScore", bound=Comparable["UtilizationScore"])
    ```

    but Comparable isn't a real type and, and a bound with a type argument
    can't be enforced (f-bounded polymorphism with contravariance). See Guido's
    comment for more details: https://github.com/python/typing/issues/59.

    This isn't just a `float`. In the case of the default scorer, it's a
    `Tuple[float, float]` which is quite difficult to map to a single number.

    """

    @abstractmethod
    def __eq__(self, other: 'UtilizationScore') -> bool:
        if False:
            return 10
        pass

    @abstractmethod
    def __lt__(self: 'UtilizationScore', other: 'UtilizationScore') -> bool:
        if False:
            for i in range(10):
                print('nop')
        pass

    def __gt__(self: 'UtilizationScore', other: 'UtilizationScore') -> bool:
        if False:
            print('Hello World!')
        return not self < other and self != other

    def __le__(self: 'UtilizationScore', other: 'UtilizationScore') -> bool:
        if False:
            i = 10
            return i + 15
        return self < other or self == other

    def __ge__(self: 'UtilizationScore', other: 'UtilizationScore') -> bool:
        if False:
            print('Hello World!')
        return not self < other

class UtilizationScorer:

    def __call__(node_resources: NodeResources, resource_demands: ResourceDemands, *, node_availability_summary: NodeAvailabilitySummary) -> Optional[UtilizationScore]:
        if False:
            print('Hello World!')
        pass

class ResourceDemandScheduler:

    def __init__(self, provider: NodeProvider, node_types: Dict[NodeType, NodeTypeConfigDict], max_workers: int, head_node_type: NodeType, upscaling_speed: float) -> None:
        if False:
            print('Hello World!')
        self.provider = provider
        self.node_types = copy.deepcopy(node_types)
        self.node_resource_updated = set()
        self.max_workers = max_workers
        self.head_node_type = head_node_type
        self.upscaling_speed = upscaling_speed
        utilization_scorer_str = os.environ.get(AUTOSCALER_UTILIZATION_SCORER_KEY, 'ray.autoscaler._private.resource_demand_scheduler._default_utilization_scorer')
        self.utilization_scorer: UtilizationScorer = load_function_or_class(utilization_scorer_str)

    def _get_head_and_workers(self, nodes: List[NodeID]) -> Tuple[NodeID, List[NodeID]]:
        if False:
            return 10
        "Returns the head node's id and the list of all worker node ids,\n        given a list `nodes` of all node ids in the cluster.\n        "
        (head_id, worker_ids) = (None, [])
        for node in nodes:
            tags = self.provider.node_tags(node)
            if tags[TAG_RAY_NODE_KIND] == NODE_KIND_HEAD:
                head_id = node
            elif tags[TAG_RAY_NODE_KIND] == NODE_KIND_WORKER:
                worker_ids.append(node)
        return (head_id, worker_ids)

    def reset_config(self, provider: NodeProvider, node_types: Dict[NodeType, NodeTypeConfigDict], max_workers: int, head_node_type: NodeType, upscaling_speed: float=1) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the class state variables.\n\n        For legacy yamls, it merges previous state and new state to make sure\n        inferered resources are not lost.\n        '
        self.provider = provider
        self.node_types = copy.deepcopy(node_types)
        self.node_resource_updated = set()
        self.max_workers = max_workers
        self.head_node_type = head_node_type
        self.upscaling_speed = upscaling_speed

    def is_feasible(self, bundle: ResourceDict) -> bool:
        if False:
            while True:
                i = 10
        for (node_type, config) in self.node_types.items():
            max_of_type = config.get('max_workers', 0)
            node_resources = config['resources']
            if (node_type == self.head_node_type or max_of_type > 0) and _fits(node_resources, bundle):
                return True
        return False

    def get_nodes_to_launch(self, nodes: List[NodeID], launching_nodes: Dict[NodeType, int], resource_demands: List[ResourceDict], unused_resources_by_ip: Dict[NodeIP, ResourceDict], pending_placement_groups: List[PlacementGroupTableData], max_resources_by_ip: Dict[NodeIP, ResourceDict], ensure_min_cluster_size: List[ResourceDict], node_availability_summary: NodeAvailabilitySummary) -> (Dict[NodeType, int], List[ResourceDict]):
        if False:
            i = 10
            return i + 15
        "Given resource demands, return node types to add to the cluster.\n\n        This method:\n            (1) calculates the resources present in the cluster.\n            (2) calculates the remaining nodes to add to respect min_workers\n                constraint per node type.\n            (3) for each strict spread placement group, reserve space on\n                available nodes and launch new nodes if necessary.\n            (4) calculates the unfulfilled resource bundles.\n            (5) calculates which nodes need to be launched to fulfill all\n                the bundle requests, subject to max_worker constraints.\n\n        Args:\n            nodes: List of existing nodes in the cluster.\n            launching_nodes: Summary of node types currently being launched.\n            resource_demands: Vector of resource demands from the scheduler.\n            unused_resources_by_ip: Mapping from ip to available resources.\n            pending_placement_groups: Placement group demands.\n            max_resources_by_ip: Mapping from ip to static node resources.\n            ensure_min_cluster_size: Try to ensure the cluster can fit at least\n                this set of resources. This differs from resources_demands in\n                that we don't take into account existing usage.\n\n            node_availability_summary: A snapshot of the current\n                NodeAvailabilitySummary.\n\n        Returns:\n            Dict of count to add for each node type, and residual of resources\n            that still cannot be fulfilled.\n        "
        utilization_scorer = partial(self.utilization_scorer, node_availability_summary=node_availability_summary)
        self._update_node_resources_from_runtime(nodes, max_resources_by_ip)
        node_resources: List[ResourceDict]
        node_type_counts: Dict[NodeType, int]
        (node_resources, node_type_counts) = self.calculate_node_resources(nodes, launching_nodes, unused_resources_by_ip)
        logger.debug('Cluster resources: {}'.format(node_resources))
        logger.debug('Node counts: {}'.format(node_type_counts))
        (node_resources, node_type_counts, adjusted_min_workers) = _add_min_workers_nodes(node_resources, node_type_counts, self.node_types, self.max_workers, self.head_node_type, ensure_min_cluster_size, utilization_scorer=utilization_scorer)
        logger.debug(f'Placement group demands: {pending_placement_groups}')
        (placement_group_demand_vector, strict_spreads) = placement_groups_to_resource_demands(pending_placement_groups)
        resource_demands = placement_group_demand_vector + resource_demands
        (spread_pg_nodes_to_add, node_resources, node_type_counts) = self.reserve_and_allocate_spread(strict_spreads, node_resources, node_type_counts, utilization_scorer)
        (unfulfilled_placement_groups_demands, _) = get_bin_pack_residual(node_resources, placement_group_demand_vector)
        max_to_add = self.max_workers + 1 - sum(node_type_counts.values())
        (pg_demands_nodes_max_launch_limit, _) = get_nodes_for(self.node_types, node_type_counts, self.head_node_type, max_to_add, unfulfilled_placement_groups_demands, utilization_scorer=utilization_scorer)
        placement_groups_nodes_max_limit = {node_type: spread_pg_nodes_to_add.get(node_type, 0) + pg_demands_nodes_max_launch_limit.get(node_type, 0) for node_type in self.node_types}
        (unfulfilled, _) = get_bin_pack_residual(node_resources, resource_demands)
        logger.debug('Resource demands: {}'.format(resource_demands))
        logger.debug('Unfulfilled demands: {}'.format(unfulfilled))
        (nodes_to_add_based_on_demand, final_unfulfilled) = get_nodes_for(self.node_types, node_type_counts, self.head_node_type, max_to_add, unfulfilled, utilization_scorer=utilization_scorer)
        logger.debug('Final unfulfilled: {}'.format(final_unfulfilled))
        total_nodes_to_add = {}
        for node_type in self.node_types:
            nodes_to_add = adjusted_min_workers.get(node_type, 0) + spread_pg_nodes_to_add.get(node_type, 0) + nodes_to_add_based_on_demand.get(node_type, 0)
            if nodes_to_add > 0:
                total_nodes_to_add[node_type] = nodes_to_add
        total_nodes_to_add = self._get_concurrent_resource_demand_to_launch(total_nodes_to_add, unused_resources_by_ip.keys(), nodes, launching_nodes, adjusted_min_workers, placement_groups_nodes_max_limit)
        logger.debug('Node requests: {}'.format(total_nodes_to_add))
        return (total_nodes_to_add, final_unfulfilled)

    def _update_node_resources_from_runtime(self, nodes: List[NodeID], max_resources_by_ip: Dict[NodeIP, ResourceDict]):
        if False:
            for i in range(10):
                print('nop')
        'Update static node type resources with runtime resources\n\n        This will update the cached static node type resources with the runtime\n        resources. Because we can not know the exact autofilled memory or\n        object_store_memory from config file.\n        '
        need_update = len(self.node_types) != len(self.node_resource_updated)
        if not need_update:
            return
        for node_id in nodes:
            tags = self.provider.node_tags(node_id)
            if TAG_RAY_USER_NODE_TYPE not in tags:
                continue
            node_type = tags[TAG_RAY_USER_NODE_TYPE]
            if node_type in self.node_resource_updated or node_type not in self.node_types:
                continue
            ip = self.provider.internal_ip(node_id)
            runtime_resources = max_resources_by_ip.get(ip)
            if runtime_resources:
                runtime_resources = copy.deepcopy(runtime_resources)
                resources = self.node_types[node_type].get('resources', {})
                for key in ['CPU', 'GPU', 'memory', 'object_store_memory']:
                    if key in runtime_resources:
                        resources[key] = runtime_resources[key]
                self.node_types[node_type]['resources'] = resources
                node_kind = tags[TAG_RAY_NODE_KIND]
                if node_kind == NODE_KIND_WORKER:
                    self.node_resource_updated.add(node_type)

    def _get_concurrent_resource_demand_to_launch(self, to_launch: Dict[NodeType, int], connected_nodes: List[NodeIP], non_terminated_nodes: List[NodeID], pending_launches_nodes: Dict[NodeType, int], adjusted_min_workers: Dict[NodeType, int], placement_group_nodes: Dict[NodeType, int]) -> Dict[NodeType, int]:
        if False:
            while True:
                i = 10
        'Updates the max concurrent resources to launch for each node type.\n\n        Given the current nodes that should be launched, the non terminated\n        nodes (running and pending) and the pending to be launched nodes. This\n        method calculates the maximum number of nodes to launch concurrently\n        for each node type as follows:\n            1) Calculates the running nodes.\n            2) Calculates the pending nodes and gets the launching nodes.\n            3) Limits the total number of pending + currently-launching +\n               to-be-launched nodes to:\n                   max(\n                       5,\n                       self.upscaling_speed * max(running_nodes[node_type], 1)\n                   ).\n\n        Args:\n            to_launch: List of number of nodes to launch based on resource\n                demand for every node type.\n            connected_nodes: Running nodes (from LoadMetrics).\n            non_terminated_nodes: Non terminated nodes (pending/running).\n            pending_launches_nodes: Nodes that are in the launch queue.\n            adjusted_min_workers: Nodes to launch to satisfy\n                min_workers and request_resources(). This overrides the launch\n                limits since the user is hinting to immediately scale up to\n                this size.\n            placement_group_nodes: Nodes to launch for placement groups.\n                This overrides the launch concurrency limits.\n        Returns:\n            Dict[NodeType, int]: Maximum number of nodes to launch for each\n                node type.\n        '
        updated_nodes_to_launch = {}
        (running_nodes, pending_nodes) = self._separate_running_and_pending_nodes(non_terminated_nodes, connected_nodes)
        for node_type in to_launch:
            max_allowed_pending_nodes = max(UPSCALING_INITIAL_NUM_NODES, int(self.upscaling_speed * max(running_nodes[node_type], 1)))
            total_pending_nodes = pending_launches_nodes.get(node_type, 0) + pending_nodes[node_type]
            upper_bound = max(max_allowed_pending_nodes - total_pending_nodes, adjusted_min_workers.get(node_type, 0) + placement_group_nodes.get(node_type, 0))
            if upper_bound > 0:
                updated_nodes_to_launch[node_type] = min(upper_bound, to_launch[node_type])
        return updated_nodes_to_launch

    def _separate_running_and_pending_nodes(self, non_terminated_nodes: List[NodeID], connected_nodes: List[NodeIP]) -> (Dict[NodeType, int], Dict[NodeType, int]):
        if False:
            return 10
        'Splits connected and non terminated nodes to pending & running.'
        running_nodes = collections.defaultdict(int)
        pending_nodes = collections.defaultdict(int)
        for node_id in non_terminated_nodes:
            tags = self.provider.node_tags(node_id)
            if TAG_RAY_USER_NODE_TYPE in tags:
                node_type = tags[TAG_RAY_USER_NODE_TYPE]
                node_ip = self.provider.internal_ip(node_id)
                if node_ip in connected_nodes:
                    running_nodes[node_type] += 1
                else:
                    pending_nodes[node_type] += 1
        return (running_nodes, pending_nodes)

    def calculate_node_resources(self, nodes: List[NodeID], pending_nodes: Dict[NodeID, int], unused_resources_by_ip: Dict[str, ResourceDict]) -> (List[ResourceDict], Dict[NodeType, int]):
        if False:
            for i in range(10):
                print('nop')
        'Returns node resource list and node type counts.\n\n        Counts the running nodes, pending nodes.\n        Args:\n             nodes: Existing nodes.\n             pending_nodes: Pending nodes.\n        Returns:\n             node_resources: a list of running + pending resources.\n                 E.g., [{"CPU": 4}, {"GPU": 2}].\n             node_type_counts: running + pending workers per node type.\n        '
        node_resources = []
        node_type_counts = collections.defaultdict(int)

        def add_node(node_type, available_resources=None):
            if False:
                print('Hello World!')
            if node_type not in self.node_types:
                logger.error(f'''Missing entry for node_type {node_type} in cluster config: {self.node_types} under entry available_node_types. This node's resources will be ignored. If you are using an unmanaged node, manually set the {TAG_RAY_NODE_KIND} tag to "{NODE_KIND_UNMANAGED}" in your cloud provider's management console.''')
                return None
            available = copy.deepcopy(self.node_types[node_type]['resources'])
            if available_resources is not None:
                available = copy.deepcopy(available_resources)
            node_resources.append(available)
            node_type_counts[node_type] += 1
        for node_id in nodes:
            tags = self.provider.node_tags(node_id)
            if TAG_RAY_USER_NODE_TYPE in tags:
                node_type = tags[TAG_RAY_USER_NODE_TYPE]
                ip = self.provider.internal_ip(node_id)
                available_resources = unused_resources_by_ip.get(ip)
                add_node(node_type, available_resources)
        for (node_type, count) in pending_nodes.items():
            for _ in range(count):
                add_node(node_type)
        return (node_resources, node_type_counts)

    def reserve_and_allocate_spread(self, strict_spreads: List[List[ResourceDict]], node_resources: List[ResourceDict], node_type_counts: Dict[NodeType, int], utilization_scorer: Callable[[NodeResources, ResourceDemands], Optional[UtilizationScore]]):
        if False:
            for i in range(10):
                print('nop')
        'For each strict spread, attempt to reserve as much space as possible\n        on the node, then allocate new nodes for the unfulfilled portion.\n\n        Args:\n            strict_spreads (List[List[ResourceDict]]): A list of placement\n                groups which must be spread out.\n            node_resources (List[ResourceDict]): Available node resources in\n                the cluster.\n            node_type_counts (Dict[NodeType, int]): The amount of each type of\n                node pending or in the cluster.\n            utilization_scorer: A function that, given a node\n                type, its resources, and resource demands, returns what its\n                utilization would be.\n\n        Returns:\n            Dict[NodeType, int]: Nodes to add.\n            List[ResourceDict]: The updated node_resources after the method.\n            Dict[NodeType, int]: The updated node_type_counts.\n\n        '
        to_add = collections.defaultdict(int)
        for bundles in strict_spreads:
            (unfulfilled, node_resources) = get_bin_pack_residual(node_resources, bundles, strict_spread=True)
            max_to_add = self.max_workers + 1 - sum(node_type_counts.values())
            (to_launch, _) = get_nodes_for(self.node_types, node_type_counts, self.head_node_type, max_to_add, unfulfilled, utilization_scorer=utilization_scorer, strict_spread=True)
            _inplace_add(node_type_counts, to_launch)
            _inplace_add(to_add, to_launch)
            new_node_resources = _node_type_counts_to_node_resources(self.node_types, to_launch)
            (unfulfilled, including_reserved) = get_bin_pack_residual(new_node_resources, unfulfilled, strict_spread=True)
            assert not unfulfilled
            node_resources += including_reserved
        return (to_add, node_resources, node_type_counts)

    def debug_string(self, nodes: List[NodeID], pending_nodes: Dict[NodeID, int], unused_resources_by_ip: Dict[str, ResourceDict]) -> str:
        if False:
            return 10
        (node_resources, node_type_counts) = self.calculate_node_resources(nodes, pending_nodes, unused_resources_by_ip)
        out = 'Worker node types:'
        for (node_type, count) in node_type_counts.items():
            out += '\n - {}: {}'.format(node_type, count)
            if pending_nodes.get(node_type):
                out += ' ({} pending)'.format(pending_nodes[node_type])
        return out

def _node_type_counts_to_node_resources(node_types: Dict[NodeType, NodeTypeConfigDict], node_type_counts: Dict[NodeType, int]) -> List[ResourceDict]:
    if False:
        return 10
    'Converts a node_type_counts dict into a list of node_resources.'
    resources = []
    for (node_type, count) in node_type_counts.items():
        resources += [node_types[node_type]['resources'].copy() for _ in range(count)]
    return resources

def _add_min_workers_nodes(node_resources: List[ResourceDict], node_type_counts: Dict[NodeType, int], node_types: Dict[NodeType, NodeTypeConfigDict], max_workers: int, head_node_type: NodeType, ensure_min_cluster_size: List[ResourceDict], utilization_scorer: Callable[[NodeResources, ResourceDemands, str], Optional[UtilizationScore]]) -> (List[ResourceDict], Dict[NodeType, int], Dict[NodeType, int]):
    if False:
        return 10
    'Updates resource demands to respect the min_workers and\n    request_resources() constraints.\n\n    Args:\n        node_resources: Resources of exisiting nodes already launched/pending.\n        node_type_counts: Counts of existing nodes already launched/pending.\n        node_types: Node types config.\n        max_workers: global max_workers constaint.\n        ensure_min_cluster_size: resource demands from request_resources().\n        utilization_scorer: A function that, given a node\n            type, its resources, and resource demands, returns what its\n            utilization would be.\n\n    Returns:\n        node_resources: The updated node resources after adding min_workers\n            and request_resources() constraints per node type.\n        node_type_counts: The updated node counts after adding min_workers\n            and request_resources() constraints per node type.\n        total_nodes_to_add_dict: The nodes to add to respect min_workers and\n            request_resources() constraints.\n    '
    total_nodes_to_add_dict = {}
    for (node_type, config) in node_types.items():
        existing = node_type_counts.get(node_type, 0)
        target = min(config.get('min_workers', 0), config.get('max_workers', 0))
        if node_type == head_node_type:
            target = target + 1
        if existing < target:
            total_nodes_to_add_dict[node_type] = target - existing
            node_type_counts[node_type] = target
            node_resources.extend([copy.deepcopy(node_types[node_type]['resources']) for _ in range(total_nodes_to_add_dict[node_type])])
    if ensure_min_cluster_size:
        max_to_add = max_workers + 1 - sum(node_type_counts.values())
        max_node_resources = []
        for node_type in node_type_counts:
            max_node_resources.extend([copy.deepcopy(node_types[node_type]['resources']) for _ in range(node_type_counts[node_type])])
        (resource_requests_unfulfilled, _) = get_bin_pack_residual(max_node_resources, ensure_min_cluster_size)
        (nodes_to_add_request_resources, _) = get_nodes_for(node_types, node_type_counts, head_node_type, max_to_add, resource_requests_unfulfilled, utilization_scorer=utilization_scorer)
        for node_type in nodes_to_add_request_resources:
            nodes_to_add = nodes_to_add_request_resources.get(node_type, 0)
            if nodes_to_add > 0:
                node_type_counts[node_type] = nodes_to_add + node_type_counts.get(node_type, 0)
                node_resources.extend([copy.deepcopy(node_types[node_type]['resources']) for _ in range(nodes_to_add)])
                total_nodes_to_add_dict[node_type] = nodes_to_add + total_nodes_to_add_dict.get(node_type, 0)
    return (node_resources, node_type_counts, total_nodes_to_add_dict)

def get_nodes_for(node_types: Dict[NodeType, NodeTypeConfigDict], existing_nodes: Dict[NodeType, int], head_node_type: NodeType, max_to_add: int, resources: List[ResourceDict], utilization_scorer: Callable[[NodeResources, ResourceDemands, str], Optional[UtilizationScore]], strict_spread: bool=False) -> (Dict[NodeType, int], List[ResourceDict]):
    if False:
        while True:
            i = 10
    'Determine nodes to add given resource demands and constraints.\n\n    Args:\n        node_types: node types config.\n        existing_nodes: counts of existing nodes already launched.\n            This sets constraints on the number of new nodes to add.\n        max_to_add: global constraint on nodes to add.\n        resources: resource demands to fulfill.\n        strict_spread: If true, each element in `resources` must be placed on a\n            different node.\n        utilization_scorer: A function that, given a node\n            type, its resources, and resource demands, returns what its\n            utilization would be.\n\n    Returns:\n        Dict of count to add for each node type, and residual of resources\n        that still cannot be fulfilled.\n    '
    nodes_to_add: Dict[NodeType, int] = collections.defaultdict(int)
    while resources and sum(nodes_to_add.values()) < max_to_add:
        utilization_scores = []
        for node_type in node_types:
            max_workers_of_node_type = node_types[node_type].get('max_workers', 0)
            if head_node_type == node_type:
                max_workers_of_node_type = max_workers_of_node_type + 1
            if existing_nodes.get(node_type, 0) + nodes_to_add.get(node_type, 0) >= max_workers_of_node_type:
                continue
            node_resources = node_types[node_type]['resources']
            if strict_spread:
                score = utilization_scorer(node_resources, [resources[0]], node_type)
            else:
                score = utilization_scorer(node_resources, resources, node_type)
            if score is not None:
                utilization_scores.append((score, node_type))
        if not utilization_scores:
            if not any((is_placement_group_resource(resource) for resources_dict in resources for resource in resources_dict)):
                logger.warning(f'The autoscaler could not find a node type to satisfy the request: {resources}. Please specify a node type with the necessary resources.')
            break
        utilization_scores = sorted(utilization_scores, reverse=True)
        best_node_type = utilization_scores[0][1]
        nodes_to_add[best_node_type] += 1
        if strict_spread:
            resources = resources[1:]
        else:
            allocated_resource = node_types[best_node_type]['resources']
            (residual, _) = get_bin_pack_residual([allocated_resource], resources)
            assert len(residual) < len(resources), (resources, residual)
            resources = residual
    return (nodes_to_add, resources)

def _resource_based_utilization_scorer(node_resources: ResourceDict, resources: List[ResourceDict], *, node_availability_summary: NodeAvailabilitySummary) -> Optional[Tuple[bool, int, float, float]]:
    if False:
        for i in range(10):
            print('nop')
    remaining = copy.deepcopy(node_resources)
    fittable = []
    resource_types = set()
    for r in resources:
        for (k, v) in r.items():
            if v > 0:
                resource_types.add(k)
        if _fits(remaining, r):
            fittable.append(r)
            _inplace_subtract(remaining, r)
    if not fittable:
        return None
    util_by_resources = []
    num_matching_resource_types = 0
    for (k, v) in node_resources.items():
        if v < 1:
            continue
        if k in resource_types:
            num_matching_resource_types += 1
        util = (v - remaining[k]) / v
        util_by_resources.append(v * util ** 3)
    if not util_by_resources:
        return None
    gpu_ok = True
    if AUTOSCALER_CONSERVE_GPU_NODES:
        is_gpu_node = 'GPU' in node_resources and node_resources['GPU'] > 0
        any_gpu_task = any(('GPU' in r for r in resources))
        if is_gpu_node and (not any_gpu_task):
            gpu_ok = False
    return (gpu_ok, num_matching_resource_types, min(util_by_resources), float(sum(util_by_resources)) / len(util_by_resources))

def _default_utilization_scorer(node_resources: ResourceDict, resources: List[ResourceDict], node_type: str, *, node_availability_summary: NodeAvailabilitySummary):
    if False:
        i = 10
        return i + 15
    return _resource_based_utilization_scorer(node_resources, resources, node_availability_summary=node_availability_summary)

def get_bin_pack_residual(node_resources: List[ResourceDict], resource_demands: List[ResourceDict], strict_spread: bool=False) -> (List[ResourceDict], List[ResourceDict]):
    if False:
        for i in range(10):
            print('nop')
    'Return a subset of resource_demands that cannot fit in the cluster.\n\n    TODO(ekl): this currently does not guarantee the resources will be packed\n    correctly by the Ray scheduler. This is only possible once the Ray backend\n    supports a placement groups API.\n\n    Args:\n        node_resources (List[ResourceDict]): List of resources per node.\n        resource_demands (List[ResourceDict]): List of resource bundles that\n            need to be bin packed onto the nodes.\n        strict_spread: If true, each element in resource_demands must be\n            placed on a different entry in `node_resources`.\n\n    Returns:\n        List[ResourceDict]: the residual list resources that do not fit.\n        List[ResourceDict]: The updated node_resources after the method.\n    '
    unfulfilled = []
    nodes = copy.deepcopy(node_resources)
    used = []
    for demand in sorted(resource_demands, key=lambda demand: (len(demand.values()), sum(demand.values()), sorted(demand.items())), reverse=True):
        found = False
        node = None
        for i in range(len(nodes)):
            node = nodes[i]
            if _fits(node, demand):
                found = True
                if strict_spread:
                    used.append(node)
                    del nodes[i]
                break
        if found and node:
            _inplace_subtract(node, demand)
        else:
            unfulfilled.append(demand)
    return (unfulfilled, nodes + used)

def _fits(node: ResourceDict, resources: ResourceDict) -> bool:
    if False:
        print('Hello World!')
    for (k, v) in resources.items():
        if v > node.get(k, 1.0 if k.startswith(ray._raylet.IMPLICIT_RESOURCE_PREFIX) else 0.0):
            return False
    return True

def _inplace_subtract(node: ResourceDict, resources: ResourceDict) -> None:
    if False:
        while True:
            i = 10
    for (k, v) in resources.items():
        if v == 0:
            continue
        if k not in node:
            assert k.startswith(ray._raylet.IMPLICIT_RESOURCE_PREFIX), (k, node)
            node[k] = 1
        assert k in node, (k, node)
        node[k] -= v
        assert node[k] >= 0.0, (node, k, v)

def _inplace_add(a: collections.defaultdict, b: Dict) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Generically adds values in `b` to `a`.\n    a[k] should be defined for all k in b.keys()'
    for (k, v) in b.items():
        a[k] += v

def placement_groups_to_resource_demands(pending_placement_groups: List[PlacementGroupTableData]):
    if False:
        i = 10
        return i + 15
    "Preprocess placement group requests into regular resource demand vectors\n    when possible. The policy is:\n        * STRICT_PACK - Convert to a single bundle.\n        * PACK - Flatten into a resource demand vector.\n        * STRICT_SPREAD - Cannot be converted.\n        * SPREAD - Flatten into a resource demand vector.\n\n    Args:\n        pending_placement_groups (List[PlacementGroupData]): List of\n        PlacementGroupLoad's.\n\n    Returns:\n        List[ResourceDict]: The placement groups which were converted to a\n            resource demand vector.\n        List[List[ResourceDict]]: The placement groups which should be strictly\n            spread.\n    "
    resource_demand_vector = []
    unconverted = []
    for placement_group in pending_placement_groups:
        shapes = [dict(bundle.unit_resources) for bundle in placement_group.bundles]
        if placement_group.strategy == PlacementStrategy.PACK or placement_group.strategy == PlacementStrategy.SPREAD:
            resource_demand_vector.extend(shapes)
        elif placement_group.strategy == PlacementStrategy.STRICT_PACK:
            combined = collections.defaultdict(float)
            for shape in shapes:
                for (label, quantity) in shape.items():
                    combined[label] += quantity
            resource_demand_vector.append(combined)
        elif placement_group.strategy == PlacementStrategy.STRICT_SPREAD:
            unconverted.append(shapes)
        else:
            logger.error(f'Unknown placement group request type: {placement_group}. Please file a bug report https://github.com/ray-project/ray/issues/new.')
    return (resource_demand_vector, unconverted)