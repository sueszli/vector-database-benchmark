from typing import Dict, List, Set, Iterable, Sequence, Optional, Deque
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupportBase
import logging
import itertools
from copy import copy
from collections import deque
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class Partition:

    def __init__(self, id: Optional[int]=None, nodes: Optional[Iterable[Node]]=None):
        if False:
            return 10
        self.id = id
        self.nodes: Set[Node] = set(nodes) if nodes is not None else set()

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return str(self.nodes)

    def add_node(self, node: Node):
        if False:
            for i in range(10):
                print('nop')
        self.nodes.add(node)

    def remove_node(self, node: Node):
        if False:
            print('Hello World!')
        self.nodes.remove(node)

    def size(self):
        if False:
            while True:
                i = 10
        return len(self.nodes)

class CapabilityBasedPartitioner:

    def __init__(self, graph_module: GraphModule, operator_support: OperatorSupportBase, allows_single_node_partition: bool=False, non_compute_ops: Optional[Sequence[str]]=None, allowed_single_node_partition_ops: Optional[Sequence[str]]=None) -> None:
        if False:
            return 10
        self.graph_module = graph_module
        self.operator_support = operator_support
        self.allows_single_node_partition = allows_single_node_partition
        self.non_compute_ops = non_compute_ops if non_compute_ops is not None else []
        self.allowed_single_node_partition_ops = allowed_single_node_partition_ops if allowed_single_node_partition_ops is not None else []

    def __is_node_supported(self, node: Node) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.operator_support.is_node_supported(dict(self.graph_module.named_modules()), node)

    def propose_partitions(self) -> List[Partition]:
        if False:
            i = 10
            return i + 15
        assignment: Dict[Node, int] = {}
        partitions_by_id: Dict[int, Partition] = {}
        new_partition_id = itertools.count()

        def maybe_merge_partition(self_id: int, other_id: int):
            if False:
                print('Hello World!')
            merged_nodes = copy(partitions_by_id[self_id].nodes)
            merged_nodes.update(partitions_by_id[other_id].nodes)
            visited: Set[Node] = set()

            def dfs_iter_find_cycle(root_node):
                if False:
                    while True:
                        i = 10
                stack: Deque[Node] = deque()
                stack.append(root_node)
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    if node in merged_nodes:
                        return True
                    if node in assignment:
                        for p_node in partitions_by_id[assignment[node]].nodes:
                            for user_node in p_node.users:
                                if user_node not in partitions_by_id[assignment[node]].nodes:
                                    stack.append(user_node)
                    else:
                        for user_node in node.users:
                            stack.append(user_node)
                    visited.add(node)
                return False
            for node in merged_nodes:
                for user_node in node.users:
                    if user_node not in merged_nodes and dfs_iter_find_cycle(user_node):
                        return False
            partitions_by_id[self_id].nodes = merged_nodes
            for node in partitions_by_id[other_id].nodes:
                assignment[node] = self_id
            del partitions_by_id[other_id]
            return True

        def merge_single_node(node: Node, id: Optional[int]):
            if False:
                while True:
                    i = 10
            if node in assignment:
                partitions_by_id[assignment[node]].remove_node(node)
            if id is None:
                assignment.pop(node)
            elif id not in partitions_by_id:
                assignment[node] = id
                partitions_by_id[id] = Partition(id=id, nodes=[node])
            else:
                assignment[node] = id
                partitions_by_id[id].add_node(node)
        logger.debug('Proposing partitions...')
        for node in reversed(self.graph_module.graph.nodes):
            merge_candidates: Dict[int, None] = {}
            if self.__is_node_supported(node) and node not in assignment:
                partition_id = next(new_partition_id)
                merge_single_node(node, partition_id)
                merge_candidates[partition_id] = None
            for node in assignment:
                merge_candidates[assignment[node]] = None
            merge_candidates_list = list(merge_candidates.keys())
            if len(merge_candidates_list) > 1:
                self_id = merge_candidates_list[0]
                for other_id in merge_candidates_list[1:]:
                    maybe_merge_partition(self_id, other_id)
        logger.debug("Reassigning getitem nodes to its producer node's partition...")
        nodes_reassignment: Dict[Node, int] = {}
        for node in self.graph_module.graph.nodes:
            is_tuple_output = True
            for user in node.users:
                if user.op != 'call_function' or _get_qualified_name(user.target) != '_operator.getitem':
                    is_tuple_output = False
                    break
            if is_tuple_output:
                id = assignment.get(node, None)
                for user in node.users:
                    if assignment.get(user, None) != id:
                        nodes_reassignment[user] = id
        for (node, id) in nodes_reassignment.items():
            merge_single_node(node, id)
        if not self.allows_single_node_partition:
            logger.debug('Filtering out single node partitions...')
            default_non_compute_ops = {'torch.ops.aten.view', '_operator.getitem'}
            non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops))
            partitions_to_remove: List[int] = []
            for (id, partition) in partitions_by_id.items():
                compute_node_count = 0
                for node in partition.nodes:
                    if node.op == 'call_function':
                        assert callable(node.target)
                        if _get_qualified_name(node.target) not in non_compute_ops:
                            compute_node_count += 1
                        if _get_qualified_name(node.target) in self.allowed_single_node_partition_ops:
                            compute_node_count += 1
                if compute_node_count <= 1:
                    partitions_to_remove.append(id)
            for id in partitions_to_remove:
                del partitions_by_id[id]
        logger.debug('Partitions proposed:')
        for (id, partition) in partitions_by_id.items():
            logger.debug('partition #%s: %s', id, [node.name for node in partition.nodes])
        return list(partitions_by_id.values())

    def fuse_partitions(self, partitions: List[Partition]) -> GraphModule:
        if False:
            i = 10
            return i + 15
        logger.debug('Fusing partitions...')
        return fuse_by_partitions(self.graph_module, [list(partition.nodes) for partition in partitions])

    def remove_bookend_non_compute_ops(self, partitions: List[Partition]):
        if False:
            i = 10
            return i + 15
        non_compute_ops = set(self.non_compute_ops)

        def is_non_compute_node(node: Node):
            if False:
                while True:
                    i = 10
            return node.op == 'call_function' and _get_qualified_name(node.target) in non_compute_ops
        transparent_input_nodes: Dict[Node, bool] = {}
        transparent_output_nodes: Dict[Node, bool] = {}

        def is_transparent_input_node(node: Node, partition: Set[Node], removed_nodes: Set[Node]):
            if False:
                for i in range(10):
                    print('nop')
            if node.op == 'placeholder' or node not in partition or node in removed_nodes:
                return True
            if node in transparent_input_nodes:
                return transparent_input_nodes[node]
            if is_non_compute_node(node):
                for input_n in node.all_input_nodes:
                    if not is_transparent_input_node(input_n, partition, removed_nodes):
                        transparent_input_nodes[node] = False
                        return False
                transparent_input_nodes[node] = True
                return True
            transparent_input_nodes[node] = False
            return False

        def is_transparent_output_node(node: Node, partition: Set[Node], removed_nodes: Set[Node]):
            if False:
                return 10
            if node.op == 'placeholder' or node not in partition or node in removed_nodes:
                return True
            if node in transparent_output_nodes:
                return transparent_output_nodes[node]
            if is_non_compute_node(node):
                for output_n in node.users:
                    if not is_transparent_output_node(output_n, partition, removed_nodes):
                        transparent_output_nodes[node] = False
                        return False
                transparent_output_nodes[node] = True
                return True
            transparent_output_nodes[node] = False
            return False
        for partition in partitions:
            remove_node: Set[Node] = set()
            for node in partition.nodes:
                if is_non_compute_node(node) and (is_transparent_input_node(node, partition.nodes, remove_node) or is_transparent_output_node(node, partition.nodes, remove_node)):
                    remove_node.add(node)
            if len(remove_node) != 0:
                partition.nodes = partition.nodes - remove_node

    def partition_and_fuse(self) -> GraphModule:
        if False:
            print('Hello World!')
        partitions = self.propose_partitions()
        fused_gm = self.fuse_partitions(partitions)
        return fused_gm