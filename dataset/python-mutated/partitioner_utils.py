from enum import Enum
from typing import NamedTuple, Dict, List, Set
from torch.fx.node import Node, map_arg

class Partition:
    """Partition class contains all the information about an individual partition.
    It also provides necessary methods for manipulation the partition.
    """

    def __init__(self, partition_id: int) -> None:
        if False:
            i = 10
            return i + 15
        self.nodes: Set[Node] = set()
        self.partition_id = partition_id
        self.parents: Set[Partition] = set()
        self.children: Set[Partition] = set()
        self.bfs_level: int = -1
        self.used_mem_bytes: int = 0
        self.logical_device_ids: List[int] = []

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.partition_id)

    def recalculate_mem_size(self):
        if False:
            i = 10
            return i + 15
        self.used_mem_bytes = 0
        for node in self.nodes:
            self.used_mem_bytes += get_extra_size_of(node, self.nodes)

    def add_node(self, node):
        if False:
            return 10
        input_nodes: Dict[Node, None] = {}
        map_arg(node.args, input_nodes.setdefault)
        map_arg(node.kwargs, input_nodes.setdefault)
        for n in input_nodes:
            if n.op in {'placeholder', 'get_attr'}:
                self.nodes.add(n)
        self.nodes.add(node)
        self.recalculate_mem_size()

    def remove_node(self, node):
        if False:
            return 10
        if node in self.nodes:
            self.nodes.remove(node)
            input_nodes: Dict[Node, None] = {}
            map_arg(node.args, input_nodes.setdefault)
            map_arg(node.kwargs, input_nodes.setdefault)
            for input_node in input_nodes:
                if all((n not in self.nodes for n in input_node.users)) and input_node.op in {'placeholder', 'get_attr'}:
                    self.nodes.remove(input_node)
            self.recalculate_mem_size()

class Device(NamedTuple):
    name: str
    available_mem_bytes: int
    logical_id: int

class NodeLatency(NamedTuple):
    mem_latency_sec: float
    computer_latency_sec: float

class PartitionLatency(NamedTuple):
    mem_latency_sec: float
    computer_latency_sec: float
    overall_latency_sec: float

class PartitionMode(Enum):
    size_based = 0
    sparse_nn = 1
    cost_aware = 2
    kl_based = 3
    aot_based = 4

class PartitionerConfig(NamedTuple):
    devices: List[Device]
    mode: PartitionMode = PartitionMode.size_based
    transfer_rate_bytes_per_sec: float = 0.0
    node_to_latency_mapping: Dict[Node, NodeLatency] = {}
    node_to_partition_mapping: Dict[Node, int] = {}
    partition_to_logical_device_mapping: Dict[int, List[int]] = {}
    saturate_host: bool = False

def get_extra_size_of(node: Node, nodes: Set[Node]) -> int:
    if False:
        print('Hello World!')
    'Given a node and a set of nodes,\n    this function return the extra size that needed\n    if this node is included in this set.\n    '
    input_nodes: Dict[Node, None] = {}
    map_arg(node.args, input_nodes.setdefault)
    map_arg(node.kwargs, input_nodes.setdefault)
    total_size_of_input_nodes = 0
    for n in input_nodes:
        if n not in nodes:
            size_bytes = getattr(n, 'size_bytes', None)
            if size_bytes:
                total_size_of_input_nodes += size_bytes.output_size
            else:
                raise RuntimeError('node has no size_bytes attr')
    size_bytes = getattr(node, 'size_bytes', None)
    if size_bytes:
        total_size_of_input_nodes += size_bytes.total_size
    else:
        raise RuntimeError('node has no size_bytes attr')
    return total_size_of_input_nodes

def get_latency_of_one_partition(partition: Partition, node_to_latency_mapping: Dict[Node, NodeLatency]) -> PartitionLatency:
    if False:
        i = 10
        return i + 15
    "Given a partition and its nodes' latency, return a PartitionLatency for this partition"

    def get_top_nodes(partition: Partition) -> List[Node]:
        if False:
            return 10
        'Given a partition, return a list of nodes on the top bfs level'
        top_nodes: List[Node] = []
        for node in partition.nodes:
            if node.op in {'placeholder', 'get_attr'}:
                continue
            input_nodes: Dict[Node, None] = {}
            map_arg(node.args, input_nodes.setdefault)
            map_arg(node.kwargs, input_nodes.setdefault)
            if not any((n in partition.nodes and n.op not in {'placeholder', 'get_attr'} for n in input_nodes)):
                top_nodes.append(node)
        return top_nodes

    def dfs_helper(node: Node, partition_latency) -> PartitionLatency:
        if False:
            for i in range(10):
                print('nop')
        'Given a top node of a partition, this function returns\n        the latency of the critical path in the partition\n        '
        node_latency = node_to_latency_mapping[node]
        overall_latency_sec = partition_latency.overall_latency_sec + max(node_latency.computer_latency_sec, node_latency.mem_latency_sec)
        mem_latency_sec = partition_latency.mem_latency_sec + node_latency.mem_latency_sec
        computer_latency_sec = partition_latency.computer_latency_sec + node_latency.computer_latency_sec
        users = set(node.users).intersection(partition.nodes)
        if users:
            max_latency = PartitionLatency(mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0)
            for n in users:
                new_partition_latency = dfs_helper(n, PartitionLatency(mem_latency_sec, computer_latency_sec, overall_latency_sec))
                if new_partition_latency.overall_latency_sec > max_latency.overall_latency_sec:
                    max_latency = new_partition_latency
            return max_latency
        return PartitionLatency(mem_latency_sec, computer_latency_sec, overall_latency_sec)
    top_nodes = get_top_nodes(partition)
    critical_path_latency = PartitionLatency(mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0)
    for node in top_nodes:
        partition_latency = dfs_helper(node, PartitionLatency(mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0))
        if partition_latency.overall_latency_sec > critical_path_latency.overall_latency_sec:
            critical_path_latency = partition_latency
    return critical_path_latency

def get_partition_to_latency_mapping(partitions: List[Partition], node_to_latency_mapping: Dict[Node, NodeLatency]) -> Dict[Partition, PartitionLatency]:
    if False:
        i = 10
        return i + 15
    'Given all the partitions and node_to_latency_mapping dictionary,\n    return a mapping dictionary of each partition to its overall latency\n    '
    partition_to_latency_mapping: Dict[Partition, PartitionLatency] = {}
    for partition in partitions:
        partition_latency = get_latency_of_one_partition(partition, node_to_latency_mapping)
        partition_to_latency_mapping[partition] = partition_latency
    return partition_to_latency_mapping

def get_comm_latency_between(parent_partition: Partition, child_partition: Partition, transfer_rate_bytes_per_sec: float):
    if False:
        i = 10
        return i + 15
    'Given two partitions (parent and child),\n    calculate the communication latency between the two.\n    '
    if parent_partition.logical_device_ids != [] and child_partition.logical_device_ids != [] and (parent_partition.logical_device_ids == child_partition.logical_device_ids):
        return 0.0
    comm_size = 0
    visited_nodes = set()
    for node in child_partition.nodes:
        input_nodes: Dict[Node, None] = {}
        map_arg(node.args, input_nodes.setdefault)
        map_arg(node.kwargs, input_nodes.setdefault)
        for n in input_nodes:
            if n in parent_partition.nodes and n not in visited_nodes:
                size_bytes = getattr(n, 'size_bytes', None)
                if size_bytes is not None:
                    comm_size += size_bytes.output_size
                visited_nodes.add(n)
    return comm_size / transfer_rate_bytes_per_sec

def get_latency_of_partitioned_graph(partitions: List[Partition], partition_to_latency_mapping: Dict[Partition, PartitionLatency], transfer_rate_bytes_per_sec: float):
    if False:
        print('Hello World!')
    'Given all partitions in a graph, find the critical path among all partitions\n    and return its latency as the latency of the whole graph\n    '

    def dfs_helper(partition: Partition, latency_so_far_sec: float) -> float:
        if False:
            i = 10
            return i + 15
        'This function helps to recursively get the latency of a path of partitions'
        latency_so_far_sec += partition_to_latency_mapping[partition].overall_latency_sec
        children = partition.children
        if partition.children:
            max_latency_sec = 0.0
            for child in partition.children:
                comm_latency_sec = get_comm_latency_between(partition, child, transfer_rate_bytes_per_sec)
                new_latency_sec = dfs_helper(child, latency_so_far_sec + comm_latency_sec)
                if new_latency_sec > max_latency_sec:
                    max_latency_sec = new_latency_sec
            return max_latency_sec
        return latency_so_far_sec

    def get_top_partitions(partitions: List[Partition]) -> List[Partition]:
        if False:
            while True:
                i = 10
        'This function is to return all the partitions without parents\n        as the starting points of all the paths\n        '
        top_partitions = []
        for partition in partitions:
            if len(partition.parents) == 0:
                top_partitions.append(partition)
        return top_partitions
    top_partitions = get_top_partitions(partitions)
    critical_path_latency_sec = 0.0
    for partition in top_partitions:
        latency_sec = dfs_helper(partition, 0.0)
        if latency_sec > critical_path_latency_sec:
            critical_path_latency_sec = latency_sec
    return critical_path_latency_sec