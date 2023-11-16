import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import Partition, Device, PartitionerConfig, get_partition_to_latency_mapping, get_latency_of_partitioned_graph, NodeLatency, get_extra_size_of, PartitionMode
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module

class DAGNode:
    """DAGNode class maintains useful information for a partition (submodule),
    and its input submodules and output submodules.
    """

    def __init__(self, submodule_node: Node, input_nodes: List[Node], output_nodes: List[Node], logical_device_ids: List[int], size_bytes: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.submodule_node: Node = submodule_node
        self.input_nodes: List[Node] = input_nodes
        self.output_nodes: List[Node] = output_nodes
        self.logical_device_ids: List[int] = logical_device_ids
        self.size_bytes = size_bytes

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return str(self.submodule_node)

class DAG:
    """DAG class contains all the DAG nodes"""

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.nodes: List[DAGNode] = []

    def create_node(self, submodule_node: Node, input_nodes: List[Node], output_nodes: List[Node], logical_devices: List[int], size_bytes: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        node = DAGNode(submodule_node, input_nodes, output_nodes, logical_devices, size_bytes)
        self.nodes.append(node)

class PartitionResult(NamedTuple):
    """NameTuple used for returning DAG and a new fx module"""
    dag: DAG
    module_with_submodules: GraphModule
'Followings are some helper functions for partition manipulation'

def reset_partition_device(partitions):
    if False:
        i = 10
        return i + 15
    for partition in partitions:
        partition.logical_device_ids = []

def combine_two_partitions(partition_0: Partition, partition_1: Partition, partitions: List[Partition]) -> None:
    if False:
        i = 10
        return i + 15
    'Given a list of partitions and its two partitions,\n    combine these two partitions into a new one appending to the partitions\n    and remove the previous two partitions from the list of partitions\n    '
    partition = Partition(len(partitions))
    partition.nodes = partition_0.nodes.union(partition_1.nodes)
    partition.recalculate_mem_size()
    partitions.append(partition)
    partitions.remove(partition_0)
    partitions.remove(partition_1)
    reorganize_partitions(partitions)
    return

def set_parents_and_children(partitions: List[Partition]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Given a list of partitions, mark parents and children for each partition'
    for partition in partitions:
        partition.children = set()
        partition.parents = set()
    for partition in partitions:
        for node in partition.nodes:
            users = node.users
            for n in users:
                for p in partitions:
                    if p != partition and n in p.nodes and (node not in p.nodes):
                        partition.children.add(p)
                        p.parents.add(partition)
    return

def reorganize_partitions(partitions: List[Partition]) -> None:
    if False:
        return 10
    'Given a list of partitions, reorganize partition id,\n    its parents and its children for each partition\n    '
    for (i, partition) in enumerate(partitions):
        partition.partition_id = i
    set_parents_and_children(partitions)
    return

def get_bfs_level_partition(partitions: List[Partition]) -> None:
    if False:
        print('Hello World!')
    'Given a list of partitions,\n    mark the bfs level for each partition\n    '
    current_level: Set[Partition] = set()
    visited: Set[Partition] = set()
    for partition in partitions:
        if len(partition.parents) == 0:
            current_level.add(partition)
    next_level: Set[Partition] = set()
    level = 0
    while current_level:
        partition = current_level.pop()
        partition.bfs_level = level
        visited.add(partition)
        children = partition.children
        for child in children:
            if child not in next_level:
                next_level.add(child)
        if not current_level:
            current_level = next_level.copy()
            next_level = set()
            level += 1
    return

def get_node_to_partition_mapping(partitions: List[Partition]) -> Dict[Node, int]:
    if False:
        i = 10
        return i + 15
    'Given a list of partitions,return node to partition mapping'
    node_to_partition: Dict[Node, int] = {}
    for partition in partitions:
        for node in partition.nodes:
            node_to_partition[node] = partition.partition_id
    return node_to_partition

def get_logical_id_to_device(devices: List[Device]) -> Dict[int, Device]:
    if False:
        print('Hello World!')
    'Get a mapping from device logical ID to Device object.'
    logical_id_to_device: Dict[int, Device] = {}
    for d in devices:
        logical_id_to_device[d.logical_id] = d
    return logical_id_to_device

def get_device_partition_stats(partitions: List[Partition], devices: List[Device]) -> Tuple[Dict[Device, List[Partition]], Dict[Device, int], List[Partition]]:
    if False:
        while True:
            i = 10
    'Given a list of partitions and a list of devices, returns:\n    1. A mapping from device to partitions on it;\n    2. A mapping from device to its remaining memory size;\n    3. A list of partitions that do not have a device.\n    '
    logical_id_to_device = get_logical_id_to_device(devices)
    device_to_partitions: Dict[Device, List[Partition]] = {}
    device_to_left_mem_bytes: Dict[Device, int] = {}
    for d in devices:
        device_to_partitions[d] = []
        device_to_left_mem_bytes[d] = d.available_mem_bytes
    no_device_partitions = []
    for partition in partitions:
        if partition.logical_device_ids != []:
            for logical_id in partition.logical_device_ids:
                device = logical_id_to_device[logical_id]
                device_to_partitions[device].append(partition)
                device_to_left_mem_bytes[device] -= partition.used_mem_bytes
        else:
            no_device_partitions.append(partition)
    return (device_to_partitions, device_to_left_mem_bytes, no_device_partitions)

def get_device_to_partitions_mapping(partitions: List[Partition], devices: List[Device]):
    if False:
        for i in range(10):
            print('nop')
    'Given a list of partitions and a list of devices,\n    map each partition into a device.\n    '

    def calculate_extra_mem_bytes_needed_for(partition: Partition, partitions: List[Partition]):
        if False:
            return 10
        all_nodes: Set[Node] = set()
        for p in partitions:
            all_nodes = all_nodes.union(p.nodes)
        if len(all_nodes) == 0:
            return partition.used_mem_bytes
        all_nodes = all_nodes.union(partition.nodes)
        extra_size_needed = 0
        for node in partition.nodes:
            extra_size_needed += get_extra_size_of(node, all_nodes)
        return extra_size_needed

    def find_device_for(partition: Partition):
        if False:
            for i in range(10):
                print('nop')
        'Given a partition, find a logical device for the partition\n        The algorithm is to put the partition on the device\n        that has just enough mem left for that partition.\n        device_to_left_mem_bytes is a dictionary between device and its left mem size\n        sorted by its left mem size\n        '
        for d in device_to_left_mem_bytes:
            extra_size_needed = calculate_extra_mem_bytes_needed_for(partition, device_to_partitions[d])
            if extra_size_needed < device_to_left_mem_bytes[d]:
                device_to_partitions[d].append(partition)
                partition.logical_device_ids.append(d.logical_id)
                device_to_left_mem_bytes[d] -= extra_size_needed
                return True
        return False
    (device_to_partitions, device_to_left_mem_bytes, no_device_partitions) = get_device_partition_stats(partitions, devices)
    found_device = True
    for partition in no_device_partitions:
        device_to_left_mem_bytes = dict(sorted(device_to_left_mem_bytes.items(), key=lambda item: item[1]))
        found_device = find_device_for(partition)
        if not found_device:
            break
    return found_device

def check_dependency(partition):
    if False:
        i = 10
        return i + 15
    'Given a partition,check if there is a circular dependency on\n    this partition using bfs\n    '
    visited: Set[Partition] = {partition}
    queue: Deque[Partition] = deque([partition])
    while queue:
        p = queue.popleft()
        for child in p.children:
            if child == partition:
                return True
            elif child not in visited:
                visited.add(child)
                queue.append(child)
    return False

class Partitioner:
    """A fx module may not fit into one device.
    Partitioner class helps partition one fx module into submodules (partitions),
    so that the submodules can be executed crossing different accelerators.
    The main function of this class is self.partition_graph.
    It partitions the fx module based on the scheme specified in partition_config
    A DAG structure is returned
    along with a new fx module with submodule nodes.
    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.partitions: List[Partition] = []
        self.node_to_partition: Dict[Node, int] = {}
        self.devices: List[Device] = []

    def partition_graph(self, fx_module: GraphModule, torch_module: torch.nn.Module, partitioner_config: PartitionerConfig) -> PartitionResult:
        if False:
            i = 10
            return i + 15
        'Given the fx module, torch module and partitioner_config,\n        find the partitions, do the partitions,\n        and then return a DAG and a new fx module with submodule nodes (partitions)\n        '
        self.graph_module = fx_module
        self.torch_module = torch_module
        self.devices = partitioner_config.devices
        if len(self.devices) == 0:
            raise RuntimeError('No devices')
        get_size_of_all_nodes(self.graph_module)
        nodes = self.graph_module.graph.nodes
        if all((node.op in {'placeholder', 'get_attr', 'output'} for node in nodes)):
            raise RuntimeError('No Partition since no operations in the module')
        total_size_of_graph = 0
        for node in nodes:
            if node.op == 'output':
                break
            total_size_of_graph += node.size_bytes.total_size
        device_with_max_mem = max(self.devices, key=lambda d: d.available_mem_bytes)
        if partitioner_config.mode == PartitionMode.aot_based:
            self.aot_based_partition(partitioner_config.node_to_partition_mapping, partitioner_config.partition_to_logical_device_mapping)
        elif total_size_of_graph <= device_with_max_mem.available_mem_bytes:
            self.find_single_partition(total_size_of_graph, logical_device_id=device_with_max_mem.logical_id)
        elif total_size_of_graph > sum([d.available_mem_bytes for d in self.devices]):
            raise RuntimeError('Devices have no enough memory for the module')
        elif partitioner_config.mode == PartitionMode.sparse_nn:
            available_mem_bytes = self.devices[0].available_mem_bytes
            if not all((device.available_mem_bytes == available_mem_bytes for device in self.devices)):
                raise RuntimeError('All devices must have same memory size!')
            self.sparse_nn_partition(available_mem_bytes)
        elif partitioner_config.mode == PartitionMode.cost_aware:
            self.cost_aware_partition(partitioner_config.transfer_rate_bytes_per_sec, partitioner_config.node_to_latency_mapping)
        elif partitioner_config.mode == PartitionMode.kl_based:
            self.kl_based_partition(partitioner_config.transfer_rate_bytes_per_sec, partitioner_config.node_to_latency_mapping)
        else:
            self.size_based_partition()
        if partitioner_config.saturate_host:
            self.saturate_host()
        module_with_submodules = self.do_partition()
        dag = self.dump_dag(module_with_submodules)
        ret = PartitionResult(dag, module_with_submodules)
        return ret

    def find_single_partition(self, total_size_of_graph, logical_device_id: int=0) -> None:
        if False:
            i = 10
            return i + 15
        'Fit the whole fx module into one device'
        partition_0 = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op == 'output':
                continue
            partition_0.nodes.add(node)
        partition_0.used_mem_bytes = total_size_of_graph
        partition_0.logical_device_ids = [logical_device_id]
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        return

    def size_based_partition(self) -> None:
        if False:
            while True:
                i = 10
        'This method is to partition the fx module based on memory size.\n        It uses greedy approach. The result may not be the best.\n        The basic idea is:\n        Step 1:\n        Find a device which has enough memory to fit the current node, create a empty partition\n        with the size of that device.\n        Then keep adding the following nodes into the partition until the partition is full.\n        Step 2:\n        Repeat Step 1 until no device left\n        Step 3:\n        If some nodes are left, create a partition for each left node (single node partition).\n        and then try to map those partitions into logical devices with enough mem left.\n        '

        def find_device_based_on_size(node) -> Device:
            if False:
                print('Hello World!')
            'Given a node, this function is to find a logical device\n            that could fit the node.\n            '
            mem_size_needed = get_extra_size_of(node, set())
            device = Device('', -1, -1)
            for d in self.devices:
                if d not in occupied_devices and d.available_mem_bytes >= mem_size_needed:
                    device = d
                    break
            if device.available_mem_bytes < 0:
                raise RuntimeError(str(node) + 'is too large to fit any device')
            occupied_devices.append(device)
            return device
        partition_to_left_mem_bytes: Dict[Partition, int] = {}
        occupied_devices: List[Device] = []
        partition = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op in {'call_module', 'call_method', 'call_function'}:
                if len(self.partitions) <= len(self.devices):
                    total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                    if partition.used_mem_bytes == 0:
                        device = find_device_based_on_size(node)
                        occupied_devices.append(device)
                        partition_to_left_mem_bytes[partition] = device.available_mem_bytes
                        partition.logical_device_ids.append(device.logical_id)
                    elif partition_to_left_mem_bytes[partition] < total_size_of_input_nodes:
                        if len(self.partitions) == len(self.devices):
                            non_single_node_partitions = self.partitions[:]
                            self.create_single_node_partition(node)
                            continue
                        device = find_device_based_on_size(node)
                        partition = self.create_partition()
                        total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                        partition_to_left_mem_bytes[partition] = device.available_mem_bytes
                        partition.logical_device_ids.append(device.logical_id)
                    partition.add_node(node)
                    partition_to_left_mem_bytes[partition] -= total_size_of_input_nodes
                else:
                    self.create_single_node_partition(node)
        reorganize_partitions(self.partitions)
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        found_partition_to_device_mapping = get_device_to_partitions_mapping(self.partitions, self.devices)
        if not found_partition_to_device_mapping:
            raise RuntimeError('Cannot Get a Valid Partition to Logical Device Mapping')
        return

    def saturate_host(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Saturate host by assigning replicates to unused devices with enough memory.\n        It uses a greedy approach to find a next available set of devices to place all split\n        partitions: For each used device, it searches for an idle device with minimal memory\n        size that can hold all the partition located on that device; If the search is successful\n        for all used devices, it then assigns the new devices' logical ID to the corresponding\n        partition.\n        "
        (device_to_partitions, device_to_left_mem_bytes, no_device_partitions) = get_device_partition_stats(self.partitions, self.devices)
        assert len(no_device_partitions) == 0, f'Expect no_device_partitions has 0 device, but get {len(no_device_partitions)}'
        used_devices = [d for d in self.devices if len(device_to_partitions[d]) > 0]
        replicated_device_to_used_device: Dict[Device, Device] = {}
        while len(used_devices) * 2 + len(replicated_device_to_used_device) <= len(self.devices):
            success = True
            idle_devices = [d for d in self.devices if d not in used_devices and d not in replicated_device_to_used_device]
            temp_replicate_mapping = {}
            for used_device in used_devices:
                available_devices = [d for d in idle_devices if d.available_mem_bytes >= used_device.available_mem_bytes - device_to_left_mem_bytes[used_device]]
                if len(available_devices) == 0:
                    success = False
                    break
                new_device = min(available_devices, key=lambda d: d.available_mem_bytes)
                idle_devices.remove(new_device)
                temp_replicate_mapping[new_device] = used_device
            if not success:
                break
            replicated_device_to_used_device.update(temp_replicate_mapping)
        for (replicate_device, original_device) in replicated_device_to_used_device.items():
            logical_id = replicate_device.logical_id
            for partition in device_to_partitions[original_device]:
                partition.logical_device_ids.append(logical_id)
        for p in self.partitions:
            print(p.logical_device_ids)

    def do_partition(self) -> GraphModule:
        if False:
            return 10
        'Return a new fx module with submodule nodes (partitions).'
        module_with_submodules = split_module(self.graph_module, self.torch_module, lambda node: self.node_to_partition[node])
        return module_with_submodules

    def dump_dag(self, module_with_submodules: GraphModule) -> DAG:
        if False:
            i = 10
            return i + 15
        'Return the dag structure and the new fx module with submodules.'
        dag = DAG()
        for node in module_with_submodules.graph.nodes:
            if node.op == 'output':
                break
            if node.op in {'placeholder', 'get_attr'}:
                continue
            if node.target == operator.__getitem__:
                continue
            input_nodes: Dict[Node, None] = {}
            map_arg(node.args, input_nodes.setdefault)
            map_arg(node.kwargs, input_nodes.setdefault)
            if len(node.users) > 1:
                output_nodes = list(node.users)
            else:
                output_nodes = [node]
            partition_id = int(node.name.rsplit('_', 1)[-1])
            device_ids = self.partitions[partition_id].logical_device_ids
            size_bytes = self.partitions[partition_id].used_mem_bytes
            dag.create_node(node, list(input_nodes), output_nodes, device_ids, size_bytes)
        return dag

    def create_partition(self) -> Partition:
        if False:
            while True:
                i = 10
        'Create a partition and append it to self.partitions.'
        partition_id = len(self.partitions)
        partition = Partition(partition_id)
        self.partitions.append(partition)
        return partition

    def create_single_node_partition(self, node):
        if False:
            while True:
                i = 10
        'Create a partition for a single node'
        partition = self.create_partition()
        partition.add_node(node)
        return

    def sparse_nn_partition(self, available_mem_bytes: int) -> None:
        if False:
            i = 10
            return i + 15
        'This method partition a sparse nn module.\n        It is size based partition but different from size_based_partition,\n        it only works when all the devices have same memory size (available_mem_bytes).\n        In the future, devices with different mem sizes will be supported like size_based_partition.\n        It first traverse all the nodes and do the partitions based on the same memory size.\n        If the current partition has no enough memory left for a new op node\n        (call_module, call_method, call_function), a new partition is created.\n        When crossing the boundary between non-embedding nodes and embedding nodes,\n        a new partition is created regardlessly.\n        For example, if the current node is a non-embedding node but the next node is an\n        embedding node, a new partition is created for the next node.\n        After the partition, the partitions are combined as much as possible.\n        The rule is that a non-embedding partition only\n        combines with another non-embedding one.\n        So as the embedding partitions.\n        '

        def combine_partitions_based_on_size(partitions: List[Partition], available_mem_bytes: int) -> None:
            if False:
                i = 10
                return i + 15
            'Combining small partitions together to keep as less partitions as possible.\n            Here is an example of the algorithm to do this:\n            Assume some partitions, we first sort them based on partition used memory size.\n            [(partition_4, 1), (partition_3, 1), (partition_2, 2), (partition_1, 7), (partition_0, 9)]\n            The available memory is 10.\n            step 1: self.find_partition_to_combine_based_on_size()\n            First, mark bfs level for each partition\n            Second, look the smallest partition, partition_4: 10 - 1 = 9\n            It means any partition has a used memory equal or less than 9 could combine this partition\n            We go from the largest and selection partition_0.\n            Check the bfs level for two partitions, if the level difference is less than 2,\n            it can be combined.\n            step 2: repeat step 1 until no partitions can be combined\n            '
            find_combination = True
            while find_combination:
                sorted_partitions = sorted(partitions, key=lambda p: p.used_mem_bytes)
                get_bfs_level_partition(self.partitions)
                (find_combination, partitions) = find_partition_to_combine_based_on_size(sorted_partitions, available_mem_bytes, partitions)
            return

        def calculate_mem_bytes_needed(p1, p2):
            if False:
                for i in range(10):
                    print('nop')
            'Given two partitions, calculate how many mem bytes\n            are needed if two partitions are combined\n            '
            nodes = p1.nodes.union(p2.nodes)
            mem_bytes_needed = 0
            for node in nodes:
                mem_bytes_needed += get_extra_size_of(node, nodes)
            return mem_bytes_needed

        def find_partition_to_combine_based_on_size(sorted_partitions: List[Partition], available_mem_bytes: int, partitions: List[Partition]) -> Tuple[bool, List[Partition]]:
            if False:
                i = 10
                return i + 15
            'step 1 in combine_partition_based_on_size()'
            find_combination = False
            smallest_partition = sorted_partitions.pop(0)
            for p in sorted_partitions[::-1]:
                if abs(smallest_partition.bfs_level - p.bfs_level) <= 1:
                    mem_bytes_needed = calculate_mem_bytes_needed(p, smallest_partition)
                    if mem_bytes_needed <= available_mem_bytes:
                        combine_two_partitions(p, smallest_partition, self.partitions)
                        partitions.remove(smallest_partition)
                        partitions.remove(p)
                        partitions.append(self.partitions[-1])
                        find_combination = True
                        break
            return (find_combination, partitions)

        def reset_partition_in_sparse_nn(partition, new_partition=True):
            if False:
                for i in range(10):
                    print('nop')
            'If crossing the boundary between non-embedding nodes and\n            embedding nodes, create a new partition\n            '
            if in_embedding_region:
                embedding_partitions.append(partition)
            else:
                non_embedding_partitions.append(partition)
            if new_partition:
                partition = self.create_partition()
                partition.left_mem_bytes = available_mem_bytes
                return partition
            return None

        def is_embedding_node(node: Node) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            'Check if a node is an embedding node'
            if node.op == 'call_module':
                submodule = self.graph_module
                for atom in str(node.target).split('.'):
                    if not hasattr(submodule, atom):
                        raise RuntimeError(f'Module {submodule} has no attribute {atom}')
                    submodule = getattr(submodule, atom)
                    if 'Embedding' in str(submodule):
                        return True
            return False
        embedding_partitions: List[Partition] = []
        non_embedding_partitions: List[Partition] = []
        in_embedding_region: bool = False
        partition = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op in {'call_module', 'call_method', 'call_function'}:
                if is_embedding_node(node) != in_embedding_region:
                    if partition.used_mem_bytes != 0:
                        partition = reset_partition_in_sparse_nn(partition)
                    in_embedding_region = not in_embedding_region
                total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                if total_size_of_input_nodes + partition.used_mem_bytes > available_mem_bytes:
                    partition = reset_partition_in_sparse_nn(partition)
                    total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                    if total_size_of_input_nodes > available_mem_bytes:
                        raise RuntimeError(node.target + 'is too large to fit into a device')
                partition.add_node(node)
        reset_partition_in_sparse_nn(partition, new_partition=False)
        set_parents_and_children(self.partitions)
        combine_partitions_based_on_size(non_embedding_partitions, available_mem_bytes)
        combine_partitions_based_on_size(embedding_partitions, available_mem_bytes)
        total_size_of_non_embedding_partitions = 0
        for partition in non_embedding_partitions:
            total_size_of_non_embedding_partitions += partition.used_mem_bytes
        if len(embedding_partitions) > len(self.devices):
            msg = 'Need ' + str(len(embedding_partitions)) + ' devices, but only ' + str(len(self.devices)) + ' provided'
            raise RuntimeError(msg)
        occupied_devices = []
        for (i, partition) in enumerate(embedding_partitions):
            if total_size_of_non_embedding_partitions + partition.used_mem_bytes > available_mem_bytes:
                raise RuntimeError('partition_' + str(partition.partition_id) + '(embedding partition) and non embedding partitions can not fit into one device')
            else:
                partition.logical_device_ids = [self.devices[i].logical_id]
                occupied_devices.append(self.devices[i].logical_id)
        for partition in non_embedding_partitions:
            partition.logical_device_ids = occupied_devices
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        return

    def cost_aware_partition(self, transfer_rate_bytes_per_sec: float, node_to_latency_mapping: Dict[Node, NodeLatency]) -> None:
        if False:
            return 10
        'This method is to partition the fx module based on the cost.\n        The cost is the total latency of running the whole fx module.\n        In partitioner_utils.py, the cost model is built.\n        The cost aware partition algorithm is:\n        #1. At every beginning, each node is a partition.\n            Then we map all the partitions to the devices\n            and calculate the cost\n        #2. Then try to pre-combine any two of the partitions if the two\n            partitions can be combined.\n            (the bfs level is less than 2 or two partitions are connected and\n            can find partition to device mapping)\n            See if any partition pair could reduce the current cost.\n            Choose the pair that shows the minimum cost and then combine them\n        #3. Repeat #2 until the cost cannot be reduced.\n        '

        def try_combining_partitions(p0_index, p1_index, partitions) -> float:
            if False:
                print('Hello World!')
            'Given two partitions and a list of partitions, combine these two partitions\n            and see what is the cost of the modified partition list\n            '
            p0 = partitions[p0_index]
            p1 = partitions[p1_index]
            "If two partitions' bfs level are less than 2 or two partitions are connected to each other,\n               then they can be combined\n            "
            if abs(p0.bfs_level - p1.bfs_level) <= 1 or p0 in p1.parents or p0 in p1.children:
                combine_two_partitions(p0, p1, partitions)
                if check_dependency(partitions[-1]):
                    return float('inf')
                reset_partition_device(partitions)
                found_deivce = get_device_to_partitions_mapping(partitions, self.devices)
                if not found_deivce:
                    return float('inf')
                partition_to_latency_mapping = get_partition_to_latency_mapping(partitions, node_to_latency_mapping)
                cost = get_latency_of_partitioned_graph(partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec)
                return cost
            return float('inf')

        def search_combination(transfer_rate_bytes_per_sec, node_to_latency_mapping) -> bool:
            if False:
                return 10
            "Given transfer rate between partitions and each node's latency,\n            find two partitions to combine so the cost of the partitions can\n            be reduced.\n            The algorithm is :\n            1. Go through all the partition pairs and see\n            if any pair of partitions can be combined.\n            2. Calculate the cost after the combination.\n            3. Select the minimum cost and combine its corresponding partition pair.\n            "
            partition_to_latency_mapping = get_partition_to_latency_mapping(self.partitions, node_to_latency_mapping)
            cost = get_latency_of_partitioned_graph(self.partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec)
            if len(self.partitions) == 1:
                return False
            partition_pair: List[int] = []
            for i in range(len(self.partitions) - 1):
                for j in range(i + 1, len(self.partitions)):
                    new_cost = try_combining_partitions(i, j, self.partitions[:])
                    if new_cost <= cost:
                        partition_pair = [i, j]
                        cost = new_cost
                    reorganize_partitions(self.partitions)
            if len(partition_pair) != 0:
                p0 = self.partitions[partition_pair[0]]
                p1 = self.partitions[partition_pair[1]]
                combine_two_partitions(p0, p1, self.partitions)
            get_bfs_level_partition(self.partitions)
            reset_partition_device(self.partitions)
            get_device_to_partitions_mapping(self.partitions, self.devices)
            return len(partition_pair) != 0
        for node in self.graph_module.graph.nodes:
            if node.op not in {'placeholder', 'get_attr', 'output'}:
                self.create_single_node_partition(node)
        set_parents_and_children(self.partitions)
        get_bfs_level_partition(self.partitions)
        find_combination = True
        while find_combination:
            find_combination = search_combination(transfer_rate_bytes_per_sec, node_to_latency_mapping)
        reorganize_partitions(self.partitions)
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        return

    def kl_based_partition(self, transfer_rate_bytes_per_sec: float, node_to_latency_mapping: Dict[Node, NodeLatency]) -> None:
        if False:
            i = 10
            return i + 15
        'This function is a cost aware partition based\n        on Kernighan-Lin algorithm.\n        First, the graph is partitioned using size_based_partition.\n        Then, each node is swapped with any other node in a different\n        partition, and at the same time, the cost is estimated after\n        the swapping.\n        For example, we have nodes n0, n1, n2, n3 and n4.\n        Using size_based_partition, n0 and n1 are in Partition p0.\n        n2, n3 and n4 in Partition p1. The current cost is estimated.\n        We first tried using n0 to swap with n2 from the other partition.\n        Then we see that swapping n0 and n2 shows a lower cost\n        than the current cost and it is the minimum among other pairs like\n        (n0, None)(This means moving n0 to Partition without swapping other nodes),\n        (n0, n3) and (n0, n4). We swap n0 and n2 and set the new cost\n        as the current cost.\n        Then We repeat this process for all the other nodes until all swapping pairs\n        are tried.\n        '

        def swap_nodes(n0, n1, p0, p1):
            if False:
                return 10
            if n0 is not None:
                p0.remove_node(n0)
                p1.add_node(n0)
            if n1 is not None:
                p0.add_node(n1)
                p1.remove_node(n1)

        def try_swap_nodes(n0, n1, p0, p1, node_to_latency_mapping, transfer_rate_per_sec):
            if False:
                print('Hello World!')
            cost = float('inf')
            swap_nodes(n0, n1, p0, p1)
            reorganize_partitions(self.partitions)
            if not check_dependency(p0) and (not check_dependency(p1)):
                reset_partition_device(self.partitions)
                partition_to_latency_mapping = get_partition_to_latency_mapping(self.partitions, node_to_latency_mapping)
                found_device = get_device_to_partitions_mapping(self.partitions, self.devices)
                if not found_device:
                    cost = float('inf')
                else:
                    cost = get_latency_of_partitioned_graph(self.partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec)
            swap_nodes(n1, n0, p0, p1)
            reorganize_partitions(self.partitions)
            reset_partition_device(self.partitions)
            get_device_to_partitions_mapping(self.partitions, self.devices)
            return cost

        def swap_node_to_partition(node, p0, p1, node_to_latency_mapping, transfer_rate_per_sec):
            if False:
                while True:
                    i = 10
            'This function helps to swap one node from partition p0\n            with all the nodes in another partition p1\n            '
            p1_nodes = list(p1.nodes) + [None]
            min_cost = float('inf')
            node_pair: List[Node] = []
            for n1 in p1_nodes:
                if n1 is not None and n1.op in {'placeholder', 'get_attr'}:
                    continue
                cost = try_swap_nodes(node, n1, p0, p1, node_to_latency_mapping, transfer_rate_per_sec)
                if cost < min_cost:
                    node_pair = [node, n1]
                    min_cost = cost
            return (cost, node_pair)
        self.size_based_partition()
        partition_to_latency_mapping = get_partition_to_latency_mapping(self.partitions, node_to_latency_mapping)
        cost = get_latency_of_partitioned_graph(self.partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec)
        node_pair: List[Node] = []
        partition_pair: List[Partition] = []
        op_nodes = []
        for n in self.graph_module.graph.nodes:
            if n.op not in {'placeholder', 'get_attr', 'output'}:
                op_nodes.append(n)
        for node in op_nodes:
            p0_index = self.node_to_partition[node]
            p0 = self.partitions[p0_index]
            for (p1_index, _) in enumerate(self.partitions):
                if p0_index != p1_index:
                    p1 = self.partitions[p1_index]
                    (new_cost, new_node_pair) = swap_node_to_partition(node, p0, p1, node_to_latency_mapping, transfer_rate_bytes_per_sec)
                    if new_cost < cost:
                        cost = new_cost
                        node_pair = new_node_pair
                        partition_pair = [p0, p1]
            if len(node_pair) != 0:
                swap_nodes(node_pair[0], node_pair[1], partition_pair[0], partition_pair[1])
                reorganize_partitions(self.partitions)
                get_device_to_partitions_mapping(self.partitions, self.devices)
        reorganize_partitions(self.partitions)
        get_device_to_partitions_mapping(self.partitions, self.devices)
        return

    def aot_based_partition(self, node_to_partition_mapping, partition_to_logical_device_mapping):
        if False:
            print('Hello World!')
        'This function helps to rebuild the partitions given the nodes and its\n        corresponding partition id\n        '
        partition_id_to_partition_mapping: Dict[int, Partition] = {}
        self.node_to_partition = node_to_partition_mapping
        for node in self.node_to_partition:
            partition_id = self.node_to_partition[node]
            if partition_id not in partition_id_to_partition_mapping:
                partition = Partition(partition_id)
                self.partitions.append(partition)
                partition_id_to_partition_mapping[partition_id] = partition
                partition.logical_device_ids = partition_to_logical_device_mapping[partition_id]
            else:
                partition = partition_id_to_partition_mapping[self.node_to_partition[node]]
            partition.add_node(node)