from typing import List, Set, Tuple, Dict
import numpy
from allennlp.common.checks import ConfigurationError

def decode_mst(energy: numpy.ndarray, length: int, has_labels: bool=True) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if False:
        while True:
            i = 10
    '\n    Note: Counter to typical intuition, this function decodes the _maximum_\n    spanning tree.\n\n    Decode the optimal MST tree with the Chu-Liu-Edmonds algorithm for\n    maximum spanning arborescences on graphs.\n\n    # Parameters\n\n    energy : `numpy.ndarray`, required.\n        A tensor with shape (num_labels, timesteps, timesteps)\n        containing the energy of each edge. If has_labels is `False`,\n        the tensor should have shape (timesteps, timesteps) instead.\n    length : `int`, required.\n        The length of this sequence, as the energy may have come\n        from a padded batch.\n    has_labels : `bool`, optional, (default = `True`)\n        Whether the graph has labels or not.\n    '
    if has_labels and energy.ndim != 3:
        raise ConfigurationError('The dimension of the energy array is not equal to 3.')
    elif not has_labels and energy.ndim != 2:
        raise ConfigurationError('The dimension of the energy array is not equal to 2.')
    input_shape = energy.shape
    max_length = input_shape[-1]
    if has_labels:
        energy = energy[:, :length, :length]
        label_id_matrix = energy.argmax(axis=0)
        energy = energy.max(axis=0)
    else:
        energy = energy[:length, :length]
        label_id_matrix = None
    original_score_matrix = energy
    score_matrix = numpy.array(original_score_matrix, copy=True)
    old_input = numpy.zeros([length, length], dtype=numpy.int32)
    old_output = numpy.zeros([length, length], dtype=numpy.int32)
    current_nodes = [True for _ in range(length)]
    representatives: List[Set[int]] = []
    for node1 in range(length):
        original_score_matrix[node1, node1] = 0.0
        score_matrix[node1, node1] = 0.0
        representatives.append({node1})
        for node2 in range(node1 + 1, length):
            old_input[node1, node2] = node1
            old_output[node1, node2] = node2
            old_input[node2, node1] = node2
            old_output[node2, node1] = node1
    final_edges: Dict[int, int] = {}
    chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives)
    heads = numpy.zeros([max_length], numpy.int32)
    if has_labels:
        head_type = numpy.ones([max_length], numpy.int32)
    else:
        head_type = None
    for (child, parent) in final_edges.items():
        heads[child] = parent
        if has_labels:
            head_type[child] = label_id_matrix[parent, child]
    return (heads, head_type)

def chu_liu_edmonds(length: int, score_matrix: numpy.ndarray, current_nodes: List[bool], final_edges: Dict[int, int], old_input: numpy.ndarray, old_output: numpy.ndarray, representatives: List[Set[int]]):
    if False:
        return 10
    "\n    Applies the chu-liu-edmonds algorithm recursively\n    to a graph with edge weights defined by score_matrix.\n\n    Note that this function operates in place, so variables\n    will be modified.\n\n    # Parameters\n\n    length : `int`, required.\n        The number of nodes.\n    score_matrix : `numpy.ndarray`, required.\n        The score matrix representing the scores for pairs\n        of nodes.\n    current_nodes : `List[bool]`, required.\n        The nodes which are representatives in the graph.\n        A representative at it's most basic represents a node,\n        but as the algorithm progresses, individual nodes will\n        represent collapsed cycles in the graph.\n    final_edges : `Dict[int, int]`, required.\n        An empty dictionary which will be populated with the\n        nodes which are connected in the maximum spanning tree.\n    old_input : `numpy.ndarray`, required.\n    old_output : `numpy.ndarray`, required.\n    representatives : `List[Set[int]]`, required.\n        A list containing the nodes that a particular node\n        is representing at this iteration in the graph.\n\n    # Returns\n\n    Nothing - all variables are modified in place.\n\n    "
    parents = [-1]
    for node1 in range(1, length):
        parents.append(0)
        if current_nodes[node1]:
            max_score = score_matrix[0, node1]
            for node2 in range(1, length):
                if node2 == node1 or not current_nodes[node2]:
                    continue
                new_score = score_matrix[node2, node1]
                if new_score > max_score:
                    max_score = new_score
                    parents[node1] = node2
    (has_cycle, cycle) = _find_cycle(parents, length, current_nodes)
    if not has_cycle:
        final_edges[0] = -1
        for node in range(1, length):
            if not current_nodes[node]:
                continue
            parent = old_input[parents[node], node]
            child = old_output[parents[node], node]
            final_edges[child] = parent
        return
    cycle_weight = 0.0
    index = 0
    for node in cycle:
        index += 1
        cycle_weight += score_matrix[parents[node], node]
    cycle_representative = cycle[0]
    for node in range(length):
        if not current_nodes[node] or node in cycle:
            continue
        in_edge_weight = float('-inf')
        in_edge = -1
        out_edge_weight = float('-inf')
        out_edge = -1
        for node_in_cycle in cycle:
            if score_matrix[node_in_cycle, node] > in_edge_weight:
                in_edge_weight = score_matrix[node_in_cycle, node]
                in_edge = node_in_cycle
            score = cycle_weight + score_matrix[node, node_in_cycle] - score_matrix[parents[node_in_cycle], node_in_cycle]
            if score > out_edge_weight:
                out_edge_weight = score
                out_edge = node_in_cycle
        score_matrix[cycle_representative, node] = in_edge_weight
        old_input[cycle_representative, node] = old_input[in_edge, node]
        old_output[cycle_representative, node] = old_output[in_edge, node]
        score_matrix[node, cycle_representative] = out_edge_weight
        old_output[node, cycle_representative] = old_output[node, out_edge]
        old_input[node, cycle_representative] = old_input[node, out_edge]
    considered_representatives: List[Set[int]] = []
    for (i, node_in_cycle) in enumerate(cycle):
        considered_representatives.append(set())
        if i > 0:
            current_nodes[node_in_cycle] = False
        for node in representatives[node_in_cycle]:
            considered_representatives[i].add(node)
            if i > 0:
                representatives[cycle_representative].add(node)
    chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives)
    found = False
    key_node = -1
    for (i, node) in enumerate(cycle):
        for cycle_rep in considered_representatives[i]:
            if cycle_rep in final_edges:
                key_node = node
                found = True
                break
        if found:
            break
    previous = parents[key_node]
    while previous != key_node:
        child = old_output[parents[previous], previous]
        parent = old_input[parents[previous], previous]
        final_edges[child] = parent
        previous = parents[previous]

def _find_cycle(parents: List[int], length: int, current_nodes: List[bool]) -> Tuple[bool, List[int]]:
    if False:
        for i in range(10):
            print('nop')
    added = [False for _ in range(length)]
    added[0] = True
    cycle = set()
    has_cycle = False
    for i in range(1, length):
        if has_cycle:
            break
        if added[i] or not current_nodes[i]:
            continue
        this_cycle = set()
        this_cycle.add(i)
        added[i] = True
        has_cycle = True
        next_node = i
        while parents[next_node] not in this_cycle:
            next_node = parents[next_node]
            if added[next_node]:
                has_cycle = False
                break
            added[next_node] = True
            this_cycle.add(next_node)
        if has_cycle:
            original = next_node
            cycle.add(original)
            next_node = parents[original]
            while next_node != original:
                cycle.add(next_node)
                next_node = parents[next_node]
            break
    return (has_cycle, list(cycle))