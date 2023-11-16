import networkx as nx
import collections
import time
import copy
from caffe2.python import workspace, core
from caffe2.proto import caffe2_pb2
import enum
import logging
import caffe2.python._import_c_extension as C
log = logging.getLogger('memonger')
log.setLevel(logging.INFO)
LiveRange = collections.namedtuple('LiveRange', ['defined', 'used', 'size'])

def share_grad_blobs(net, losses, param_grads, namescope, dont_share_blobs=None, share_activations=False, blob_shapes=None):
    if False:
        return 10
    "\n    Implements similar optimization as Torch's shareGradInput():\n    for the gradients that are passed between layers, share blobs between\n    operators when possible. This yields significant memory savings with\n    deep networks.\n\n    Returns an optimized protobuf (assign to net._net)\n    "

    def is_grad_blob(b):
        if False:
            for i in range(10):
                print('nop')
        name = str(b)
        return name.endswith('_grad') and name.startswith((namescope, '_' + namescope)) and (name not in param_grads)

    def is_grad_op(op):
        if False:
            i = 10
            return i + 15
        for b in list(op.input) + list(op.output):
            if is_grad_blob(b):
                return True
        return False
    log.warn('NOTE: Executing memonger to optimize gradient memory')
    if namescope != '' and (not namescope.endswith('/')):
        namescope += '/'
    netproto = copy.deepcopy(net.Proto())
    activations = []
    external_output = set(net.Proto().external_output)
    for op in net.Proto().op:
        for b in op.output:
            if b + '_w' in op.input and b not in external_output:
                activations.append(b)
    activations = set(activations[:-2])
    grad_op_indices = []
    for (idx, op) in enumerate(netproto.op):
        if is_grad_op(op):
            grad_op_indices.append(idx)
    shared_blobs = set()
    for op in net.Proto().op:
        for b in list(op.input) + list(op.output):
            if is_grad_blob(b) or (share_activations and b in activations):
                shared_blobs.add(b)
    start_time = time.time()
    optim_str = C.memonger_compute_blob_recycling_for_dag(netproto.SerializeToString(), [str(s).encode('utf-8') for s in losses], grad_op_indices, set((str(s).encode('utf-8') for s in shared_blobs)), namescope.encode('utf-8'), set() if dont_share_blobs is None else dont_share_blobs, {} if blob_shapes is None else blob_shapes)
    log.info('Memonger memory optimization took {} secs'.format(time.time() - start_time))
    optim = caffe2_pb2.NetDef()
    optim.ParseFromString(optim_str)
    assert verify_graph_equality(net.Proto(), optim), 'Memonger graph is not equal to original.'
    assert verify_inplace_blobs(net.Proto(), optim), 'Inplace assignments differ in memonger net.'
    return optim

def optimize_inference_for_dag(net, input_blobs, namescope=''):
    if False:
        i = 10
        return i + 15
    netproto = copy.deepcopy(net.Proto())
    external_input = set(net.Proto().external_input)
    external_output = set(net.Proto().external_output)

    def is_activation_blob(b):
        if False:
            while True:
                i = 10
        return b not in external_input and b not in external_output
    activation_blobs = set()
    seen_as_output = set()
    ops = list(net.Proto().op)
    op_indices = [index for (index, op) in enumerate(net.Proto().op)]
    for op in ops:
        for b in op.input:
            if is_activation_blob(b):
                activation_blobs.add(b)
                if b not in seen_as_output:
                    raise AssertionError('{} not in external input'.format(b))
        for b in op.output:
            if is_activation_blob(b):
                activation_blobs.add(b)
        seen_as_output = seen_as_output.union(set(op.output))
        assert not op.is_gradient_op, 'You can only pass inference-only nets to optimize_inference_for_dag'
    start_time = time.time()
    optim_str = C.memonger_compute_blob_recycling_for_dag(netproto.SerializeToString(), [str(s).encode('utf-8') for s in input_blobs], op_indices, set((str(s).encode('utf-8') for s in activation_blobs)), namescope.encode('utf-8'), set(), {})
    log.info('Memonger memory optimization took {} secs'.format(time.time() - start_time))
    optim = caffe2_pb2.NetDef()
    optim.ParseFromString(optim_str)
    assert verify_graph_equality(net.Proto(), optim), 'Memonger graph is not equal to original.'
    assert verify_inplace_blobs(net.Proto(), optim), 'Inplace assignments differ in memonger net.'
    return optim

def estimate_memory_usage(protos, shapes, types, devicescope):
    if False:
        for i in range(10):
            print('nop')
    import numpy as np
    '\n    Estimate memory usage of a model. This is an estimate because\n    we assume a single threaded execution and miss some internal\n    memory usage of operators. Only estimates the memory for a given\n    device scope.\n\n    Also, currently it does not handle correctly if blob sizes vary\n    during execution, as it uses only the final blob size.\n\n    Returns (total, highwater, by op type) memory allocation in bytes.\n    '
    sizeofs = {caffe2_pb2.TensorProto.DOUBLE: 8, caffe2_pb2.TensorProto.FLOAT: 4, caffe2_pb2.TensorProto.FLOAT16: 2, caffe2_pb2.TensorProto.INT32: 4, caffe2_pb2.TensorProto.INT8: 1, caffe2_pb2.TensorProto.UINT8: 1, caffe2_pb2.TensorProto.UINT16: 2, caffe2_pb2.TensorProto.INT16: 2, caffe2_pb2.TensorProto.BOOL: 1, caffe2_pb2.TensorProto.INT64: 8}

    def split_net(proto):
        if False:
            for i in range(10):
                print('nop')
        ops = [op for op in proto.op if op.device_option == devicescope or op.type in {'Free', 'Alias'}]
        del proto.op[:]
        proto.op.extend(ops)
        return proto

    def num_bytes(blob):
        if False:
            while True:
                i = 10
        if blob not in shapes or blob not in types:
            log.warning('Unknown blob encountered: {}'.format(blob))
            return 0
        sizeof = sizeofs[types[blob]]
        return sizeof * np.prod(shapes[blob])
    protos = [split_net(proto) for proto in protos]
    allocs_by_ops = collections.defaultdict(lambda : 0)
    current_allocated = 0
    max_allocated = 0
    total_allocated = 0
    allocated = set()
    for proto in protos:
        for op in proto.op:
            if op.type == 'Free' or op.type == 'Alias':
                for o in op.output:
                    if o in allocated:
                        current_allocated -= num_bytes(o)
                        allocated.remove(o)
            else:
                for output in op.output:
                    if output not in allocated:
                        nbytes = num_bytes(output)
                        total_allocated += nbytes
                        current_allocated += nbytes
                        max_allocated = max(max_allocated, current_allocated)
                        allocated.add(output)
                        allocs_by_ops[op.type] += nbytes
    return (total_allocated, max_allocated, allocs_by_ops)

def release_blobs_when_used(netproto, dont_free_blobs, selector_fun=None):
    if False:
        print('Hello World!')
    '\n    Insert Free-ops after a blob has been used the last time, so that its\n    memory can be reclaimed. Use this only with efficient caching memory\n    managers (such as CUB, --caffe2_cuda_memory_pool=cub).\n\n    Blobs used with Alias op won\'t be freed.\n\n    @dont_free_blobs:  is a set of blobs that should not be freed\n    @selector_fun:     optional lambda that return True if blob name\n                       can be released. Use for easy special filtering, like\n                       excluding blobs with "loss" in the name.\n\n    Returns a new protobuffer. To use with a model, use:\n        model.net._net = memonger.release_blobs_when_used(..)\n    '
    input_blobs = set()
    can_release = set()
    alias_blobs = set()
    netproto = copy.deepcopy(netproto)
    for op in netproto.op:
        if op.type == 'Alias':
            alias_blobs.add(op.input[0])
            continue
        for inp in op.input:
            input_blobs.add(inp)
        for outp in op.output:
            if outp not in input_blobs:
                if selector_fun is None or selector_fun(outp):
                    can_release.add(outp)
    can_release = can_release - set(netproto.external_output)
    can_release = can_release.intersection(input_blobs)
    can_release = can_release - dont_free_blobs
    can_release = can_release - alias_blobs
    ops = list(netproto.op)
    for j in reversed(range(0, len(netproto.op))):
        op = netproto.op[j]
        for inp in op.input:
            if inp in can_release:
                can_release.remove(inp)
                ops.insert(j + 1, core.CreateOperator('Free', [inp], [inp]))
    del netproto.op[:]
    netproto.op.extend(ops)
    return netproto

def _find_source_nodes(g):
    if False:
        return 10
    ' Return nodes without predecessors '
    ret = []
    for cn in g:
        cur_pred = list(g.predecessors(cn))
        if not cur_pred:
            ret.append(cn)
    return ret

def _find_target_nodes(g):
    if False:
        for i in range(10):
            print('nop')
    ' Return nodes without successors '
    ret = []
    for cn in g:
        cur_succ = list(g.successors(cn))
        if not cur_succ:
            ret.append(cn)
    return ret

def _add_single_target_ifneeded(g):
    if False:
        while True:
            i = 10
    targets = _find_target_nodes(g)
    assert len(targets) >= 1
    if len(targets) == 1:
        return g
    ret = copy.deepcopy(g)

    def _next_available_idx(g):
        if False:
            i = 10
            return i + 15
        ret = -1
        for cn in g:
            if cn > ret:
                ret = cn
        ret += 1
        return ret
    target_node_idx = _next_available_idx(g)
    ret.add_node(target_node_idx)
    for cn in targets:
        ret.add_edge(cn, target_node_idx)
    return ret

def _get_path(pred_list, dist_list):
    if False:
        i = 10
        return i + 15
    " Get the path from nx.bellman_ford()'s output "
    assert all((dist_list[x] <= 0 for x in dist_list))
    target = min(dist_list, key=lambda x: dist_list[x])
    ret = []
    cur = target
    while cur is not None:
        ret.append(cur)
        try:
            cur = pred_list[cur][0] if pred_list[cur] else None
        except TypeError:
            cur = pred_list[cur]
    return list(reversed(ret))

def _get_longest_paths(g, source_nodes):
    if False:
        i = 10
        return i + 15
    " Get the longest path for nodes in 'source_nodes'\n        Find with bellman_ford() by setting weight = -1\n    "
    ng = copy.deepcopy(g)
    for (u, v) in ng.edges():
        ng[u][v]['weight'] = -1
    ret = {}
    for cn in source_nodes:
        (pred, dist) = nx.bellman_ford_predecessor_and_distance(ng, cn, weight='weight')
        path = _get_path(pred, dist)
        assert path[0] == cn
        assert len(path) - 1 == -dist[path[-1]]
        ret[cn] = path
    return ret

def _build_tree(paths):
    if False:
        return 10
    ' Build a tree for given paths based on common elements.\n        Last elements of all paths are the same, which is the root of the tree.\n    '
    assert all((cp[-1] == paths[0][-1] for cp in paths))
    g = nx.DiGraph()
    node_set = {y for x in paths for y in x}
    g.add_nodes_from(node_set)
    for cp in paths:
        for ce in zip(cp[0:-1], cp[1:]):
            g.add_edge(ce[1], ce[0])
    root = paths[0][-1]
    _compute_tree_height(g, root)
    return (g, root)

def _compute_tree_height(g, root):
    if False:
        for i in range(10):
            print('nop')
    ' Compute the heights of the tree for all nodes\n        Height of leaves are 0\n    '

    def _get_height(root):
        if False:
            print('Hello World!')
        children = list(g.successors(root))
        height = 0
        if children:
            child_heights = [_get_height(x) for x in children]
            height = max(child_heights) + 1
        g.nodes[root]['height'] = height
        return height
    _get_height(root)

def _sort_tree_leaves(g, root):
    if False:
        i = 10
        return i + 15
    ' For each node, sort its child nodes based on the height of the nodes.\n        Return the leaf nodes of the tree after sorting.\n    '

    def _get_height(root):
        if False:
            while True:
                i = 10
        return g.nodes[root]['height']

    def _get_sorted_leaves(root):
        if False:
            while True:
                i = 10
        children = list(g.successors(root))
        if not children:
            return [root]
        child_heights = [_get_height(x) for x in children]
        order = sorted(range(len(children)), key=lambda x: child_heights[x])
        ret = []
        for co in order:
            cr = children[co]
            ret += _get_sorted_leaves(cr)
        return ret
    return _get_sorted_leaves(root)

def topological_sort_traversal_longest_path(g):
    if False:
        i = 10
        return i + 15
    " The graph 'g' may contain several source nodes (nodes without incoming\n        edge), which could be in any order and still be a valid\n        topological sorting result. We would like to arrange these source nodes\n        so that the average live spans of the computed blobs are shorter.\n        The idea is to sort the source nodes based on the length of their path to\n        the target node so that the one with longer path is used first.\n        This is done by:\n        - Add a single target node if there are multiple target nodes in 'g'.\n        - Find the longest path between each source and the target node.\n        - Convert the longest paths to a tree with the target node being the root\n          and source nodes being the leaves.\n        - Sort the nodes of the tree based on the height of the tree.\n    "
    gt = _add_single_target_ifneeded(g)
    source_nodes = _find_source_nodes(gt)
    lpaths = _get_longest_paths(gt, source_nodes)
    (tree, root) = _build_tree(list(lpaths.values()))
    sorted_sources = _sort_tree_leaves(tree, root)
    assert sorted(sorted_sources) == sorted(source_nodes)
    if nx.__version__ < '2.0':
        ret = nx.topological_sort(g, sorted_sources)
    else:
        dependency_order = list(sorted_sources)
        seen_nodes = set(sorted_sources)
        for s in sorted_sources:
            desc = nx.descendants(g, s)
            for d in desc:
                if d not in seen_nodes:
                    seen_nodes.add(d)
                    dependency_order.append(d)
        sort_key = dict(((v, len(dependency_order) - i) for (i, v) in enumerate(dependency_order)))
        ret = nx.algorithms.dag.lexicographical_topological_sort(g, key=lambda x: sort_key[x])
        ret = list(ret)
    assert len(ret) == len(g.nodes)
    return ret

def topological_sort_traversal(g):
    if False:
        for i in range(10):
            print('nop')
    return list(nx.topological_sort(g))

def compute_ranges(linearized_ops, blob_sizes=None):
    if False:
        for i in range(10):
            print('nop')
    if not blob_sizes:
        log.warning('Provide blob sizes to get more accurate assignments.')
    blobs = collections.defaultdict(lambda : LiveRange(defined=None, used=None, size=None))
    for (i, op) in enumerate(linearized_ops):
        for blob in op.input:
            used = blobs[blob].used
            if used is None:
                used = i
            else:
                used = max(used, i)
            blobs[blob] = blobs[blob]._replace(used=used)
            blob_size = blob_sizes[blob] if blob_sizes else None
            assert not blob_sizes or blob_size is not None
            blobs[blob] = blobs[blob]._replace(size=blob_size)
        for blob in op.output:
            defined = blobs[blob].defined
            if defined is None:
                defined = i
            else:
                defined = min(defined, i)
            blobs[blob] = blobs[blob]._replace(defined=defined)
            blob_size = blob_sizes[blob] if blob_sizes else None
            assert not blob_sizes or blob_size is not None
            blobs[blob] = blobs[blob]._replace(size=blob_size)
    return blobs

def is_compatible(candidate_range, assignment, static_blobs):
    if False:
        i = 10
        return i + 15
    (name, range_) = assignment[-1]
    if name in static_blobs:
        return False
    if candidate_range.defined is None or range_.defined is None or range_.used is None:
        return False
    return candidate_range.defined > range_.used

def compute_blob_assignments(assignments):
    if False:
        print('Hello World!')
    blob_assignments = {}
    for assignment in assignments:
        if len(assignment) == 1:
            continue
        (last_blob, _) = assignment[-1]
        for (blob, _) in assignment:
            blob_assignments[blob] = last_blob
    return blob_assignments

def _get_max_size(assignment):
    if False:
        i = 10
        return i + 15
    if not assignment:
        return 0
    ret = max([x[1].size for x in assignment])
    ret = 0 if ret is None else ret
    return ret

def get_memory_usage(assignments):
    if False:
        print('Hello World!')
    ret = 0
    for cur in assignments:
        ret += _get_max_size(cur)
    return ret

def compute_assignments_greedy(ranges_sorted, init_assignments=None):
    if False:
        i = 10
        return i + 15
    assignments = init_assignments or []
    visited = {y[0] for x in assignments for y in x}
    for (name, range_) in ranges_sorted:
        if name in visited:
            continue
        assigned = False
        best_assignment = 0
        min_dist = float('inf')
        candidate_size = range_.size or 0
        for (idx, assignment) in enumerate(assignments):
            if is_compatible(range_, assignment, []):
                assigned = True
                dist = abs(_get_max_size(assignment) - candidate_size)
                if dist < min_dist:
                    min_dist = dist
                    best_assignment = idx
        if assigned:
            assignment = assignments[best_assignment]
            assignment.append((name, range_))
        else:
            assignments.append([(name, range_)])
    return assignments

def _get_count(assignments):
    if False:
        for i in range(10):
            print('nop')
    ' Return number of blobs in assignments '
    if assignments:
        return sum([len(x) for x in assignments])
    return 0

def compute_assignments_dp(ranges_sorted, init_assignment, counter=None):
    if False:
        return 10
    " Compute assignment for blobs in 'ranges_sorted' on top of 'init_assignment'\n        using dynamic programming + recursion.\n\n        ranges_sorted: blobs sorted by 'used'\n        init_assignment: assignment to start with, blobs in 'ranges_sorted' should\n                         not be used in 'init_assignment'\n\n        Using f(b, k, init) to represent the best assignment for blobs b[0:k]\n        given initial assignment 'init', we have\n            f(b, k, init) = f(b, j, init) +\n                            find_best(b[j:k], f(b, j, init))\n        where j is the index of the last best assignment that is independent of\n        blob b[k - 1] (b[k - 1] is compatible with all assignments in\n        f(b, j, init)), and find_best(b1, init1) gives the best assignment\n        for blobs in 'b1' based on the initial assignment 'init1', and blobs\n        b1[0:-1] should be incompatible with b1[-1]. f(b, len(b), []) gives\n        the best assignment for blobs 'b'.\n\n        For find_best(b, init), since b[0:-1] are not compatible with b[-1], we\n        could reduce it to a smaller problem to find best assignment for b[0:-1]\n        as\n            find_best(b, init) = min {\n                f(b[0:-1], len(b) - 1, init - x) + [x, b[-1]] for x in init, or\n                f(b[0:-1], len(b) - 1, init) + [b[-1]]\n            }\n        where min{} gives the assignment with minimum memory usage.\n    "

    def _get_compatible_prev(candidate_range, best_assignments, cur_idx):
        if False:
            print('Hello World!')
        ' Find closest position k of best_assignments that is independent of\n            candidate_range that candiate_range is compatible with all assignments\n            in best_assignments[k].\n            Return -1 if not found.\n        '

        def is_compatible_all(candidate_range, assignments):
            if False:
                while True:
                    i = 10
            ' return true if compatible for all assignments in assignments '
            return all([is_compatible(candidate_range[1], x, []) for x in assignments])
        ii = cur_idx - 1
        while ii >= 0:
            cba = best_assignments[ii]
            if is_compatible_all(candidate_range, cba):
                return ii
            ii -= 1
        return -1

    def _find_best(ranges, init_assignment, prev_best_assignment, counter):
        if False:
            while True:
                i = 10
        " Find the best assignment for blobs 'ranges' given an initialized\n            assignment 'init_assignment'.\n\n            Blobs in ranges[0:-1] should be incompatible with blob range[-1].\n            'prev_best_assignment': best assignment for blobs in ranges[:-1]\n\n            By assigning ranges[-1] to each assignment k in 'init_assignment' or\n            in a new assignment, the problem becomes a smaller problem to find\n            the best assignment for ranges[0:-1] given the initial assignment\n            init_assigment[0:k, (k+1):-1].\n        "
        find_range = ranges[-1]
        assert all((not is_compatible(x[1], [find_range], []) for x in ranges[0:-1]))
        sz = len(init_assignment)
        best_candidates = []
        for ii in range(sz):
            if not is_compatible(find_range[1], init_assignment[ii], []):
                continue
            cur_best = copy.deepcopy(init_assignment)
            cur_best[ii].append(find_range)
            if len(ranges) > 1:
                cur_best_tmp = [x for (i, x) in enumerate(cur_best) if i != ii]
                cur_best_tmp = compute_assignments_dp(ranges[:-1], cur_best_tmp, counter)
                cur_best = cur_best_tmp + [cur_best[ii]]
            best_candidates.append(cur_best)
        best_candidates.append(prev_best_assignment + [[find_range]])
        ret = min(best_candidates, key=get_memory_usage)
        return ret
    if not counter:
        counter = [0]
    counter[0] += 1
    if counter and counter[0] % 5000 == 0:
        rs = [ranges_sorted[0][1].defined, ranges_sorted[-1][1].used]
        log.info('Finding assignments {} ({} -> {})...'.format(counter[0], rs[0], rs[1]))
    init_assignment = init_assignment or []
    best_assignments = []
    for (ii, cur_range) in enumerate(ranges_sorted):
        prev_idx = _get_compatible_prev(cur_range, best_assignments, ii)
        prev_best = copy.deepcopy(init_assignment) if prev_idx < 0 else copy.deepcopy(best_assignments[prev_idx])
        ranges_part = ranges_sorted[prev_idx + 1:ii + 1]
        cur_best = _find_best(ranges_part, prev_best, best_assignments[-1] if best_assignments else init_assignment, counter)
        assert _get_count(cur_best) == _get_count(prev_best) + len(ranges_part)
        best_assignments.append(copy.deepcopy(cur_best))
    assert len(best_assignments) == len(ranges_sorted)
    best = best_assignments[-1]
    return best

def get_updated_ranges(ranges, max_live=None):
    if False:
        while True:
            i = 10
    ' Set LiveRange.defined = -1 if it is None\n        Set LiveRange.used = max_live if it is None\n        Set LiveRanee.size = 1 if it is None\n    '

    def _get_max_live(ranges):
        if False:
            while True:
                i = 10
        max_live = max((x[1].used for x in ranges if x[1].used)) + 1
        return max_live

    def _update_range(x, max_live, size):
        if False:
            return 10
        cx = x
        if x[1].defined is None:
            cx = (cx[0], cx[1]._replace(defined=-1))
        if x[1].used is None:
            cx = (cx[0], cx[1]._replace(used=max_live))
        if x[1].size is None:
            cx = (cx[0], cx[1]._replace(size=size))
        return cx
    if max_live is None:
        max_live = _get_max_live(ranges)
    ranges = [_update_range(x, max_live, 1) for x in ranges]
    return ranges

def compute_assignments(ranges, static_blobs, algo):
    if False:
        i = 10
        return i + 15
    "\n    algo: Method used to find assignments (AssignmentAlgorithm.GREEDY or\n          AssignmentAlgorithm.DYNAMIC_PROGRAMMING).\n          AssignmentAlgorithm.DYNAMIC_PROGRAMMING gives optimal solution at the\n          cost of more computation.\n          AssignmentAlgorithm.GREEDY may be better in the case 'blob_sizes' is\n          not provided.\n    "
    ranges = sorted(ranges.items(), key=lambda p: (p[1].used is None, p[1].used))
    ranges = get_updated_ranges(ranges)
    ranges_sharable = [x for x in ranges if x[0] not in static_blobs]
    ranges_static = [x for x in ranges if x[0] in static_blobs]
    log.info('Total sharable blobs {}'.format(len(ranges_sharable)))
    best_assignment = []
    if algo == AssignmentAlgorithm.DYNAMIC_PROGRAMMING:
        best_assignment = compute_assignments_dp(ranges_sharable, [])
    elif algo == AssignmentAlgorithm.GREEDY:
        best_assignment = compute_assignments_greedy(ranges_sharable, [])
    else:
        assert 'Invalid algo name {}'.format(algo)
    best_assignment += [[x] for x in ranges_static]
    return best_assignment

def verify_assignments(assignments):
    if False:
        i = 10
        return i + 15
    for cur in assignments:
        for (x, y) in zip(cur[0:-1], cur[1:]):
            assert x[1].used < y[1].defined

def compute_interference_graph(ops):
    if False:
        while True:
            i = 10
    g = nx.DiGraph()
    for (i, op) in enumerate(ops):
        g.add_node(i, op=op)
    for (i, parent_op) in enumerate(ops):
        for (j, child_op) in enumerate(ops):
            if i >= j:
                continue
            if any((output in child_op.input for output in parent_op.output)):
                deps = set(child_op.input).intersection(parent_op.output)
                g.add_edge(i, j, deps=deps)
                assert nx.is_directed_acyclic_graph(g), child_op
    return g
Optimization = collections.namedtuple('Optimization', ['net', 'assignments', 'blob_assignments'])

def apply_assignments(net, blob_assignments):
    if False:
        while True:
            i = 10

    def canonical_name(blob):
        if False:
            while True:
                i = 10
        if blob not in blob_assignments:
            return blob
        return blob_assignments[blob]
    for op in net.op:
        if op.type.startswith('RecurrentNetwork'):
            apply_recurrent_blob_assignments(op, blob_assignments, canonical_name)
        for (i, input_) in enumerate(op.input):
            op.input[i] = canonical_name(input_)
        for (i, output) in enumerate(op.output):
            op.output[i] = canonical_name(output)

def apply_recurrent_blob_assignments(op, blob_assignments, canonical_name):
    if False:
        while True:
            i = 10
    log.debug('Applying assignments to recurrent op: {}'.format(op.type))
    alias_dst_args = [a for a in op.arg if a.name.endswith('alias_dst')]
    for alias_dst in alias_dst_args:
        for (i, blob) in enumerate(alias_dst.strings):
            alias_dst.strings[i] = canonical_name(blob.decode()).encode()
    link_external_args = [a for a in op.arg if a.name.endswith('link_external')]
    for link_external in link_external_args:
        for (i, blob) in enumerate(link_external.strings):
            link_external.strings[i] = canonical_name(blob.decode()).encode()
    step_args = [a for a in op.arg if a.name.endswith('step_net')]
    for step_arg in step_args:
        apply_assignments(step_arg.n, blob_assignments)
        for (i, einp) in enumerate(step_arg.n.external_input):
            if einp in blob_assignments:
                step_arg.n.external_input[i] = canonical_name(einp)
    for (blob, renamed) in blob_assignments.items():
        if blob in list(op.input) + list(op.output):
            a = caffe2_pb2.Argument()
            a.name = blob + '.rename'
            a.s = str(renamed).encode('ascii')
            op.arg.extend([a])

class AssignmentAlgorithm(enum.Enum):
    GREEDY = 0
    DYNAMIC_PROGRAMMING = 1

def optimize_inference_fast(net, static_blobs):
    if False:
        i = 10
        return i + 15
    optim = caffe2_pb2.NetDef()
    optim_str = C.memonger_optimize_inference_net(net.SerializeToString(), [str(s).encode('utf-8') for s in static_blobs])
    optim.ParseFromString(optim_str)
    return optim

def optimize_interference(net, static_blobs, ordering_function=topological_sort_traversal, blob_sizes=None, algo=AssignmentAlgorithm.GREEDY):
    if False:
        for i in range(10):
            print('nop')
    "\n    ordering_function: topological_sort_traversal or\n                       topological_sort_traversal_longest_path.\n                       topological_sort_traversal_longest_path gives better\n                       results but needs a bit more computation.\n    algo: Method used to find assignments (AssignmentAlgorithm.GREEDY or\n          AssignmentAlgorithm.DYNAMIC_PROGRAMMING).\n          AssignmentAlgorithm.DYNAMIC_PROGRAMMING gives optimal solution at the\n          cost of more computation.\n          AssignmentAlgorithm.GREEDY may be better in the case 'blob_sizes' is\n          not provided.\n    "
    '\n    1) Use a BFS traversal of the execution graph to generate an\n       ordering of the node executions.\n    2) Generate use-def ranges for each `blob` in the BFS traversal\n       order.\n    3) Assign blobs to `canonical blobs`\n    4) Rename blobs to canonical blobs\n    '
    net = copy.deepcopy(net)
    g = compute_interference_graph(net.op)
    ordering = ordering_function(g)
    linearized_ops = [net.op[i] for i in ordering]
    del net.op[:]
    net.op.extend(linearized_ops)
    ranges = compute_ranges(linearized_ops, blob_sizes)
    assignments = compute_assignments(ranges, static_blobs, algo)
    blob_assignments = compute_blob_assignments(assignments)
    apply_assignments(net, blob_assignments)
    return Optimization(net=net, blob_assignments=blob_assignments, assignments=assignments)

def verify_inplace_blobs(net_a, net_b):
    if False:
        return 10
    '\n    Verifies that net_a and net_b have the same in-place blob assignments.\n    Particularly, that memonger did not add an in-place assignment when that\n    did not exist before.\n    '

    def get_inplaces(op):
        if False:
            return 10
        out = list(op.output)
        inplaces = []
        for (j, inp) in enumerate(op.input):
            if inp in out:
                inplaces.append([j, out.index(inp)])
        return inplaces
    for (op_a, op_b) in zip(net_a.op, net_b.op):
        if op_a.type != op_b.type:
            return False
        if get_inplaces(op_a) != get_inplaces(op_b):
            return False
    return True

def verify_graph_equality(net_a, net_b):
    if False:
        print('Hello World!')
    '\n    Determines if the execution of two graphs are identical.\n    That is, all inputs blobs are mapped to the same output blobs\n    for each operator in their respective positions.\n\n    This is meant to check the output of memonger with the original graph.\n    It assumes that the nets have same external input and output.\n\n    O(E) runtime + O(1) amortized cost to hash for python dict\n    '

    def parent_list(ops):
        if False:
            while True:
                i = 10
        parent_list = [[] for _ in ops]
        edge_owner = {}
        for (i, op) in enumerate(ops):
            for blob in op.input:
                parent_id = edge_owner.get(blob)
                if parent_id is not None:
                    parent_list[i].append(parent_id)
            for blob in op.output:
                edge_owner[blob] = i
        return parent_list
    if len(net_a.op) != len(net_b.op):
        return False
    for (op_a, op_b) in zip(net_a.op, net_b.op):
        if op_a.type != op_b.type or op_a.device_option != op_b.device_option or op_a.engine != op_b.engine:
            return False
    parent_list_a = parent_list(net_a.op)
    parent_list_b = parent_list(net_b.op)
    if parent_list_a != parent_list_b:
        j = 0
        for (a, b) in zip(parent_list_a, parent_list_b):
            if a != b:
                print('Difference {} vs {} \n {}'.format(j, net_a.op[j], net_b.op[j]))
                print('Parents: {} vs {}'.format(a, b))
            j += 1
    return parent_list_a == parent_list_b
Statistics = collections.namedtuple('Statistics', ['baseline_nbytes', 'optimized_nbytes'])

def blob_nbytes(blob):
    if False:
        for i in range(10):
            print('nop')
    sz = 0
    try:
        sz = workspace.FetchBlob(blob).nbytes
    except Exception:
        log.warning('Error when fetching blob {}'.format(blob))
    return sz

def compute_statistics(assignments):
    if False:
        while True:
            i = 10
    blob_bytes = {blob: blob_nbytes(blob) for assignment in assignments for (blob, _) in assignment}
    baseline_nbytes = sum(blob_bytes.values())
    optimized_nbytes = sum((max((blob_bytes[blob] for (blob, _) in assignment)) for assignment in assignments))
    return Statistics(baseline_nbytes=baseline_nbytes, optimized_nbytes=optimized_nbytes)

def collect_blob_sizes(net):
    if False:
        i = 10
        return i + 15
    blobs = {}
    for op in net.op:
        for blob in op.input:
            blobs[blob] = blob_nbytes(blob)
        for blob in op.output:
            blobs[blob] = blob_nbytes(blob)
    return blobs