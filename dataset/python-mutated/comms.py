from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
overlap_log = torch._logging.getArtifactLogger(__name__, 'overlap')

def sink_waits(snodes: List['scheduler.BaseSchedulerNode']) -> List['scheduler.BaseSchedulerNode']:
    if False:
        while True:
            i = 10
    '\n    Greedily moves waits as late as possible (i.e. until we reach a use). Optimal in terms of\n    communication overlap.\n    '
    new_order = []
    cur_waits = set()
    for snode in snodes:
        if isinstance(snode.node, ir.Wait):
            cur_waits.add(snode)
        else:
            for wait in tuple_sorted(cur_waits):
                if snode in wait.node_users:
                    new_order.append(wait)
                    cur_waits.remove(wait)
            new_order.append(snode)
    for snode in tuple_sorted(cur_waits):
        new_order.append(snode)
    return new_order

def raise_comms(snodes: List['scheduler.BaseSchedulerNode']) -> List['scheduler.BaseSchedulerNode']:
    if False:
        print('Hello World!')
    "\n    Greedily moves comms as early as possible (i.e. until we reach an input).\n    Optimal in terms of communication overlap.\n\n    TODO: We might want to adjust this in the future to account for memory limitations.\n    e.g. when we are compiling FSDP, this heuristics will cause the all-gathers to be prefetched as soon as possible,\n    which is the beginning of the forwards pass. We'll have to either do a special pass for FSDP,\n    or we'll want to redo this pass with memory considerations so we handle the FSDP case in a general way.\n    "
    new_order_reversed: List['scheduler.BaseSchedulerNode'] = []
    cur_comms: List['scheduler.BaseSchedulerNode'] = []
    for snode in reversed(snodes):
        if isinstance(snode.node, ir.CollectiveKernel):
            cur_comms.append(snode)
        else:
            for comm in cur_comms:
                assert len(comm.inverse_users) > 0
            while len(cur_comms) > 0 and any((snode in comm.inverse_users for comm in cur_comms)):
                comm = cur_comms.pop(0)
                new_order_reversed.append(comm)
            new_order_reversed.append(snode)
    assert len(cur_comms) <= 1
    for snode in tuple_sorted(cur_comms):
        new_order_reversed.append(snode)
    return new_order_reversed[::-1]

def get_ancestors(node):
    if False:
        while True:
            i = 10
    ancestors = set()
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in node.inverse_users:
                if inp not in ancestors:
                    ancestors.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return ancestors

def get_descendants(node):
    if False:
        print('Hello World!')
    descendants = set()
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in node.node_users:
                if inp not in descendants:
                    descendants.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return descendants

def decide_global_ordering_of_comms(nodes: List['scheduler.BaseSchedulerNode']):
    if False:
        for i in range(10):
            print('nop')
    "\n    Decide global ordering of comms, by just enforcing the ordering that's in the input graph\n    (might not be the same ordering as the eager mode program).\n    TODO: Come up with a better approach\n    "
    comm_nodes = [n for n in nodes if isinstance(n.node, ir.CollectiveKernel)]
    for i in range(1, len(comm_nodes)):
        comm_nodes[i].add_fake_dep(WeakDep(comm_nodes[i - 1].get_name()))

def assert_no_comm_nodes(snodes: List['scheduler.BaseSchedulerNode']) -> None:
    if False:
        for i in range(10):
            print('nop')
    assert not any((isinstance(snode.node, ir.CollectiveKernel) for snode in snodes))

def estimate_op_runtime(snode: 'scheduler.BaseSchedulerNode') -> float:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns estimated op runtime in nanoseconds (ns)\n    '
    if config.estimate_op_runtime == 'default':
        runtime = snode.get_estimated_runtime()
    else:
        runtime = config.estimate_op_runtime(snode)
    return runtime

def reorder_compute_for_overlap(snodes: List['scheduler.BaseSchedulerNode']) -> List['scheduler.BaseSchedulerNode']:
    if False:
        while True:
            i = 10
    "\n    Decides a global ordering of all compute and communication nodes,\n    assuming that we already have a global ordering of communication nodes.\n\n    Overall scheduling procedure is:\n        Step 1: Given that we've currently scheduled comm N, we now schedule all compute nodes\n            that are required for comm N + 1 but do not depend on comm N, to run at the same time with comm N.\n        Step 2: If all those compute nodes are sufficient to overlap comm N, we're done.\n            Otherwise, we now need to look elsewhere to find compute that overlaps with comm N.\n            We prioritize compute nodes that are needed sooner.\n        Step 3: We schedule the compute nodes dependent on comm N and required for comm N + 1.\n        Step 4: We schedule comm N + 1.\n        Repeat this for subsequent comm nodes.\n    "
    final_order = []
    comm_nodes = []
    for snode in snodes:
        if isinstance(snode.node, ir.CollectiveKernel):
            comm_nodes.append(snode)
    if len(comm_nodes) == 0:
        return snodes
    comm_ancestors = {node: get_ancestors(node) for node in comm_nodes}
    comm_descendants = {node: get_descendants(node) for node in comm_nodes}
    indeg = {k: 0 for k in snodes}
    for snode in snodes:
        for user in snode.node_users:
            if user in indeg:
                indeg[user] += 1
    ready_to_schedule_nodes = {node for node in snodes if indeg[node] == 0}
    unscheduled_nodes = set()
    unscheduled_nodes = set(snodes)

    def schedule_node(snode):
        if False:
            print('Hello World!')
        '\n        Schedule a single node.\n        '
        assert snode in unscheduled_nodes
        assert snode in ready_to_schedule_nodes
        ready_to_schedule_nodes.remove(snode)
        unscheduled_nodes.remove(snode)
        final_order.append(snode)
        for user in tuple_sorted(snode.node_users):
            if user in indeg:
                indeg[user] -= 1
                if indeg[user] == 0:
                    ready_to_schedule_nodes.add(user)

    def schedule_nodes(snodes):
        if False:
            while True:
                i = 10
        '\n        Schedules all nodes in `snodes` in an arbitrary topologically valid order.\n        '
        all_nodes = set(snodes)
        assert all((node in unscheduled_nodes for node in all_nodes))
        while len(all_nodes) > 0:
            progress = False
            for node in tuple_sorted(all_nodes):
                if node in ready_to_schedule_nodes:
                    schedule_node(node)
                    all_nodes.remove(node)
                    progress = True
            if not progress:
                raise Exception('Unable to find a free node (indeg == 0). This is an impossible state to reach. Please report a bug to PyTorch.')
    assert len(comm_nodes) > 0
    schedule_nodes(list(comm_ancestors[comm_nodes[0]]) + [comm_nodes[0]])
    rolled_over_compute_cost = 0
    for idx in range(1, len(comm_ancestors)):
        needed_by_next_comm_and_ready_compute_nodes = unscheduled_nodes & comm_ancestors[comm_nodes[idx]] - comm_descendants[comm_nodes[idx - 1]]
        assert_no_comm_nodes(needed_by_next_comm_and_ready_compute_nodes)
        total_compute_runtime_cost = rolled_over_compute_cost + sum([estimate_op_runtime(node) for node in needed_by_next_comm_and_ready_compute_nodes])
        prev_comm_runtime_cost = estimate_op_runtime(comm_nodes[idx - 1])
        schedule_nodes(tuple_sorted(needed_by_next_comm_and_ready_compute_nodes))
        step1_runtime_cost = total_compute_runtime_cost
        if step1_runtime_cost >= prev_comm_runtime_cost:
            pass
        else:
            ready_to_schedule_compute_nodes = tuple_sorted(ready_to_schedule_nodes - comm_descendants[comm_nodes[idx - 1]])
            assert_no_comm_nodes(ready_to_schedule_compute_nodes)

            def earliest_comm_descendant(node):
                if False:
                    i = 10
                    return i + 15
                for idx in range(len(comm_nodes)):
                    if node in comm_ancestors[comm_nodes[idx]]:
                        return idx
                return len(comm_nodes)
            ready_to_schedule_compute_nodes = sorted(ready_to_schedule_compute_nodes, key=earliest_comm_descendant)
            for snode in ready_to_schedule_compute_nodes:
                if total_compute_runtime_cost >= prev_comm_runtime_cost:
                    break
                compute_runtime_cost = estimate_op_runtime(snode)
                if prev_comm_runtime_cost - total_compute_runtime_cost <= compute_runtime_cost / 2:
                    continue
                schedule_node(snode)
                total_compute_runtime_cost += compute_runtime_cost
        rollable_compute_cost = total_compute_runtime_cost - step1_runtime_cost
        needed_by_next_comm_nodes = unscheduled_nodes & comm_ancestors[comm_nodes[idx]]
        schedule_nodes(list(needed_by_next_comm_nodes))
        schedule_nodes([comm_nodes[idx]])
        is_prev_comm_blocking_next_comm = len(needed_by_next_comm_nodes) > 0
        if is_prev_comm_blocking_next_comm:
            rolled_over_compute_cost = 0
        else:
            rolled_over_compute_cost = rollable_compute_cost
    schedule_nodes(unscheduled_nodes)
    return final_order

def node_summary(snode):
    if False:
        i = 10
        return i + 15
    detail = ''
    if isinstance(snode.node, ir.ExternKernelOut):
        detail = f' ({snode.node.kernel})'
    out_tensor_info = ''
    if hasattr(snode.node, 'layout') and hasattr(snode.node.layout, 'size') and hasattr(snode.node.layout, 'stride'):
        out_tensor_info = f' (size={snode.node.layout.size}, stride={snode.node.layout.stride})'
    node_name = ''
    if hasattr(snode.node, 'name'):
        node_name = snode.node.name
    return f'{snode.node.__class__.__name__}{detail}{out_tensor_info} ({node_name})'

def visualize_overlap(order):
    if False:
        print('Hello World!')
    total_est_runtime: float = 0.0
    cur_comm_node = None
    for snode in order:
        if cur_comm_node is None:
            if isinstance(snode.node, ir.CollectiveKernel):
                total_est_runtime += estimate_op_runtime(snode)
                cur_comm_node = snode.node
            elif isinstance(snode.node, ir.Wait):
                raise Exception('Wait is not expected when there is no collective running')
            else:
                total_est_runtime += estimate_op_runtime(snode)
            overlap_log.debug(f'{node_summary(snode)}')
        elif isinstance(snode.node, ir.CollectiveKernel):
            raise Exception('Found two collectives running at the same time. `visualize_overlap` needs to be updated to handle this case')
        elif isinstance(snode.node, ir.Wait):
            overlap_log.debug(f'{node_summary(snode)}')
            cur_comm_node = None
        else:
            overlap_log.debug(f'| {node_summary(snode)}')
    overlap_log.debug(f'Est. runtime (ms): {total_est_runtime / 1000 / 1000}')

def reorder_compute_and_comm_for_overlap(snodes: List['scheduler.BaseSchedulerNode']) -> List['scheduler.BaseSchedulerNode']:
    if False:
        for i in range(10):
            print('nop')
    order = snodes
    for p in config.reorder_for_compute_comm_overlap_passes:
        if isinstance(p, str) and p in globals():
            p = globals()[p]
        if torch.distributed.get_rank() == 0:
            overlap_log.debug(f'==== Visualize overlap before reordering pass {p} ====')
            try:
                visualize_overlap(order)
            except Exception as e:
                overlap_log.debug(str(e))
        order = p(order)
        if torch.distributed.get_rank() == 0:
            overlap_log.debug(f'==== Visualize overlap after reordering pass {p} ====')
            try:
                visualize_overlap(order)
            except Exception as e:
                overlap_log.debug(str(e))
    return order