from operator import itemgetter
from functorch.compile import make_boxed_func
import torch
import torch.nn as nn
from torch._functorch.compilers import aot_module
from torch._inductor.decomposition import select_decomp_table
from torch.distributed._tensor import DTensor
inductor_decomps = select_decomp_table()
graphs = []

def fwd_bwd_compiler(fx_g, _):
    if False:
        i = 10
        return i + 15
    graphs.append(fx_g)
    return make_boxed_func(fx_g)

def get_inductor_decomp_graphs(model: nn.Module, args, kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Obtain forward and backward graphs of a model with inductor decompositions using tracing and aot_module.\n\n    Convenient util to get the fwd and bwd graphs of an arbitrary model\n    with inductor decompositions. Note that this would simply do tracing\n    with aot_module and don't ensure correctness. This is useful to track\n    the ops needed in DTensor.\n    "
    compiled_mod = aot_module(model, fw_compiler=fwd_bwd_compiler, decompositions=inductor_decomps)
    output = compiled_mod(*args, **kwargs)
    if output.ndim != 0:
        output = output.sum()
    output.backward()
    assert len(graphs) == 2
    return graphs

def print_op_coverage_summary(model: nn.Module, args, kwargs, *, output_csv=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Util to print the operator coverage summary of a certain model with tabulute.\n\n    Must have tabulate module installed.\n    '
    import csv
    from tabulate import tabulate
    (fwd_graph, bwd_graph) = get_inductor_decomp_graphs(model, args, kwargs)
    op_counts = {}
    for node in fwd_graph.graph.nodes:
        if node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload):
            if node.target not in op_counts:
                op_counts[node.target] = 0
            op_counts[node.target] += 1
    for node in bwd_graph.graph.nodes:
        if node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload):
            if node.target not in op_counts:
                op_counts[node.target] = 0
            op_counts[node.target] += 1
    op_infos = []
    for (op, count) in op_counts.items():
        supported = op in DTensor._op_dispatcher.sharding_propagator.op_to_rules
        op_infos.append([op, str(op._schema), count, supported])
    count_idx = 2
    op_infos.sort(key=itemgetter(count_idx), reverse=True)
    headers = ['Operator', 'Schema', 'Total Count', 'Supported']
    print(tabulate(op_infos, headers=headers))
    if output_csv:
        with open('op_summary.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(headers)
            for row in op_infos:
                csv_writer.writerow(row)