import math
import sympy
import torch
from torch.utils._sympy.value_ranges import ValueRanges
from .ir import LoopBody
from .utils import dominated_nodes

def val_expressable_in_32_bits(val):
    if False:
        while True:
            i = 10
    if getattr(val, 'is_Boolean', False):
        return True
    if isinstance(val, sympy.Expr):
        assert val.is_constant()
        if val.is_Integer or val.is_Boolean:
            val = int(val)
        else:
            val = float(val)
    if isinstance(val, float):
        return val <= 2 ** 24 and val >= -2 ** 24
    if isinstance(val, int):
        iinfo = torch.iinfo(torch.int32)
        return val <= iinfo.max and val >= iinfo.min
    raise Exception(f'Unexpected value {val}')

def range_expressable_in_32_bits(range):
    if False:
        print('Hello World!')
    return val_expressable_in_32_bits(range.lower) and val_expressable_in_32_bits(range.upper)

def try_to_reduce_precision(node, bounds, indirect_vars, indices, replacement_vals):
    if False:
        for i in range(10):
            print('nop')

    def skip_filter(node):
        if False:
            return 10
        return node.target == 'to_dtype' and node.args[2] in (torch.int32, torch.float32, torch.float64)
    for dominated in dominated_nodes([node], skip_filter):
        if dominated.target in ['store', 'output']:
            continue
        if isinstance(dominated.target, str) and 'set_indirect' in dominated.target:
            idx = int(dominated.target[len('set_indirect'):])
            indirect_var = indirect_vars[idx]
            for (index, expr) in indices.items():
                if indirect_var in expr.free_symbols:
                    index_val = replacement_vals[index]
                    if math.isinf(index_val.lower) or math.isinf(index_val.upper):
                        return
                    index_val_int = ValueRanges(int(index_val.lower), int(index_val.upper))
                    if not range_expressable_in_32_bits(index_val_int):
                        return
        if not range_expressable_in_32_bits(bounds[dominated]):
            return
    args = list(node.args)
    args[2] = torch.int32
    node.args = tuple(args)

def indexing_dtype_strength_reduction(loop_body: LoopBody):
    if False:
        for i in range(10):
            print('nop')
    "\n    Performs Value Range Analysis on LoopBody's fx graph to reduce precision of\n    intermediaries from int64 to int32\n    "
    bv = loop_body.bounds()
    int64_dtype_nodes = [node for node in loop_body.get_nodes() if node.target == 'to_dtype' and node.args[2] == torch.int64 and (node not in bv.unbounded_vars)]
    if not int64_dtype_nodes:
        return
    bounds = bv.get_bounds()
    for node in int64_dtype_nodes:
        try_to_reduce_precision(node, bounds, loop_body.indirect_vars, loop_body.indexing_exprs, bv.replacement_vals)