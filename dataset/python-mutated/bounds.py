import operator
from functools import partial
from typing import Any, Callable, Dict
from sympy import Expr
import torch
from torch.utils._sympy.value_ranges import bound_sympy, ValueRangeAnalysis, ValueRanges
from .ir import InterpreterShim, LoopBody, LoopBodyBlock
from .utils import cache_on_self, dominated_nodes
from .virtualized import V

class BoundVars:
    """
    Performs Value Range Analysis on LoopBody's fx graph by calling BoundVars.run()
    It exposes the ranges of the nodes in the `bounds` variable

    Note. A current limitation of this analysis is that it just works on a per-loop basis.
    We should be able to propagate the bounds between across the whole graph. This may benefit
    the case a bounded variable is returned by a kernel and fed into another.
    """

    def __init__(self, loop_body: LoopBody) -> None:
        if False:
            while True:
                i = 10
        self.loop_body = loop_body
        self.replacement_vals = {k: ValueRanges(0, v - 1) if isinstance(v, int) or v.is_number else bound_sympy(v) for (k, v) in loop_body.var_ranges.items()}
        self.unbounded_vars = dominated_nodes((node for node in self.loop_body.get_nodes() if node.target in ['load', 'reduction', operator.getitem] or 'masked_subblock' in node.target))
        self._bounds: Dict[torch.fx.Node, ValueRanges] = {}

    @cache_on_self
    def get_bounds(self) -> Dict[torch.fx.Node, ValueRanges]:
        if False:
            i = 10
            return i + 15
        submodules = self.swap_submodules(self.loop_body.submodules)
        for node in self.unbounded_vars:
            if not isinstance(node.target, str) or ('masked_subblock' not in node.target and 'set_indirect' not in node.target):
                self._bounds[node] = ValueRanges.unknown()
        with V.set_ops_handler(ValueRangeAnalysis()):
            interpreter = InterpreterShim(self.loop_body.root_block.graph, submodules)
            interpreter.run(V.get_ops_handler(), initial_env=self._bounds)
        return self._bounds

    def swap_submodules(self, submodules: Dict[str, Callable[..., Any]]) -> Dict[str, Callable[..., ValueRanges]]:
        if False:
            for i in range(10):
                print('nop')
        result: Dict[str, Callable[..., ValueRanges]] = {}
        for key in submodules.keys():
            if key == 'get_index':
                result[key] = self.get_index
            elif 'masked_subblock' in key:
                subblock = self.loop_body.subblocks[key]

                def make_fn(subblock):
                    if False:
                        for i in range(10):
                            print('nop')
                    return lambda mask, value: self.masked_subblock(subblock, self._bounds, mask, value, result)
                result[key] = make_fn(subblock)
            else:
                assert 'set_indirect' in key
                idx = int(key[len('set_indirect'):])
                var = self.loop_body.indirect_vars[idx]
                indirect = partial(self.set_indirect, var)
                result[key] = indirect
        return result

    def masked_subblock(self, subblock: LoopBodyBlock, env: Dict[torch.fx.Node, ValueRanges], mask: Any, value: Any, submodules: Dict[str, Callable[..., Any]]) -> ValueRanges:
        if False:
            i = 10
            return i + 15
        interp = InterpreterShim(subblock.graph, submodules)
        interp.run(V.get_ops_handler(), initial_env=env)
        output = [node for node in subblock.graph.nodes if node.target == 'output']
        assert len(output) == 1
        return interp.env[output[0]]

    def set_indirect(self, old: Expr, new: ValueRanges) -> ValueRanges:
        if False:
            while True:
                i = 10
        assert isinstance(new, ValueRanges)
        self.replacement_vals[old] = new
        return new

    def get_index(self, name: Expr) -> ValueRanges:
        if False:
            print('Hello World!')
        expr = self.loop_body.indexing_exprs[name]
        bound = self.replacement_vals.get(expr)
        if bound is None:
            bound = bound_sympy(expr, self.replacement_vals)
        self.replacement_vals[name] = bound
        return bound