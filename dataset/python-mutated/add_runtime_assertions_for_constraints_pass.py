import copy
import math
import operator
import traceback
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Set, Tuple
import sympy
import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import SymInt
from torch._export.pass_base import _ExportPassBase, ProxyValue, PassResult
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._sympy.value_ranges import ValueRanges
__all__ = ['_AddRuntimeAssertionsForConstraintsPass', 'InputDim']

class InputDim(NamedTuple):
    input_name: str
    dim: int

def _convert_to_int(val):
    if False:
        return 10
    if val == sympy.oo:
        return math.inf
    if val == -sympy.oo:
        return -math.inf
    if isinstance(val, sympy.Integer):
        return int(val)
    raise RuntimeError('Export constraints cannot be non-integer expressions')

def _convert_range_to_int(range: ValueRanges):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(range, ValueRanges)
    min_val = _convert_to_int(range.lower)
    max_val = _convert_to_int(range.upper)
    return (min_val, max_val)

class _AddRuntimeAssertionsForInlineConstraintsPass(_ExportPassBase):

    def __init__(self, range_constraints: Dict[sympy.Symbol, ValueRanges], equality_constraints: List[Tuple[InputDim, InputDim]]):
        if False:
            while True:
                i = 10
        super().__init__()
        self.range_constraints: Dict[sympy.Symbol, ValueRanges] = range_constraints
        self.equality_constraints: List[Tuple[InputDim, InputDim]] = equality_constraints
        self._asserts_generated_unbacked_symbols: Set[sympy.Symbol] = set()
        self.counter = 0

    def _assert_range_constraint(self, proxy, lower, upper, assert_msg):
        if False:
            print('Hello World!')
        if lower > -math.inf:
            self._insert_assert_async(operator.ge, proxy, lower, assert_msg)
        if upper < math.inf:
            self._insert_assert_async(operator.le, proxy, upper, assert_msg)

    def _insert_assert_async(self, operator, lower, upper, assert_msg):
        if False:
            print('Hello World!')
        '\n        Inserts assert_async call_function nodes in the graph. This function is\n        called **during** the interpreter-based pass.\n        '
        self.counter += 1
        cmp = super().call_operator(operator, (lower, upper), {}, self._create_dummy_node_metadata())
        cmp_tensor = super().call_operator(torch.ops.aten.scalar_tensor.default, (cmp,), {}, self._create_dummy_node_metadata())
        super().call_operator(torch.ops.aten._assert_async.msg, (cmp_tensor, assert_msg), {}, self._create_dummy_node_metadata())

    def call_operator(self, op, args, kwargs, meta) -> ProxyValue:
        if False:
            while True:
                i = 10
        ret = super().call_operator(op, args, kwargs, meta)
        if 'val' not in meta:
            return ret
        val = meta['val']

        def add_assertions(val):
            if False:
                print('Hello World!')
            call_backs: List[Callable] = []
            messages: List[str] = []
            if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                symbol = val.node._expr
                if isinstance(symbol, sympy.Symbol) and symbol.name.startswith('i'):
                    if symbol in self._asserts_generated_unbacked_symbols:
                        return (call_backs, messages)
                    constraint = self.range_constraints[symbol]
                    (min_val, max_val) = _convert_range_to_int(constraint)
                    assert_msg = f' is outside of inline constraint [{min_val}, {max_val}].'
                    call_backs.append(partial(self._assert_range_constraint, lower=min_val, upper=max_val))
                    messages.append(assert_msg)
                    self._asserts_generated_unbacked_symbols.add(symbol)
            elif isinstance(val, torch.Tensor):
                for (i, sym) in enumerate(val.shape):
                    (cbs, msgs) = add_assertions(sym)
                    for (cb, msg) in zip(cbs, msgs):

                        def sym_size_cb(proxy, assert_msg, dim):
                            if False:
                                for i in range(10):
                                    print('nop')
                            dim_proxy = super(_AddRuntimeAssertionsForInlineConstraintsPass, self).call_operator(torch.ops.aten.sym_size.int, (proxy, dim), {}, self._create_dummy_node_metadata())
                            cb(proxy=dim_proxy, assert_msg=assert_msg)
                        call_backs.append(partial(sym_size_cb, dim=i))
                        messages.append(f'.shape[{i}]' + msg)
            return (call_backs, messages)
        (callbacks, messages) = add_assertions(val)
        for (cb, msg) in zip(callbacks, messages):
            cb(proxy=ret, assert_msg=f'{ret.node}' + msg)
        return ret

    def call(self, graph_module):
        if False:
            i = 10
            return i + 15
        val = super().call(graph_module)
        if self.counter == 0 and type(self) is _AddRuntimeAssertionsForInlineConstraintsPass:
            return PassResult(graph_module, False)
        for node in val.graph_module.graph.nodes:
            if not node.meta.get('stack_trace', None):
                node.meta['stack_trace'] = ''.join(traceback.format_stack(limit=1))
        return PassResult(val.graph_module, val.modified)

class _AddRuntimeAssertionsForConstraintsPass(_AddRuntimeAssertionsForInlineConstraintsPass):

    def __init__(self, range_constraints: Dict[sympy.Symbol, ValueRanges], equality_constraints: List[Tuple[InputDim, InputDim]]):
        if False:
            while True:
                i = 10
        super().__init__(range_constraints, equality_constraints)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        if False:
            for i in range(10):
                print('nop')
        graph_module = copy.deepcopy(graph_module)
        graph = graph_module.graph
        insert_loc = None
        for node in graph.nodes:
            if node.op != 'placeholder':
                continue
            insert_loc = node
        if insert_loc is None:
            return super().call(graph_module)
        inputdim_to_node: Dict[InputDim, torch.fx.Node] = OrderedDict()
        for node in graph.nodes:
            if node.op != 'placeholder':
                continue
            if 'val' not in node.meta or node.meta['val'] is None:
                continue
            if not isinstance(node.meta['val'], FakeTensor):
                self._insert_prim_assert_inplace(graph, node, node.meta['val'])
            else:
                fake_tensor_shape = node.meta['val'].shape
                for (dim, shape) in enumerate(fake_tensor_shape):
                    with graph.inserting_after(insert_loc):
                        dim_node = graph.call_function(torch.ops.aten.sym_size.int, (node, dim))
                    input_dim = InputDim(node.name, dim)
                    inputdim_to_node[input_dim] = dim_node
                    insert_loc = dim_node
                    if isinstance(shape, SymInt):
                        symbol = shape.node._expr
                        if symbol in self.range_constraints:
                            self._insert_range_assert_inplace(graph, input_dim, dim_node, self.range_constraints[symbol])
                    else:
                        assert isinstance(shape, int)
                        self._insert_specialized_shape_assert_inplace(graph, input_dim, dim_node, shape)
        if len(inputdim_to_node) > 0:
            with graph.inserting_after(list(inputdim_to_node.values())[-1]):
                self._insert_equality_assert_inplace(graph, inputdim_to_node)
        return super().call(graph_module)

    def _insert_specialized_shape_assert_inplace(self, graph: torch.fx.Graph, input_dim: InputDim, dim_node: torch.fx.Node, shape: int):
        if False:
            while True:
                i = 10
        assert_msg = f'Input {input_dim.input_name}.shape[{input_dim.dim}] is specialized at {shape}'
        with graph.inserting_after(dim_node):
            eq_node = graph.call_function(operator.eq, (dim_node, shape))
        with graph.inserting_after(eq_node):
            tensor_eq_node = graph.call_function(torch.ops.aten.scalar_tensor.default, (eq_node,))
        with graph.inserting_after(tensor_eq_node):
            _ = graph.call_function(torch.ops.aten._assert_async.msg, (tensor_eq_node, assert_msg))

    def _insert_prim_assert_inplace(self, graph, node: torch.fx.Node, value: Any):
        if False:
            while True:
                i = 10
        assert_msg = f'Input {node.name} is specialized to be {value} at tracing time,it is not supported to pass in a different value at run time.'
        with graph.inserting_after(node):
            eq_node = graph.call_function(operator.eq, (node, value))
        with graph.inserting_after(eq_node):
            tensor_eq_node = graph.call_function(torch.ops.aten.scalar_tensor.default, (eq_node,))
        with graph.inserting_after(tensor_eq_node):
            _ = graph.call_function(torch.ops.aten._assert_async.msg, (tensor_eq_node, assert_msg))

    def _insert_range_assert_inplace(self, graph: torch.fx.Graph, input_dim: InputDim, dim_node: torch.fx.Node, range: ValueRanges):
        if False:
            for i in range(10):
                print('nop')
        "\n        Add runtime asserts for user-specified range constraints for\n        each placeholder's dynamic dimension.\n        "
        (min_val, max_val) = _convert_range_to_int(range)
        assert_msg = f'Input {input_dim.input_name}.shape[{input_dim.dim}] is outside of specified dynamic range [{min_val}, {max_val}]'
        with graph.inserting_after(dim_node):
            if min_val > 2:
                self._insert_assert_async_inplace(graph, operator.ge, (dim_node, min_val), assert_msg)
            if max_val < math.inf:
                self._insert_assert_async_inplace(graph, operator.le, (dim_node, max_val), assert_msg)

    def _insert_equality_assert_inplace(self, graph: torch.fx.Graph, inputdim_to_node: Dict[InputDim, torch.fx.Node]):
        if False:
            return 10
        for (input_dim, other_input_dim) in self.equality_constraints:
            dim_node = inputdim_to_node[input_dim]
            assert_msg = f'Input {input_dim.input_name}.shape[{input_dim.dim}] is not equal to input {other_input_dim.input_name}.shape[{other_input_dim.dim}]'
            other_dim_node = inputdim_to_node[other_input_dim]
            self._insert_assert_async_inplace(graph, operator.eq, (dim_node, other_dim_node), assert_msg)

    def _insert_assert_async_inplace(self, graph, operator, args, assert_msg):
        if False:
            for i in range(10):
                print('nop')
        '\n        Inserts assert_async call_function nodes in the graph. This function is\n        called before we run the interpreter-based pass and does an inplace\n        insertion.\n        '
        cmp_node = graph.call_function(operator, args)
        with graph.inserting_after(cmp_node):
            cmp_tensor_node = graph.call_function(torch.ops.aten.scalar_tensor.default, (cmp_node,))
        with graph.inserting_after(cmp_tensor_node):
            _ = graph.call_function(torch.ops.aten._assert_async.msg, (cmp_tensor_node, assert_msg))