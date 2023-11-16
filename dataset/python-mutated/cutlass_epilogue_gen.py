from typing import Dict, List
from unittest.mock import patch
import sympy
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str
_MAGIC_SYMPY_ERROR_STRING = '[!sympy: unsupported expr!]'

def _arg_str(a):
    if False:
        i = 10
        return i + 15
    if isinstance(a, sympy.Expr):
        return f"{_MAGIC_SYMPY_ERROR_STRING}('{sympy_str(a)}')"
    return str(a)

class CUTLASSEVTOpNotImplementedError(NotImplementedError):
    pass

class CutlassEVTEpilogueTypeFormatter:
    """
    Codegen class, which provides an entry point to generate
    Cutlass "Epilogue Visitor Tree" (EVT) functor declarations.

    See https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder
    for more about EVTs and how they are declared and used to generate.

    Notes:
        * Used by CUTLASSGemmTemplate.
        * This class should not be instantiated by users, it is intended to be used
            by calling CutlassEVTEpilogueTypeFormatter.ir_to_evt_string(...)
            which instantiates this class as an ops handler for virtualized.V.ops.[op-name]
        * Extend this with more _op_<whatever> nodes to add support for new pointwise operations.


    """

    def __init__(self, accumulator_node_name, evt_type_name):
        if False:
            print('Hello World!')
        '\n\n        Initialize an instance of CutlassEVTEpilogueTypeFormatter.\n\n        Parameters:\n        - accumulator_node_name (str): The name of the output Buffer for the GEMM operation in the original (unfused)\n                                       IR graph.\n        - evt_type_name (str):      The output name of the EVT type we are generating.\n\n        '
        self.accumulator_node_name = accumulator_node_name
        self.output = IndentedBuffer(0)
        self.var_counter = 0
        self.evt_type_name = evt_type_name
        self.aliases = dict()

    @staticmethod
    def ir_to_evt_string(template_output_node_name: str, evt_type_name: str, epilogue_nodes: List[IRNode]):
        if False:
            i = 10
            return i + 15
        '\n        Formats IR nodes into a string representation compatible with Cutlass EVT format.\n\n        Args:\n            template_output_node_name (str): The name of the template output node.\n            evt_type_name (str): The name of the EVT type.\n            epilogue_nodes (List[IRNode]): A list of IR nodes representing the epilogue nodes. As of now, these must be\n                ComputedBuffer nodes wrapping Pointwise nodes.\n\n        Returns:\n            A string representation of the IR nodes formatted according to the Cutlass EVT format.\n        '
        formatter = CutlassEVTEpilogueTypeFormatter(template_output_node_name, evt_type_name)
        with virtualized.V.set_ops_handler(formatter), patch.object(FlexibleLayout, 'allow_indexing', True):
            for node in epilogue_nodes:
                if isinstance(node, ComputedBuffer):
                    pnode = node.data
                else:
                    raise RuntimeError('Epilogue nodes must be Pointwise nodes, wrapped in a named ComputedBuffer')
                assert isinstance(pnode, Pointwise)
                index = pnode._index(pnode.ranges)
                result = pnode.inner_fn(index)
                formatter.aliases[node.name] = result
            res = formatter.getvalue(result)
            if _MAGIC_SYMPY_ERROR_STRING in res:
                raise CUTLASSEVTOpNotImplementedError('sympy / indexing expressions not yet supported in EVT fusion')
            else:
                return res

    def __getattr__(self, name):
        if False:
            return 10
        '\n        Resolve V.ops.<whatever> calls, after this instance has been installed as V.ops handler.\n        '

        def inner(*args, **kwargs):
            if False:
                return 10
            fargs = [_arg_str(a) for a in args]
            fkwargs = {key: _arg_str(a) for (key, a) in kwargs.items()}
            fn = getattr(self, f'_op_{name}')
            line = fn(*fargs, **fkwargs)
            self.var_counter += 1
            varname = f'EVT_expr_{self.var_counter}'
            self.output.writeline(f'using {varname} = {line};')
            return varname
        if name.startswith('_'):
            raise CUTLASSEVTOpNotImplementedError(name)
        if hasattr(self, f'_op_{name}'):
            return inner
        else:
            raise CUTLASSEVTOpNotImplementedError(name)

    def _op_load(self, name, index_expr):
        if False:
            while True:
                i = 10
        if name == self.accumulator_node_name:
            return f'cutlass::epilogue::fusion::Sm90AccFetch /* :={name} (matmul output in accumulator) */'
        elif name in self.aliases:
            return self.aliases[name]
        else:
            raise CUTLASSEVTOpNotImplementedError(f'Operand {name} not found. Auxiliary inputs not supported yet.')

    def _op_constant(self, value, dtype):
        if False:
            print('Hello World!')
        if str(dtype) in ('torch.float16', 'torch.float32'):
            return f'cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementAcc> /* value={value}, dtype={dtype} */'
        else:
            raise CUTLASSEVTOpNotImplementedError(f'Unsupported dtype for constant: {dtype}')

    def _cutlass_binary_functional_op(self, op, a, b):
        if False:
            i = 10
            return i + 15
        return f'cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::{op}, ElementAcc, ElementAcc, RoundStyle>,{a},{b}>'

    def _convert_to_output_dtype(self, a):
        if False:
            return 10
        return f'cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<identity_op, ElementD, ElementAcc, RoundStyle>,{a}>'

    def _op_to_dtype(self, a, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return a

    def _op_mul(self, a, b):
        if False:
            print('Hello World!')
        return self._cutlass_binary_functional_op('multiplies', a, b)

    def _op_div(self, a, b):
        if False:
            while True:
                i = 10
        return self._cutlass_binary_functional_op('divides', a, b)

    def _op_truediv(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        return self._cutlass_binary_functional_op('divides', a, b)

    def _op_ge(self, a, b):
        if False:
            while True:
                i = 10
        return self._cutlass_binary_functional_op('greater_equal', a, b)

    def _op_add(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        return self._cutlass_binary_functional_op('plus', a, b)

    def _op_sub(self, a, b):
        if False:
            print('Hello World!')
        return self._cutlass_binary_functional_op('minus', a, b)

    def _op_minimum(self, a, b):
        if False:
            print('Hello World!')
        return self._cutlass_binary_functional_op('minimum', a, b)

    def _op_maximum(self, a, b):
        if False:
            while True:
                i = 10
        return self._cutlass_binary_functional_op('maximum', a, b)

    def _op_relu(self, a):
        if False:
            for i in range(10):
                print('nop')
        const_zero = self._op_constant(0.0, 'torch.float32')
        return f'cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::maximum, ElementAcc, ElementAcc, RoundStyle>,{a}, {const_zero}>'

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if False:
            return 10
        raise CUTLASSEVTOpNotImplementedError()

    def getvalue(self, result) -> str:
        if False:
            return 10
        dtype_converted_expr = self._convert_to_output_dtype(f'EVT_expr_{self.var_counter}')
        self.output.writeline(f'using {self.evt_type_name} = {dtype_converted_expr};')
        return self.output.getvalue()

class CutlassEVTEpilogueArgumentFormatter:
    """
    Codegen class, which provides an entry point to generate
    Cutlass "Epilogue Visitor Tree" (EVT) Argument initializers

    See https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder
    for more about EVTs and how they are declared and used to generate.

    Notes:
        * Used by CUTLASSGemmTemplate.
        * This class should not be instantiated by users, it is intended to be used
            by calling CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string(...)
            which instantiates this class as an ops handler for virtualized.V.ops.[op-name]
        * Extend this with more _op_<whatever> nodes to add support for new pointwise operations.


    """

    def __init__(self, accumulator_node_name: str):
        if False:
            i = 10
            return i + 15
        '\n\n        Initializes a CutlassEVTEpilogueArgumentFormatter object. Do not instantiate directly.\n        Use the CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string static method.\n\n        Args:\n            accumulator_node_name (str): The name of the accumulator node which should contain\n                                          the Matmul result before fusion according to the IR graph.\n        '
        self.accumulator_node_name: str = accumulator_node_name
        self.output: IndentedBuffer = IndentedBuffer(0)
        self.var_counter: int = 0
        self.aliases: Dict[str, str] = dict()

    @staticmethod
    def ir_to_evt_argument_string(template_output_node_name: str, epilogue_nodes: List[IRNode]) -> str:
        if False:
            return 10
        formatter = CutlassEVTEpilogueArgumentFormatter(template_output_node_name)
        with virtualized.V.set_ops_handler(formatter), patch.object(FlexibleLayout, 'allow_indexing', True):
            for node in epilogue_nodes:
                assert isinstance(node, ComputedBuffer)
                pnode = node.data
                assert isinstance(pnode, Pointwise)
                index = pnode._index(pnode.ranges)
                result = pnode.inner_fn(index)
                if node.name is not None:
                    formatter.aliases[node.name] = result
            res: str = formatter.getvalue(result)
            if _MAGIC_SYMPY_ERROR_STRING in res:
                raise CUTLASSEVTOpNotImplementedError('sympy / indexing expressions not yet supported in EVT fusion')
            else:
                return res

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')

        def inner(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            fargs = [_arg_str(a) for a in args]
            fkwargs = {key: _arg_str(a) for (key, a) in kwargs.items()}
            fn = getattr(self, f'_op_{name}')
            line = fn(*fargs, **fkwargs)
            return line
        if name.startswith('_'):
            raise CUTLASSEVTOpNotImplementedError(name)
        if hasattr(self, f'_op_{name}'):
            return inner
        else:
            raise CUTLASSEVTOpNotImplementedError(name)

    def _op_load(self, name, index_expr):
        if False:
            while True:
                i = 10
        if name == self.accumulator_node_name:
            return '{}'
        elif name in self.aliases:
            return self.aliases[name]
        else:
            raise CUTLASSEVTOpNotImplementedError(f'Operand {name} not found. Auxiliary inputs not supported yet.')

    def _op_constant(self, value, dtype):
        if False:
            i = 10
            return i + 15
        if str(dtype) in ('torch.float16', 'torch.float32'):
            return '{ static_cast<ElementAcc>(' + str(value) + ') }'
        else:
            raise CUTLASSEVTOpNotImplementedError(f'Unsupported dtype for constant: {dtype}')

    def _cutlass_binary_functional_op(self, op, a, b):
        if False:
            i = 10
            return i + 15
        return f'{{ /*{op}: */ {a}, {b} }}'

    def _op_mul(self, a, b):
        if False:
            i = 10
            return i + 15
        return self._cutlass_binary_functional_op('multiplies', a, b)

    def _op_div(self, a, b):
        if False:
            i = 10
            return i + 15
        return self._cutlass_binary_functional_op('divides', a, b)

    def _op_truediv(self, a, b):
        if False:
            print('Hello World!')
        return self._cutlass_binary_functional_op('divides', a, b)

    def _op_ge(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        return self._cutlass_binary_functional_op('greater_equal', a, b)

    def _op_add(self, a, b):
        if False:
            print('Hello World!')
        return self._cutlass_binary_functional_op('plus', a, b)

    def _op_sub(self, a, b):
        if False:
            print('Hello World!')
        return self._cutlass_binary_functional_op('minus', a, b)

    def _op_minimum(self, a, b):
        if False:
            print('Hello World!')
        return self._cutlass_binary_functional_op('minimum', a, b)

    def _op_maximum(self, a, b):
        if False:
            print('Hello World!')
        return self._cutlass_binary_functional_op('maximum', a, b)

    def _op_relu(self, a):
        if False:
            for i in range(10):
                print('nop')
        const_zero = self._op_constant(0.0, 'torch.float32')
        return '{' + str(a) + ', ' + const_zero + '}'

    def _op_to_dtype(self, a, dtype, src_dtype=None):
        if False:
            i = 10
            return i + 15
        assert dtype in ('torch.float32', 'torch.float16'), f'Unsupported dtype: {dtype}'
        assert src_dtype in (None, 'torch.float32', 'torch.float16'), f'Unsupported source dtype: {src_dtype}'
        return a

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if False:
            print('Hello World!')
        raise CUTLASSEVTOpNotImplementedError()

    def getvalue(self, result) -> str:
        if False:
            while True:
                i = 10
        return '{' + str(result) + '}'