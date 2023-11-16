import functools
import itertools
import logging
from typing import List, Optional
from unittest.mock import patch
import sympy
import torch
from ...autotune_process import CUDABenchmarkRequest, TensorMeta
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .cuda_kernel import CUDATemplateCaller, CUDATemplateKernel
log = logging.getLogger(__name__)

class CUDATemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(self, name: str, input_nodes: List[Buffer], layout: Layout, input_reorder: Optional[List[int]]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Baseclass for CUDA C++ Templates, derived from KernelTemplate. Not to be instantiated directly.\n\n        Args:\n            name (str): The name of the CUDATemplate object.\n            input_nodes (List[IRNode]): A list of input IRNodes.\n            layout (Layout): The layout of the output buffer / tensor.\n            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.\n\n        '
        super().__init__(name)
        self.input_nodes = input_nodes
        self.output_node: Buffer = Buffer('buf_out', layout)
        self.input_reorder = input_reorder
        self.layout = layout

    def generate(self, **kwargs) -> CUDATemplateCaller:
        if False:
            while True:
                i = 10
        '\n        Generates the CUDA template caller object for the given GEMM template and operation. This CUDATemplateCaller\n        may be used to call and benchmark the generated CUDA kernel in a standalone manner to enable Autotuning.\n\n        Args:\n            kwargs: Additional keyword arguments.\n\n        Returns:\n            A CUDATemplateCaller object representing the generated CUDA template caller.\n        '
        kernel_name = f'cuda_{self.name}'
        with patch.object(V.graph, 'get_dtype', self._fake_get_dtype(self.output_node)), CUDATemplateKernel(kernel_name=kernel_name) as kernel:
            code = self.render(kernel=kernel, **kwargs)
            (_, call_args, _) = kernel.args.python_argdefs()
            log.debug('Generated Code:\n%s', code)
            log.debug('Args: cpp_argdefs: %s, python_argdefs: %s', kernel.args.cpp_argdefs(), kernel.args.python_argdefs())
        input_reorder = self.input_reorder if self.input_reorder is not None else list(range(len(self.input_nodes)))
        expected_args = list(unique((self.input_nodes[idx].get_name() for idx in input_reorder)))
        expected_args.extend([self.output_node.get_name()])
        assert list(call_args)[:len(expected_args)] == expected_args, (call_args, expected_args)
        extra_args = V.graph.sizevars.size_hints(map(sympy.expand, call_args[len(expected_args):]))
        kernel_hash_name = f'cuda_{self.name}_{next(self.index_counter)}'
        bmreq = CUDABenchmarkRequest(kernel_name=kernel_name, input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes), output_tensor_meta=TensorMeta.from_irnodes(self.output_node), extra_args=extra_args, source_code=code)

        def make_kernel_render(template_node: CUDATemplateBuffer, epilogue_nodes: Optional[List[IRNode]]=None):
            if False:
                for i in range(10):
                    print('nop')
            kernel = CUDATemplateKernel(kernel_name='KERNEL_NAME')
            render = functools.partial(self.render, kernel=kernel, template_buffer_node=template_node, epilogue_nodes=epilogue_nodes, **kwargs)
            return (kernel, render)
        return CUDATemplateCaller(kernel_hash_name, self.name, self.input_nodes, self.output_node.get_layout(), make_kernel_render, bmreq, self)

    def header(self) -> IndentedBuffer:
        if False:
            return 10
        res = IndentedBuffer()
        res.splice('\n                #include <exception>\n                #include <iostream>\n                #include <memory>\n                #include <random>\n                #include <vector>\n            ')
        return res

    def globals(self) -> IndentedBuffer:
        if False:
            i = 10
            return i + 15
        res = IndentedBuffer()
        res.splice('\n                // We compile all models with -fvisibility=hidden. Any symbols that need to be\n                // exposed in the final shared library must be declared with PT_EXPORT to make\n                // them visible.\n                #ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)\n                #define PT_EXPORT __attribute__((__visibility__("default")))\n                #else\n                #ifdef _WIN32\n                #define PT_EXPORT __declspec(dllexport)\n                #else\n                #define PT_EXPORT\n                #endif\n                #endif\n                using bfloat16 = nv_bfloat16;\n            ')
        return res

    def render(self, **kwargs) -> str:
        if False:
            return 10
        raise NotImplementedError

class CUTLASSTemplate(CUDATemplate):
    """
    CUTLASSTemplate is a class that provides a template for generating CUTLASS Templates. Used as a baseclass for the
    CUTLASSGemmTemplate, providing functionality that might also be relevant for non-GEMM CUTLASS Kernels.
    """

    def header(self) -> IndentedBuffer:
        if False:
            print('Hello World!')
        res = super().header()
        res.splice('\n                #include "cute/tensor.hpp"\n                #include "cutlass/cutlass.h"\n                #include "cutlass/numeric_types.h"\n                #include "cutlass/tensor_ref.h"\n                #include "cutlass/util/host_tensor.h"\n                #include "cutlass/util/reference/host/tensor_fill.h"\n                #include "cutlass/util/reference/device/tensor_fill.h"\n                #include "cutlass/util/device_memory.h"\n            ')
        return res

    def globals(self) -> IndentedBuffer:
        if False:
            print('Hello World!')
        res = super().globals()
        res.splice('\n                using namespace cute;\n                #define CUTLASS_CHECK(status)                                                      \\\n                {                                                                                  \\\n                  cutlass::Status error = status;                                                  \\\n                  if (error != cutlass::Status::kSuccess) {                                        \\\n                    auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +             \\\n                        cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);        \\\n                    throw std::runtime_error(msg);                                                 \\\n                  }                                                                                \\\n                }\n\n                // Used as pass-through functor in EVT just for type casting / rounding\n                template <typename T>\n                struct identity_op {\n                  CUTLASS_HOST_DEVICE\n                  T operator()(T val) const { return val; }\n                };\n\n            ')
        return res

    def cute_int(self, int_str: str, var_name: str) -> str:
        if False:
            i = 10
            return i + 15
        res = ''
        if int_str in {'1', '1L'}:
            res = 'cute::Int<1>{}'
        else:
            res = int_str
        return f'{res} /* {var_name} */'
    _DTYPE_TO_CUTLASS = {torch.float32: 'float', torch.float64: 'double', torch.float16: 'cutlass::half_t', torch.int32: 'int', torch.int8: 'int8_t', torch.uint8: 'uint8_t', torch.bool: 'bool', torch.bfloat16: 'cutlass::bfloat16_t'}

    def cutlass_type_cast(self, node: IRNode, ptr: str) -> str:
        if False:
            i = 10
            return i + 15
        if node is None:
            return ptr
        else:
            return f'({self._DTYPE_TO_CUTLASS.get(node.get_dtype())}*)({ptr})'