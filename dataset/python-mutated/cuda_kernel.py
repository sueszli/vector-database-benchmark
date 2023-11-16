import logging
from typing import Callable, Dict, List, Optional
from ... import ir
from ...autotune_process import CUDABenchmarkRequest
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout, TensorBox
from ...select_algorithm import ChoiceCaller
from ...utils import sympy_product
from ...virtualized import V
from ..common import IndentedBuffer, Kernel, OpOverrides
from ..cpp import CppPrinter, DTYPE_TO_CPP
log = logging.getLogger(__name__)
cexpr = CppPrinter().doprint

def _normalize_idx(index: int, total_length: int) -> int:
    if False:
        while True:
            i = 10
    return index if index >= 0 else index + total_length

class CUDAKernel(Kernel):
    """
    Baseclass for CUDA / Cutlass based Kernels
    """
    overrides = OpOverrides

class CUDATemplateKernel(CUDAKernel):
    """
    Template kernels defined by CUDA / Cutlass in C++.
    """
    _EXTRA_CPP_ARGS = 'size_t* workspace_size, uint8_t* workspace, cudaStream_t stream'

    def __init__(self, kernel_name):
        if False:
            i = 10
            return i + 15
        '\n        Initializes a new instance of the CUDATemplateKernel class.\n\n        Args:\n            kernel_name (str): The name of the kernel.\n        '
        super().__init__()
        self.kernel_name = kernel_name
        self.named_nodes: Dict[str, IRNode] = {}

    def arg_name(self, node: IRNode) -> Optional[str]:
        if False:
            print('Hello World!')
        '\n        Returns arg name of a given input or output node.\n        '
        if node is None:
            return None
        return {**self.args.input_buffers, **self.args.output_buffers}.get(node.get_name(), None)

    def check_not_null(self, node: IRNode) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates code to check that a node is not null.\n        '
        if node is None:
            return ''
        size_str = self.size(node, 0, -1)
        name_str = self.arg_name(node)
        if name_str is None:
            return ''
        res = IndentedBuffer(initial_indent=2)
        res.tabwidth = 1
        res.splice(f'\n            {{\n              if (!{name_str}) {{\n                int64_t {name_str}_size = {size_str};\n                if ({name_str}_size > 0) {{\n                  throw std::runtime_error("input {name_str} is null but size is not 0!");\n                }}\n              }}\n            }}\n            ')
        return res.getvalue()

    def def_kernel(self, inputs: List[IRNode], outputs: List[IRNode], names_str: str='', input_reorder: Optional[List[int]]=None) -> str:
        if False:
            while True:
                i = 10
        '\n        Hook called from template code to generate function definition and\n        needed args.\n\n        Args:\n            inputs: List of input IRNodes\n            outputs: List of output IRNodes\n            names_str: Comma separated list of input + output argument names.\n            input_reorder: The actual order of input nodes.\n                           e.g. The template might have input argument defined as [X, W, Bias],\n                           and the actual input passed into this template could be [Bias, X, W].\n                           In this case, the `input_reorder` would be [2, 0, 1].\n        '
        names = [x.strip() for x in names_str.strip().split(',')]
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(f'len(inputs) + len(outputs)={len(inputs) + len(outputs)!r} != len(names)={len(names)!r}, inputs={inputs!r}, outputs={outputs!r}, names={names!r}')
        if input_reorder is not None:
            assert len(inputs) == len(input_reorder)
        else:
            input_reorder = list(range(len(inputs)))
        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.args.input_buffers[node.get_name()] = name
        for (name, node) in zip(names[len(inputs):len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name
        (arg_defs, *_) = self.args.cpp_argdefs()
        return f"PT_EXPORT int {self.kernel_name}({', '.join(arg_defs)}, {self._EXTRA_CPP_ARGS})"

    def call_kernel(self, name: str, node: 'CUDATemplateBuffer', epilogue_nodes: List[ir.Buffer]) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Generates code to call the kernel through V.graph.wrapper_code.\n        used from within torch._inductor.wrapper.WrapperCodeGen\n\n        name: Name of kernel function.\n        node: The CUDATemplateBuffer node which contains information about the kernel, it's fused epilogue nodes\n        as well as all required inputs and outputs.\n        "
        wrapper = V.graph.wrapper_code
        (_, call_args, _) = self.args.python_argdefs()
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + '.item()'
            else:
                call_args[i] = f'c_void_p({call_args[i]}.data_ptr())'
        call_args.append('None')
        if node.get_workspace_size() > 0:
            call_args.append(f'c_void_p({node.get_name()}_workspace.data_ptr())')
        else:
            call_args.append('None')
        wrapper.generate_kernel_call(name, call_args, device_index=V.graph.scheduler.current_device.index, cuda=True, triton=False)

    def dtype(self, node: IRNode) -> Optional[str]:
        if False:
            return 10
        '\n        Generates code which represents dtype of a given node.\n        '
        if node is None:
            return 'void'
        return DTYPE_TO_CPP.get(node.get_layout().dtype)

    def offset(self, node: IRNode) -> str:
        if False:
            print('Hello World!')
        '\n        Generates code which represents offset of a given node.\n        '
        if node is None:
            return '0'
        return str(node.get_layout().offset)

    def ptr(self, node: IRNode) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates code which represents pointer of a given node.\n        '
        if node is None:
            return 'nullptr'
        arg_name = self.arg_name(node)
        if arg_name is None:
            return 'nullptr'
        offset = self.offset(node)
        return arg_name if offset == '0' else f'{arg_name} + {offset}'

    def size(self, node: IRNode, start_index: int, end_index: Optional[int]=None, default_value: int=0) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Hook called from template code to get the size of an arg.\n        Generates code which represents size of a given node in [start_index, end_index).\n        If node is None, returns default_value.\n\n        TODO: Will add needed args to pass it in if it is dynamic.\n        '
        if node is None:
            return str(default_value)
        start_index = _normalize_idx(start_index, len(node.get_size()))
        if end_index is None:
            end_index = start_index
        end_index = _normalize_idx(end_index, len(node.get_size()))
        sizes = node.get_size()[start_index:end_index + 1]
        if len(sizes) == 0:
            return str(default_value)
        val = sympy_product(sizes)
        return cexpr(self.rename_indexing(val))

    def stride(self, node: IRNode, index: int, default_value: int=0) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Hook called from template code to get the stride of an arg.\n        Generates code which represents stride of a given node at index.\n        If node is None, returns default_value.\n\n        TODO: Will add needed args to pass it in if it is dynamic.\n        '
        if node is None:
            return str(default_value)
        index = _normalize_idx(index, len(node.get_size()))
        if index < 0:
            return str(default_value)
        stride = node.get_stride()[index]
        return cexpr(self.rename_indexing(stride))

    def row_or_column_stride(self, node: IRNode, default_value: int=0) -> str:
        if False:
            return 10
        '\n        Hook called from template code to get the row or column stride of an arg.\n        This is required by some CUTLASS 2.X APIs.\n        If the node is in row_major, it returns stride[-2].\n        If the node is in column_major, it returns stride[-1].\n\n        TODO: Will add needed args to pass it in if it is dynamic.\n        '
        if node is None or len(node.get_stride()) < 2:
            return str(default_value)
        stride0 = node.get_stride()[-1]
        stride1 = node.get_stride()[-2]
        if stride0 == 1:
            return cexpr(self.rename_indexing(stride1))
        elif stride1 == 1:
            return cexpr(self.rename_indexing(stride0))
        else:
            raise RuntimeError(f'At least 1 stride should be 1. Strides: node.get_stride()={node.get_stride()!r}')

class CUDATemplateCaller(ChoiceCaller):
    """
    CUDATemplateCaller

    This class represents a caller for CUDA template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CUDABenchmarkRequest): The benchmark request for the caller.
        template_buffer (CUDATemplateBuffer): The template buffer for the caller.
    """

    def __init__(self, name: str, category: str, input_nodes: List[Buffer], layout: Layout, make_kernel_render: Callable[[CUDATemplateBuffer, Optional[List[IRNode]]], str], bmreq: CUDABenchmarkRequest, template: 'CUDATemplate'):
        if False:
            while True:
                i = 10
        super().__init__(name, input_nodes, layout)
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template

    def benchmark(self, *args, out) -> float:
        if False:
            while True:
                i = 10
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'CUDATemplateCaller(source_file={self.bmreq.source_file})'

    def call_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'cuda_template_kernels.{self.name}'

    def hash_key(self) -> str:
        if False:
            return 10
        return '-'.join([self.category, self.bmreq.hash_key])

    def output_node(self) -> TensorBox:
        if False:
            while True:
                i = 10
        return TensorBox.create(CUDATemplateBuffer(layout=self.layout, inputs=self.input_nodes, make_kernel_render=self.make_kernel_render, workspace_size=self.bmreq.workspace_size, template=self.template))