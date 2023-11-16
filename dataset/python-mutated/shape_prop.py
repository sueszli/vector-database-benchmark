import torch
import torch.fx
import traceback
from torch._dispatch.python import enable_python_dispatcher
from torch.fx.node import Node, map_aggregate
from typing import Any, Tuple, NamedTuple, Optional, Dict
from torch.fx._compatibility import compatibility
from torch._guards import detect_fake_mode
__all__ = ['TensorMetadata', 'ShapeProp']

@compatibility(is_backward_compatible=True)
class TensorMetadata(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool
    stride: Tuple[int, ...]
    memory_format: Optional[torch.memory_format]
    is_quantized: bool
    qparams: Dict[str, Any]

def _extract_tensor_metadata(result: torch.Tensor) -> TensorMetadata:
    if False:
        i = 10
        return i + 15
    '\n    Extract a TensorMetadata NamedTuple describing `result`.\n    '
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride()
    memory_formats = {torch.contiguous_format, torch.channels_last, torch.channels_last_3d}
    memory_format = None
    for query_format in memory_formats:
        if result.is_contiguous(memory_format=query_format):
            memory_format = query_format
            break
    is_quantized = result.is_quantized
    qparams: Dict[str, Any] = {}
    if is_quantized:
        qscheme = result.qscheme()
        qparams['qscheme'] = qscheme
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            qparams['scale'] = result.q_scale()
            qparams['zero_point'] = result.q_zero_point()
        elif qscheme in {torch.per_channel_affine, torch.per_channel_affine_float_qparams, torch.per_channel_symmetric}:
            qparams['scale'] = result.q_per_channel_scales().tolist()
            qparams['zero_point'] = result.q_per_channel_zero_points().tolist()
            qparams['axis'] = result.q_per_channel_axis()
    return TensorMetadata(shape, dtype, requires_grad, stride, memory_format, is_quantized, qparams)

@compatibility(is_backward_compatible=True)
class ShapeProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and
    record the shape and type of the result
    into the corresponding node.

    Example:
         In this example, we record the shape
         and data type of a module given
         an example input ``torch.randn(50, D_in)``.
         We print the name, shape and dtype of each node.

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super().__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)
            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred
        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        model = TwoLayerNet(D_in, H, D_out)
        gm = torch.fx.symbolic_trace(model)
        sample_input = torch.randn(50, D_in)
        ShapeProp(gm).propagate(sample_input)

        for node in gm.graph.nodes:
            print(node.name, node.meta['tensor_meta'].dtype,
                node.meta['tensor_meta'].shape)

        The output of this code is:

        x torch.float32 torch.Size([50, 1000])
        linear1 torch.float32 torch.Size([50, 100])
        clamp_1 torch.float32 torch.Size([50, 100])
        linear2 torch.float32 torch.Size([50, 10])
        output torch.float32 torch.Size([50, 10])

    Args:
         module (GraphModule): The module to be executed
         fake_mode (FakeTensorMode): A fake mode for copying the gm

    """

    def __init__(self, gm, fake_mode=None):
        if False:
            while True:
                i = 10
        super().__init__(gm)
        if fake_mode is None:
            fake_mode = detect_fake_mode()
        if fake_mode is not None:
            from torch._dynamo.utils import deepcopy_to_fake_tensor
            self.fake_module = deepcopy_to_fake_tensor(self.module, fake_mode)
            self.fake_mode = fake_mode
        else:
            self.fake_module = None
            self.fake_mode = None
        self.real_module = self.module

    def run_node(self, n: Node) -> Any:
        if False:
            while True:
                i = 10
        try:
            if self.fake_module is not None:
                self.module = self.fake_module
            try:
                if self.fake_mode is not None:
                    with self.fake_mode, enable_python_dispatcher():
                        result = super().run_node(n)
                else:
                    result = super().run_node(n)
            finally:
                self.module = self.real_module
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f'ShapeProp error for: node={n.format_node()} with meta={n.meta}') from e
        found_tensor = False

        def extract_tensor_meta(obj):
            if False:
                print('Hello World!')
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj
        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta['tensor_meta'] = meta
        n.meta['type'] = type(result)
        return result

    def propagate(self, *args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Run `module` via interpretation and return the result and\n        record the shape and type of each node.\n\n        Args:\n            *args (Tensor): the sample input.\n\n        Returns:\n            Any: The value returned from executing the Module\n        '
        if self.fake_mode is not None:
            fake_args = [self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args]
        else:
            fake_args = args
        return super().run(*fake_args)