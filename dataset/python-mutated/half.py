from contextlib import contextmanager
from typing import Any, ContextManager, Generator, Literal
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import Module
from lightning.fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
from lightning.pytorch.plugins.precision.precision import Precision

class HalfPrecision(Precision):
    """Plugin for training with half precision.

    Args:
        precision: Whether to use ``torch.float16`` (``'16-true'``) or ``torch.bfloat16`` (``'bf16-true'``).

    """
    precision: Literal['bf16-true', '16-true'] = '16-true'

    def __init__(self, precision: Literal['bf16-true', '16-true']='16-true') -> None:
        if False:
            print('Hello World!')
        self.precision = precision
        self._desired_input_dtype = torch.bfloat16 if precision == 'bf16-true' else torch.float16

    def convert_module(self, module: Module) -> Module:
        if False:
            for i in range(10):
                print('nop')
        return module.to(dtype=self._desired_input_dtype)

    def tensor_init_context(self) -> ContextManager:
        if False:
            i = 10
            return i + 15
        return _DtypeContextManager(self._desired_input_dtype)

    def module_init_context(self) -> ContextManager:
        if False:
            for i in range(10):
                print('nop')
        return self.tensor_init_context()

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        if False:
            i = 10
            return i + 15
        "A context manager to change the default tensor type when tensors get created during the module's forward.\n\n        See: :meth:`torch.set_default_tensor_type`\n\n        "
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self._desired_input_dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(default_dtype)

    def convert_input(self, data: Any) -> Any:
        if False:
            print('Hello World!')
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self._desired_input_dtype)