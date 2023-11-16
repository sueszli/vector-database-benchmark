from typing import Any, List, Optional, Union
import torch
from torch.nn import Module
from typing_extensions import Self

class _DeviceDtypeModuleMixin(Module):
    __jit_unused_properties__: List[str] = ['device', 'dtype']

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self._dtype: Union[str, torch.dtype] = torch.get_default_dtype()
        self._device = torch.device('cpu')

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        if False:
            print('Hello World!')
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: Union[str, torch.dtype]) -> None:
        if False:
            print('Hello World!')
        raise RuntimeError('Cannot set the dtype explicitly. Please use module.to(new_dtype).')

    @property
    def device(self) -> torch.device:
        if False:
            return 10
        device = self._device
        if device.type == 'cuda' and device.index is None:
            return torch.device(f'cuda:{torch.cuda.current_device()}')
        return device

    def to(self, *args: Any, **kwargs: Any) -> Self:
        if False:
            return 10
        'See :meth:`torch.nn.Module.to`.'
        (device, dtype) = torch._C._nn._parse_to(*args, **kwargs)[:2]
        self.__update_properties(device=device, dtype=dtype)
        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[Union[torch.device, int]]=None) -> Self:
        if False:
            return 10
        'Moves all model parameters and buffers to the GPU. This also makes associated parameters and buffers\n        different objects. So it should be called before constructing optimizer if the module will live on GPU while\n        being optimized.\n\n        Arguments:\n            device: If specified, all parameters will be copied to that device. If `None`, the current CUDA device\n                index will be used.\n\n        Returns:\n            Module: self\n\n        '
        if device is None:
            device = torch.device('cuda', torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device('cuda', index=device)
        self.__update_properties(device=device)
        return super().cuda(device=device)

    def cpu(self) -> Self:
        if False:
            return 10
        'See :meth:`torch.nn.Module.cpu`.'
        self.__update_properties(device=torch.device('cpu'))
        return super().cpu()

    def type(self, dst_type: Union[str, torch.dtype]) -> Self:
        if False:
            print('Hello World!')
        'See :meth:`torch.nn.Module.type`.'
        self.__update_properties(dtype=dst_type)
        return super().type(dst_type=dst_type)

    def float(self) -> Self:
        if False:
            print('Hello World!')
        'See :meth:`torch.nn.Module.float`.'
        self.__update_properties(dtype=torch.float)
        return super().float()

    def double(self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'See :meth:`torch.nn.Module.double`.'
        self.__update_properties(dtype=torch.double)
        return super().double()

    def half(self) -> Self:
        if False:
            while True:
                i = 10
        'See :meth:`torch.nn.Module.half`.'
        self.__update_properties(dtype=torch.half)
        return super().half()

    def __update_properties(self, device: Optional[torch.device]=None, dtype: Optional[Union[str, torch.dtype]]=None) -> None:
        if False:
            while True:
                i = 10

        def apply_fn(module: Union[_DeviceDtypeModuleMixin, Module]) -> None:
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(module, _DeviceDtypeModuleMixin):
                return
            if device is not None:
                module._device = device
            if dtype is not None:
                module._dtype = dtype
        self.apply(apply_fn)