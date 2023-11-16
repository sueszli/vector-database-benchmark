"""Base class for deriving trainable modules."""
from typing import Union, Optional
import ivy
from ivy.stateful.module import Module

class Sequential(Module):

    def __init__(self, *sub_modules: Module, device: Optional[Union[ivy.Device, ivy.NativeDevice]]=None, v: Optional[Union[ivy.Array, ivy.NativeArray]]=None, dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]]=None):
        if False:
            return 10
        "\n        Initialize a sequential container. Modules will be added to it in the order they\n        are passed in the constructor.\n\n        Parameters\n        ----------\n        submodules\n            Submodules to chain together into a sequence.\n        device\n            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'\n            etc.\n        v\n            the variables for each submodule in the sequence, constructed internally by\n            default.\n        "
        if v is not None:
            for (i, submod) in enumerate(sub_modules):
                try:
                    submod.v = v['submodules'][f'v{str(i)}']
                except KeyError:
                    if submod.v:
                        raise ivy.utils.exceptions.IvyException('variables v passed to Sequential class must have key chains in the form of "submodules/v{}", where {} is an idx')
        self._submodules = list(sub_modules)
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self._submodules)

    def _forward(self, inputs):
        if False:
            while True:
                i = 10
        '\n        Perform forward pass of the Sequential container.\n\n        Parameters\n        ----------\n        inputs\n            Inputs to process.\n\n        Returns\n        -------\n        ret\n            The output after each of the layers in the Sequential has been applied.\n        '
        x = inputs
        for (i, submod) in enumerate(self._submodules):
            try:
                x = submod(x, v=self.v.submodules[f'v{str(i)}'])
            except KeyError:
                if submod.v:
                    raise ivy.utils.exceptions.IvyException('variables v passed to Sequential class must have key chains in the form of "submodules/v{}", where {} is an idx')
                x = submod(x)
        return x