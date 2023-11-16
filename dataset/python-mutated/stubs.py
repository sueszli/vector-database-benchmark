from torch import nn

class QuantStub(nn.Module):
    """Quantize stub module, before calibration, this is same as an observer,
    it will be swapped as `nnq.Quantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

    def __init__(self, qconfig=None):
        if False:
            return 10
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return x

class DeQuantStub(nn.Module):
    """Dequantize stub module, before calibration, this is same as identity,
    this will be swapped as `nnq.DeQuantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

    def __init__(self, qconfig=None):
        if False:
            while True:
                i = 10
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x

class QuantWrapper(nn.Module):
    """A wrapper class that wraps the input module, adds QuantStub and
    DeQuantStub and surround the call to module with call to quant and dequant
    modules.

    This is used by the `quantization` utility functions to add the quant and
    dequant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `QuantStub`
    will be swapped to `nnq.Quantize` which does actual quantization. Similarly
    for `DeQuantStub`.
    """
    quant: QuantStub
    dequant: DeQuantStub
    module: nn.Module

    def __init__(self, module):
        if False:
            print('Hello World!')
        super().__init__()
        qconfig = getattr(module, 'qconfig', None)
        self.add_module('quant', QuantStub(qconfig))
        self.add_module('dequant', DeQuantStub(qconfig))
        self.add_module('module', module)
        self.train(module.training)

    def forward(self, X):
        if False:
            print('Hello World!')
        X = self.quant(X)
        X = self.module(X)
        return self.dequant(X)