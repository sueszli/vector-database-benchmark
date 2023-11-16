from .module import Module

class QuantStub(Module):
    """A helper :class:`~.Module` simply returning input. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.QuantStub` using :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        if False:
            while True:
                i = 10
        return inp

class DequantStub(Module):
    """A helper :class:`~.Module` simply returning input. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.DequantStub` using :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        if False:
            while True:
                i = 10
        return inp