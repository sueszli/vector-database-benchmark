from torch import nn

class ModelForwardWrapper(nn.Module):
    """Model that calls specified method instead of forward.

    Args:
        model: @TODO: docs
        method_name: @TODO: docs

    (Workaround, single method tracing is not supported)
    """

    def __init__(self, model, method_name):
        if False:
            return 10
        'Init'
        super().__init__()
        self.model = model
        self.method_name = method_name

    def forward(self, *args, **kwargs):
        if False:
            return 10
        'Forward pass.\n\n        Args:\n            *args: some args\n            **kwargs: some kwargs\n\n        Returns:\n            output: specified method output\n        '
        return getattr(self.model, self.method_name)(*args, **kwargs)
__all__ = ['ModelForwardWrapper']