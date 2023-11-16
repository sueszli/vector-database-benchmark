import copy
from paddle.nn import Layer
from .config import QuantConfig
from .quantize import Quantization

class QAT(Quantization):
    """
    Tools used to prepare model for quantization-aware training.
    Args:
        config(QuantConfig) - Quantization configuration

    Examples:
        .. code-block:: python

            >>> from paddle.quantization import QAT, QuantConfig
            >>> from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            >>> quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
            >>> q_config = QuantConfig(activation=quanter, weight=quanter)
            >>> qat = QAT(q_config)
    """

    def __init__(self, config: QuantConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)

    def quantize(self, model: Layer, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a model for quantization-aware training.\n\n        The quantization configuration will be propagated in the model.\n        And it will insert fake quanters into the model to simulate the quantization.\n\n        Args:\n            model(Layer) - The model to be quantized.\n            inplace(bool) - Whether to modify the model in-place.\n\n        Return: The prepared model for quantization-aware training.\n\n        Examples:\n            .. code-block:: python\n\n                >>> from paddle.quantization import QAT, QuantConfig\n                >>> from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver\n                >>> from paddle.vision.models import LeNet\n\n                >>> quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)\n                >>> q_config = QuantConfig(activation=quanter, weight=quanter)\n                >>> qat = QAT(q_config)\n                >>> model = LeNet()\n                >>> quant_model = qat.quantize(model)\n                >>> print(quant_model)\n                LeNet(\n                  (features): Sequential(\n                    (0): QuantedConv2D(\n                      (weight_quanter): FakeQuanterWithAbsMaxObserverLayer()\n                      (activation_quanter): FakeQuanterWithAbsMaxObserverLayer()\n                    )\n                    (1): ObserveWrapper(\n                      (_observer): FakeQuanterWithAbsMaxObserverLayer()\n                      (_observed): ReLU()\n                    )\n                    (2): ObserveWrapper(\n                      (_observer): FakeQuanterWithAbsMaxObserverLayer()\n                      (_observed): MaxPool2D(kernel_size=2, stride=2, padding=0)\n                    )\n                    (3): QuantedConv2D(\n                      (weight_quanter): FakeQuanterWithAbsMaxObserverLayer()\n                      (activation_quanter): FakeQuanterWithAbsMaxObserverLayer()\n                    )\n                    (4): ObserveWrapper(\n                      (_observer): FakeQuanterWithAbsMaxObserverLayer()\n                      (_observed): ReLU()\n                    )\n                    (5): ObserveWrapper(\n                      (_observer): FakeQuanterWithAbsMaxObserverLayer()\n                      (_observed): MaxPool2D(kernel_size=2, stride=2, padding=0)\n                    )\n                  )\n                  (fc): Sequential(\n                    (0): QuantedLinear(\n                      (weight_quanter): FakeQuanterWithAbsMaxObserverLayer()\n                      (activation_quanter): FakeQuanterWithAbsMaxObserverLayer()\n                    )\n                    (1): QuantedLinear(\n                      (weight_quanter): FakeQuanterWithAbsMaxObserverLayer()\n                      (activation_quanter): FakeQuanterWithAbsMaxObserverLayer()\n                    )\n                    (2): QuantedLinear(\n                      (weight_quanter): FakeQuanterWithAbsMaxObserverLayer()\n                      (activation_quanter): FakeQuanterWithAbsMaxObserverLayer()\n                    )\n                  )\n                )\n        '
        assert model.training, 'Quantization-Aware Training shoud work on training models. Please set training mode by model.train().'
        _model = model if inplace else copy.deepcopy(model)
        self._config._specify(_model)
        self._convert_to_quant_layers(_model, self._config)
        self._insert_activation_observers(_model, self._config)
        return _model