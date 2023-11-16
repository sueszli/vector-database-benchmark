import copy
from paddle.distributed import fleet
from paddle.nn import Layer
from .config import QuantConfig
from .quantize import Quantization

class PTQ(Quantization):
    """
    Applying post training quantization to the model.
    """

    def __init__(self, config: QuantConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)

    def _is_parallel_training(self):
        if False:
            print('Hello World!')
        try:
            if fleet.worker_num() > 2:
                return True
            else:
                return False
        except Exception:
            return False

    def quantize(self, model: Layer, inplace=False):
        if False:
            return 10
        '\n        Create a model for post-training quantization.\n\n        The quantization configuration will be propagated in the model.\n        And it will insert observers into the model to collect and compute\n        quantization parameters.\n\n        Args:\n            model(Layer) - The model to be quantized.\n            inplace(bool) - Whether to modify the model in-place.\n\n        Return: The prepared model for post-training quantization.\n\n        Examples:\n            .. code-block:: python\n\n                >>> from paddle.quantization import PTQ, QuantConfig\n                >>> from paddle.quantization.observers import AbsmaxObserver\n                >>> from paddle.vision.models import LeNet\n\n                >>> observer = AbsmaxObserver()\n                >>> q_config = QuantConfig(activation=observer, weight=observer)\n                >>> ptq = PTQ(q_config)\n                >>> model = LeNet()\n                >>> model.eval()\n                >>> quant_model = ptq.quantize(model)\n                >>> print(quant_model)\n                LeNet(\n                  (features): Sequential(\n                    (0): QuantedConv2D(\n                      (weight_quanter): AbsmaxObserverLayer()\n                      (activation_quanter): AbsmaxObserverLayer()\n                    )\n                    (1): ObserveWrapper(\n                      (_observer): AbsmaxObserverLayer()\n                      (_observed): ReLU()\n                    )\n                    (2): ObserveWrapper(\n                      (_observer): AbsmaxObserverLayer()\n                      (_observed): MaxPool2D(kernel_size=2, stride=2, padding=0)\n                    )\n                    (3): QuantedConv2D(\n                      (weight_quanter): AbsmaxObserverLayer()\n                      (activation_quanter): AbsmaxObserverLayer()\n                    )\n                    (4): ObserveWrapper(\n                      (_observer): AbsmaxObserverLayer()\n                      (_observed): ReLU()\n                    )\n                    (5): ObserveWrapper(\n                      (_observer): AbsmaxObserverLayer()\n                      (_observed): MaxPool2D(kernel_size=2, stride=2, padding=0)\n                    )\n                  )\n                  (fc): Sequential(\n                    (0): QuantedLinear(\n                      (weight_quanter): AbsmaxObserverLayer()\n                      (activation_quanter): AbsmaxObserverLayer()\n                    )\n                    (1): QuantedLinear(\n                      (weight_quanter): AbsmaxObserverLayer()\n                      (activation_quanter): AbsmaxObserverLayer()\n                    )\n                    (2): QuantedLinear(\n                      (weight_quanter): AbsmaxObserverLayer()\n                      (activation_quanter): AbsmaxObserverLayer()\n                    )\n                  )\n                )\n        '
        _model = model
        if not inplace:
            assert not self._is_parallel_training(), "'inplace' is not compatible with parallel training."
            _model = copy.deepcopy(model)
            _model.eval()
        assert not model.training, 'Post-Training Quantization shoud not work on training models. Please set evaluation mode by model.eval().'
        self._config._specify(_model)
        self._convert_to_quant_layers(_model, self._config)
        self._insert_activation_observers(_model, self._config)
        return _model