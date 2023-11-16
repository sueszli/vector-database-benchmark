from paddle import nn
__all__ = []

class MetaParallelBase(nn.Layer):

    def __init__(self, layers, hcg, strategy):
        if False:
            print('Hello World!')
        super().__init__(layers.full_name() + '_meta_parallel_base')
        self._layers = layers
        self._hcg = hcg
        self._strategy = strategy
        self._prepare_for_model()

    def _prepare_for_model(self):
        if False:
            i = 10
            return i + 15
        pass

    def _pre_forward(self, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        pass

    def forward(self, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        self._pre_forward(*inputs, **kwargs)
        output = self._layers(*inputs, **kwargs)
        self._post_forward(output)
        return output

    def _post_forward(self, output):
        if False:
            print('Hello World!')
        pass