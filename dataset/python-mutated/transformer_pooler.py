from typing import Union, TYPE_CHECKING
import torch
from allennlp.common import FromParams
from allennlp.modules.transformer.activation_layer import ActivationLayer
if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

class TransformerPooler(ActivationLayer, FromParams):
    _pretrained_relevant_module = ['pooler', 'bert.pooler', 'roberta.pooler']

    def __init__(self, hidden_size: int, intermediate_size: int, activation: Union[str, torch.nn.Module]='relu'):
        if False:
            while True:
                i = 10
        super().__init__(hidden_size, intermediate_size, activation, pool=True)

    @classmethod
    def _from_config(cls, config: 'PretrainedConfig', **kwargs):
        if False:
            return 10
        return cls(config.hidden_size, config.hidden_size, 'tanh')