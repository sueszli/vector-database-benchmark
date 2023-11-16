import gymnasium as gym
from typing import Dict, List, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
(_, nn) = try_import_torch()

@PublicAPI
class TorchModelV2(ModelV2):
    """Torch version of ModelV2.

    Note that this class by itself is not a valid model unless you
    inherit from nn.Module and implement forward() in a subclass."""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a TorchModelV2.\n\n        Here is an example implementation for a subclass\n        ``MyModelClass(TorchModelV2, nn.Module)``::\n\n            def __init__(self, *args, **kwargs):\n                TorchModelV2.__init__(self, *args, **kwargs)\n                nn.Module.__init__(self)\n                self._hidden_layers = nn.Sequential(...)\n                self._logits = ...\n                self._value_branch = ...\n        '
        if not isinstance(self, nn.Module):
            raise ValueError('Subclasses of TorchModelV2 must also inherit from nn.Module, e.g., MyModel(TorchModelV2, nn.Module)')
        ModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, framework='torch')
        self.tower_stats = {}

    @override(ModelV2)
    def variables(self, as_dict: bool=False) -> Union[List[TensorType], Dict[str, TensorType]]:
        if False:
            i = 10
            return i + 15
        p = list(self.parameters())
        if as_dict:
            return {k: p[i] for (i, k) in enumerate(self.state_dict().keys())}
        return p

    @override(ModelV2)
    def trainable_variables(self, as_dict: bool=False) -> Union[List[TensorType], Dict[str, TensorType]]:
        if False:
            print('Hello World!')
        if as_dict:
            return {k: v for (k, v) in self.variables(as_dict=True).items() if v.requires_grad}
        return [v for v in self.variables() if v.requires_grad]