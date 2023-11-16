import contextlib
import gymnasium as gym
import re
from typing import Dict, List, Union
from ray.util import log_once
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
(tf1, tf, tfv) = try_import_tf()

@PublicAPI
class TFModelV2(ModelV2):
    """TF version of ModelV2, which should contain a tf keras Model.

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        if False:
            return 10
        'Initializes a TFModelV2 instance.\n\n        Here is an example implementation for a subclass\n        ``MyModelClass(TFModelV2)``::\n\n            def __init__(self, *args, **kwargs):\n                super(MyModelClass, self).__init__(*args, **kwargs)\n                input_layer = tf.keras.layers.Input(...)\n                hidden_layer = tf.keras.layers.Dense(...)(input_layer)\n                output_layer = tf.keras.layers.Dense(...)(hidden_layer)\n                value_layer = tf.keras.layers.Dense(...)(hidden_layer)\n                self.base_model = tf.keras.Model(\n                    input_layer, [output_layer, value_layer])\n        '
        super().__init__(obs_space, action_space, num_outputs, model_config, name, framework='tf')
        self.var_list = []
        if tf1.executing_eagerly():
            self.graph = None
        else:
            self.graph = tf1.get_default_graph()

    def context(self) -> contextlib.AbstractContextManager:
        if False:
            return 10
        'Returns a contextmanager for the current TF graph.'
        if self.graph:
            return self.graph.as_default()
        else:
            return ModelV2.context(self)

    def update_ops(self) -> List[TensorType]:
        if False:
            while True:
                i = 10
        'Return the list of update ops for this model.\n\n        For example, this should include any BatchNorm update ops.'
        return []

    def register_variables(self, variables: List[TensorType]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Register the given list of variables with this model.'
        if log_once('deprecated_tfmodelv2_register_variables'):
            deprecation_warning(old='TFModelV2.register_variables', error=False)
        self.var_list.extend(variables)

    @override(ModelV2)
    def variables(self, as_dict: bool=False) -> Union[List[TensorType], Dict[str, TensorType]]:
        if False:
            while True:
                i = 10
        if as_dict:
            if self.var_list:
                return {v.name: v for v in self.var_list}
            else:
                return self._find_sub_modules('', self.__dict__)
        if self.var_list:
            return list(self.var_list)
        else:
            return list(self.variables(as_dict=True).values())

    @override(ModelV2)
    def trainable_variables(self, as_dict: bool=False) -> Union[List[TensorType], Dict[str, TensorType]]:
        if False:
            print('Hello World!')
        if as_dict:
            return {k: v for (k, v) in self.variables(as_dict=True).items() if v.trainable}
        return [v for v in self.variables() if v.trainable]

    @staticmethod
    def _find_sub_modules(current_key, struct):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(struct, tf.keras.models.Model) or isinstance(struct, tf.Module):
            ret = {}
            for var in struct.variables:
                name = re.sub('/', '.', var.name)
                key = current_key + '.' + name
                ret[key] = var
            return ret
        elif isinstance(struct, TFModelV2):
            return {current_key + '.' + key: var for (key, var) in struct.variables(as_dict=True).items()}
        elif isinstance(struct, tf.Variable):
            return {current_key: struct}
        elif isinstance(struct, (tuple, list)):
            ret = {}
            for (i, value) in enumerate(struct):
                sub_vars = TFModelV2._find_sub_modules(current_key + '_{}'.format(i), value)
                ret.update(sub_vars)
            return ret
        elif isinstance(struct, dict):
            if current_key:
                current_key += '_'
            ret = {}
            for (key, value) in struct.items():
                sub_vars = TFModelV2._find_sub_modules(current_key + str(key), value)
                ret.update(sub_vars)
            return ret
        return {}