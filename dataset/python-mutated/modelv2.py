from collections import OrderedDict
import contextlib
import gymnasium as gym
from gymnasium.spaces import Space
import numpy as np
from typing import Dict, List, Any, Union
from ray.rllib.models.preprocessors import get_preprocessor, RepeatedValuesPreprocessor
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import NullContextManager
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.framework import try_import_tf, try_import_torch, TensorType
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.typing import ModelConfigDict, ModelInputDict, TensorStructType
(tf1, tf, tfv) = try_import_tf()
(torch, _) = try_import_torch()

@PublicAPI
class ModelV2:
    """Defines an abstract neural network model for use with RLlib.

    Custom models should extend either TFModelV2 or TorchModelV2 instead of
    this class directly.

    Data flow:
        obs -> forward() -> model_out
            \\-> value_function() -> V(s)
    """

    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int, model_config: ModelConfigDict, name: str, framework: str):
        if False:
            return 10
        'Initializes a ModelV2 instance.\n\n        This method should create any variables used by the model.\n\n        Args:\n            obs_space: Observation space of the target gym\n                env. This may have an `original_space` attribute that\n                specifies how to unflatten the tensor into a ragged tensor.\n            action_space: Action space of the target gym\n                env.\n            num_outputs: Number of output units of the model.\n            model_config: Config for the model, documented\n                in ModelCatalog.\n            name: Name (scope) for the model.\n            framework: Either "tf" or "torch".\n        '
        self.obs_space: Space = obs_space
        self.action_space: Space = action_space
        self.num_outputs: int = num_outputs
        self.model_config: ModelConfigDict = model_config
        self.name: str = name or 'default_model'
        self.framework: str = framework
        self._last_output = None
        self.time_major = self.model_config.get('_time_major')
        self.view_requirements = {SampleBatch.OBS: ViewRequirement(shift=0, space=self.obs_space)}

    @PublicAPI
    def get_initial_state(self) -> List[TensorType]:
        if False:
            i = 10
            return i + 15
        'Get the initial recurrent state values for the model.\n\n        Returns:\n            List of np.array (for tf) or Tensor (for torch) objects containing the\n            initial hidden state of an RNN, if applicable.\n\n        .. testcode::\n            :skipif: True\n\n            import numpy as np\n            from ray.rllib.models.modelv2 import ModelV2\n            class MyModel(ModelV2):\n                # ...\n                def get_initial_state(self):\n                    return [\n                        np.zeros(self.cell_size, np.float32),\n                        np.zeros(self.cell_size, np.float32),\n                    ]\n        '
        return []

    @PublicAPI
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        if False:
            for i in range(10):
                print('nop')
        'Call the model with the given input tensors and state.\n\n        Any complex observations (dicts, tuples, etc.) will be unpacked by\n        __call__ before being passed to forward(). To access the flattened\n        observation tensor, refer to input_dict["obs_flat"].\n\n        This method can be called any number of times. In eager execution,\n        each call to forward() will eagerly evaluate the model. In symbolic\n        execution, each call to forward creates a computation graph that\n        operates over the variables of this model (i.e., shares weights).\n\n        Custom models should override this instead of __call__.\n\n        Args:\n            input_dict: dictionary of input tensors, including "obs",\n                "obs_flat", "prev_action", "prev_reward", "is_training",\n                "eps_id", "agent_id", "infos", and "t".\n            state: list of state tensors with sizes matching those\n                returned by get_initial_state + the batch dimension\n            seq_lens: 1d tensor holding input sequence lengths\n\n        Returns:\n            A tuple consisting of the model output tensor of size\n            [BATCH, num_outputs] and the list of new RNN state(s) if any.\n\n        .. testcode::\n            :skipif: True\n\n            import numpy as np\n            from ray.rllib.models.modelv2 import ModelV2\n            class MyModel(ModelV2):\n                # ...\n                def forward(self, input_dict, state, seq_lens):\n                    model_out, self._value_out = self.base_model(\n                        input_dict["obs"])\n                    return model_out, state\n        '
        raise NotImplementedError

    @PublicAPI
    def value_function(self) -> TensorType:
        if False:
            return 10
        'Returns the value function output for the most recent forward pass.\n\n        Note that a `forward` call has to be performed first, before this\n        methods can return anything and thus that calling this method does not\n        cause an extra forward pass through the network.\n\n        Returns:\n            Value estimate tensor of shape [BATCH].\n        '
        raise NotImplementedError

    @PublicAPI
    def custom_loss(self, policy_loss: TensorType, loss_inputs: Dict[str, TensorType]) -> Union[List[TensorType], TensorType]:
        if False:
            i = 10
            return i + 15
        "Override to customize the loss function used to optimize this model.\n\n        This can be used to incorporate self-supervised losses (by defining\n        a loss over existing input and output tensors of this model), and\n        supervised losses (by defining losses over a variable-sharing copy of\n        this model's layers).\n\n        You can find an runnable example in examples/custom_loss.py.\n\n        Args:\n            policy_loss: List of or single policy loss(es) from the policy.\n            loss_inputs: map of input placeholders for rollout data.\n\n        Returns:\n            List of or scalar tensor for the customized loss(es) for this\n            model.\n        "
        return policy_loss

    @PublicAPI
    def metrics(self) -> Dict[str, TensorType]:
        if False:
            i = 10
            return i + 15
        'Override to return custom metrics from your model.\n\n        The stats will be reported as part of the learner stats, i.e.,\n        info.learner.[policy_id, e.g. "default_policy"].model.key1=metric1\n\n        Returns:\n            The custom metrics for this model.\n        '
        return {}

    def __call__(self, input_dict: Union[SampleBatch, ModelInputDict], state: List[Any]=None, seq_lens: TensorType=None) -> (TensorType, List[TensorType]):
        if False:
            while True:
                i = 10
        'Call the model with the given input tensors and state.\n\n        This is the method used by RLlib to execute the forward pass. It calls\n        forward() internally after unpacking nested observation tensors.\n\n        Custom models should override forward() instead of __call__.\n\n        Args:\n            input_dict: Dictionary of input tensors.\n            state: list of state tensors with sizes matching those\n                returned by get_initial_state + the batch dimension\n            seq_lens: 1D tensor holding input sequence lengths.\n\n        Returns:\n            A tuple consisting of the model output tensor of size\n                [BATCH, output_spec.size] or a list of tensors corresponding to\n                output_spec.shape_list, and a list of state tensors of\n                [BATCH, state_size_i] if any.\n        '
        if isinstance(input_dict, SampleBatch):
            restored = input_dict.copy(shallow=True)
        else:
            restored = input_dict.copy()
        if not state:
            state = []
            i = 0
            while 'state_in_{}'.format(i) in input_dict:
                state.append(input_dict['state_in_{}'.format(i)])
                i += 1
        if seq_lens is None:
            seq_lens = input_dict.get(SampleBatch.SEQ_LENS)
        if self.model_config.get('_disable_preprocessor_api'):
            restored['obs_flat'] = input_dict['obs']
        else:
            restored['obs'] = restore_original_dimensions(input_dict['obs'], self.obs_space, self.framework)
            try:
                if len(input_dict['obs'].shape) > 2:
                    restored['obs_flat'] = flatten(input_dict['obs'], self.framework)
                else:
                    restored['obs_flat'] = input_dict['obs']
            except AttributeError:
                restored['obs_flat'] = input_dict['obs']
        with self.context():
            res = self.forward(restored, state or [], seq_lens)
        if isinstance(input_dict, SampleBatch):
            input_dict.accessed_keys = restored.accessed_keys - {'obs_flat'}
            input_dict.deleted_keys = restored.deleted_keys
            input_dict.added_keys = restored.added_keys - {'obs_flat'}
        if not isinstance(res, list) and (not isinstance(res, tuple)) or len(res) != 2:
            raise ValueError('forward() must return a tuple of (output, state) tensors, got {}'.format(res))
        (outputs, state_out) = res
        if not isinstance(state_out, list):
            raise ValueError('State output is not a list: {}'.format(state_out))
        self._last_output = outputs
        return (outputs, state_out if len(state_out) > 0 else state or [])

    def import_from_h5(self, h5_file: str) -> None:
        if False:
            i = 10
            return i + 15
        'Imports weights from an h5 file.\n\n        Args:\n            h5_file: The h5 file name to import weights from.\n\n        .. testcode::\n            :skipif: True\n\n            from ray.rllib.algorithms.ppo import PPO\n            algo = PPO(...)\n            algo.import_policy_model_from_h5("/tmp/weights.h5")\n            for _ in range(10):\n                algo.train()\n        '
        raise NotImplementedError

    @PublicAPI
    def last_output(self) -> TensorType:
        if False:
            while True:
                i = 10
        'Returns the last output returned from calling the model.'
        return self._last_output

    @PublicAPI
    def context(self) -> contextlib.AbstractContextManager:
        if False:
            print('Hello World!')
        'Returns a contextmanager for the current forward pass.'
        return NullContextManager()

    @PublicAPI
    def variables(self, as_dict: bool=False) -> Union[List[TensorType], Dict[str, TensorType]]:
        if False:
            print('Hello World!')
        'Returns the list (or a dict) of variables for this model.\n\n        Args:\n            as_dict: Whether variables should be returned as dict-values\n                (using descriptive str keys).\n\n        Returns:\n            The list (or dict if `as_dict` is True) of all variables of this\n            ModelV2.\n        '
        raise NotImplementedError

    @PublicAPI
    def trainable_variables(self, as_dict: bool=False) -> Union[List[TensorType], Dict[str, TensorType]]:
        if False:
            while True:
                i = 10
        'Returns the list of trainable variables for this model.\n\n        Args:\n            as_dict: Whether variables should be returned as dict-values\n                (using descriptive keys).\n\n        Returns:\n            The list (or dict if `as_dict` is True) of all trainable\n            (tf)/requires_grad (torch) variables of this ModelV2.\n        '
        raise NotImplementedError

    @PublicAPI
    def is_time_major(self) -> bool:
        if False:
            return 10
        'If True, data for calling this ModelV2 must be in time-major format.\n\n        Returns\n            Whether this ModelV2 requires a time-major (TxBx...) data\n            format.\n        '
        return self.time_major is True

    @Deprecated(new='ModelV2.__call__()', error=True)
    def from_batch(self, train_batch: SampleBatch, is_training: bool=True) -> (TensorType, List[TensorType]):
        if False:
            print('Hello World!')
        'Convenience function that calls this model with a tensor batch.\n\n        All this does is unpack the tensor batch to call this model with the\n        right input dict, state, and seq len arguments.\n        '
        input_dict = train_batch.copy()
        input_dict.set_training(is_training)
        states = []
        i = 0
        while 'state_in_{}'.format(i) in input_dict:
            states.append(input_dict['state_in_{}'.format(i)])
            i += 1
        ret = self.__call__(input_dict, states, input_dict.get(SampleBatch.SEQ_LENS))
        return ret

@DeveloperAPI
def flatten(obs: TensorType, framework: str) -> TensorType:
    if False:
        while True:
            i = 10
    'Flatten the given tensor.'
    if framework in ['tf2', 'tf']:
        return tf1.keras.layers.Flatten()(obs)
    elif framework == 'torch':
        assert torch is not None
        return torch.flatten(obs, start_dim=1)
    else:
        raise NotImplementedError('flatten', framework)

@DeveloperAPI
def restore_original_dimensions(obs: TensorType, obs_space: Space, tensorlib: Any=tf) -> TensorStructType:
    if False:
        for i in range(10):
            print('nop')
    'Unpacks Dict and Tuple space observations into their original form.\n\n    This is needed since we flatten Dict and Tuple observations in transit\n    within a SampleBatch. Before sending them to the model though, we should\n    unflatten them into Dicts or Tuples of tensors.\n\n    Args:\n        obs: The flattened observation tensor.\n        obs_space: The flattened obs space. If this has the\n            `original_space` attribute, we will unflatten the tensor to that\n            shape.\n        tensorlib: The library used to unflatten (reshape) the array/tensor.\n\n    Returns:\n        single tensor or dict / tuple of tensors matching the original\n        observation space.\n    '
    if tensorlib in ['tf', 'tf2']:
        assert tf is not None
        tensorlib = tf
    elif tensorlib == 'torch':
        assert torch is not None
        tensorlib = torch
    elif tensorlib == 'numpy':
        assert np is not None
        tensorlib = np
    original_space = getattr(obs_space, 'original_space', obs_space)
    return _unpack_obs(obs, original_space, tensorlib=tensorlib)
_cache = {}

def _unpack_obs(obs: TensorType, space: Space, tensorlib: Any=tf) -> TensorStructType:
    if False:
        while True:
            i = 10
    'Unpack a flattened Dict or Tuple observation array/tensor.\n\n    Args:\n        obs: The flattened observation tensor, with last dimension equal to\n            the flat size and any number of batch dimensions. For example, for\n            Box(4,), the obs may have shape [B, 4], or [B, N, M, 4] in case\n            the Box was nested under two Repeated spaces.\n        space: The original space prior to flattening\n        tensorlib: The library used to unflatten (reshape) the array/tensor\n    '
    if isinstance(space, (gym.spaces.Dict, gym.spaces.Tuple, Repeated)):
        if isinstance(space, gym.spaces.Tuple) and isinstance(obs, (list, tuple)) or (isinstance(space, gym.spaces.Dict) and isinstance(obs, dict)):
            return obs
        if id(space) in _cache:
            prep = _cache[id(space)]
        else:
            prep = get_preprocessor(space)(space)
            if len(_cache) < 999:
                _cache[id(space)] = prep
        if len(obs.shape) < 2 or obs.shape[-1] != prep.shape[0]:
            raise ValueError('Expected flattened obs shape of [..., {}], got {}'.format(prep.shape[0], obs.shape))
        offset = 0
        if tensorlib == tf:

            def get_value(v):
                if False:
                    i = 10
                    return i + 15
                if v is None:
                    return -1
                elif isinstance(v, int):
                    return v
                elif v.value is None:
                    return -1
                else:
                    return v.value
            batch_dims = [get_value(v) for v in obs.shape[:-1]]
        else:
            batch_dims = list(obs.shape[:-1])
        if isinstance(space, gym.spaces.Tuple):
            assert len(prep.preprocessors) == len(space.spaces), len(prep.preprocessors) == len(space.spaces)
            u = []
            for (p, v) in zip(prep.preprocessors, space.spaces):
                obs_slice = obs[..., offset:offset + p.size]
                offset += p.size
                u.append(_unpack_obs(tensorlib.reshape(obs_slice, batch_dims + list(p.shape)), v, tensorlib=tensorlib))
        elif isinstance(space, gym.spaces.Dict):
            assert len(prep.preprocessors) == len(space.spaces), len(prep.preprocessors) == len(space.spaces)
            u = OrderedDict()
            for (p, (k, v)) in zip(prep.preprocessors, space.spaces.items()):
                obs_slice = obs[..., offset:offset + p.size]
                offset += p.size
                u[k] = _unpack_obs(tensorlib.reshape(obs_slice, batch_dims + list(p.shape)), v, tensorlib=tensorlib)
        else:
            assert isinstance(prep, RepeatedValuesPreprocessor), prep
            child_size = prep.child_preprocessor.size
            lengths = obs[..., 0]
            with_repeat_dim = tensorlib.reshape(obs[..., 1:], batch_dims + [space.max_len, child_size])
            u = _unpack_obs(with_repeat_dim, space.child_space, tensorlib=tensorlib)
            return RepeatedValues(u, lengths=lengths, max_len=prep._obs_space.max_len)
        return u
    else:
        return obs