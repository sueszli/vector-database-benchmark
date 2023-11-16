import json
import logging
import pathlib
from typing import Any, Callable, Hashable, Mapping, Optional, Sequence, Tuple, Union
from ray.rllib.core.learner.learner import FrameworkHyperparameters, Learner, LearnerHyperparameters
from ray.rllib.core.rl_module.rl_module import RLModule, ModuleID, SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.tf.tf_rl_module import TfRLModule
from ray.rllib.policy.eager_tf_policy import _convert_to_tf
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import override, OverrideToImplementCustomLogic
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.serialization import convert_numpy_to_python_primitives
from ray.rllib.utils.typing import Optimizer, Param, ParamDict, TensorType
(tf1, tf, tfv) = try_import_tf()
logger = logging.getLogger(__name__)

class TfLearner(Learner):
    framework: str = 'tf2'

    def __init__(self, *, framework_hyperparameters: Optional[FrameworkHyperparameters]=None, **kwargs):
        if False:
            return 10
        try:
            tf1.enable_v2_behavior()
        except ValueError:
            pass
        super().__init__(framework_hyperparameters=framework_hyperparameters or FrameworkHyperparameters(), **kwargs)
        self._enable_tf_function = self._framework_hyperparameters.eager_tracing
        self._strategy: tf.distribute.Strategy = None

    @OverrideToImplementCustomLogic
    @override(Learner)
    def configure_optimizers_for_module(self, module_id: ModuleID, hps: LearnerHyperparameters) -> None:
        if False:
            print('Hello World!')
        module = self._module[module_id]
        optimizer = tf.keras.optimizers.Adam()
        params = self.get_parameters(module)
        optimizer.build(module.trainable_variables)
        self.register_optimizer(module_id=module_id, optimizer=optimizer, params=params, lr_or_lr_schedule=hps.learning_rate)

    @override(Learner)
    def compute_gradients(self, loss_per_module: Mapping[str, TensorType], gradient_tape: 'tf.GradientTape', **kwargs) -> ParamDict:
        if False:
            for i in range(10):
                print('nop')
        grads = gradient_tape.gradient(loss_per_module[ALL_MODULES], self._params)
        return grads

    @override(Learner)
    def apply_gradients(self, gradients_dict: ParamDict) -> None:
        if False:
            i = 10
            return i + 15
        for optimizer in self._optimizer_parameters:
            optim_grad_dict = self.filter_param_dict_for_optimizer(optimizer=optimizer, param_dict=gradients_dict)
            variable_list = []
            gradient_list = []
            for (param_ref, grad) in optim_grad_dict.items():
                if grad is not None:
                    variable_list.append(self._params[param_ref])
                    gradient_list.append(grad)
            optimizer.apply_gradients(zip(gradient_list, variable_list))

    @override(Learner)
    def load_state(self, path: Union[str, pathlib.Path]) -> None:
        if False:
            while True:
                i = 10
        with self._strategy.scope():
            super().load_state(path)

    def _save_optimizer_hparams(self, path: pathlib.Path, optim: 'tf.keras.optimizers.Optimizer', optim_name: str) -> None:
        if False:
            print('Hello World!')
        'Save the hyperparameters of optim to path/optim_name_hparams.json.\n\n        Args:\n            path: The path to the directory to save the hyperparameters to.\n            optim: The optimizer to save the hyperparameters of.\n            optim_name: The name of the optimizer.\n\n        '
        hparams = tf.keras.optimizers.serialize(optim)
        hparams = tf.nest.map_structure(convert_numpy_to_python_primitives, hparams)
        with open(path / f'{optim_name}_hparams.json', 'w') as f:
            json.dump(hparams, f)

    def _save_optimizer_state(self, path: pathlib.Path, optim: 'tf.keras.optimizers.Optimizer', optim_name: str) -> None:
        if False:
            print('Hello World!')
        'Save the state variables of optim to path/optim_name_state.txt.\n\n        Args:\n            path: The path to the directory to save the state to.\n            optim: The optimizer to save the state of.\n            optim_name: The name of the optimizer.\n\n        '
        state = optim.variables()
        serialized_tensors = [tf.io.serialize_tensor(tensor) for tensor in state]
        contents = tf.strings.join(serialized_tensors, separator='tensor: ')
        tf.io.write_file(str(path / f'{optim_name}_state.txt'), contents)

    @override(Learner)
    def _save_optimizers(self, path: Union[str, pathlib.Path]) -> None:
        if False:
            for i in range(10):
                print('nop')
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for (name, optim) in self._named_optimizers.items():
            self._save_optimizer_hparams(path, optim, name)
            self._save_optimizer_state(path, optim, name)

    def _load_optimizer_from_hparams(self, path: pathlib.Path, optim_name: str) -> 'tf.keras.optimizers.Optimizer':
        if False:
            print('Hello World!')
        'Load an optimizer from the hyperparameters saved at path/optim_name_hparams.json.\n\n        Args:\n            path: The path to the directory to load the hyperparameters from.\n            optim_name: The name of the optimizer.\n\n        Returns:\n            The optimizer loaded from the hyperparameters.\n\n        '
        with open(path / f'{optim_name}_hparams.json', 'r') as f:
            state = json.load(f)
        return tf.keras.optimizers.deserialize(state)

    def _load_optimizer_state(self, path: pathlib.Path, optim: 'tf.keras.optimizers.Optimizer', optim_name: str) -> None:
        if False:
            return 10
        'Load the state of optim from the state saved at path/optim_name_state.txt.\n\n        Args:\n            path: The path to the directory to load the state from.\n            optim: The optimizer to load the state into.\n            optim_name: The name of the optimizer.\n\n        '
        contents = tf.io.read_file(str(path / f'{optim_name}_state.txt'))
        serialized_tensors = tf.strings.split(contents, sep='tensor: ')
        unserialized_optim_state = []
        for (serialized_tensor, optim_tensor) in zip(serialized_tensors, optim.variables()):
            unserialized_optim_state.append(tf.io.parse_tensor(serialized_tensor, optim_tensor.dtype))
        optim.set_weights(unserialized_optim_state)

    @override(Learner)
    def _load_optimizers(self, path: Union[str, pathlib.Path]) -> None:
        if False:
            for i in range(10):
                print('nop')
        path = pathlib.Path(path)
        for name in self._named_optimizers.keys():
            new_optim = self._load_optimizer_from_hparams(path, name)
            old_optim = self._named_optimizers[name]
            self._named_optimizers[name] = new_optim
            param_seq = self._optimizer_parameters.pop(old_optim)
            self._optimizer_parameters[new_optim] = []
            for param_ref in param_seq:
                self._optimizer_parameters[new_optim].append(param_ref)
            del old_optim
            variable_list = [self._params[param_ref] for param_ref in self._optimizer_parameters[new_optim]]
            new_optim.build(variable_list)
            self._load_optimizer_state(path, new_optim, name)

    @override(Learner)
    def set_module_state(self, state: Mapping[str, Any]) -> None:
        if False:
            return 10
        self._module.set_state(state)

    @override(Learner)
    def get_optimizer_state(self) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        optim_state = {}
        with tf.init_scope():
            for (name, optim) in self._named_optimizers.items():
                optim_state[name] = [var.numpy() for var in optim.variables()]
        return optim_state

    @override(Learner)
    def set_optimizer_state(self, state: Mapping[str, Any]) -> None:
        if False:
            print('Hello World!')
        for (name, state_array) in state.items():
            if name not in self._named_optimizers:
                raise ValueError(f'Optimizer {name} in weights is not known.Known optimizers are {self._named_optimizers.keys()}')
            optim = self._named_optimizers[name]
            optim.set_weights(state_array)

    @override(Learner)
    def get_param_ref(self, param: Param) -> Hashable:
        if False:
            print('Hello World!')
        return param.ref()

    @override(Learner)
    def get_parameters(self, module: RLModule) -> Sequence[Param]:
        if False:
            while True:
                i = 10
        return list(module.trainable_variables)

    def _is_module_compatible_with_learner(self, module: RLModule) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return isinstance(module, TfRLModule)

    @override(Learner)
    def _check_registered_optimizer(self, optimizer: Optimizer, params: Sequence[Param]) -> None:
        if False:
            while True:
                i = 10
        super()._check_registered_optimizer(optimizer, params)
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise ValueError(f'The optimizer ({optimizer}) is not a tf keras optimizer! Only use tf.keras.optimizers.Optimizer subclasses for TfLearner.')
        for param in params:
            if not isinstance(param, tf.Variable):
                raise ValueError(f'One of the parameters ({param}) in the registered optimizer is not a tf.Variable!')

    @override(Learner)
    def _convert_batch_type(self, batch: MultiAgentBatch) -> MultiAgentBatch:
        if False:
            return 10
        batch = _convert_to_tf(batch.policy_batches)
        length = max((len(b) for b in batch.values()))
        batch = MultiAgentBatch(batch, env_steps=length)
        return batch

    @override(Learner)
    def add_module(self, *, module_id: ModuleID, module_spec: SingleAgentRLModuleSpec) -> None:
        if False:
            i = 10
            return i + 15
        with self._strategy.scope():
            super().add_module(module_id=module_id, module_spec=module_spec)
        if self._enable_tf_function:
            self._possibly_traced_update = tf.function(self._untraced_update, reduce_retracing=True)

    @override(Learner)
    def remove_module(self, module_id: ModuleID) -> None:
        if False:
            return 10
        with self._strategy.scope():
            super().remove_module(module_id)
        if self._enable_tf_function:
            self._possibly_traced_update = tf.function(self._untraced_update, reduce_retracing=True)

    def _make_distributed_strategy_if_necessary(self) -> 'tf.distribute.Strategy':
        if False:
            while True:
                i = 10
        'Create a distributed strategy for the learner.\n\n        A stratgey is a tensorflow object that is used for distributing training and\n        gradient computation across multiple devices. By default a no-op strategy is\n        used that is not distributed.\n\n        Returns:\n            A strategy for the learner to use for distributed training.\n\n        '
        if self._distributed:
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
        elif self._use_gpu:
            devices = tf.config.list_logical_devices('GPU')
            assert self._local_gpu_idx < len(devices), f'local_gpu_idx {self._local_gpu_idx} is not a valid GPU id or is not available.'
            local_gpu = [devices[self._local_gpu_idx].name]
            strategy = tf.distribute.MirroredStrategy(devices=local_gpu)
        else:
            strategy = tf.distribute.get_strategy()
        return strategy

    @override(Learner)
    def build(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Build the TfLearner.\n\n        This method is specific TfLearner. Before running super() it sets the correct\n        distributing strategy with the right device, so that computational graph is\n        placed on the correct device. After running super(), depending on eager_tracing\n        flag it will decide whether to wrap the update function with tf.function or not.\n        '
        if not self._strategy:
            self._strategy = self._make_distributed_strategy_if_necessary()
        with self._strategy.scope():
            super().build()
        if self._enable_tf_function:
            self._possibly_traced_update = tf.function(self._untraced_update, reduce_retracing=True)
        else:
            self._possibly_traced_update = self._untraced_update

    @override(Learner)
    def _update(self, batch: NestedDict) -> Tuple[Any, Any, Any]:
        if False:
            i = 10
            return i + 15
        return self._possibly_traced_update(batch)

    def _untraced_update(self, batch: NestedDict, _ray_trace_ctx=None):
        if False:
            i = 10
            return i + 15

        def helper(_batch):
            if False:
                print('Hello World!')
            _batch = NestedDict(_batch)
            with tf.GradientTape(persistent=True) as tape:
                fwd_out = self._module.forward_train(_batch)
                loss_per_module = self.compute_loss(fwd_out=fwd_out, batch=_batch)
            gradients = self.compute_gradients(loss_per_module, gradient_tape=tape)
            del tape
            postprocessed_gradients = self.postprocess_gradients(gradients)
            self.apply_gradients(postprocessed_gradients)
            return (fwd_out, loss_per_module, dict(self._metrics))
        return self._strategy.run(helper, args=(batch,))

    @override(Learner)
    def _get_tensor_variable(self, value, dtype=None, trainable=False) -> 'tf.Tensor':
        if False:
            i = 10
            return i + 15
        return tf.Variable(value, trainable=trainable, dtype=dtype or (tf.float32 if isinstance(value, float) else tf.int32 if isinstance(value, int) else None))

    @staticmethod
    @override(Learner)
    def _get_optimizer_lr(optimizer: 'tf.Optimizer') -> float:
        if False:
            for i in range(10):
                print('nop')
        return optimizer.lr

    @staticmethod
    @override(Learner)
    def _set_optimizer_lr(optimizer: 'tf.Optimizer', lr: float) -> None:
        if False:
            return 10
        optimizer.lr = lr

    @staticmethod
    @override(Learner)
    def _get_clip_function() -> Callable:
        if False:
            print('Hello World!')
        from ray.rllib.utils.tf_utils import clip_gradients
        return clip_gradients