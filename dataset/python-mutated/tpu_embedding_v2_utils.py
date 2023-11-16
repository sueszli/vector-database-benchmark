"""Companion classes for mid level API for TPU Embeddings in TF2."""
import abc
import math
import typing
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Text, Tuple, TypeVar, Union
from absl import logging
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export
TableVariable = TypeVar('TableVariable', sharded_variable.ShardedVariable, tf_variables.Variable)
SlotVarCreationFnType = Callable[[TableVariable, List[Text], List[init_ops_v2.Initializer]], Dict[Text, TableVariable]]
ClipValueType = Union[Tuple[float, float], float]

class _Optimizer(metaclass=abc.ABCMeta):
    """Base class for all optimizers, with common parameters."""

    def __init__(self, learning_rate: Union[float, Callable[[], float]], use_gradient_accumulation: bool, clip_weight_min: Optional[float], clip_weight_max: Optional[float], weight_decay_factor: Optional[float], multiply_weight_decay_factor_by_learning_rate: bool, clipvalue: Optional[ClipValueType]=None, slot_variable_creation_fn: Optional[SlotVarCreationFnType]=None, low_dimensional_packing_status: bool=False):
        if False:
            i = 10
            return i + 15
        self.learning_rate = learning_rate
        self.use_gradient_accumulation = use_gradient_accumulation
        self.clip_weight_min = clip_weight_min
        self.clip_weight_max = clip_weight_max
        if not use_gradient_accumulation and clipvalue is not None:
            raise ValueError(f'When `use_gradient_accumulation` is False, gradient clipping cannot be used and `clipvalue` should be left as None. Received value {clipvalue} for argument `clipvalue`.')
        if clipvalue is None:
            clipvalue = (None, None)
        elif not isinstance(clipvalue, tuple):
            clipvalue = (-1.0 * clipvalue, clipvalue)
        (self.clip_gradient_min, self.clip_gradient_max) = clipvalue
        self.weight_decay_factor = weight_decay_factor
        self.multiply_weight_decay_factor_by_learning_rate = multiply_weight_decay_factor_by_learning_rate
        if slot_variable_creation_fn is not None and (not callable(slot_variable_creation_fn)):
            raise ValueError(f'Argument `slot_variable_creation_fn` must be either None or a callable. Received: {slot_variable_creation_fn}')
        self.slot_variable_creation_fn = slot_variable_creation_fn
        self.low_dimensional_packing_status = low_dimensional_packing_status

    @abc.abstractmethod
    def _slot_names(self) -> List[Text]:
        if False:
            print('Hello World!')
        "Returns the name of all the slot variables.\n\n    This does not include the 'parameters' variable and these names must match\n    the names of the slots variables as used in the corresponding\n    `tpu_ops.load_tpu_embedding_*` ops.\n    "
        raise NotImplementedError

    @abc.abstractmethod
    def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
        if False:
            for i in range(10):
                print('nop')
        'Returns initializers for slot variables.\n\n    This returns a parallel list to self._slot_names().\n    '
        raise NotImplementedError

    def _set_optimization_parameters(self, parameters: optimization_parameters_pb2.OptimizationParameters):
        if False:
            print('Hello World!')
        'Sets the optimizer fields in the OptimizationParameters.'
        if self.use_gradient_accumulation:
            parameters.gradient_accumulation_status = optimization_parameters_pb2.GradientAccumulationStatus.ENABLED
        else:
            parameters.gradient_accumulation_status = optimization_parameters_pb2.GradientAccumulationStatus.DISABLED
        if self.clip_weight_min is not None:
            parameters.clipping_limits.lower.value = self.clip_weight_min
        if self.clip_weight_max is not None:
            parameters.clipping_limits.upper.value = self.clip_weight_max
        if self.clip_gradient_min is not None:
            parameters.gradient_clipping_limits.lower.value = self.clip_gradient_min
        if self.clip_gradient_max is not None:
            parameters.gradient_clipping_limits.upper.value = self.clip_gradient_max
        if self.weight_decay_factor:
            parameters.weight_decay_factor = self.weight_decay_factor
            if self.multiply_weight_decay_factor_by_learning_rate:
                parameters.multiply_weight_decay_factor_by_learning_rate = True
        parameters.low_dimensional_packing_status = self.low_dimensional_packing_status

    @abc.abstractmethod
    def _load(self) -> Callable[..., ops.Operation]:
        if False:
            print('Hello World!')
        'Returns the load function for the optimizer.'
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve(self) -> Callable[..., core.Tensor]:
        if False:
            print('Hello World!')
        'Returns the retrieve function for the optimizer.'
        raise NotImplementedError

    def _create_slots(self, table: 'TableConfig', variable_creator: Callable[[Text, init_ops_v2.Initializer], tf_variables.Variable]) -> Dict[Text, tf_variables.Variable]:
        if False:
            return 10
        "Creates slot variables for table.\n\n    Args:\n      table: The table variable to create slots for.\n      variable_creator: A function which creates variables. Takes parameters\n        'name', 'initializer'.\n\n    Returns:\n      A dict of variables, keyed by self._slot_names().\n    "
        if self.slot_variable_creation_fn is not None:
            return self.slot_variable_creation_fn(table, self._slot_names(), self._slot_initializers())
        else:
            slots = {}
            for (slot, initializer) in zip(self._slot_names(), self._slot_initializers()):
                slots[slot] = variable_creator(slot, initializer)
            return slots

    def __eq__(self, other: Any) -> Union[Any, bool]:
        if False:
            return 10
        if isinstance(other, self.__class__):
            return all([attr1 == attr2 for (attr1, attr2) in zip(self.__dict__.items(), other.__dict__.items())])
        else:
            return False

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return hash(tuple(self.__dict__.items()))

@tf_export('tpu.experimental.embedding.SGD')
class SGD(_Optimizer):
    """Optimization parameters for stochastic gradient descent for TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.SGD(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

    def __init__(self, learning_rate: Union[float, Callable[[], float]]=0.01, use_gradient_accumulation: bool=True, clip_weight_min: Optional[float]=None, clip_weight_max: Optional[float]=None, weight_decay_factor: Optional[float]=None, multiply_weight_decay_factor_by_learning_rate: bool=None, clipvalue: Optional[ClipValueType]=None, low_dimensional_packing_status: bool=False):
        if False:
            for i in range(10):
                print('nop')
        "Optimization parameters for stochastic gradient descent.\n\n    Args:\n      learning_rate: The learning rate. It should be a floating point value or a\n        callable taking no arguments for a dynamic learning rate.\n      use_gradient_accumulation: setting this to `False` makes embedding\n        gradients calculation less accurate but faster.\n      clip_weight_min: the minimum value to clip by; None means -infinity.\n      clip_weight_max: the maximum value to clip by; None means +infinity.\n      weight_decay_factor: amount of weight decay to apply; None means that the\n        weights are not decayed. Weights are decayed by multiplying the weight\n        by this factor each step.\n      multiply_weight_decay_factor_by_learning_rate: if true,\n        `weight_decay_factor` is multiplied by the current learning rate.\n      clipvalue: Controls clipping of the gradient. Set to either a single\n        positive scalar value to get clipping or a tiple of scalar values (min,\n        max) to set a separate maximum or minimum. If one of the two entries is\n        None, then there will be no clipping that direction. Note if this is\n        set, you may see a decrease in performance as  gradient accumulation\n        will be enabled (it is normally off for SGD as it has no affect on\n        accuracy). See\n        'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for more\n        information on gradient accumulation and its impact on tpu embeddings.\n      low_dimensional_packing_status: Status of the low-dimensional embedding\n        packing optimization controls whether to optimize the packing of\n        1-dimensional, 2-dimensional, and 4-dimensional embedding tables in\n        memory.\n    "
        super().__init__(learning_rate, use_gradient_accumulation, clip_weight_min, clip_weight_max, weight_decay_factor, multiply_weight_decay_factor_by_learning_rate, clipvalue, None, low_dimensional_packing_status)

    def _slot_names(self) -> List[Text]:
        if False:
            while True:
                i = 10
        return []

    def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
        if False:
            while True:
                i = 10
        return []

    def _set_optimization_parameters(self, parameters: optimization_parameters_pb2.OptimizationParameters):
        if False:
            for i in range(10):
                print('nop')
        super()._set_optimization_parameters(parameters)
        parameters.stochastic_gradient_descent.SetInParent()

    def _load(self) -> Callable[..., ops.Operation]:
        if False:
            return 10
        return tpu_ops.load_tpu_embedding_stochastic_gradient_descent_parameters

    def _retrieve(self) -> Callable[..., core.Tensor]:
        if False:
            i = 10
            return i + 15
        return tpu_ops.retrieve_tpu_embedding_stochastic_gradient_descent_parameters

@tf_export('tpu.experimental.embedding.Adagrad')
class Adagrad(_Optimizer):
    """Optimization parameters for Adagrad with TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

    def __init__(self, learning_rate: Union[float, Callable[[], float]]=0.001, initial_accumulator_value: float=0.1, use_gradient_accumulation: bool=True, clip_weight_min: Optional[float]=None, clip_weight_max: Optional[float]=None, weight_decay_factor: Optional[float]=None, multiply_weight_decay_factor_by_learning_rate: bool=None, slot_variable_creation_fn: Optional[SlotVarCreationFnType]=None, clipvalue: Optional[ClipValueType]=None, low_dimensional_packing_status: bool=False):
        if False:
            return 10
        'Optimization parameters for Adagrad.\n\n    Args:\n      learning_rate: The learning rate. It should be a floating point value or a\n        callable taking no arguments for a dynamic learning rate.\n      initial_accumulator_value: initial accumulator for Adagrad.\n      use_gradient_accumulation: setting this to `False` makes embedding\n        gradients calculation less accurate but faster.\n      clip_weight_min: the minimum value to clip by; None means -infinity.\n      clip_weight_max: the maximum value to clip by; None means +infinity.\n      weight_decay_factor: amount of weight decay to apply; None means that the\n        weights are not decayed.\n      multiply_weight_decay_factor_by_learning_rate: if true,\n        `weight_decay_factor` is multiplied by the current learning rate.\n      slot_variable_creation_fn: If you wish do directly control the creation of\n        the slot variables, set this to a callable taking three parameters: a\n        table variable, a list of slot names to create for it, and a list of\n        initializers. This function should return a dict with the slot names as\n        keys and the created variables as values with types matching the table\n        variable. When set to None (the default), uses the built-in variable\n        creation.\n      clipvalue: Controls clipping of the gradient. Set to either a single\n        positive scalar value to get clipping or a tuple of scalar values (min,\n        max) to set a separate maximum or minimum. If one of the two entries is\n        None, then there will be no clipping that direction.\n      low_dimensional_packing_status: Status of the low-dimensional embedding\n        packing optimization controls whether to optimize the packing of\n        1-dimensional, 2-dimensional, and 4-dimensional embedding tables in\n        memory.\n    '
        super().__init__(learning_rate, use_gradient_accumulation, clip_weight_min, clip_weight_max, weight_decay_factor, multiply_weight_decay_factor_by_learning_rate, clipvalue, slot_variable_creation_fn, low_dimensional_packing_status)
        if initial_accumulator_value <= 0:
            raise ValueError(f'Argument `initial_accumulator_value` must be a positive float. Received: {initial_accumulator_value}')
        self.initial_accumulator_value = initial_accumulator_value

    def _slot_names(self) -> List[Text]:
        if False:
            while True:
                i = 10
        return ['accumulators']

    def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
        if False:
            for i in range(10):
                print('nop')
        return [init_ops_v2.Constant(self.initial_accumulator_value, support_partition=True)]

    def _set_optimization_parameters(self, parameters: optimization_parameters_pb2.OptimizationParameters):
        if False:
            while True:
                i = 10
        super()._set_optimization_parameters(parameters)
        parameters.adagrad.SetInParent()

    def _load(self) -> Callable[..., ops.Operation]:
        if False:
            for i in range(10):
                print('nop')
        return tpu_ops.load_tpu_embedding_adagrad_parameters

    def _retrieve(self) -> Callable[..., core.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        return tpu_ops.retrieve_tpu_embedding_adagrad_parameters

@tf_export('tpu.experimental.embedding.AdagradMomentum')
class AdagradMomentum(_Optimizer):
    """Optimization parameters for Adagrad + Momentum with TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.AdagradMomentum(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.AdagradMomentum(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.AdagradMomentum(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

    def __init__(self, learning_rate: Union[float, Callable[[], float]]=0.001, momentum: float=0.0, use_nesterov: bool=False, exponent: float=2, beta2: float=1, epsilon: float=1e-10, use_gradient_accumulation: bool=True, clip_weight_min: Optional[float]=None, clip_weight_max: Optional[float]=None, weight_decay_factor: Optional[float]=None, multiply_weight_decay_factor_by_learning_rate: bool=None, slot_variable_creation_fn: Optional[SlotVarCreationFnType]=None, clipvalue: Optional[ClipValueType]=None, low_dimensional_packing_status: bool=False):
        if False:
            return 10
        'Optimization parameters for Adagrad + Momentum.\n\n    Args:\n      learning_rate: The learning rate. It should be a floating point value or a\n        callable taking no arguments for a dynamic learning rate.\n      momentum: Moving average parameter for the momentum accumulator.\n      use_nesterov: Whether to use the Nesterov variant of momentum. See\n        Sutskever et al., 2013.\n      exponent: Exponent for the Adagrad accumulator.\n      beta2: Moving average parameter for the Adagrad accumulator.\n      epsilon: initial accumulator for Adagrad accumulator.\n      use_gradient_accumulation: setting this to `False` makes embedding\n        gradients calculation less accurate but faster.\n      clip_weight_min: the minimum value to clip by; None means -infinity.\n      clip_weight_max: the maximum value to clip by; None means +infinity.\n      weight_decay_factor: amount of weight decay to apply; None means that the\n        weights are not decayed.\n      multiply_weight_decay_factor_by_learning_rate: if true,\n        `weight_decay_factor` is multiplied by the current learning rate.\n      slot_variable_creation_fn: If you wish do directly control the creation of\n        the slot variables, set this to a callable taking three parameters: a\n        table variable, a list of slot names to create for it, and a list of\n        initializers. This function should return a dict with the slot names as\n        keys and the created variables as values with types matching the table\n        variable. When set to None (the default), uses the built-in variable\n        creation.\n      clipvalue: Controls clipping of the gradient. Set to either a single\n        positive scalar value to get clipping or a tuple of scalar values (min,\n        max) to set a separate maximum or minimum. If one of the two entries is\n        None, then there will be no clipping that direction.\n      low_dimensional_packing_status: Status of the low-dimensional embedding\n        packing optimization controls whether to optimize the packing of\n        1-dimensional, 2-dimensional, and 4-dimensional embedding tables in\n        memory.\n    '
        super().__init__(learning_rate, use_gradient_accumulation, clip_weight_min, clip_weight_max, weight_decay_factor, multiply_weight_decay_factor_by_learning_rate, clipvalue, slot_variable_creation_fn, low_dimensional_packing_status)
        if epsilon <= 0:
            raise ValueError('Adagrad momentum: epsilon must be positive')
        if exponent <= 0:
            raise ValueError('Adagrad momentum: Precondition exponent must >0')
        self.momentum = momentum
        self.use_nesterov = use_nesterov
        self.exponent = exponent
        self.beta2 = beta2
        self.epsilon = epsilon

    def _slot_names(self) -> List[Text]:
        if False:
            i = 10
            return i + 15
        return ['accumulators', 'momenta']

    def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
        if False:
            return 10
        return [init_ops_v2.Constant(support_partition=True), init_ops_v2.Constant(support_partition=True)]

    def _set_optimization_parameters(self, parameters: optimization_parameters_pb2.OptimizationParameters):
        if False:
            print('Hello World!')
        super()._set_optimization_parameters(parameters)
        parameters.adagrad_momentum.SetInParent()
        parameters.adagrad_momentum.momentum = self.momentum
        parameters.adagrad_momentum.use_nesterov = self.use_nesterov
        parameters.adagrad_momentum.exponent = self.exponent
        parameters.adagrad_momentum.beta2 = self.beta2
        parameters.adagrad_momentum.epsilon = self.epsilon

    def _load(self) -> Callable[..., ops.Operation]:
        if False:
            while True:
                i = 10
        return tpu_ops.load_tpu_embedding_adagrad_momentum_parameters

    def _retrieve(self) -> Callable[..., core.Tensor]:
        if False:
            return 10
        return tpu_ops.retrieve_tpu_embedding_adagrad_momentum_parameters

@tf_export('tpu.experimental.embedding.FTRL')
class FTRL(_Optimizer):
    """Optimization parameters for FTRL with TPU embeddings.

  See Algorithm 1 of this
  [paper](https://research.google.com/pubs/archive/41159.pdf).

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.FTRL(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.FTRL(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.FTRL(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

    def __init__(self, learning_rate: Union[float, Callable[[], float]]=0.001, learning_rate_power: float=-0.5, l1_regularization_strength: float=0.0, l2_regularization_strength: float=0.0, beta: float=0.0, initial_accumulator_value: float=0.1, use_gradient_accumulation: bool=True, clip_weight_min: Optional[float]=None, clip_weight_max: Optional[float]=None, weight_decay_factor: Optional[float]=None, multiply_weight_decay_factor_by_learning_rate: bool=None, slot_variable_creation_fn: Optional[SlotVarCreationFnType]=None, clipvalue: Optional[ClipValueType]=None, multiply_linear_by_learning_rate: bool=False, allow_zero_accumulator: bool=False, low_dimensional_packing_status: bool=False):
        if False:
            for i in range(10):
                print('nop')
        'Optimization parameters for Adagrad.\n\n    Args:\n      learning_rate: The learning rate. It should be a floating point value or a\n        callable taking no arguments for a dynamic learning rate.\n      learning_rate_power: A float value, must be less or equal to zero.\n        Controls how the learning rate decreases during training. Use zero for a\n        fixed learning rate.\n      l1_regularization_strength: A float value, must be greater than or equal\n        to zero.\n      l2_regularization_strength: A float value, must be greater than or equal\n        to zero.\n      beta: A float value, representing the beta value from the paper.\n      initial_accumulator_value: The starting value for accumulators. Only zero\n        or positive values are allowed.\n      use_gradient_accumulation: setting this to `False` makes embedding\n        gradients calculation less accurate but faster.\n      clip_weight_min: the minimum value to clip by; None means -infinity.\n      clip_weight_max: the maximum value to clip by; None means +infinity.\n      weight_decay_factor: amount of weight decay to apply; None means that the\n        weights are not decayed.\n      multiply_weight_decay_factor_by_learning_rate: if true,\n        `weight_decay_factor` is multiplied by the current learning rate.\n      slot_variable_creation_fn: If you wish do directly control the creation of\n        the slot variables, set this to a callable taking three parameters: a\n        table variable, a list of slot names to create for it, and a list of\n        initializers. This function should return a dict with the slot names as\n        keys and the created variables as values with types matching the table\n        variable. When set to None (the default), uses the built-in variable\n        creation.\n      clipvalue: Controls clipping of the gradient. Set to either a single\n        positive scalar value to get clipping or a tuple of scalar values (min,\n        max) to set a separate maximum or minimum. If one of the two entries is\n        None, then there will be no clipping that direction.\n      multiply_linear_by_learning_rate: If set to True, a modified formula is\n        used for FTRL that treats the "linear" accumulator as being\n        pre-multiplied by the learning rate (i.e., the accumulator named\n        "linear" actually stores "linear * learning_rate"). Other than\n        checkpoint compatibility, this is mathematically equivalent for a static\n        learning rate; for a dynamic learning rate, it is nearly the same as\n        long as the learning rate does not change quickly. The benefit of this\n        is that the modified formula handles zero and near-zero learning rates\n        without producing NaNs, improving flexibility for learning rate ramp-up.\n      allow_zero_accumulator: If set to True, changes some internal formulas to\n        allow zero and near-zero accumulator values at the cost of some\n        performance; this only needs to be set if you are using an initial\n        accumulator value of zero, which is uncommon.\n      low_dimensional_packing_status: Status of the low-dimensional embedding\n        packing optimization controls whether to optimize the packing of\n        1-dimensional, 2-dimensional, and 4-dimensional embedding tables in\n        memory.\n    '
        super().__init__(learning_rate, use_gradient_accumulation, clip_weight_min, clip_weight_max, weight_decay_factor, multiply_weight_decay_factor_by_learning_rate, clipvalue, slot_variable_creation_fn, low_dimensional_packing_status)
        if initial_accumulator_value <= 0:
            raise ValueError(f'Argument `initial_accumulator_value` must be a positive float. Received: {initial_accumulator_value}')
        self.initial_accumulator_value = initial_accumulator_value
        self.learning_rate_power = learning_rate_power
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength
        self.beta = beta
        self.multiply_linear_by_learning_rate = multiply_linear_by_learning_rate
        self.allow_zero_accumulator = allow_zero_accumulator

    def _slot_names(self) -> List[Text]:
        if False:
            return 10
        return ['accumulators', 'linears']

    def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
        if False:
            print('Hello World!')
        return [init_ops_v2.Constant(self.initial_accumulator_value, support_partition=True), init_ops_v2.Constant(support_partition=True)]

    def _set_optimization_parameters(self, parameters: optimization_parameters_pb2.OptimizationParameters):
        if False:
            i = 10
            return i + 15
        super()._set_optimization_parameters(parameters)
        ftrl = parameters.ftrl
        ftrl.l1 = self.l1_regularization_strength
        ftrl.l2 = self.l2_regularization_strength
        ftrl.lr_power = self.learning_rate_power
        ftrl.beta = self.beta
        ftrl.multiply_linear_by_lr = self.multiply_linear_by_learning_rate
        ftrl.allow_zero_accumulator = self.allow_zero_accumulator

    def _load(self) -> Callable[..., ops.Operation]:
        if False:
            print('Hello World!')
        return tpu_ops.load_tpu_embedding_ftrl_parameters

    def _retrieve(self) -> Callable[..., core.Tensor]:
        if False:
            i = 10
            return i + 15
        return tpu_ops.retrieve_tpu_embedding_ftrl_parameters

@tf_export('tpu.experimental.embedding.Adam')
class Adam(_Optimizer):
    """Optimization parameters for Adam with TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  NOTE: By default this optimizer is lazy, i.e. it will not apply the gradient
  update of zero to rows that were not looked up. You can change this behavior
  by setting `lazy_adam` to `False`.

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.Adam(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

    def __init__(self, learning_rate: Union[float, Callable[[], float]]=0.001, beta_1: float=0.9, beta_2: float=0.999, epsilon: float=1e-07, lazy_adam: bool=True, sum_inside_sqrt: bool=True, use_gradient_accumulation: bool=True, clip_weight_min: Optional[float]=None, clip_weight_max: Optional[float]=None, weight_decay_factor: Optional[float]=None, multiply_weight_decay_factor_by_learning_rate: bool=None, slot_variable_creation_fn: Optional[SlotVarCreationFnType]=None, clipvalue: Optional[ClipValueType]=None, low_dimensional_packing_status: bool=False):
        if False:
            i = 10
            return i + 15
        "Optimization parameters for Adam.\n\n    See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a\n    complete description of these parameters and their impacts on the optimizer\n    algorithm.\n\n    Args:\n      learning_rate: The learning rate. It should be a floating point value or a\n        callable taking no arguments for a dynamic learning rate.\n      beta_1: A float value. The exponential decay rate for the 1st moment\n        estimates.\n      beta_2: A float value. The exponential decay rate for the 2nd moment\n        estimates.\n      epsilon: A small constant for numerical stability.\n      lazy_adam: Use lazy Adam instead of Adam. Lazy Adam trains faster.\n      sum_inside_sqrt: When this is true, the Adam update formula is changed\n        from `m / (sqrt(v) + epsilon)` to `m / sqrt(v + epsilon**2)`. This\n        option improves the performance of TPU training and is not expected to\n        harm model quality.\n      use_gradient_accumulation: Setting this to `False` makes embedding\n        gradients calculation less accurate but faster.\n      clip_weight_min: the minimum value to clip by; None means -infinity.\n      clip_weight_max: the maximum value to clip by; None means +infinity.\n      weight_decay_factor: amount of weight decay to apply; None means that the\n        weights are not decayed.\n      multiply_weight_decay_factor_by_learning_rate: if true,\n        `weight_decay_factor` is multiplied by the current learning rate.\n      slot_variable_creation_fn: If you wish do directly control the creation of\n        the slot variables, set this to a callable taking three parameters: a\n        table variable, a list of slot names to create for it, and a list of\n        initializers. This function should return a dict with the slot names as\n        keys and the created variables as values with types matching the table\n        variable. When set to None (the default), uses the built-in variable\n        creation.\n      clipvalue: Controls clipping of the gradient. Set to either a single\n        positive scalar value to get clipping or a tiple of scalar values (min,\n        max) to set a separate maximum or minimum. If one of the two entries is\n        None, then there will be no clipping that direction.\n      low_dimensional_packing_status: Status of the low-dimensional embedding\n        packing optimization controls whether to optimize the packing of\n        1-dimensional, 2-dimensional, and 4-dimensional embedding tables in\n        memory.\n    "
        super(Adam, self).__init__(learning_rate, use_gradient_accumulation, clip_weight_min, clip_weight_max, weight_decay_factor, multiply_weight_decay_factor_by_learning_rate, clipvalue, slot_variable_creation_fn, low_dimensional_packing_status)
        if beta_1 < 0.0 or beta_1 >= 1.0:
            raise ValueError(f'Argument `beta_1` must be >= 0 and < 1. Received: {beta_1}.')
        if beta_2 < 0.0 or beta_2 >= 1.0:
            raise ValueError(f'Argument `beta_2` must be >= 0 and < 1. Received: {beta_1}.')
        if epsilon <= 0.0:
            raise ValueError('epsilon must be positive; got {}.'.format(epsilon))
        if not use_gradient_accumulation and (not lazy_adam):
            raise ValueError('When disabling lazy Adam (`lazy_adam=False`), gradient accumulation must be used. Set `use_gradient_accumulation` to False.')
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.lazy_adam = lazy_adam
        self.sum_inside_sqrt = sum_inside_sqrt

    def _slot_names(self) -> List[Text]:
        if False:
            while True:
                i = 10
        return ['momenta', 'velocities']

    def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
        if False:
            i = 10
            return i + 15
        return [init_ops_v2.Constant(support_partition=True), init_ops_v2.Constant(support_partition=True)]

    def _set_optimization_parameters(self, parameters: optimization_parameters_pb2.OptimizationParameters):
        if False:
            print('Hello World!')
        super(Adam, self)._set_optimization_parameters(parameters)
        parameters.adam.beta1 = self.beta_1
        parameters.adam.beta2 = self.beta_2
        parameters.adam.epsilon = self.epsilon
        parameters.adam.use_non_lazy_adam = not self.lazy_adam
        parameters.adam.use_sum_inside_sqrt = self.sum_inside_sqrt

    def _load(self) -> Callable[..., ops.Operation]:
        if False:
            i = 10
            return i + 15
        return tpu_ops.load_tpu_embedding_adam_parameters

    def _retrieve(self) -> Callable[..., core.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        return tpu_ops.retrieve_tpu_embedding_adam_parameters

@tf_export('tpu.experimental.embedding.QuantizationConfig')
class QuantizationConfig:
    """Settings for simulated quantization of the tpu embedding table.

  When simulated quantization is enabled, the results of the embedding lookup
  are clipped and quantized according to the settings here before the combiner
  is applied.

  For example, to quantize `input` the following is done:
  ```python
  if input < lower
    input = lower
  if input > upper
    input = upper
  quantum = (upper - lower) / (num_buckets - 1)
  input = math.floor((input - lower) / quantum + 0.5) * quantium + lower
  ```

  See tensorflow/core/protobuf/tpu/optimization_parameters.proto for more
  details.

  NOTE: This does not change the storage type of the embedding table, that will
  continue to be float32 as will the saved variable in the checkpoint. You will
  have to manually quantize the variable (typically with the same algorithm and
  settings as above) manually.
  """

    def __init__(self, num_buckets: int, lower: float, upper: float):
        if False:
            i = 10
            return i + 15
        'Simulated quantizaiton configuration.\n\n    Args:\n      num_buckets: The number of quantization buckets, must be atleast 2.\n      lower: The lower bound for the quantization range.\n      upper: The upper bound for the quantization range.\n\n    Returns:\n      `QuantizationConfig`.\n\n    Raises:\n      ValueError: if `num_buckets` is less than 2.\n    '
        if num_buckets < 2:
            raise ValueError(f'num_buckets is {num_buckets}, must be at least 2 for simulated quantization.')
        self.num_buckets = num_buckets
        self.lower = lower
        self.upper = upper

    def _set_optimization_parameters(self, parameters: optimization_parameters_pb2.OptimizationParameters):
        if False:
            i = 10
            return i + 15
        parameters.simulated_quantization.enabled = True
        parameters.simulated_quantization.num_buckets = self.num_buckets
        parameters.simulated_quantization.clipping_limits.lower.value = self.lower
        parameters.simulated_quantization.clipping_limits.upper.value = self.upper

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'QuantizationConfig(num_buckets={num_buckets!r}, lower={lower!r}, upper={upper!r})'.format(num_buckets=self.num_buckets, lower=self.lower, upper=self.upper)

@tf_export('tpu.experimental.embedding.TableConfig')
class TableConfig:
    """Configuration data for one embedding table.

  This class holds the configuration data for a single embedding table. It is
  used as the `table` parameter of a
  `tf.tpu.experimental.embedding.FeatureConfig`. Multiple
  `tf.tpu.experimental.embedding.FeatureConfig` objects can use the same
  `tf.tpu.experimental.embedding.TableConfig` object. In this case a shared
  table will be created for those feature lookups.

  ```python
  table_config_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  table_config_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = {
      'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_two)}
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  The above configuration has 2 tables, and three features. The first two
  features will be looked up in the first table and the third feature will be
  looked up in the second table.

  """

    def __init__(self, vocabulary_size: int, dim: int, initializer: Optional[Callable[[Any], None]]=None, optimizer: Optional[_Optimizer]=None, combiner: Text='mean', name: Optional[Text]=None, quantization_config: QuantizationConfig=None, layout: Optional[Any]=None):
        if False:
            return 10
        "Embedding table configuration.\n\n    Args:\n      vocabulary_size: Size of the table's vocabulary (number of rows).\n      dim: The embedding dimension (width) of the table.\n      initializer: A callable initializer taking one parameter, the shape of the\n        variable that will be initialized. Will be called once per task, to\n        initialize that task's shard of the embedding table. If not specified,\n        defaults to `truncated_normal_initializer` with mean `0.0` and standard\n        deviation `1/sqrt(dim)`.\n      optimizer: An optional instance of an optimizer parameters class, instance\n        of one of `tf.tpu.experimental.embedding.SGD`,\n        `tf.tpu.experimental.embedding.Adagrad` or\n        `tf.tpu.experimental.embedding.Adam`. If set will override the global\n        optimizer passed to `tf.tpu.experimental.embedding.TPUEmbedding`.\n      combiner: A string specifying how to reduce if there are multiple entries\n        in a single row. Currently 'mean', 'sqrtn', 'sum' are supported, with\n        'mean' the default. 'sqrtn' often achieves good accuracy, in particular\n        with bag-of-words columns. For more information, see\n        `tf.nn.embedding_lookup_sparse`.\n      name: An optional string used to name the table. Must be defined if\n        running on SparseCore.\n      quantization_config: The simulated quantization config. An instance of\n        `tf.tpu.experimental.embedding.QuantizationConfig`. See the class for\n        more documentation.\n      layout: If the table already has its layout computed, you can pass it in\n        here. Otherwise, we will compute it for you. Most users should leave\n        this as None.\n\n    Returns:\n      `TableConfig`.\n\n    Raises:\n      ValueError: if `vocabulary_size` is not a positive integer.\n      ValueError: if `dim` is not a positive integer.\n      ValueError: if `initializer` is specified and is not callable.\n      ValueError: if `combiner` is not supported.\n    "
        if not isinstance(vocabulary_size, int) or vocabulary_size < 1:
            raise ValueError(f'Argument `vocabulary_size` must be an int and must be >= 1. Received: {vocabulary_size}')
        if not isinstance(dim, int) or dim < 1:
            raise ValueError(f'Argument `dim` (embedding dimension) must be an int and must be >= 1. Received: {dim}')
        if initializer is not None and (not callable(initializer)):
            raise ValueError(f'Argument `initializer` must be a callable (or None). Received: {initializer}')
        if initializer is None:
            initializer = init_ops_v2.TruncatedNormal(mean=0.0, stddev=1 / math.sqrt(dim))
        accepted_combiners = ('mean', 'sum', 'sqrtn')
        if combiner not in accepted_combiners:
            raise ValueError(f'Argument `combiner` must be one of {accepted_combiners}. Received: {combiner}')
        if name is None:
            logging.warning('Name of the table config must be specified for running on SparseCore. Different table configs must have unique names.')
        self.vocabulary_size = vocabulary_size
        self.dim = dim
        self.initializer = initializer
        self.optimizer = optimizer
        self.combiner = combiner
        self.name = name
        self.quantization_config = quantization_config
        self.layout = layout

    def __repr__(self):
        if False:
            while True:
                i = 10
        initializer = self.initializer
        if isinstance(initializer, init_ops_v2.TruncatedNormal):
            initializer = typing.cast(init_ops_v2.TruncatedNormal, initializer)
            if initializer.mean == 0.0 and math.isclose(initializer.stddev, 1 / math.sqrt(self.dim)):
                initializer = None
        return 'TableConfig(vocabulary_size={vocabulary_size!r}, dim={dim!r}, initializer={initializer!r}, optimizer={optimizer!r}, combiner={combiner!r}, name={name!r}, quantization_config={quantization!r})'.format(vocabulary_size=self.vocabulary_size, dim=self.dim, initializer=initializer, optimizer=self.optimizer, combiner=self.combiner, name=self.name, quantization=self.quantization_config)

    def _set_table_descriptor(self, table_descriptor: tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration.TableDescriptor, num_hosts: int, learning_rate_index: Dict[Callable[[], Any], int]):
        if False:
            i = 10
            return i + 15
        'Set the table descriptor from the table data.'
        table_descriptor.name = self.name
        table_descriptor.vocabulary_size = max(self.vocabulary_size, num_hosts)
        table_descriptor.dimension = self.dim
        parameters = table_descriptor.optimization_parameters
        if self.optimizer:
            if callable(self.optimizer.learning_rate):
                parameters.learning_rate.dynamic.tag = learning_rate_index[self.optimizer.learning_rate]
            else:
                parameters.learning_rate.constant = self.optimizer.learning_rate
            if self.optimizer.low_dimensional_packing_status:
                parameters.low_dimensional_packing_status = optimization_parameters_pb2.LowDimensionalPackingStatus.Status.ENABLED
            self.optimizer._set_optimization_parameters(parameters)
        if self.quantization_config:
            self.quantization_config._set_optimization_parameters(parameters)

@tf_export('tpu.experimental.embedding.RowIdInitializer')
class RowIdInitializer:
    """An initializer that initializes the table with vocabulary ids."""

    def __init__(self, offset: int=0):
        if False:
            for i in range(10):
                print('nop')
        self.offset = offset

    def __call__(self, shape: Union[Sequence[int], TensorShape], dtype: dtypes.DType) -> core.Tensor:
        if False:
            print('Hello World!')
        return math_ops.range(start=self.offset, limit=self.offset + shape[0], delta=1, dtype=dtype)[:, None] * array_ops.ones(shape, dtype=dtype)

@tf_export('tpu.experimental.embedding.FeatureConfig')
class FeatureConfig:
    """Configuration data for one embedding feature.

  This class holds the configuration data for a single embedding feature. The
  main use is to assign features to `tf.tpu.experimental.embedding.TableConfig`s
  via the table parameter:

  ```python
  table_config_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  table_config_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = {
      'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_two)}
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  The above configuration has 2 tables, and three features. The first two
  features will be looked up in the first table and the third feature will be
  looked up in the second table.

  You can also specify the output shape for each feature. The output shape
  should be the expected activation shape excluding the table dimension. For
  dense and sparse tensor, the output shape should be the same as the input
  shape excluding the last dimension. For ragged tensor, the output shape can
  mismatch the input shape.

  NOTE: The `max_sequence_length` will be only used when the input tensor has
  rank 2 and the `output_shape` is not set in the feature config.

  When feeding features into `embedding.enqueue` they can be `tf.Tensor`s,
  `tf.SparseTensor`s or `tf.RaggedTensor`s. When the argument
  `max_sequence_length` is 0, the default, you should expect a output of
  `embedding.dequeue` for this feature of shape `(batch_size, dim)`. If
  `max_sequence_length` is greater than 0, the feature is embedded as a sequence
  and padded up to the given length. The shape of the output for this feature
  will be `(batch_size, max_sequence_length, dim)`.
  """

    def __init__(self, table: TableConfig, max_sequence_length: int=0, validate_weights_and_indices: bool=True, output_shape: Optional[Union[List[int], TensorShape]]=None, name: Optional[Text]=None):
        if False:
            while True:
                i = 10
        'Feature configuration.\n\n    Args:\n      table: An instance of `tf.tpu.experimental.embedding.TableConfig`,\n        describing the table in which this feature should be looked up.\n      max_sequence_length: If positive, the feature is a sequence feature with\n        the corresponding maximum sequence length. If the sequence is longer\n        than this, it will be truncated. If 0, the feature is not a sequence\n        feature.\n      validate_weights_and_indices: If true, uses safe_embedding_lookup during\n        serving which ensures there are no empty rows and all weights and ids\n        are positive at the expense of extra compute cost.\n      output_shape: Optional argument to config the output shape of the feature\n        activation. If provided, the feature feeding to the `embedding.enqueue`\n        has to match the shape (for ragged tensor, the input shape and output\n        shape can mismatch). If not provided, the shape can be either provided\n        to the `embedding.build` or auto detected at the runtime.\n      name: An optional string used to name the table. Must be defined if\n        running on SparseCore.\n\n    Returns:\n      `FeatureConfig`.\n\n    Raises:\n      ValueError: if `table` is not an instance of\n        `tf.tpu.experimental.embedding.TableConfig`.\n      ValueError: if `max_sequence_length` not an integer or is negative.\n    '
        if not isinstance(table, TableConfig):
            raise ValueError(f'Argument `table` has invalid type {type(table)}. Expected `tf.tpu.experimental.embedding.TableConfig`.')
        if not isinstance(max_sequence_length, int) or max_sequence_length < 0:
            raise ValueError(f'Argument `max_sequence_length` must be an int and must be >= 0. Received: {max_sequence_length}')
        self.table = table
        self.max_sequence_length = max_sequence_length
        self.name = name
        self.output_shape = TensorShape(output_shape)
        if not isinstance(validate_weights_and_indices, bool):
            raise ValueError(f'Argument `validate_weights_and_indices` must be a boolean. Received: {validate_weights_and_indices}')
        self.validate_weights_and_indices = validate_weights_and_indices

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'FeatureConfig(table={table!r}, max_sequence_length={max_sequence_length!r}, validate_weights_and_indices={validate_weights_and_indices!r}, output_shape={output_shape!r}, name={name!r})'.format(table=self.table, max_sequence_length=self.max_sequence_length, validate_weights_and_indices=self.validate_weights_and_indices, output_shape=self.output_shape, name=self.name)

def log_tpu_embedding_configuration(config: tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration) -> None:
    if False:
        return 10
    'Logs a TPUEmbeddingConfiguration proto across multiple statements.\n\n  Args:\n    config: TPUEmbeddingConfiguration proto to log.  Necessary because\n      logging.info has a maximum length to each log statement, which\n      particularly large configs can exceed.\n  '
    logging.info('Beginning log of TPUEmbeddingConfiguration.')
    for line in str(config).splitlines():
        logging.info(line)
    logging.info('Done with log of TPUEmbeddingConfiguration.')

def _sort_device_spec_strings(device_strings: Iterable[str]) -> List[str]:
    if False:
        return 10
    sorted_specs = sorted((device_spec.DeviceSpecV2.from_string(spec) for spec in device_strings), key=lambda s: (s.replica, s.task, s.device_index))
    return [spec.to_string() for spec in sorted_specs]

def get_list_of_hosts(strategy: tpu_strategy.TPUStrategy) -> List[Text]:
    if False:
        print('Hello World!')
    'Returns a sorted list of CPU devices for the remote jobs.\n\n  Args:\n    strategy: A TPUStrategy object.\n\n  Returns:\n    A sorted list of device host strings.\n  '
    list_of_hosts = []
    for tpu_device in _sort_device_spec_strings(strategy.extended.worker_devices):
        host = device_util.get_host_for_device(tpu_device)
        if host not in list_of_hosts:
            list_of_hosts.append(host)
    assert len(list_of_hosts) == strategy.extended.num_hosts
    return list_of_hosts