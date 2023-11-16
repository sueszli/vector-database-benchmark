from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
__all__ = ['BackendConfig', 'BackendPatternConfig', 'DTypeConfig', 'DTypeWithConstraints', 'ObservationType']
INPUT_DTYPE_DICT_KEY = 'input_dtype'
OUTPUT_DTYPE_DICT_KEY = 'output_dtype'
WEIGHT_DTYPE_DICT_KEY = 'weight_dtype'
BIAS_DTYPE_DICT_KEY = 'bias_dtype'
IS_DYNAMIC_DICT_KEY = 'is_dynamic'
NAME_DICT_KEY = 'name'
CONFIGS_DICT_KEY = 'configs'
PATTERN_DICT_KEY = 'pattern'
PATTERN_COMPLEX_FORMAT_DICT_KEY = 'pattern_complex_format'
OBSERVATION_TYPE_DICT_KEY = 'observation_type'
DTYPE_CONFIGS_DICT_KEY = 'dtype_configs'
ROOT_MODULE_DICT_KEY = 'root_module'
QAT_MODULE_DICT_KEY = 'qat_module'
REFERENCE_QUANTIZED_MODULE_DICT_KEY = 'reference_quantized_module_for_root'
FUSED_MODULE_DICT_KEY = 'fused_module'
FUSER_METHOD_DICT_KEY = 'fuser_method'
ROOT_NODE_GETTER_DICT_KEY = 'root_node_getter'
EXTRA_INPUTS_GETTER_DICT_KEY = 'extra_inputs_getter'
NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY = 'num_tensor_args_to_observation_type'
INPUT_TYPE_TO_INDEX_DICT_KEY = 'input_type_to_index'

class ObservationType(Enum):
    """ An enum that represents different ways of how an operator/operator pattern
    should be observed
    """
    OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT = 0
    'this means input and output are observed with different observers, based\n    on qconfig.activation\n    example: conv, linear, softmax\n    '
    OUTPUT_SHARE_OBSERVER_WITH_INPUT = 1
    'this means the output will use the same observer instance as input, based\n    on qconfig.activation\n    example: torch.cat, maxpool\n    '
    INPUT_OUTPUT_NOT_OBSERVED = 2
    'this means the input and output are never observed\n    example: x.shape, x.size\n    '

@dataclass
class DTypeWithConstraints:
    """
    Config for specifying additional constraints for a given dtype, such as quantization
    value ranges, scale value ranges, and fixed quantization params, to be used in
    :class:`~torch.ao.quantization.backend_config.DTypeConfig`.

    The constraints currently supported are:

    * `quant_min_lower_bound` and `quant_max_upper_bound`: Lower and upper
      bounds for the minimum and maximum quantized values respectively. If
      the QConfig’s `quant_min` and `quant_max` fall outside this range,
      then the QConfig will be ignored.

    * `scale_min_lower_bound` and `scale_max_upper_bound`: Lower and upper
      bounds for the minimum and maximum scale values respectively. If the
      QConfig’s minimum scale value (currently exposed as `eps`) falls below
      the lower bound, then the QConfig will be ignored. Note that the upper
      bound is currently not enforced.

    * `scale_exact_match` and `zero_point_exact_match`: Exact match requirements
      for scale and zero point, to be used for operators with fixed quantization
      parameters such as sigmoid and tanh. If the observer specified in the QConfig
      is neither `FixedQParamsObserver` nor `FixedQParamsFakeQuantize`, or if
      the quantization parameters don't match, then the QConfig will be ignored.
    """
    dtype: Optional[torch.dtype] = None
    quant_min_lower_bound: Union[int, float, None] = None
    quant_max_upper_bound: Union[int, float, None] = None
    scale_min_lower_bound: Union[int, float, None] = None
    scale_max_upper_bound: Union[int, float, None] = None
    scale_exact_match: Optional[float] = None
    zero_point_exact_match: Optional[int] = None

@dataclass
class DTypeConfig:
    """
    Config object that specifies the supported data types passed as arguments to
    quantize ops in the reference model spec, for input and output activations,
    weights, and biases.

    For example, consider the following reference model:

      quant1 - [dequant1 - fp32_linear - quant2] - dequant2

    The pattern in the square brackets refers to the reference pattern of
    statically quantized linear. Setting the input dtype as `torch.quint8`
    in the DTypeConfig means we pass in `torch.quint8` as the dtype argument
    to the first quantize op (quant1). Similarly, setting the output dtype as
    `torch.quint8` means we pass in `torch.quint8` as the dtype argument to
    the second quantize op (quant2).

    Note that the dtype here does not refer to the interface dtypes of the
    op. For example, the "input dtype" here is not the dtype of the input
    tensor passed to the quantized linear op. Though it can still be the
    same as the interface dtype, this is not always the case, e.g. the
    interface dtype is fp32 in dynamic quantization but the "input dtype"
    specified in the DTypeConfig would still be quint8. The semantics of
    dtypes here are the same as the semantics of the dtypes specified in
    the observers.

    These dtypes are matched against the ones specified in the user’s
    QConfig. If there is a match, and the QConfig satisfies the constraints
    specified in the DTypeConfig (if any), then we will quantize the given
    pattern using this DTypeConfig. Otherwise, the QConfig is ignored and
    the pattern will not be quantized.

    Example usage::

        >>> # xdoctest: +SKIP(failing)
        >>> dtype_config1 = DTypeConfig(
        ...     input_dtype=torch.quint8,
        ...     output_dtype=torch.quint8,
        ...     weight_dtype=torch.qint8,
        ...     bias_dtype=torch.float)

        >>> dtype_config2 = DTypeConfig(
        ...     input_dtype=DTypeWithConstraints(
        ...         dtype=torch.quint8,
        ...         quant_min_lower_bound=0,
        ...         quant_max_upper_bound=255,
        ...     ),
        ...     output_dtype=DTypeWithConstraints(
        ...         dtype=torch.quint8,
        ...         quant_min_lower_bound=0,
        ...         quant_max_upper_bound=255,
        ...     ),
        ...     weight_dtype=DTypeWithConstraints(
        ...         dtype=torch.qint8,
        ...         quant_min_lower_bound=-128,
        ...         quant_max_upper_bound=127,
        ...     ),
        ...     bias_dtype=torch.float)

        >>> dtype_config1.input_dtype
        torch.quint8

        >>> dtype_config2.input_dtype
        torch.quint8

        >>> dtype_config2.input_dtype_with_constraints
        DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None)
    """
    input_dtype_with_constraints: DTypeWithConstraints
    output_dtype_with_constraints: DTypeWithConstraints
    weight_dtype_with_constraints: DTypeWithConstraints
    bias_dtype: Optional[torch.dtype]
    is_dynamic: Optional[bool]

    def __init__(self, input_dtype: Union[torch.dtype, DTypeWithConstraints, None]=None, output_dtype: Union[torch.dtype, DTypeWithConstraints, None]=None, weight_dtype: Union[torch.dtype, DTypeWithConstraints, None]=None, bias_dtype: Optional[torch.dtype]=None, is_dynamic: Optional[bool]=None):
        if False:
            while True:
                i = 10
        if isinstance(input_dtype, DTypeWithConstraints):
            self.input_dtype_with_constraints = input_dtype
        else:
            self.input_dtype_with_constraints = DTypeWithConstraints(dtype=input_dtype)
        if isinstance(output_dtype, DTypeWithConstraints):
            self.output_dtype_with_constraints = output_dtype
        else:
            self.output_dtype_with_constraints = DTypeWithConstraints(dtype=output_dtype)
        if isinstance(weight_dtype, DTypeWithConstraints):
            self.weight_dtype_with_constraints = weight_dtype
        else:
            self.weight_dtype_with_constraints = DTypeWithConstraints(dtype=weight_dtype)
        self.bias_dtype = bias_dtype
        self.is_dynamic = is_dynamic

    @property
    def input_dtype(self) -> Optional[torch.dtype]:
        if False:
            i = 10
            return i + 15
        return self.input_dtype_with_constraints.dtype

    @property
    def output_dtype(self) -> Optional[torch.dtype]:
        if False:
            print('Hello World!')
        return self.output_dtype_with_constraints.dtype

    @property
    def weight_dtype(self) -> Optional[torch.dtype]:
        if False:
            while True:
                i = 10
        return self.weight_dtype_with_constraints.dtype

    @classmethod
    def from_dict(cls, dtype_config_dict: Dict[str, Any]) -> DTypeConfig:
        if False:
            return 10
        '\n        Create a ``DTypeConfig`` from a dictionary with the following items (all optional):\n            "input_dtype": torch.dtype or ``DTypeWithConstraints``\n            "output_dtype": torch.dtype or ``DTypeWithConstraints``\n            "weight_dtype": torch.dtype or ``DTypeWithConstraints``\n            "bias_type": torch.dtype\n            "is_dynamic": bool\n        '
        input_dtype = dtype_config_dict.get(INPUT_DTYPE_DICT_KEY, None)
        if input_dtype is not None and (not isinstance(input_dtype, (torch.dtype, DTypeWithConstraints))):
            raise ValueError('Expected input_dtype to be a torch.dtype or DTypeWithConstraints')
        output_dtype = dtype_config_dict.get(OUTPUT_DTYPE_DICT_KEY, None)
        if output_dtype is not None and (not isinstance(output_dtype, (torch.dtype, DTypeWithConstraints))):
            raise ValueError('Expected output_dtype to be a torch.dtype or DTypeWithConstraints')
        weight_dtype = dtype_config_dict.get(WEIGHT_DTYPE_DICT_KEY, None)
        if weight_dtype is not None and (not isinstance(weight_dtype, (torch.dtype, DTypeWithConstraints))):
            raise ValueError('Expected weight_dtype to be a torch.dtype or DTypeWithConstraints')
        bias_dtype = dtype_config_dict.get(BIAS_DTYPE_DICT_KEY, None)
        is_dynamic = dtype_config_dict.get(IS_DYNAMIC_DICT_KEY, None)
        return cls(input_dtype, output_dtype, weight_dtype, bias_dtype, is_dynamic)

    def to_dict(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        Convert this ``DTypeConfig`` to a dictionary with the items described in\n        :func:`~torch.ao.quantization.backend_config.DTypeConfig.from_dict`.\n        '
        dtype_config_dict: Dict[str, Any] = {}
        if self.input_dtype is not None:
            dtype_config_dict[INPUT_DTYPE_DICT_KEY] = self.input_dtype_with_constraints
        if self.output_dtype is not None:
            dtype_config_dict[OUTPUT_DTYPE_DICT_KEY] = self.output_dtype_with_constraints
        if self.weight_dtype is not None:
            dtype_config_dict[WEIGHT_DTYPE_DICT_KEY] = self.weight_dtype_with_constraints
        if self.bias_dtype is not None:
            dtype_config_dict[BIAS_DTYPE_DICT_KEY] = self.bias_dtype
        if self.is_dynamic is not None:
            dtype_config_dict[IS_DYNAMIC_DICT_KEY] = self.is_dynamic
        return dtype_config_dict

class BackendConfig:
    """Config that defines the set of patterns that can be quantized on a given backend, and how reference
    quantized models can be produced from these patterns.

    A pattern in this context refers to a module, a functional, an operator, or a directed acyclic graph
    of the above. Each pattern supported on the target backend can be individually configured through
    :class:`~torch.ao.quantization.backend_config.BackendPatternConfig` in terms of:

    (1) The supported input/output activation, weight, and bias data types

    (2) How observers and quant/dequant ops are inserted in order to construct the reference pattern, and

    (3) (Optionally) Fusion, QAT, and reference module mappings.

    The format of the patterns is described in:
    https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md

    Example usage::

        import torch
        from torch.ao.quantization.backend_config import (
            BackendConfig,
            BackendPatternConfig,
            DTypeConfig,
            ObservationType,
        )

        weighted_int8_dtype_config = DTypeConfig(
            input_dtype=torch.quint8,
            output_dtype=torch.quint8,
            weight_dtype=torch.qint8,
            bias_dtype=torch.float)

        def fuse_conv2d_relu(is_qat, conv, relu):
            return torch.ao.nn.intrinsic.ConvReLU2d(conv, relu)

        # For quantizing Linear
        linear_config = BackendPatternConfig(torch.nn.Linear)             .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)             .add_dtype_config(weighted_int8_dtype_config)             .set_root_module(torch.nn.Linear)             .set_qat_module(torch.ao.nn.qat.Linear)             .set_reference_quantized_module(torch.ao.nn.quantized.reference.Linear)

        # For fusing Conv2d + ReLU into ConvReLU2d
        conv_relu_config = BackendPatternConfig((torch.nn.Conv2d, torch.nn.ReLU))             .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)             .add_dtype_config(weighted_int8_dtype_config)             .set_fused_module(torch.ao.nn.intrinsic.ConvReLU2d)             .set_fuser_method(fuse_conv2d_relu)

        # For quantizing ConvReLU2d
        fused_conv_relu_config = BackendPatternConfig(torch.ao.nn.intrinsic.ConvReLU2d)             .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)             .add_dtype_config(weighted_int8_dtype_config)             .set_root_module(torch.nn.Conv2d)             .set_qat_module(torch.ao.nn.intrinsic.qat.ConvReLU2d)             .set_reference_quantized_module(torch.ao.nn.quantized.reference.Conv2d)

        backend_config = BackendConfig("my_backend")             .set_backend_pattern_config(linear_config)             .set_backend_pattern_config(conv_relu_config)             .set_backend_pattern_config(fused_conv_relu_config)

    """

    def __init__(self, name: str=''):
        if False:
            i = 10
            return i + 15
        self.name = name
        self._pattern_complex_format_to_config: Dict[Pattern, BackendPatternConfig] = {}

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'BackendConfig({self.__dict__})'

    def set_name(self, name: str) -> BackendConfig:
        if False:
            i = 10
            return i + 15
        '\n        Set the name of the target backend.\n        '
        self.name = name
        return self

    def set_backend_pattern_config(self, config: BackendPatternConfig) -> BackendConfig:
        if False:
            return 10
        '\n        Set the config for an pattern that can be run on the target backend.\n        This overrides any existing config for the given pattern.\n        '
        pattern_complex_format = torch.ao.quantization.backend_config.utils._get_pattern_in_reversed_nested_tuple_format(config)
        self._pattern_complex_format_to_config[pattern_complex_format] = config
        return self

    def set_backend_pattern_configs(self, configs: List[BackendPatternConfig]) -> BackendConfig:
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the configs for patterns that can be run on the target backend.\n        This overrides any existing config for a given pattern if it was previously registered already.\n        '
        for conf in configs:
            self.set_backend_pattern_config(conf)
        return self

    @property
    def configs(self) -> List[BackendPatternConfig]:
        if False:
            i = 10
            return i + 15
        '\n        Return a copy of the list of configs set in this `BackendConfig`.\n        '
        return list(self._pattern_complex_format_to_config.values())

    @classmethod
    def from_dict(cls, backend_config_dict: Dict[str, Any]) -> BackendConfig:
        if False:
            print('Hello World!')
        '\n        Create a ``BackendConfig`` from a dictionary with the following items:\n\n            "name": the name of the target backend\n\n            "configs": a list of dictionaries that each represents a `BackendPatternConfig`\n\n        '
        conf = cls(backend_config_dict.get(NAME_DICT_KEY, ''))
        for d in backend_config_dict.get(CONFIGS_DICT_KEY, []):
            if isinstance(d, BackendPatternConfig):
                conf.set_backend_pattern_config(d)
            elif isinstance(d, Dict):
                conf.set_backend_pattern_config(BackendPatternConfig.from_dict(d))
            else:
                raise ValueError(f"Expected backend_config_dict['{CONFIGS_DICT_KEY}'] to be a dictionary")
        return conf

    def to_dict(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        Convert this ``BackendConfig`` to a dictionary with the items described in\n        :func:`~torch.ao.quantization.backend_config.BackendConfig.from_dict`.\n        '
        return {NAME_DICT_KEY: self.name, CONFIGS_DICT_KEY: [c.to_dict() for c in self.configs]}

class BackendPatternConfig:
    """
    Config object that specifies quantization behavior for a given operator pattern.
    For a detailed example usage, see :class:`~torch.ao.quantization.backend_config.BackendConfig`.
    """

    def __init__(self, pattern: Optional[Pattern]=None):
        if False:
            while True:
                i = 10
        self.pattern: Optional[Pattern] = pattern
        self.observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
        self.dtype_configs: List[DTypeConfig] = []
        self.root_module: Optional[Type[torch.nn.Module]] = None
        self.qat_module: Optional[Type[torch.nn.Module]] = None
        self.reference_quantized_module: Optional[Type[torch.nn.Module]] = None
        self.fused_module: Optional[Type[torch.nn.Module]] = None
        self.fuser_method: Optional[Callable] = None
        self._root_node_getter: Optional[Callable] = None
        self._extra_inputs_getter: Optional[Callable] = None
        self._num_tensor_args_to_observation_type: Dict[int, ObservationType] = {}
        self._input_type_to_index: Dict[str, int] = {}
        self._pattern_complex_format: Optional[Pattern] = None

    def __repr__(self):
        if False:
            while True:
                i = 10
        dict_nonempty = {k: v for (k, v) in self.__dict__.items() if not isinstance(v, (list, dict)) and v is not None or (isinstance(v, (list, dict)) and len(v) > 0)}
        return f'BackendPatternConfig({dict_nonempty})'

    def set_pattern(self, pattern: Pattern) -> BackendPatternConfig:
        if False:
            while True:
                i = 10
        '\n        Set the pattern to configure.\n\n        The pattern can be a float module, functional operator, pytorch operator, or a tuple\n        combination of the above. Tuple patterns are treated as sequential patterns, and\n        currently only tuples of 2 or 3 elements are supported.\n        '
        if self._pattern_complex_format is not None:
            raise ValueError("Only one of 'pattern' or 'pattern_complex_format' can be set")
        self.pattern = pattern
        return self

    def set_observation_type(self, observation_type: ObservationType) -> BackendPatternConfig:
        if False:
            i = 10
            return i + 15
        '\n        Set how observers should be inserted in the graph for this pattern.\n\n        Observation type here refers to how observers (or quant-dequant ops) will be placed\n        in the graph. This is used to produce the desired reference patterns understood by\n        the backend. Weighted ops such as linear and conv require different observers\n        (or quantization parameters passed to quantize ops in the reference model) for the\n        input and the output.\n\n        There are two observation types:\n\n            `OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT` (default): the output observer instance\n            will be different from the input. This is the most common observation type.\n\n            `OUTPUT_SHARE_OBSERVER_WITH_INPUT`: the output observer instance will be the\n            same as the input. This is useful for operators like `cat`.\n\n        Note: This will be renamed in the near future, since we will soon insert QuantDeQuantStubs\n        with observers (and fake quantizes) attached instead of observers themselves.\n        '
        self.observation_type = observation_type
        return self

    def add_dtype_config(self, dtype_config: DTypeConfig) -> BackendPatternConfig:
        if False:
            print('Hello World!')
        '\n        Add a set of supported data types passed as arguments to quantize ops in the\n        reference model spec.\n        '
        self.dtype_configs.append(dtype_config)
        return self

    def set_dtype_configs(self, dtype_configs: List[DTypeConfig]) -> BackendPatternConfig:
        if False:
            while True:
                i = 10
        '\n        Set the supported data types passed as arguments to quantize ops in the\n        reference model spec, overriding all previously registered data types.\n        '
        self.dtype_configs = dtype_configs
        return self

    def set_root_module(self, root_module: Type[torch.nn.Module]) -> BackendPatternConfig:
        if False:
            i = 10
            return i + 15
        '\n        Set the module that represents the root for this pattern.\n\n        When we construct the reference quantized model during the convert phase,\n        the root modules (e.g. torch.nn.Linear for torch.ao.nn.intrinsic.LinearReLU)\n        will be swapped to the corresponding reference quantized modules (e.g.\n        torch.ao.nn.reference.quantized.Linear). This allows custom backends to\n        specify custom reference quantized module implementations to match the\n        numerics of their lowered operators. Since this is a one-to-one mapping,\n        both the root module and the reference quantized module must be specified\n        in the same BackendPatternConfig in order for the conversion to take place.\n        '
        self.root_module = root_module
        return self

    def set_qat_module(self, qat_module: Type[torch.nn.Module]) -> BackendPatternConfig:
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the module that represents the QAT implementation for this pattern.\n        '
        self.qat_module = qat_module
        return self

    def set_reference_quantized_module(self, reference_quantized_module: Type[torch.nn.Module]) -> BackendPatternConfig:
        if False:
            while True:
                i = 10
        "\n        Set the module that represents the reference quantized implementation for\n        this pattern's root module.\n\n        For more detail, see :func:`~torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module`.\n        "
        self.reference_quantized_module = reference_quantized_module
        return self

    def set_fused_module(self, fused_module: Type[torch.nn.Module]) -> BackendPatternConfig:
        if False:
            return 10
        '\n        Set the module that represents the fused implementation for this pattern.\n        '
        self.fused_module = fused_module
        return self

    def set_fuser_method(self, fuser_method: Callable) -> BackendPatternConfig:
        if False:
            i = 10
            return i + 15
        "\n        Set the function that specifies how to fuse this BackendPatternConfig's pattern.\n\n        The first argument of this function should be `is_qat`, and the rest of the arguments\n        should be the items in the tuple pattern. The return value of this function should be\n        the resulting fused module.\n\n        For example, the fuser method for the pattern `(torch.nn.Linear, torch.nn.ReLU)` can be:\n\n            def fuse_linear_relu(is_qat, linear, relu):\n                return torch.ao.nn.intrinsic.LinearReLU(linear, relu)\n\n        For a more complicated example, see https://gist.github.com/jerryzh168/8bea7180a8ba3c279f2c9b050f2a69a6.\n        "
        self.fuser_method = fuser_method
        return self

    def _set_root_node_getter(self, root_node_getter: Callable) -> BackendPatternConfig:
        if False:
            for i in range(10):
                print('nop')
        self._root_node_getter = root_node_getter
        return self

    def _set_extra_inputs_getter(self, extra_inputs_getter: Callable) -> BackendPatternConfig:
        if False:
            return 10
        self._extra_inputs_getter = extra_inputs_getter
        return self

    def _set_num_tensor_args_to_observation_type(self, num_tensor_args_to_observation_type: Dict[int, ObservationType]) -> BackendPatternConfig:
        if False:
            return 10
        self._num_tensor_args_to_observation_type = num_tensor_args_to_observation_type
        return self

    def _set_input_type_to_index(self, input_type_to_index: Dict[str, int]) -> BackendPatternConfig:
        if False:
            print('Hello World!')
        self._input_type_to_index = input_type_to_index
        return self

    def _set_pattern_complex_format(self, pattern: Pattern) -> BackendPatternConfig:
        if False:
            while True:
                i = 10
        '\n        Set the pattern to configure, using the reversed nested tuple format.\n\n        See the BackendConfig README for more detail:\n        https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md#advanced-pattern-specification\n        '
        if self.pattern is not None:
            raise ValueError("Only one of 'pattern' or 'pattern_complex_format' can be set")
        self._pattern_complex_format = pattern
        return self

    @classmethod
    def from_dict(cls, backend_pattern_config_dict: Dict[str, Any]) -> BackendPatternConfig:
        if False:
            return 10
        '\n        Create a ``BackendPatternConfig`` from a dictionary with the following items:\n\n            "pattern": the pattern being configured\n            "observation_type": the :class:`~torch.ao.quantization.backend_config.ObservationType` that specifies how\n            observers should be inserted for this pattern\n            "dtype_configs": a list of dictionaries that represents :class:`~torch.ao.quantization.backend_config.DTypeConfig` s\n            "root_module": a :class:`torch.nn.Module` that represents the root for this pattern\n            "qat_module": a :class:`torch.nn.Module` that represents the QAT implementation for this pattern\n            "reference_quantized_module": a :class:`torch.nn.Module` that represents the reference quantized\n            implementation for this pattern\'s root module.\n            "fused_module": a :class:`torch.nn.Module` that represents the fused implementation for this pattern\n            "fuser_method": a function that specifies how to fuse the pattern for this pattern\n            "pattern_complex_format": the pattern specified in the reversed nested tuple format (deprecated)\n\n        '

        def _get_dtype_config(obj: Any) -> DTypeConfig:
            if False:
                for i in range(10):
                    print('nop')
            '\n            Convert the given object into a ``DTypeConfig`` if possible, else throw an exception.\n            '
            if isinstance(obj, DTypeConfig):
                return obj
            if isinstance(obj, Dict):
                return DTypeConfig.from_dict(obj)
            raise ValueError(f"""Expected a list of DTypeConfigs in backend_pattern_config_dict["{DTYPE_CONFIGS_DICT_KEY}"], got '{type(obj)}'""")
        conf = cls()
        if PATTERN_DICT_KEY in backend_pattern_config_dict:
            conf.set_pattern(backend_pattern_config_dict[PATTERN_DICT_KEY])
        if OBSERVATION_TYPE_DICT_KEY in backend_pattern_config_dict:
            conf.set_observation_type(backend_pattern_config_dict[OBSERVATION_TYPE_DICT_KEY])
        for d in backend_pattern_config_dict.get(DTYPE_CONFIGS_DICT_KEY, []):
            conf.add_dtype_config(_get_dtype_config(d))
        conf.set_root_module(backend_pattern_config_dict.get(ROOT_MODULE_DICT_KEY, None))
        conf.set_qat_module(backend_pattern_config_dict.get(QAT_MODULE_DICT_KEY, None))
        conf.set_reference_quantized_module(backend_pattern_config_dict.get(REFERENCE_QUANTIZED_MODULE_DICT_KEY, None))
        conf.set_fused_module(backend_pattern_config_dict.get(FUSED_MODULE_DICT_KEY, None))
        conf.set_fuser_method(backend_pattern_config_dict.get(FUSER_METHOD_DICT_KEY, None))
        conf._set_root_node_getter(backend_pattern_config_dict.get(ROOT_NODE_GETTER_DICT_KEY, None))
        conf._set_extra_inputs_getter(backend_pattern_config_dict.get(EXTRA_INPUTS_GETTER_DICT_KEY, None))
        conf._set_num_tensor_args_to_observation_type(backend_pattern_config_dict.get(NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY, {}))
        conf._set_input_type_to_index(backend_pattern_config_dict.get(INPUT_TYPE_TO_INDEX_DICT_KEY, {}))
        if PATTERN_COMPLEX_FORMAT_DICT_KEY in backend_pattern_config_dict:
            conf._set_pattern_complex_format(backend_pattern_config_dict[PATTERN_COMPLEX_FORMAT_DICT_KEY])
        return conf

    def to_dict(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Convert this ``BackendPatternConfig`` to a dictionary with the items described in\n        :func:`~torch.ao.quantization.backend_config.BackendPatternConfig.from_dict`.\n        '
        backend_pattern_config_dict: Dict[str, Any] = {OBSERVATION_TYPE_DICT_KEY: self.observation_type, DTYPE_CONFIGS_DICT_KEY: [c.to_dict() for c in self.dtype_configs]}
        if self.pattern is not None:
            backend_pattern_config_dict[PATTERN_DICT_KEY] = self.pattern
        if self.root_module is not None:
            backend_pattern_config_dict[ROOT_MODULE_DICT_KEY] = self.root_module
        if self.qat_module is not None:
            backend_pattern_config_dict[QAT_MODULE_DICT_KEY] = self.qat_module
        if self.reference_quantized_module is not None:
            backend_pattern_config_dict[REFERENCE_QUANTIZED_MODULE_DICT_KEY] = self.reference_quantized_module
        if self.fused_module is not None:
            backend_pattern_config_dict[FUSED_MODULE_DICT_KEY] = self.fused_module
        if self.fuser_method is not None:
            backend_pattern_config_dict[FUSER_METHOD_DICT_KEY] = self.fuser_method
        if self._root_node_getter is not None:
            backend_pattern_config_dict[ROOT_NODE_GETTER_DICT_KEY] = self._root_node_getter
        if self._extra_inputs_getter is not None:
            backend_pattern_config_dict[EXTRA_INPUTS_GETTER_DICT_KEY] = self._extra_inputs_getter
        if len(self._num_tensor_args_to_observation_type) > 0:
            backend_pattern_config_dict[NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY] = self._num_tensor_args_to_observation_type
        if len(self._input_type_to_index) > 0:
            backend_pattern_config_dict[INPUT_TYPE_TO_INDEX_DICT_KEY] = self._input_type_to_index
        if self._pattern_complex_format is not None:
            backend_pattern_config_dict[PATTERN_COMPLEX_FORMAT_DICT_KEY] = self._pattern_complex_format
        return backend_pattern_config_dict