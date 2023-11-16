from typing import Any, Dict, Optional, Tuple, Union
import warnings
import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY
from .fx.tracer import QuantizationTracer
from .fx.tracer import Scope, ScopeContextManager
from .fx.fuse import fuse
from .fx.prepare import prepare
from .fx.convert import convert
from .backend_config import BackendConfig, get_tensorrt_backend_config
from .fx.graph_module import ObservedGraphModule
from .fx.custom_config import ConvertCustomConfig, FuseCustomConfig, PrepareCustomConfig
from .fx.utils import get_custom_module_class_keys
from .fx.utils import get_skipped_module_name_and_classes
from .qconfig_mapping import QConfigMapping

def attach_preserved_attrs_to_model(model: Union[GraphModule, torch.nn.Module], preserved_attrs: Dict[str, Any]) -> None:
    if False:
        while True:
            i = 10
    ' Store preserved attributes to the model.meta so that it can be preserved during deepcopy\n    '
    model.meta[_USER_PRESERVED_ATTRIBUTES_KEY] = copy.copy(preserved_attrs)
    for (attr_name, attr) in model.meta[_USER_PRESERVED_ATTRIBUTES_KEY].items():
        setattr(model, attr_name, attr)

def _check_is_graph_module(model: torch.nn.Module) -> None:
    if False:
        i = 10
        return i + 15
    if not isinstance(model, GraphModule):
        raise ValueError('input model must be a GraphModule, ' + 'Got type:' + str(type(model)) + ' Please make ' + 'sure to follow the tutorials.')

def _attach_meta_to_node_if_not_exist(model: GraphModule) -> None:
    if False:
        return 10
    ' Attach meta field to all nodes of the graph if it does not exist,\n    meta field is a field stores some meta information about the node, such\n    as dtype and shape information for output of the node, this only exists\n    if the program is captured by make_fx (used in quantize_pt2e flow), if\n    the program is captured by torch.fx symbolic tracing, this field may not exist,\n    so we add it here to avoid checking this all over the places\n    '
    for node in model.graph.nodes:
        if not hasattr(node, 'meta'):
            node.meta = {}

def _swap_ff_with_fxff(model: torch.nn.Module) -> None:
    if False:
        i = 10
        return i + 15
    ' Swap FloatFunctional with FXFloatFunctional\n    '
    modules_to_swap = []
    for (name, module) in model.named_children():
        if isinstance(module, torch.ao.nn.quantized.FloatFunctional):
            modules_to_swap.append(name)
        else:
            _swap_ff_with_fxff(module)
    for name in modules_to_swap:
        del model._modules[name]
        model._modules[name] = torch.ao.nn.quantized.FXFloatFunctional()

def _fuse_fx(model: GraphModule, is_qat: bool, fuse_custom_config: Union[FuseCustomConfig, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    if False:
        print('Hello World!')
    ' Internal helper function to fuse modules in preparation for quantization\n\n    Args:\n        model: GraphModule object from symbolic tracing (torch.fx.symbolic_trace)\n    '
    _check_is_graph_module(model)
    return fuse(model, is_qat, fuse_custom_config, backend_config)

def _prepare_fx(model: torch.nn.Module, qconfig_mapping: Union[QConfigMapping, Dict[str, Any]], is_qat: bool, example_inputs: Tuple[Any, ...], prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None]=None, _equalization_config: Optional[Union[QConfigMapping, Dict[str, Any]]]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None, is_standalone_module: bool=False) -> GraphModule:
    if False:
        for i in range(10):
            print('nop')
    ' Internal helper function for prepare_fx\n    Args:\n      `model`, `qconfig_mapping`, `prepare_custom_config`, `_equalization_config`:\n      see docs for :func:`~torch.ao.quantization.prepare_fx`\n      `is_standalone_module`: a boolean flag indicates whether we are\n      quantizing a standalone module or not, a standalone module\n      is a submodule of the parent module that is not inlined in the\nforward graph of the parent module,\n      the way we quantize standalone module is described in:\n      :func:`~torch.ao.quantization._prepare_standalone_module_fx`\n    '
    if prepare_custom_config is None:
        prepare_custom_config = PrepareCustomConfig()
    if _equalization_config is None:
        _equalization_config = QConfigMapping()
    if isinstance(prepare_custom_config, Dict):
        warnings.warn('Passing a prepare_custom_config_dict to prepare is deprecated and will not be supported in a future version. Please pass in a PrepareCustomConfig instead.')
        prepare_custom_config = PrepareCustomConfig.from_dict(prepare_custom_config)
    _swap_ff_with_fxff(model)
    (skipped_module_names, skipped_module_classes) = get_skipped_module_name_and_classes(prepare_custom_config, is_standalone_module)
    preserved_attr_names = prepare_custom_config.preserved_attributes
    preserved_attrs = {attr: getattr(model, attr) for attr in preserved_attr_names if hasattr(model, attr)}
    tracer = QuantizationTracer(skipped_module_names, skipped_module_classes)
    graph_module = GraphModule(model, tracer.trace(model))
    _attach_meta_to_node_if_not_exist(graph_module)
    fuse_custom_config = FuseCustomConfig().set_preserved_attributes(prepare_custom_config.preserved_attributes)
    graph_module = _fuse_fx(graph_module, is_qat, fuse_custom_config, backend_config)
    prepared = prepare(graph_module, qconfig_mapping, is_qat, tracer.node_name_to_scope, example_inputs=example_inputs, prepare_custom_config=prepare_custom_config, _equalization_config=_equalization_config, backend_config=backend_config, is_standalone_module=is_standalone_module)
    attach_preserved_attrs_to_model(prepared, preserved_attrs)
    return prepared

def _prepare_standalone_module_fx(model: torch.nn.Module, qconfig_mapping: Union[QConfigMapping, Dict[str, Any]], is_qat: bool, example_inputs: Tuple[Any, ...], prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    if False:
        for i in range(10):
            print('nop')
    ' [Internal use only] Prepare a standalone module, so that it can be used when quantizing the\n    parent module.\n    standalone_module means it a submodule that is not inlined in parent module,\n    and will be quantized separately as one unit.\n\n    How the standalone module is observed is specified by `input_quantized_idxs` and\n    `output_quantized_idxs` in the prepare_custom_config for the standalone module\n\n    Returns:\n\n        * model(GraphModule): prepared standalone module. It has these attributes in\n          model.meta:\n\n            * `standalone_module_input_quantized_idxs(List[Int])`: a list of\n              indexes for the graph input that is expected to be quantized,\n              same as input_quantized_idxs configuration provided\n              for the standalone module\n            * `standalone_module_output_quantized_idxs(List[Int])`: a list of\n              indexs for the graph output that is quantized\n              same as input_quantized_idxs configuration provided\n              for the standalone module\n\n    '
    return _prepare_fx(model, qconfig_mapping, is_qat, example_inputs, prepare_custom_config, backend_config=backend_config, is_standalone_module=True)

def fuse_fx(model: torch.nn.Module, fuse_custom_config: Union[FuseCustomConfig, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    if False:
        return 10
    ' Fuse modules like conv+bn, conv+bn+relu etc, model must be in eval mode.\n    Fusion rules are defined in torch.ao.quantization.fx.fusion_pattern.py\n\n    Args:\n\n        * `model` (torch.nn.Module): a torch.nn.Module model\n        * `fuse_custom_config` (FuseCustomConfig): custom configurations for fuse_fx.\n            See :class:`~torch.ao.quantization.fx.custom_config.FuseCustomConfig` for more details\n    Example::\n\n        from torch.ao.quantization import fuse_fx\n        m = Model().eval()\n        m = fuse_fx(m)\n\n    '
    if fuse_custom_config is None:
        fuse_custom_config = FuseCustomConfig()
    if isinstance(fuse_custom_config, Dict):
        warnings.warn('Passing a fuse_custom_config_dict to fuse is deprecated and will not be supported in a future version. Please pass in a FuseCustomConfig instead.')
        fuse_custom_config = FuseCustomConfig.from_dict(fuse_custom_config)
    torch._C._log_api_usage_once('quantization_api.quantize_fx.fuse_fx')
    preserved_attr_names = fuse_custom_config.preserved_attributes
    preserved_attrs = {attr: getattr(model, attr) for attr in preserved_attr_names if hasattr(model, attr)}
    graph_module = torch.fx.symbolic_trace(model)
    _attach_meta_to_node_if_not_exist(graph_module)
    graph_module = _fuse_fx(graph_module, False, fuse_custom_config, backend_config)
    attach_preserved_attrs_to_model(graph_module, preserved_attrs)
    return graph_module

def prepare_fx(model: torch.nn.Module, qconfig_mapping: Union[QConfigMapping, Dict[str, Any]], example_inputs: Tuple[Any, ...], prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None]=None, _equalization_config: Optional[Union[QConfigMapping, Dict[str, Any]]]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    if False:
        return 10
    ' Prepare a model for post training quantization\n\n    Args:\n      * `model` (torch.nn.Module): torch.nn.Module model\n\n      * `qconfig_mapping` (QConfigMapping): QConfigMapping object to configure how a model is\n         quantized, see :class:`~torch.ao.quantization.qconfig_mapping.QConfigMapping`\n         for more details\n\n      * `example_inputs` (Tuple[Any, ...]): Example inputs for forward function of the model,\n         Tuple of positional args (keyword args can be passed as positional args as well)\n\n      * `prepare_custom_config` (PrepareCustomConfig): customization configuration for quantization tool.\n          See :class:`~torch.ao.quantization.fx.custom_config.PrepareCustomConfig` for more details\n\n      * `_equalization_config`: config for specifying how to perform equalization on the model\n\n      * `backend_config` (BackendConfig): config that specifies how operators are quantized\n         in a backend, this includes how the operators are observed,\n         supported fusion patterns, how quantize/dequantize ops are\n         inserted, supported dtypes etc. See :class:`~torch.ao.quantization.backend_config.BackendConfig` for more details\n\n    Return:\n      A GraphModule with observer (configured by qconfig_mapping), ready for calibration\n\n    Example::\n\n        import torch\n        from torch.ao.quantization import get_default_qconfig_mapping\n        from torch.ao.quantization.quantize_fx import prepare_fx\n\n        class Submodule(torch.nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.linear = torch.nn.Linear(5, 5)\n            def forward(self, x):\n                x = self.linear(x)\n                return x\n\n        class M(torch.nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.linear = torch.nn.Linear(5, 5)\n                self.sub = Submodule()\n\n            def forward(self, x):\n                x = self.linear(x)\n                x = self.sub(x) + x\n                return x\n\n        # initialize a floating point model\n        float_model = M().eval()\n\n        # define calibration function\n        def calibrate(model, data_loader):\n            model.eval()\n            with torch.no_grad():\n                for image, target in data_loader:\n                    model(image)\n\n        # qconfig is the configuration for how we insert observers for a particular\n        # operator\n        # qconfig = get_default_qconfig("fbgemm")\n        # Example of customizing qconfig:\n        # qconfig = torch.ao.quantization.QConfig(\n        #    activation=MinMaxObserver.with_args(dtype=torch.qint8),\n        #    weight=MinMaxObserver.with_args(dtype=torch.qint8))\n        # `activation` and `weight` are constructors of observer module\n\n        # qconfig_mapping is a collection of quantization configurations, user can\n        # set the qconfig for each operator (torch op calls, functional calls, module calls)\n        # in the model through qconfig_mapping\n        # the following call will get the qconfig_mapping that works best for models\n        # that target "fbgemm" backend\n        qconfig_mapping = get_default_qconfig_mapping("fbgemm")\n\n        # We can customize qconfig_mapping in different ways.\n        # e.g. set the global qconfig, which means we will use the same qconfig for\n        # all operators in the model, this can be overwritten by other settings\n        # qconfig_mapping = QConfigMapping().set_global(qconfig)\n        # e.g. quantize the linear submodule with a specific qconfig\n        # qconfig_mapping = QConfigMapping().set_module_name("linear", qconfig)\n        # e.g. quantize all nn.Linear modules with a specific qconfig\n        # qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear, qconfig)\n        # for a more complete list, please see the docstring for :class:`torch.ao.quantization.QConfigMapping`\n        # argument\n\n        # example_inputs is a tuple of inputs, that is used to infer the type of the\n        # outputs in the model\n        # currently it\'s not used, but please make sure model(*example_inputs) runs\n        example_inputs = (torch.randn(1, 3, 224, 224),)\n\n        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack\n        # e.g. backend_config = get_default_backend_config("fbgemm")\n        # `prepare_fx` inserts observers in the model based on qconfig_mapping and\n        # backend_config. If the configuration for an operator in qconfig_mapping\n        # is supported in the backend_config (meaning it\'s supported by the target\n        # hardware), we\'ll insert observer modules according to the qconfig_mapping\n        # otherwise the configuration in qconfig_mapping will be ignored\n        #\n        # Example:\n        # in qconfig_mapping, user sets linear module to be quantized with quint8 for\n        # activation and qint8 for weight:\n        # qconfig = torch.ao.quantization.QConfig(\n        #     observer=MinMaxObserver.with_args(dtype=torch.quint8),\n        #     weight=MinMaxObserver.with-args(dtype=torch.qint8))\n        # Note: current qconfig api does not support setting output observer, but\n        # we may extend this to support these more fine grained control in the\n        # future\n        #\n        # qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear, qconfig)\n        # in backend config, linear module also supports in this configuration:\n        # weighted_int8_dtype_config = DTypeConfig(\n        #   input_dtype=torch.quint8,\n        #   output_dtype=torch.quint8,\n        #   weight_dtype=torch.qint8,\n        #   bias_type=torch.float)\n\n        # linear_pattern_config = BackendPatternConfig(torch.nn.Linear) \\\n        #    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \\\n        #    .add_dtype_config(weighted_int8_dtype_config) \\\n        #    ...\n\n        # backend_config = BackendConfig().set_backend_pattern_config(linear_pattern_config)\n        # `prepare_fx` will check that the setting requested by suer in qconfig_mapping\n        # is supported by the backend_config and insert observers and fake quant modules\n        # in the model\n        prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)\n        # Run calibration\n        calibrate(prepared_model, sample_inference_data)\n    '
    torch._C._log_api_usage_once('quantization_api.quantize_fx.prepare_fx')
    return _prepare_fx(model, qconfig_mapping, False, example_inputs, prepare_custom_config, _equalization_config, backend_config)

def prepare_qat_fx(model: torch.nn.Module, qconfig_mapping: Union[QConfigMapping, Dict[str, Any]], example_inputs: Tuple[Any, ...], prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    if False:
        print('Hello World!')
    ' Prepare a model for quantization aware training\n\n    Args:\n      * `model` (torch.nn.Module): torch.nn.Module model\n      * `qconfig_mapping` (QConfigMapping): see :func:`~torch.ao.quantization.prepare_fx`\n      * `example_inputs` (Tuple[Any, ...]): see :func:`~torch.ao.quantization.prepare_fx`\n      * `prepare_custom_config` (PrepareCustomConfig): see :func:`~torch.ao.quantization.prepare_fx`\n      * `backend_config` (BackendConfig): see :func:`~torch.ao.quantization.prepare_fx`\n\n    Return:\n      A GraphModule with fake quant modules (configured by qconfig_mapping and backend_config), ready for\n      quantization aware training\n\n    Example::\n\n        import torch\n        from torch.ao.quantization import get_default_qat_qconfig_mapping\n        from torch.ao.quantization.quantize_fx import prepare_qat_fx\n\n        class Submodule(torch.nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.linear = torch.nn.Linear(5, 5)\n            def forward(self, x):\n                x = self.linear(x)\n                return x\n\n        class M(torch.nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.linear = torch.nn.Linear(5, 5)\n                self.sub = Submodule()\n\n            def forward(self, x):\n                x = self.linear(x)\n                x = self.sub(x) + x\n                return x\n\n        # initialize a floating point model\n        float_model = M().train()\n        # (optional, but preferred) load the weights from pretrained model\n        # float_model.load_weights(...)\n\n        # define the training loop for quantization aware training\n        def train_loop(model, train_data):\n            model.train()\n            for image, target in data_loader:\n                ...\n\n        # qconfig is the configuration for how we insert observers for a particular\n        # operator\n        # qconfig = get_default_qconfig("fbgemm")\n        # Example of customizing qconfig:\n        # qconfig = torch.ao.quantization.QConfig(\n        #    activation=FakeQuantize.with_args(observer=MinMaxObserver.with_args(dtype=torch.qint8)),\n        #    weight=FakeQuantize.with_args(observer=MinMaxObserver.with_args(dtype=torch.qint8)))\n        # `activation` and `weight` are constructors of observer module\n\n        # qconfig_mapping is a collection of quantization configurations, user can\n        # set the qconfig for each operator (torch op calls, functional calls, module calls)\n        # in the model through qconfig_mapping\n        # the following call will get the qconfig_mapping that works best for models\n        # that target "fbgemm" backend\n        qconfig_mapping = get_default_qat_qconfig("fbgemm")\n\n        # We can customize qconfig_mapping in different ways, please take a look at\n        # the docstring for :func:`~torch.ao.quantization.prepare_fx` for different ways\n        # to configure this\n\n        # example_inputs is a tuple of inputs, that is used to infer the type of the\n        # outputs in the model\n        # currently it\'s not used, but please make sure model(*example_inputs) runs\n        example_inputs = (torch.randn(1, 3, 224, 224),)\n\n        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack\n        # e.g. backend_config = get_default_backend_config("fbgemm")\n        # `prepare_qat_fx` inserts observers in the model based on qconfig_mapping and\n        # backend_config, if the configuration for an operator in qconfig_mapping\n        # is supported in the backend_config (meaning it\'s supported by the target\n        # hardware), we\'ll insert fake_quantize modules according to the qconfig_mapping\n        # otherwise the configuration in qconfig_mapping will be ignored\n        # see :func:`~torch.ao.quantization.prepare_fx` for a detailed explanation of\n        # how qconfig_mapping interacts with backend_config\n        prepared_model = prepare_qat_fx(float_model, qconfig_mapping, example_inputs)\n        # Run training\n        train_loop(prepared_model, train_loop)\n\n    '
    torch._C._log_api_usage_once('quantization_api.quantize_fx.prepare_qat_fx')
    return _prepare_fx(model, qconfig_mapping, True, example_inputs, prepare_custom_config, backend_config=backend_config)

def _convert_fx(graph_module: GraphModule, is_reference: bool, convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None]=None, is_standalone_module: bool=False, _remove_qconfig: bool=True, qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None, is_decomposed: bool=False) -> GraphModule:
    if False:
        for i in range(10):
            print('nop')
    ' `is_standalone_module`: see docs in :func:`~torch.ao.quantization.prepare_standalone_module_fx`\n    '
    if convert_custom_config is None:
        convert_custom_config = ConvertCustomConfig()
    if isinstance(convert_custom_config, Dict):
        warnings.warn('Passing a convert_custom_config_dict to convert is deprecated and will not be supported in a future version. Please pass in a ConvertCustomConfig instead.')
        convert_custom_config = ConvertCustomConfig.from_dict(convert_custom_config)
    _check_is_graph_module(graph_module)
    preserved_attr_names = convert_custom_config.preserved_attributes
    preserved_attrs = {attr: getattr(graph_module, attr) for attr in preserved_attr_names if hasattr(graph_module, attr)}
    quantized = convert(graph_module, is_reference, convert_custom_config, is_standalone_module, _remove_qconfig_flag=_remove_qconfig, qconfig_mapping=qconfig_mapping, backend_config=backend_config, is_decomposed=is_decomposed)
    attach_preserved_attrs_to_model(quantized, preserved_attrs)
    return quantized

def convert_fx(graph_module: GraphModule, convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None]=None, _remove_qconfig: bool=True, qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    if False:
        print('Hello World!')
    ' Convert a calibrated or trained model to a quantized model\n\n    Args:\n        * `graph_module` (torch.fx.GraphModule): A prepared and calibrated/trained model (GraphModule)\n\n        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.\n            See :class:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig` for more details\n\n        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.\n\n        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.\n\n           The keys must include the ones in the qconfig_mapping passed to `prepare_fx` or `prepare_qat_fx`,\n           with the same values or `None`. Additional keys can be specified with values set to `None`.\n\n          For each entry whose value is set to None, we skip quantizing that entry in the model::\n\n            qconfig_mapping = QConfigMapping\n                .set_global(qconfig_from_prepare)\n                .set_object_type(torch.nn.functional.add, None)  # skip quantizing torch.nn.functional.add\n                .set_object_type(torch.nn.functional.linear, qconfig_from_prepare)\n                .set_module_name("foo.bar", None)  # skip quantizing module "foo.bar"\n\n         * `backend_config` (BackendConfig): A configuration for the backend which describes how\n            operators should be quantized in the backend, this includes quantization\n            mode support (static/dynamic/weight_only), dtype support (quint8/qint8 etc.),\n            observer placement for each operators and fused operators.\n            See :class:`~torch.ao.quantization.backend_config.BackendConfig` for more details\n\n    Return:\n        A quantized model (torch.nn.Module)\n\n    Example::\n\n        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training\n        # convert_fx converts a calibrated/trained model to a quantized model for the\n        # target hardware, this includes converting the model first to a reference\n        # quantized model, and then lower the reference quantized model to a backend\n        # Currently, the supported backends are fbgemm (onednn), qnnpack (xnnpack) and\n        # they share the same set of quantized operators, so we are using the same\n        # lowering procedure\n        #\n        # backend_config defines the corresponding reference quantized module for\n        # the weighted modules in the model, e.g. nn.Linear\n        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack\n        # e.g. backend_config = get_default_backend_config("fbgemm")\n        quantized_model = convert_fx(prepared_model)\n\n    '
    torch._C._log_api_usage_once('quantization_api.quantize_fx.convert_fx')
    return _convert_fx(graph_module, is_reference=False, convert_custom_config=convert_custom_config, _remove_qconfig=_remove_qconfig, qconfig_mapping=qconfig_mapping, backend_config=backend_config)

def convert_to_reference_fx(graph_module: GraphModule, convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None]=None, _remove_qconfig: bool=True, qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    if False:
        for i in range(10):
            print('nop')
    ' Convert a calibrated or trained model to a reference quantized model,\n    see https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md for more details,\n    reference quantized model is a standard representation of a quantized model provided\n    by FX Graph Mode Quantization, it can be further lowered to run on the target\n    hardware, like accelerators\n\n    Args:\n        * `graph_module` (GraphModule): A prepared and calibrated/trained model (GraphModule)\n\n        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.\n            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.\n\n        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.\n\n        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.\n            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.\n\n         * `backend_config` (BackendConfig): A configuration for the backend which describes how\n            operators should be quantized in the backend. See\n            :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.\n\n    Return:\n        A reference quantized model (GraphModule)\n\n    Example::\n\n        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training\n        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack\n        # e.g. backend_config = get_default_backend_config("fbgemm")\n        reference_quantized_model = convert_to_reference_fx(prepared_model)\n\n    '
    torch._C._log_api_usage_once('quantization_api.quantize_fx.convert_to_reference_fx')
    return _convert_fx(graph_module, is_reference=True, convert_custom_config=convert_custom_config, _remove_qconfig=_remove_qconfig, qconfig_mapping=qconfig_mapping, backend_config=backend_config)

def _convert_to_reference_decomposed_fx(graph_module: GraphModule, convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None]=None, qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    if False:
        return 10
    ' Convert a calibrated or trained model to a reference quantized model, with\n    decomposed representation for quantized Tensor\n    see https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md for more details,\n    reference quantized model is a standard representation of a quantized model provided\n    by FX Graph Mode Quantization, it can be further lowered to run on the target\n    hardware, like accelerators\n\n    Note: this is not public API\n\n    Args:\n        * `graph_module` (GraphModule): A prepared and calibrated/trained model (GraphModule)\n\n        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.\n            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.\n\n        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.\n\n        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.\n            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.\n\n         * `backend_config` (BackendConfig): A configuration for the backend which describes how\n            operators should be quantized in the backend. See\n            :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.\n\n    Return:\n        A reference quantized model (GraphModule) with operators working with decomposed quantized Tensor\n\n    Example::\n\n        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training\n        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack\n        # e.g. backend_config = get_default_backend_config("fbgemm")\n        reference_quantized_model = _convert_to_reference_decomposed_fx(prepared_model)\n\n    '
    torch._C._log_api_usage_once('quantization_api.quantize_fx._convert_to_reference_decomposed_fx')
    return _convert_fx(graph_module, is_reference=True, convert_custom_config=convert_custom_config, _remove_qconfig=False, qconfig_mapping=qconfig_mapping, backend_config=backend_config, is_decomposed=True)

def _convert_standalone_module_fx(graph_module: GraphModule, is_reference: bool=False, convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None]=None) -> GraphModule:
    if False:
        i = 10
        return i + 15
    ' [Internal use only] Convert a model produced by :func:`~torch.ao.quantization.prepare_standalone_module_fx`\n    and convert it to a quantized model\n\n    Returns a quantized standalone module, whether input/output is quantized is\n    specified by prepare_custom_config, with\n    input_quantized_idxs, output_quantized_idxs, please\n    see docs for prepare_fx for details\n    '
    return _convert_fx(graph_module, is_reference, convert_custom_config, is_standalone_module=True)