from __future__ import annotations
import abc
import contextlib
import dataclasses
import io
import logging
import os
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Final, List, Mapping, Optional, Protocol, runtime_checkable, Sequence, Set, Tuple, TYPE_CHECKING, TypeVar, Union
from typing_extensions import Self
import torch
import torch._ops
import torch.export as torch_export
import torch.utils._pytree as pytree
from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, io_adapter
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import decomposition_table, patcher as patcher, registration, serialization as fx_serialization
if TYPE_CHECKING:
    import onnx
    import onnxscript
    from onnxscript.function_libs.torch_lib import registration as torchlib_registry
    from torch.onnx._internal.fx import diagnostics
else:
    try:
        from torch.onnx._internal.fx import diagnostics
    except ImportError:
        pass
_DEFAULT_OPSET_VERSION: Final[int] = 18
'The default ONNX opset version the exporter will use if one is not specified explicitly\nthrough :class:`ExportOptions`. This should NEVER be accessed outside of this module! Users\nshould reference :attr:`ExportOptions.opset_version`.'
_PYTORCH_GITHUB_ISSUES_URL = 'https://github.com/pytorch/pytorch/issues'
'The URL to the PyTorch GitHub issues page.'
_DEFAULT_FAILED_EXPORT_SARIF_LOG_PATH = 'report_dynamo_export.sarif'
'The default path to write the SARIF log to if the export fails.'
_PROTOBUF_SIZE_MAX_LIMIT = 2 * 1024 * 1024 * 1024
'The maximum size of a Protobuf file in bytes. This is used to determine whether to\nserialize the model with external data or not.'
log = logging.getLogger(__name__)
DiagnosticOptions = infra.DiagnosticOptions

@dataclasses.dataclass
class ONNXFakeContext:
    """A dataclass used to store context for model export using FakeTensor.

    This dataclass stores the FakeTensorMode instance used to convert
    real tensors and model parameters into fake tensors. This :attr:`ONNXFakeContext.fake_mode` is
    reused internally during tracing of a :class:`torch.nn.Module` into a FX :class:`GraphModule`.
    """
    fake_mode: fake_tensor.FakeTensorMode
    'The fake tensor mode used for tracing model using fake tensors and parameters.'
    state_dict_paths: Optional[Tuple[Union[str, io.BytesIO]]] = None
    'List of paths of files that contain the model :meth:`state_dict`'

class OnnxRegistry:
    """Registry for ONNX functions.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It supports registering custom onnx-script functions and for
    dispatcher to dispatch calls to the appropriate function.

    """

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        'Initializes the registry'
        self._registry: Dict[registration.OpName, List[registration.ONNXFunction]] = defaultdict(list)
        from onnxscript.function_libs.torch_lib import ops, registration
        self._opset_version = _DEFAULT_OPSET_VERSION
        warnings.warn(f'torch.onnx.dynamo_export only implements opset version {self._opset_version} for now. If you need to use a different opset version, please register them with register_custom_op.')
        self._initiate_registry_from_torchlib(registration.default_registry)

    @property
    def opset_version(self) -> int:
        if False:
            while True:
                i = 10
        'The ONNX opset version the exporter should target. Defaults to the latest\n        supported ONNX opset version: 18. The default version will increment over time as\n        ONNX continues to evolve.'
        return self._opset_version

    def _initiate_registry_from_torchlib(self, torchlib_registry: torchlib_registry.Registry):
        if False:
            for i in range(10):
                print('nop')
        'Populates the registry with ATen functions from torchlib.\n\n        Args:\n            torchlib_registry: The torchlib registry to use for populating the registry.\n        '
        for (aten_name, aten_overloads_func) in torchlib_registry.items():
            internal_name_instance = registration.OpName.from_qualified_name(aten_name)
            for overload_func in aten_overloads_func.overloads:
                symbolic_function = registration.ONNXFunction(onnx_function=overload_func, op_full_name=internal_name_instance.qualified_name(), is_custom=False, is_complex=False)
                self._register(internal_name_instance, symbolic_function)
            for complex_func in aten_overloads_func.complex:
                symbolic_function = registration.ONNXFunction(onnx_function=complex_func, op_full_name=internal_name_instance.qualified_name(), is_custom=False, is_complex=True)
                self._register(internal_name_instance, symbolic_function)

    @_beartype.beartype
    def _register(self, internal_qualified_name: registration.OpName, symbolic_function: registration.ONNXFunction) -> None:
        if False:
            print('Hello World!')
        'Registers a ONNXFunction to an operator.\n\n        Args:\n            internal_qualified_name: The qualified name of the operator to register: OpName.\n            symbolic_function: The ONNXFunction to register.\n        '
        self._registry[internal_qualified_name].append(symbolic_function)

    @_beartype.beartype
    def register_op(self, function: Union['onnxscript.OnnxFunction', 'onnxscript.TracedOnnxFunction'], namespace: str, op_name: str, overload: Optional[str]=None, is_complex: bool=False) -> None:
        if False:
            print('Hello World!')
        "Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.\n\n        Args:\n            function: The onnx-sctip function to register.\n            namespace: The namespace of the operator to register.\n            op_name: The name of the operator to register.\n            overload: The overload of the operator to register. If it's default overload,\n                leave it to None.\n            is_complex: Whether the function is a function that handles complex valued inputs.\n\n        Raises:\n            ValueError: If the name is not in the form of 'namespace::op'.\n        "
        internal_name_instance = registration.OpName.from_name_parts(namespace=namespace, op_name=op_name, overload=overload)
        symbolic_function = registration.ONNXFunction(onnx_function=function, op_full_name=internal_name_instance.qualified_name(), is_custom=True, is_complex=is_complex)
        self._register(internal_name_instance, symbolic_function)

    @_beartype.beartype
    def get_op_functions(self, namespace: str, op_name: str, overload: Optional[str]=None) -> Optional[List[registration.ONNXFunction]]:
        if False:
            return 10
        "Returns a list of ONNXFunctions for the given op: torch.ops.<namespace>.<op_name>.<overload>.\n\n        The list is ordered by the time of registration. The custom operators should be\n        in the second half of the list.\n\n        Args:\n            namespace: The namespace of the operator to get.\n            op_name: The name of the operator to get.\n            overload: The overload of the operator to get. If it's default overload,\n                leave it to None.\n        Returns:\n            A list of ONNXFunctions corresponding to the given name, or None if\n            the name is not in the registry.\n        "
        internal_name_instance = registration.OpName.from_name_parts(namespace=namespace, op_name=op_name, overload=overload)
        return self._registry.get(internal_name_instance)

    @_beartype.beartype
    def is_registered_op(self, namespace: str, op_name: str, overload: Optional[str]=None) -> bool:
        if False:
            i = 10
            return i + 15
        "Returns whether the given op is registered: torch.ops.<namespace>.<op_name>.<overload>.\n\n        Args:\n            namespace: The namespace of the operator to check.\n            op_name: The name of the operator to check.\n            overload: The overload of the operator to check. If it's default overload,\n                leave it to None.\n\n        Returns:\n            True if the given op is registered, otherwise False.\n        "
        functions = self.get_op_functions(namespace=namespace, op_name=op_name, overload=overload)
        return functions is not None

    @_beartype.beartype
    def _all_registered_ops(self) -> Set[str]:
        if False:
            print('Hello World!')
        'Returns the set of all registered function names.'
        return {op_name_class.qualified_name() for op_name_class in self._registry.keys()}

class ExportOptions:
    """Options to influence the TorchDynamo ONNX exporter.

    Attributes:
        dynamic_shapes: Shape information hint for input/output tensors.
            When ``None``, the exporter determines the most compatible setting.
            When ``True``, all input shapes are considered dynamic.
            When ``False``, all input shapes are considered static.
        op_level_debug: Whether to export the model with op-level debug information
        diagnostic_options: The diagnostic options for the exporter.
        fake_context: The fake context used for symbolic tracing.
        onnx_registry: The ONNX registry used to register ATen operators to ONNX functions.
    """
    dynamic_shapes: Optional[bool] = None
    'Shape information hint for input/output tensors.\n\n    - ``None``: the exporter determines the most compatible setting.\n    - ``True``: all input shapes are considered dynamic.\n    - ``False``: all input shapes are considered static.\n    '
    op_level_debug: Optional[bool] = None
    'When True export the model with op-level debug running ops through ONNX Runtime.'
    diagnostic_options: DiagnosticOptions
    'The diagnostic options for the exporter.'
    fake_context: Optional[ONNXFakeContext] = None
    'The fake context used for symbolic tracing.'
    onnx_registry: Optional[OnnxRegistry] = None
    'The ONNX registry used to register ATen operators to ONNX functions.'

    @_beartype.beartype
    def __init__(self, *, dynamic_shapes: Optional[bool]=None, op_level_debug: Optional[bool]=None, fake_context: Optional[ONNXFakeContext]=None, onnx_registry: Optional[OnnxRegistry]=None, diagnostic_options: Optional[DiagnosticOptions]=None):
        if False:
            for i in range(10):
                print('nop')
        self.dynamic_shapes = dynamic_shapes
        self.op_level_debug = op_level_debug
        self.fake_context = fake_context
        self.onnx_registry = onnx_registry
        self.diagnostic_options = diagnostic_options or DiagnosticOptions()

class ResolvedExportOptions(ExportOptions):
    """Consolidates :class:`ExportOptions` with default values.
    All unspecified options from :class:`ExportOptions` are assigned a default value.
    This is an internal class and its API may be changed at any time without notice.
    """
    dynamic_shapes: bool
    op_level_debug: bool
    diagnostic_options: DiagnosticOptions
    fake_context: ONNXFakeContext
    onnx_registry: OnnxRegistry
    decomposition_table: Dict[torch._ops.OpOverload, Callable]
    'A dictionary that maps operators to their decomposition functions.'
    onnxfunction_dispatcher: torch.onnx._internal.fx.onnxfunction_dispatcher.OnnxFunctionDispatcher
    'The ONNX dispatcher used to dispatch ATen operators to ONNX functions.'
    fx_tracer: FXGraphExtractor
    'The FXGraphExtractor instance used to extract the FX graph from the model.'
    diagnostic_context: diagnostics.DiagnosticContext
    'The diagnostics context for the export. Responsible for recording diagnostics,\n    logging diagnostics, and generating the SARIF log.'

    @_beartype.beartype
    def __init__(self, options: Union[ExportOptions, 'ResolvedExportOptions'], model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None):
        if False:
            for i in range(10):
                print('nop')
        from torch.onnx._internal.fx import diagnostics, dynamo_graph_extractor, torch_export_graph_extractor
        if isinstance(options, ResolvedExportOptions):
            self.dynamic_shapes = options.dynamic_shapes
            self.op_level_debug = options.op_level_debug
            self.diagnostic_options = options.diagnostic_options
            self.fake_context = options.fake_context
            if isinstance(model, torch_export.ExportedProgram) and (not isinstance(options.fx_tracer, torch_export_graph_extractor.TorchExport)):
                message = "'model' of type 'ExportedProgram' is only supported with 'TorchExport' FX Tracer"
                e = InvalidExportOptionsError(message)
                raise InvalidExportOptionsError(ONNXProgram._from_failure(e, options.diagnostic_context), message)
            self.fx_tracer = options.fx_tracer
            self.onnx_registry = options.onnx_registry
            self.onnxfunction_dispatcher = options.onnxfunction_dispatcher
            self.decomposition_table = options.decomposition_table
            self.diagnostic_context = options.diagnostic_context
        else:
            T = TypeVar('T')

            @_beartype.beartype
            def resolve(value: Optional[T], fallback: Union[T, Callable[[], T]]) -> T:
                if False:
                    while True:
                        i = 10
                if value is not None:
                    return value
                if callable(fallback):
                    return fallback()
                return fallback
            self.dynamic_shapes = resolve(options.dynamic_shapes, False)
            self.diagnostic_options = resolve(options.diagnostic_options, DiagnosticOptions())
            if isinstance(model, torch_export.ExportedProgram):
                self.fx_tracer = torch_export_graph_extractor.TorchExport()
            else:
                self.fx_tracer = dynamo_graph_extractor.DynamoExport()
            self.fake_context = resolve(options.fake_context, None)
            self.diagnostic_context = diagnostics.DiagnosticContext('torch.onnx.dynamo_export', torch.__version__, self.diagnostic_options)
            self.onnx_registry = resolve(options.onnx_registry, OnnxRegistry())
            self.decomposition_table = decomposition_table.create_onnx_friendly_decomposition_table(self.onnx_registry)
            from torch.onnx._internal.fx import onnxfunction_dispatcher
            self.op_level_debug = resolve(options.op_level_debug, False)
            self.onnxfunction_dispatcher = onnxfunction_dispatcher.OnnxFunctionDispatcher(self.onnx_registry, self.diagnostic_context)
            for key in dir(options):
                if not key.startswith('_'):
                    assert hasattr(self, key), f"Unresolved option '{key}'"

@contextlib.contextmanager
def enable_fake_mode():
    if False:
        while True:
            i = 10
    'Enable fake mode for the duration of the context.\n\n    Internally it instantiates a :class:`torch._subclasses.fake_tensor.FakeTensorMode` context manager\n    that converts user input and model parameters into :class:`torch._subclasses.fake_tensor.FakeTensor`.\n\n    A :class:`torch._subclasses.fake_tensor.FakeTensor`\n    is a :class:`torch.Tensor` with the ability to run PyTorch code without having to\n    actually do computation through tensors allocated on a ``meta`` device. Because\n    there is no actual data being allocated on the device, this API allows for\n    exporting large models without the actual memory footprint needed for executing it.\n\n    It is highly recommended to enable fake mode when exporting models that\n    are too large to fit into memory.\n\n    Returns:\n        A :class:`ONNXFakeContext` object that must be passed to :func:`dynamo_export`\n        through the :attr:`ExportOptions.fake_context` argument.\n\n    Example::\n\n        # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)\n        >>> import torch\n        >>> import torch.onnx\n        >>> class MyModel(torch.nn.Module):  # Dummy model\n        ...     def __init__(self) -> None:\n        ...         super().__init__()\n        ...         self.linear = torch.nn.Linear(2, 2)\n        ...     def forward(self, x):\n        ...         out = self.linear(x)\n        ...         return out\n        >>> with torch.onnx.enable_fake_mode() as fake_context:\n        ...     my_nn_module = MyModel()\n        ...     arg1 = torch.randn(2, 2, 2)  # positional input 1\n        >>> export_options = torch.onnx.ExportOptions(fake_context=fake_context)\n        >>> onnx_program = torch.onnx.dynamo_export(\n        ...     my_nn_module,\n        ...     arg1,\n        ...     export_options=export_options\n        ... )\n        >>> # Saving model WITHOUT initializers\n        >>> onnx_program.save("my_model_without_initializers.onnx")\n        >>> # Saving model WITH initializers\n        >>> onnx_program.save("my_model_with_initializers.onnx", model_state_dict=MyModel().state_dict())\n\n    .. warning::\n        This API is experimental and is *NOT* backward-compatible.\n\n    '
    from torch._subclasses import fake_tensor
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    fake_mode = fake_tensor.FakeTensorMode(allow_non_fake_inputs=not torch._guards.detect_fake_mode(), shape_env=ShapeEnv(allow_scalar_outputs=False, allow_dynamic_output_shape_ops=False))
    patcher_context = patcher.ONNXTorchPatcher()
    fake_context = ONNXFakeContext(fake_mode=fake_mode)
    with fake_mode, patcher_context:
        yield fake_context
    fake_context.state_dict_paths = tuple(patcher_context.paths)

@runtime_checkable
class ONNXProgramSerializer(Protocol):
    """Protocol for serializing an ONNX graph into a specific format (e.g. Protobuf).
    Note that this is an advanced usage scenario."""

    def serialize(self, onnx_program: ONNXProgram, destination: io.BufferedIOBase) -> None:
        if False:
            return 10
        'Protocol method that must be implemented for serialization.\n\n        Args:\n            onnx_program: Represents the in-memory exported ONNX model\n            destination: A binary IO stream or pre-allocated buffer into which\n                the serialized model should be written.\n\n        Example:\n\n            A simple serializer that writes the exported :py:obj:`onnx.ModelProto` in Protobuf\n            format to ``destination``:\n\n            ::\n\n                # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)\n                >>> import io\n                >>> import torch\n                >>> import torch.onnx\n                >>> class MyModel(torch.nn.Module):  # Dummy model\n                ...     def __init__(self) -> None:\n                ...         super().__init__()\n                ...         self.linear = torch.nn.Linear(2, 2)\n                ...     def forward(self, x):\n                ...         out = self.linear(x)\n                ...         return out\n                >>> class ProtobufONNXProgramSerializer:\n                ...     def serialize(\n                ...         self, onnx_program: torch.onnx.ONNXProgram, destination: io.BufferedIOBase\n                ...     ) -> None:\n                ...         destination.write(onnx_program.model_proto.SerializeToString())\n                >>> model = MyModel()\n                >>> arg1 = torch.randn(2, 2, 2)  # positional input 1\n                >>> torch.onnx.dynamo_export(model, arg1).save(\n                ...     destination="exported_model.onnx",\n                ...     serializer=ProtobufONNXProgramSerializer(),\n                ... )\n        '
        ...

class ProtobufONNXProgramSerializer:
    """Serializes ONNX graph as Protobuf."""

    @_beartype.beartype
    def serialize(self, onnx_program: ONNXProgram, destination: io.BufferedIOBase) -> None:
        if False:
            print('Hello World!')
        import onnx
        if not isinstance(onnx_program.model_proto, onnx.ModelProto):
            raise ValueError('onnx_program.ModelProto is not an onnx.ModelProto')
        destination.write(onnx_program.model_proto.SerializeToString())

class LargeProtobufONNXProgramSerializer:
    """Serializes ONNX graph as Protobuf.

    Fallback to serializing as Protobuf with external data for models larger than 2GB.
    """
    _destination_path: Final[str]

    def __init__(self, destination_path: str):
        if False:
            for i in range(10):
                print('nop')
        self._destination_path = destination_path

    @_beartype.beartype
    def serialize(self, onnx_program: ONNXProgram, destination: io.BufferedIOBase) -> None:
        if False:
            print('Hello World!')
        '`destination` is ignored. The model is saved to `self._destination_path` instead.'
        import onnx
        if onnx_program.model_proto.ByteSize() < _PROTOBUF_SIZE_MAX_LIMIT:
            onnx.save_model(onnx_program.model_proto, self._destination_path)
        else:
            onnx.save_model(onnx_program.model_proto, self._destination_path, save_as_external_data=True, all_tensors_to_one_file=True)

class ONNXProgram:
    """An in-memory representation of a PyTorch model that has been exported to ONNX."""
    _model_proto: Final[onnx.ModelProto]
    _input_adapter: Final[io_adapter.InputAdapter]
    _output_adapter: Final[io_adapter.OutputAdapter]
    _diagnostic_context: Final[diagnostics.DiagnosticContext]
    _fake_context: Final[Optional[ONNXFakeContext]]
    _export_exception: Final[Optional[Exception]]

    @_beartype.beartype
    def __init__(self, model_proto: onnx.ModelProto, input_adapter: io_adapter.InputAdapter, output_adapter: io_adapter.OutputAdapter, diagnostic_context: diagnostics.DiagnosticContext, *, fake_context: Optional[ONNXFakeContext]=None, export_exception: Optional[Exception]=None):
        if False:
            print('Hello World!')
        self._model_proto = model_proto
        self._input_adapter = input_adapter
        self._output_adapter = output_adapter
        self._diagnostic_context = diagnostic_context
        self._fake_context = fake_context
        self._export_exception = export_exception

    @property
    def model_proto(self) -> onnx.ModelProto:
        if False:
            print('Hello World!')
        'The exported ONNX model as an :py:obj:`onnx.ModelProto`.'
        if self._export_exception is not None:
            raise self._export_exception
        return self._model_proto

    @property
    def diagnostic_context(self) -> diagnostics.DiagnosticContext:
        if False:
            while True:
                i = 10
        'The diagnostic context associated with the export.'
        return self._diagnostic_context

    @property
    def fake_context(self) -> Optional[ONNXFakeContext]:
        if False:
            while True:
                i = 10
        'The fake context associated with the export.'
        return self._fake_context

    @_beartype.beartype
    def adapt_torch_inputs_to_onnx(self, *model_args, **model_kwargs) -> Sequence[Union[torch.Tensor, int, float, bool]]:
        if False:
            return 10
        'Converts the PyTorch model inputs to exported ONNX model inputs format.\n\n        Due to design differences, input/output format between PyTorch model and exported\n        ONNX model are often not the same. E.g., None is allowed for PyTorch model, but are\n        not supported by ONNX. Nested constructs of tensors are allowed for PyTorch model,\n        but only flattened tensors are supported by ONNX, etc.\n\n        The actual adapting steps are associated with each individual export. It\n        depends on the PyTorch model, the particular set of model_args and model_kwargs\n        used for the export, and export options.\n\n        This method replays the adapting steps recorded during export.\n\n        Args:\n            model_args: The PyTorch model inputs.\n            model_kwargs: The PyTorch model keyword inputs.\n\n        Returns:\n            A sequence of tensors converted from PyTorch model inputs.\n\n        Example::\n\n            # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)\n            >>> import torch\n            >>> import torch.onnx\n            >>> from typing import Dict, Tuple\n            >>> def func_with_nested_input_structure(\n            ...     x_dict: Dict[str, torch.Tensor],\n            ...     y_tuple: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]\n            ... ):\n            ...     if "a" in x_dict:\n            ...         x = x_dict["a"]\n            ...     elif "b" in x_dict:\n            ...         x = x_dict["b"]\n            ...     else:\n            ...         x = torch.randn(3)\n            ...\n            ...     y1, (y2, y3) = y_tuple\n            ...\n            ...     return x + y1 + y2 + y3\n            >>> x_dict = {"a": torch.tensor(1.)}\n            >>> y_tuple = (torch.tensor(2.), (torch.tensor(3.), torch.tensor(4.)))\n            >>> onnx_program = torch.onnx.dynamo_export(func_with_nested_input_structure, x_dict, y_tuple)\n            >>> print(x_dict, y_tuple)\n            {\'a\': tensor(1.)} (tensor(2.), (tensor(3.), tensor(4.)))\n            >>> print(onnx_program.adapt_torch_inputs_to_onnx(x_dict, y_tuple))\n            (tensor(1.), tensor(2.), tensor(3.), tensor(4.))\n\n        .. warning::\n            This API is experimental and is *NOT* backward-compatible.\n\n        '
        return self._input_adapter.apply(*model_args, **model_kwargs)

    @_beartype.beartype
    def adapt_torch_outputs_to_onnx(self, model_outputs: Any) -> Sequence[Union[torch.Tensor, int, float, bool]]:
        if False:
            for i in range(10):
                print('nop')
        'Converts the PyTorch model outputs to exported ONNX model outputs format.\n\n        Due to design differences, input/output format between PyTorch model and exported\n        ONNX model are often not the same. E.g., None is allowed for PyTorch model, but are\n        not supported by ONNX. Nested constructs of tensors are allowed for PyTorch model,\n        but only flattened tensors are supported by ONNX, etc.\n\n        The actual adapting steps are associated with each individual export. It\n        depends on the PyTorch model, the particular set of model_args and model_kwargs\n        used for the export, and export options.\n\n        This method replays the adapting steps recorded during export.\n\n        Args:\n            model_outputs: The PyTorch model outputs.\n\n        Returns:\n            PyTorch model outputs in exported ONNX model outputs format.\n\n        Example::\n\n            # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)\n            >>> import torch\n            >>> import torch.onnx\n            >>> def func_returning_tuples(x, y, z):\n            ...     x = x + y\n            ...     y = y + z\n            ...     z = x + y\n            ...     return (x, (y, z))\n            >>> x = torch.tensor(1.)\n            >>> y = torch.tensor(2.)\n            >>> z = torch.tensor(3.)\n            >>> onnx_program = torch.onnx.dynamo_export(func_returning_tuples, x, y, z)\n            >>> pt_output = func_returning_tuples(x, y, z)\n            >>> print(pt_output)\n            (tensor(3.), (tensor(5.), tensor(8.)))\n            >>> print(onnx_program.adapt_torch_outputs_to_onnx(pt_output))\n            [tensor(3.), tensor(5.), tensor(8.)]\n\n        .. warning::\n            This API is experimental and is *NOT* backward-compatible.\n\n        '
        return self._output_adapter.apply(model_outputs)

    @_beartype.beartype
    def save(self, destination: Union[str, io.BufferedIOBase], *, model_state_dict: Optional[Union[Dict[str, Any], str]]=None, serializer: Optional[ONNXProgramSerializer]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Saves the in-memory ONNX model to ``destination`` using specified ``serializer``.\n\n        Args:\n            destination: The destination to save the ONNX model. It can be either a string or a file-like object.\n                When used with ``model_state_dict``, it must be a string with a full path to the destination.\n                In that case, besides saving the ONNX model, a folder with "_initializers" suffix (without extension)\n                will be created to store the each initializer of the ONNX model in a separate file. For example, if the\n                destination is "/path/model.onnx", the initializers will be saved in "/path/model_initializers/" folder.\n            model_state_dict: The state_dict of the PyTorch model containing all weights on it.\n                It can be either a dict as returned by :meth:`model.state_dict`, or a string with a file name.\n                Required when :func:`enable_fake_mode` is used but real initializers are needed on the ONNX graph.\n                It can be either a string with the path to a checkpoint or a dictionary with the actual model state.\n\n            serializer: The serializer to use. If not specified, the model will be serialized as Protobuf.\n        '
        if serializer is None:
            if isinstance(destination, str):
                serializer = LargeProtobufONNXProgramSerializer(destination)
            else:
                serializer = ProtobufONNXProgramSerializer()
        _model_state_dict_files: List[Union[str, io.BytesIO]] = []
        if model_state_dict is not None:
            if isinstance(model_state_dict, dict):
                model_state_dict_file = io.BytesIO()
                torch.save(model_state_dict, model_state_dict_file)
                model_state_dict_file.seek(0)
                _model_state_dict_files.append(model_state_dict_file)
            else:
                (isinstance(model_state_dict, str), "model_state_dict must be a path to the model's state_dict or the actual state_dict")
                _model_state_dict_files.append(model_state_dict)
        elif self._fake_context and self._fake_context.state_dict_paths:
            for path in self._fake_context.state_dict_paths:
                if path in _model_state_dict_files:
                    continue
                try:
                    extra_state_dict = torch.load(path)
                    extra_state_dict_file = io.BytesIO()
                    torch.save(extra_state_dict, extra_state_dict_file)
                    extra_state_dict_file.seek(0)
                    _model_state_dict_files.append(extra_state_dict_file)
                except FileNotFoundError:
                    pass
        if _model_state_dict_files:
            if not isinstance(destination, str):
                raise RuntimeError('`destination` must be a string with a path when `model_state_dict` is specified.')
            (destination_path, destination_filename) = os.path.split(destination)
            onnx_model_location = destination_filename
            onnx_initializer_location = destination_filename.split('.')[0] + '_initializers'
            fx_serialization.save_model_with_external_data(destination_path, onnx_model_location, onnx_initializer_location, tuple(_model_state_dict_files), self.model_proto)
        elif isinstance(destination, str):
            with open(destination, 'wb') as f:
                serializer.serialize(self, f)
        else:
            try:
                serializer.serialize(self, destination)
            except ValueError as exc:
                raise ValueError("'destination' should be provided as a path-like string when saving a model larger than 2GB. External tensor data will be saved alongside the model on disk.") from exc

    @_beartype.beartype
    def save_diagnostics(self, destination: str) -> None:
        if False:
            while True:
                i = 10
        'Saves the export diagnostics as a SARIF log to the specified destination path.\n\n        Args:\n            destination: The destination to save the diagnostics SARIF log.\n                It must have a `.sarif` extension.\n\n        Raises:\n            ValueError: If the destination path does not end with `.sarif` extension.\n        '
        if not destination.endswith('.sarif'):
            message = f"'destination' must have a .sarif extension, got {destination}"
            log.fatal(message)
            raise ValueError(message)
        self.diagnostic_context.dump(destination)

    @classmethod
    def _from_failure(cls, export_exception: Exception, diagnostic_context: diagnostics.DiagnosticContext) -> Self:
        if False:
            return 10
        '\n        Creates an instance of :class:`ONNXProgram` when the export process encounters a failure.\n\n        In case of a failed export, this method is used to encapsulate the exception\n        and associated diagnostic context within an :class:`ONNXProgram` instance for\n        easier handling and debugging.\n\n        Args:\n            export_exception: The exception raised during the export process.\n            diagnostic_context: The context associated with diagnostics during export.\n\n        Returns:\n            An instance of :class:`ONNXProgram` representing the failed ONNX program.\n        '
        import onnx
        return ONNXProgram(onnx.ModelProto(), io_adapter.InputAdapter(), io_adapter.OutputAdapter(), diagnostic_context, export_exception=export_exception)

class FXGraphExtractor(abc.ABC):
    """Abstract interface for FX graph extractor engines.
    This class isolates FX extraction logic from the rest of the export logic.
    That allows a single ONNX exporter that can leverage different FX graphs."""

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.input_adapter: io_adapter.InputAdapter = io_adapter.InputAdapter()
        self.output_adapter: io_adapter.OutputAdapter = io_adapter.OutputAdapter()

    @abc.abstractmethod
    def generate_fx(self, options: ResolvedExportOptions, model: Union[torch.nn.Module, Callable], model_args: Sequence[Any], model_kwargs: Mapping[str, Any]) -> torch.fx.GraphModule:
        if False:
            return 10
        "Analyzes user ``model`` and generates a FX graph.\n        Args:\n            options: The export options.\n            model: The user model.\n            model_args: The model's positional input arguments.\n            model_kwargs: The model's keyword input arguments.\n        Returns:\n            The generated FX Graph.\n        "
        ...

    @abc.abstractmethod
    def pre_export_passes(self, options: ResolvedExportOptions, original_model: Union[torch.nn.Module, Callable], fx_module: torch.fx.GraphModule, fx_module_args: Sequence[Any]):
        if False:
            return 10
        'Applies pre-export passes to the FX graph.\n\n        Pre-export passes are FX-to-FX graph transformations that make the graph\n        more palatable for the FX-to-ONNX conversion.\n        For example, it can be used to flatten model input/output, add explicit\n        casts to the graph, replace/decompose operators, functionalize the graph, etc.\n        '
        ...

class Exporter:

    @_beartype.beartype
    def __init__(self, options: ResolvedExportOptions, model: Union[torch.nn.Module, Callable], model_args: Sequence[Any], model_kwargs: Mapping[str, Any]):
        if False:
            print('Hello World!')
        self.options = options
        assert self.options is not None
        self.model = model
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        from torch.onnx._internal.fx import fx_symbolic_graph_extractor
        if not isinstance(self.options.fx_tracer, fx_symbolic_graph_extractor.FXSymbolicTracer):
            self._assert_fake_tensor_mode()

    def export(self) -> ONNXProgram:
        if False:
            i = 10
            return i + 15
        with self.options.diagnostic_context:
            graph_module = self.options.fx_tracer.generate_fx(self.options, self.model, self.model_args, self.model_kwargs)
            from torch.onnx._internal.fx import fx_onnx_interpreter
            fx_interpreter = fx_onnx_interpreter.FxOnnxInterpreter(diagnostic_context=self.options.diagnostic_context)
            onnxscript_graph = fx_interpreter.run(fx_graph_module=graph_module, onnxfunction_dispatcher=self.options.onnxfunction_dispatcher, op_level_debug=self.options.op_level_debug)
            if self.options.fake_context is not None:
                initializers_with_real_tensors: Dict[str, torch.Tensor] = {}
                for (initializer_name, initializer) in onnxscript_graph.initializers.items():
                    if not isinstance(initializer, torch._subclasses.FakeTensor):
                        initializers_with_real_tensors[initializer_name] = initializer
                onnxscript_graph.initializers = initializers_with_real_tensors
            onnx_model = onnxscript_graph.to_model_proto(self.options.onnx_registry.opset_version)
            return torch.onnx.ONNXProgram(onnx_model, self.options.fx_tracer.input_adapter, self.options.fx_tracer.output_adapter, self.options.diagnostic_context, fake_context=self.options.fake_context)

    def _assert_fake_tensor_mode(self):
        if False:
            for i in range(10):
                print('nop')
        'Asserts that the model and its input do not contain fake tensors.'
        has_any_fake_tensor = pytree.tree_any(lambda x: isinstance(x, torch._subclasses.FakeTensor), (self.model_args, self.model_kwargs))
        has_any_fake_param_or_buffer = False
        if isinstance(self.model, torch.nn.Module):
            has_any_fake_param_or_buffer = pytree.tree_any(lambda x: isinstance(x, torch._subclasses.FakeTensor), (self.model.parameters(), self.model.buffers()))
        if (has_any_fake_tensor or has_any_fake_param_or_buffer) and (not self.options.fake_context):
            raise RuntimeError('Cannot export a model with fake inputs/weights without enabling fake mode.')
        has_any_non_fake_tensors = pytree.tree_any(lambda x: isinstance(x, torch.Tensor) and (not isinstance(x, torch._subclasses.FakeTensor)), (self.model_args, self.model_kwargs))
        has_any_non_fake_param_or_buffer = False
        if isinstance(self.model, torch.nn.Module):
            has_any_non_fake_param_or_buffer = pytree.tree_any(lambda x: isinstance(x, torch.Tensor) and (not isinstance(x, torch._subclasses.FakeTensor)), (self.model.parameters(), self.model.buffers()))
        if (has_any_non_fake_tensors or has_any_non_fake_param_or_buffer) and self.options.fake_context:
            raise RuntimeError('Cannot export a model with non fake inputs/weights and enabled fake mode.')

class UnsatisfiedDependencyError(RuntimeError):
    """Raised when an ONNX exporter dependency cannot be satisfied."""

    def __init__(self, package_name: str, message: str):
        if False:
            return 10
        super().__init__(message)
        self.package_name = package_name

class OnnxExporterError(RuntimeError):
    """Raised when an ONNX exporter error occurs.

    This exception is thrown when there's an error during the ONNX export process.
    It encapsulates the :class:`ONNXProgram` object generated until the failure, allowing
    access to the partial export results and associated metadata.
    """
    onnx_program: Final[ONNXProgram]

    def __init__(self, onnx_program: ONNXProgram, message: str):
        if False:
            return 10
        '\n        Initializes the OnnxExporterError with the given ONNX program and message.\n\n        Args:\n            onnx_program (ONNXProgram): The partial results of the ONNX export.\n            message (str): The error message to be displayed.\n        '
        super().__init__(message)
        self.onnx_program = onnx_program

class InvalidExportOptionsError(RuntimeError):
    """Raised when user specified an invalid value for the :class:`ExportOptions`."""
    pass

@_beartype.beartype
def _assert_dependencies(export_options: ResolvedExportOptions):
    if False:
        print('Hello World!')
    opset_version = export_options.onnx_registry.opset_version

    def missing_package(package_name: str, exc_info: logging._ExcInfoType):
        if False:
            i = 10
            return i + 15
        message = f'Please install the `{package_name}` package (e.g. `python -m pip install {package_name}`).'
        log.fatal(message, exc_info=exc_info)
        return UnsatisfiedDependencyError(package_name, message)

    def missing_opset(package_name: str):
        if False:
            i = 10
            return i + 15
        message = f'The installed `{package_name}` does not support the specified ONNX opset version {opset_version}. Install a newer `{package_name}` package or specify an older opset version.'
        log.fatal(message)
        return UnsatisfiedDependencyError(package_name, message)
    try:
        import onnx
    except ImportError as e:
        raise missing_package('onnx', e) from e
    if onnx.defs.onnx_opset_version() < opset_version:
        raise missing_opset('onnx')
    try:
        import onnxscript
    except ImportError as e:
        raise missing_package('onnxscript', e) from e
    if not isinstance(onnxscript.onnx_opset.all_opsets['', opset_version], onnxscript.values.Opset):
        raise missing_opset('onnxscript')

@_beartype.beartype
def dynamo_export(model: Union[torch.nn.Module, Callable, torch_export.ExportedProgram], /, *model_args, export_options: Optional[ExportOptions]=None, **model_kwargs) -> ONNXProgram:
    if False:
        return 10
    'Export a torch.nn.Module to an ONNX graph.\n\n    Args:\n        model: The PyTorch model to be exported to ONNX.\n        model_args: Positional inputs to ``model``.\n        model_kwargs: Keyword inputs to ``model``.\n        export_options: Options to influence the export to ONNX.\n\n    Returns:\n        An in-memory representation of the exported ONNX model.\n\n    **Example 1 - Simplest export**\n    ::\n\n        class MyModel(torch.nn.Module):\n            def __init__(self) -> None:\n                super().__init__()\n                self.linear = torch.nn.Linear(2, 2)\n            def forward(self, x, bias=None):\n                out = self.linear(x)\n                out = out + bias\n                return out\n        model = MyModel()\n        kwargs = {"bias": 3.}\n        args = (torch.randn(2, 2, 2),)\n        onnx_program = torch.onnx.dynamo_export(\n            model,\n            *args,\n            **kwargs).save("my_simple_model.onnx")\n\n    **Example 2 - Exporting with dynamic shapes**\n    ::\n\n        # The previous model can be exported with dynamic shapes\n        export_options = torch.onnx.ExportOptions(dynamic_shapes=True)\n        onnx_program = torch.onnx.dynamo_export(\n            model,\n            *args,\n            **kwargs,\n            export_options=export_options)\n        onnx_program.save("my_dynamic_model.onnx")\n\n\n    By printing input dynamic dimensions we can see the input shape is no longer (2,2,2)\n    ::\n\n        >>> print(onnx_program.model_proto.graph.input[0])\n        name: "arg0"\n        type {\n          tensor_type {\n            elem_type: 1\n            shape {\n              dim {\n                dim_param: "arg0_dim_0"\n              }\n              dim {\n                dim_param: "arg0_dim_1"\n              }\n              dim {\n                dim_param: "arg0_dim_2"\n              }\n            }\n          }\n        }\n    '
    if export_options is not None:
        resolved_export_options = export_options if isinstance(export_options, ResolvedExportOptions) else ResolvedExportOptions(export_options, model=model)
    else:
        resolved_export_options = ResolvedExportOptions(ExportOptions(), model=model)
    _assert_dependencies(resolved_export_options)
    try:
        return Exporter(options=resolved_export_options, model=model, model_args=model_args, model_kwargs=model_kwargs).export()
    except Exception as e:
        sarif_report_path = _DEFAULT_FAILED_EXPORT_SARIF_LOG_PATH
        resolved_export_options.diagnostic_context.dump(sarif_report_path)
        message = f"Failed to export the model to ONNX. Generating SARIF report at '{sarif_report_path}'. SARIF is a standard format for the output of static analysis tools. SARIF logs can be loaded in VS Code SARIF viewer extension, or SARIF web viewer (https://microsoft.github.io/sarif-web-component/). Please report a bug on PyTorch Github: {_PYTORCH_GITHUB_ISSUES_URL}"
        raise OnnxExporterError(ONNXProgram._from_failure(e, resolved_export_options.diagnostic_context), message) from e

def common_pre_export_passes(options: ResolvedExportOptions, original_model: Union[torch.nn.Module, Callable], fx_module: torch.fx.GraphModule, fx_module_args: Sequence[Any]):
    if False:
        for i in range(10):
            print('nop')
    from torch.onnx._internal.fx import analysis, passes
    diagnostic_context = options.diagnostic_context
    module = passes.Decompose(diagnostic_context, fx_module, options.decomposition_table, enable_dynamic_axes=options.dynamic_shapes, allow_fake_constant=options.fake_context is not None).run(*fx_module_args)
    module = passes.Functionalize(diagnostic_context, module, enable_dynamic_axes=options.dynamic_shapes, allow_fake_constant=options.fake_context is not None).run(*fx_module_args)
    module = passes.RemoveInputMutation(diagnostic_context, module).run(*fx_module_args)
    module = passes.InsertTypePromotion(diagnostic_context, module).run()
    analysis.UnsupportedFxNodesAnalysis(diagnostic_context, module, options.onnxfunction_dispatcher).analyze(infra.levels.ERROR)
    if isinstance(original_model, torch.nn.Module):
        module = passes.RestoreParameterAndBufferNames(diagnostic_context, module, original_model).run()
    module = passes.Modularize(diagnostic_context, module).run()
    options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNoneInputStep())
    options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNonTensorInputStep())
    options.fx_tracer.input_adapter.append_step(io_adapter.ConvertComplexToRealRepresentationInputStep())
    options.fx_tracer.output_adapter.append_step(io_adapter.FlattenOutputStep())
    options.fx_tracer.output_adapter.append_step(io_adapter.ConvertComplexToRealRepresentationOutputStep())
    return module
__all__ = ['DiagnosticOptions', 'ExportOptions', 'ONNXProgram', 'ONNXProgramSerializer', 'InvalidExportOptionsError', 'OnnxExporterError', 'OnnxRegistry', 'UnsatisfiedDependencyError', 'dynamo_export', 'enable_fake_mode']