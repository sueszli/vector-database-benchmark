"""Dispatcher for AtenLib functions from onnx-script."""
from __future__ import annotations
import logging
import operator
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import torch
import torch._ops
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import diagnostics, registration, type_utils as fx_type_utils
if TYPE_CHECKING:
    import onnxscript
    from torch.onnx import OnnxRegistry
from onnxscript.function_libs.torch_lib import graph_building as onnxscript_graph_building

@_beartype.beartype
def _find_opschema_matched_symbolic_function_disagnostic_message_formatter(fn: Callable, self, node: torch.fx.Node, default_and_custom_functions: List[registration.ONNXFunction], *args, **kwargs) -> str:
    if False:
        i = 10
        return i + 15
    'Format the diagnostic message for the nearest match warning.'
    all_function_overload_names = ''
    for symbolic_func in default_and_custom_functions:
        overload_func = symbolic_func.onnx_function
        all_function_overload_names += f'ONNX Node: {overload_func.name}[opset={overload_func.opset};is_custom={symbolic_func.is_custom}]. \n'
    return f'FX Node: {node.target}. \n{all_function_overload_names}'

@_beartype.beartype
def _find_operator_overloads_in_onnx_registry_disagnostic_message_formatter(fn: Callable, self, node: torch.fx.Node, *args, **kwargs) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Format the diagnostic message for the nearest match warning.'
    return f"Searching operator overload: '{node.target}' in onnx registry...\n"

class OnnxFunctionDispatcher:
    """A dispatcher that finds the best ONNX Function for ATen/Custom operators.

    It uses the `torch.ops` name to find the function. If not found, it falls back to default.
    Otherwise, the best match is found among all function overloads. An exact match has
    higher precedence over the closest ones.

    Below is a breakdown on how the dispatch mechanism works:

    1. Use the torch.ops name to find the function:
        a. Check if the ATen overload exists in the registry.
        b. If not, check if the default overload exists in the registry.

    2. Find the nearest match among all overloaded functions:
        a. If the types match perfectly, select the function.
        b. Otherwise, find the nearest one with the highest matching score. Because of
            the potential wrongly annotated dtypes and attributes matching, we use
            nearest match to find the best function once the aten name is targeted.

    3. Tie-breaker: If there are multiple nearest matches, we will select the one with
        the highest matching score.

    NOTE: The nearest match `doesn't guarantee` a correct match, and a warning message is logged.
    """

    def __init__(self, onnx_registry: 'OnnxRegistry', diagnostic_context: diagnostics.DiagnosticContext):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the ONNX Function dispatcher.\n\n        Args:\n            onnx_registry: The ONNX registry.\n            diagnostic_context: The diagnostic context to use for reporting errors.\n        '
        self.onnx_registry = onnx_registry
        self.diagnostic_context = diagnostic_context

    @_beartype.beartype
    def dispatch(self, node: torch.fx.Node, onnx_args: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], onnx_kwargs: Dict[str, fx_type_utils.Argument], diagnostic_context: diagnostics.DiagnosticContext) -> Union['onnxscript.OnnxFunction', 'onnxscript.TracedOnnxFunction']:
        if False:
            while True:
                i = 10
        'Dispatches an ONNX function based on the given FX node, arguments, and keyword arguments.\n        Args:\n            node: The TorchFX node to dispatch the function for.\n            onnx_args: The arguments of the ONNX function.\n            onnx_kwargs: The keyword arguments of the ONNX function.\n            diagnostic_context: The diagnostic context to use for reporting errors.\n        Returns:\n            Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.\n        Raises:\n            RuntimeError: If there are no overloaded functions available for the given FX node.\n        '
        default_and_custom_functions = self.get_function_overloads(node, diagnostic_context)
        return self._find_the_perfect_or_nearest_match_onnxfunction(node, default_and_custom_functions, onnx_args, onnx_kwargs, diagnostic_context)

    @_beartype.beartype
    def _filter_or_keep_complex(self, node, default_and_custom_functions: List[registration.ONNXFunction], diagnostic_context: diagnostics.DiagnosticContext) -> List[registration.ONNXFunction]:
        if False:
            for i in range(10):
                print('nop')
        if any((torch.is_complex(arg.meta['val']) for arg in node.args if isinstance(arg, torch.fx.Node) and 'val' in arg.meta and isinstance(arg.meta['val'], torch.Tensor))):
            default_and_custom_functions = [func for func in default_and_custom_functions if func.is_complex]
            if not default_and_custom_functions:
                op_full_name = self._get_aten_name(node, diagnostic_context).qualified_name()
                diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Cannot find any COMPLEX symbolic function for {op_full_name}, which should be registered under {node.target}.', unsupported_fx_node=node)
                diagnostic_context.log(diagnostic)
                raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
        else:
            default_and_custom_functions = [func for func in default_and_custom_functions if not func.is_complex]
            if not default_and_custom_functions:
                op_full_name = self._get_aten_name(node, diagnostic_context).qualified_name()
                diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Can ONLY find COMPLEX symbolic function for {op_full_name}, which should be registered under {node.target}.', unsupported_fx_node=node)
                diagnostic_context.log(diagnostic)
                raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
        return default_and_custom_functions

    @_beartype.beartype
    @diagnostics.diagnose_call(diagnostics.rules.find_opschema_matched_symbolic_function, diagnostic_message_formatter=_find_opschema_matched_symbolic_function_disagnostic_message_formatter)
    def _find_the_perfect_or_nearest_match_onnxfunction(self, node: torch.fx.Node, default_and_custom_functions: List[registration.ONNXFunction], onnx_args: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], onnx_kwargs: Dict[str, fx_type_utils.Argument], diagnostic_context: diagnostics.DiagnosticContext):
        if False:
            for i in range(10):
                print('nop')
        'Find the perfect/nearest matched OnnxFunction for the given FX node, arguments, and keyword arguments.\n\n        Args:\n            default_and_custom_functions: The list includes overloaded functions, with\n                custom ones appearing after the default ones.\n            onnx_args: Arguments organized in PyTorch inputs way.\n            onnx_kwargs: Keyword arguments organized in PyTorch inputs way.\n            diagnostic_context: The diagnostic context to use for reporting errors.\n\n            Returns:\n                Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.\n            Raises:\n                RuntimeError: If there are no overloaded functions available for the given FX node.\n        '
        overload_match_ranking: Dict[registration.ONNXFunction, Optional[int]] = {}
        diagnostic = diagnostic_context.inflight_diagnostic()
        for symbolic_function in reversed(default_and_custom_functions):
            function_opschema = _OnnxSchemaChecker(symbolic_function.onnx_function)
            if function_opschema.perfect_match_inputs(diagnostic, onnx_args, onnx_kwargs):
                return symbolic_function.onnx_function
            overload_match_ranking[symbolic_function] = function_opschema.match_score
        overload_match_ranking = {k: v for (k, v) in overload_match_ranking.items() if v is not None}
        if not overload_match_ranking:
            op_full_name = self._get_aten_name(node, diagnostic_context).qualified_name()
            diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Cannot find any perfect/nearest match of symbolic function for {op_full_name},which should be registered under {node.target}.', unsupported_fx_node=node)
            diagnostic_context.log(diagnostic)
            raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
        diagnostic.warning('### Exact match is not found!\nCannot find a perfect match of symbolic overload, a nearest match is found. Please check the ONNX output carefully. \n')
        diagnostic.level = diagnostics.levels.WARNING
        symbolic_function_list: List[registration.ONNXFunction] = sorted(overload_match_ranking, key=lambda k: (overload_match_ranking[k], k.is_custom, default_and_custom_functions.index(k)), reverse=True)
        return symbolic_function_list[0].onnx_function

    @_beartype.beartype
    def _get_aten_name(self, node: torch.fx.Node, diagnostic_context: diagnostics.DiagnosticContext) -> registration.OpName:
        if False:
            while True:
                i = 10
        'Get the OpName from the target.\n\n        Args:\n            node: The TorchFX node to get the aten name for.\n            diagnostic_context: The diagnostic context to use for reporting errors.\n\n        Returns:\n            The internal op name within dataclass: registration.OpName.\n        '
        if node.target == operator.getitem:
            return registration.OpName.from_name_parts(namespace='aten', op_name='getitem')
        if isinstance(node.target, torch._ops.OpOverloadPacket):
            if node.target != torch.ops.aten.sym_size:
                diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Unsupported OverloadPacket: {node.target}, aten.sym_size is the only allowed OverloadPacket!', unsupported_fx_node=node)
                diagnostic_context.log(diagnostic)
                raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
            aten_op_default = node.target.default
            return registration.OpName.from_op_overload(op_overload=aten_op_default)
        if isinstance(node.target, types.BuiltinFunctionType):
            for node_arg in node.args:
                if not isinstance(node_arg, (torch.fx.Node, int, float)) or (isinstance(node_arg, torch.fx.Node) and (not fx_type_utils.is_torch_symbolic_type(node_arg.meta['val']))):
                    diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Unsupported node arg: {node_arg} (type {type(node_arg)}) with builtin function: {node.target}, only int/float/SymInt/SymFloat is supported with built-in ops!', unsupported_fx_node=node)
                    diagnostic_context.log(diagnostic)
                    raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
            return registration.OpName.from_builtin_function(node.target)
        if isinstance(node.target, torch._ops.OpOverload):
            return registration.OpName.from_op_overload(op_overload=node.target)
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Unknown call_function target: {node.target}', unsupported_fx_node=node)
        diagnostic_context.log(diagnostic)
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)

    @_beartype.beartype
    @diagnostics.diagnose_call(diagnostics.rules.find_operator_overloads_in_onnx_registry, diagnostic_message_formatter=_find_operator_overloads_in_onnx_registry_disagnostic_message_formatter)
    def get_function_overloads(self, node: torch.fx.Node, diagnostic_context: diagnostics.DiagnosticContext) -> List[registration.ONNXFunction]:
        if False:
            print('Hello World!')
        'Get the function overloads from the registry.\n\n        Args:\n            node: The node to get the function overloads for.\n            diagnostic_context: The diagnostic context to use for reporting errors.\n\n        Returns:\n            The list contains ONNXFunctions, starting with the default ones and\n            followed by any custom ones.\n        '
        internal_opname: registration.OpName = self._get_aten_name(node=node, diagnostic_context=diagnostic_context)
        function_group: Optional[List[registration.ONNXFunction]] = None
        function_group = self.onnx_registry.get_op_functions(namespace=internal_opname.namespace, op_name=internal_opname.op_name, overload=internal_opname.overload)
        if function_group is None:
            function_group = self.onnx_registry.get_op_functions(namespace=internal_opname.namespace, op_name=internal_opname.op_name, overload=None)
            if function_group is not None:
                op_full_name = internal_opname.qualified_name()
                diagnostic = diagnostic_context.inflight_diagnostic()
                diagnostic.warning('### The operator overload is not found in onnx registry!\nCannot find the operator overload in onnx registry, but the default overload is found. Please check the ONNX output carefully. \n')
                diagnostic.level = diagnostics.levels.WARNING
        if function_group is not None:
            function_group = self._filter_or_keep_complex(node, function_group, diagnostic_context)
            return function_group
        op_full_name = internal_opname.qualified_name()
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Cannot find symbolic function for {op_full_name}, which should be registered under {node.target}.', unsupported_fx_node=node)
        diagnostic_context.log(diagnostic)
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)

class _OnnxSchemaChecker:
    """
    The OnnxSchemaChecker class is a checker for ONNX OpSchema and param schema.

    It provides methods to check for input compatibility based on the OpSchema. It also
    provides a matching score to indicate how well the OpSchema matches the input and
    kwargs types. A function will be evaluated as perfect match, nearest match eligible,
    or no match.

    Here are some common examples in categories:

    1. [NOTE: Perfect match]: The number of inputs and attributes are exactly the same as
        the OpSchema. The types of inputs and attributes are exactly the same as the
        OpSchema.

        ```python
        inputs = (Tensor[2, 3], Tensor[2, 3])
        attributes = {"alpha": 1.0}

        @torch_op("aten::op")
        def aten_op(self: TReal, other: TReal, alpha: float = 1) -> TReal:
            ...

        ```
        Result: Perfect match.

    2. [NOTE: Optional input]: The dispatcher recognizes optional inputs. However,
        the input can't be ignored. None must be provided.

        ```python
        inputs = (Tensor([2, 3]), None)
        attributes = {}

        aten_op(X: TTensor, Y: Optional[INT64]):
            ...
        ```
        Result: Perfect match.
        Real example: `aten::convolution`.

    3. [NOTE: Different attributes]: If an attribute is provided with value, it's
        a must to match the attribute in function signature.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a":1, "b":2}

        aten_op(X: TTensor, a: int):
            ...
        ```
        Result: No match.
        Real example: `aten::div` vs `aten::div.Tensor_mode`.

    4. [NOTE: Default attributes]: Default attribute will fill in the value into
        inputs/attributes.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {}

        aten_op(X: TTensor, a: int = 3):
            ...
        ```
        Result: Perfect match.
        Real example: `aten::clone`

    5. [NOTE: Ignore attribute with None value]: The attributes with None value
        will be ignored in matching.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a": None}

        aten_op(X: TTensor):
            ...
        ```
        Result: Perfect match.

        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a": None}

        aten_op(X: TTensor, a: int = 3):
            ...
        ```
        Result: Nearest match eligible.

        Real example: `aten::div` vs `aten::div.Tensor_mode`.

    Attributes:
        onnxfunction: The OnnxFunction.
        param_schema: The parameter schema defined in the OnnxFunction.
        op_schema: The ONNX OpSchema.
        type_constraints: The type constraints defined in the OpSchema.
        attributes: The attributes defined in the OpSchema.
        _matching_score: The matching score of the OnnxSchemaChecker .

    """

    def __init__(self, onnxfunction: Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction]):
        if False:
            while True:
                i = 10
        'Initialize the OnnxSchemaChecker .\n\n        Args:\n            onnxfunction: The OnnxFunction.\n        '
        self.onnxfunction = onnxfunction
        self.param_schema = self.onnxfunction.param_schemas()
        op_schema = self.onnxfunction.op_schema
        assert op_schema is not None
        self.op_schema = op_schema
        self.type_constraints = {constraint.type_param_str: set(constraint.allowed_type_strs) for constraint in self.op_schema.type_constraints}
        self.attributes = self.op_schema.attributes
        self._matching_score: Optional[int] = None

    @property
    def match_score(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        "The matching score of the OnnxSchemaChecker .\n\n        If this remains None, it means the matching score has not been calculated,\n        and it's not a nearest match candidate.\n\n        Returns:\n            The matching score of the OnnxSchemaChecker .\n        "
        return self._matching_score

    @_beartype.beartype
    def perfect_match_inputs(self, diagnostic: diagnostics.Diagnostic, args: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], kwargs: Dict[str, fx_type_utils.Argument]) -> bool:
        if False:
            while True:
                i = 10
        'Check if the inputs perfectly match the OpSchema requirements.\n\n        The definition of perfect match is that the input types are all in the type\n        constraints and the number of inputs matches the number of inputs in the\n        OpSchema.\n\n        Checking steps:\n        1. The function signature matches the inputs number, and attribute names.\n        2. The input/attribute types are all in the type constraints.\n\n        A function should at least pass the first step to be eligible for the\n        nearest matching.\n\n        Args:\n            diagnostic: The diagnostic to use for logging detailed info.\n            args: The input arguments organized in PyTorch inputs way.\n            kwargs: The input keyword arguments organized in PyTorch inputs way.\n\n        Returns:\n            True if the inputs match the requirements, False otherwise.\n        '
        (function_inputs, function_attributes) = self._separate_input_attributes_from_arguments(self.param_schema, args, kwargs, fill_defaults=True)
        with diagnostic.log_section(logging.INFO, 'Checking perfect match...'):
            diagnostic.info('%s', diagnostics.LazyString(diagnostics.format_argument, self.onnxfunction))
            is_perfect_match = True
            if len(function_inputs) != len(self.op_schema.inputs):
                with diagnostic.log_section(logging.INFO, 'Failed: input number mismatch!'):
                    diagnostic.info('Actual %d vs expected %d', len(function_inputs), len(self.op_schema.inputs))
                diagnostic.info('The function is not a nearest match candidate.')
                is_perfect_match = False
            if set(function_attributes) != set(self.attributes):
                with diagnostic.log_section(logging.INFO, 'Failed: attribute mismatch!'):
                    diagnostic.info('%s', diagnostics.LazyString(lambda : f'Actual {set(function_attributes)} vs expected {set(self.attributes)}'))
                diagnostic.info('The function is not a nearest match candidate.')
                is_perfect_match = False
            if not is_perfect_match:
                return False
            for (schema_input, torch_input) in zip(self.op_schema.inputs, function_inputs):
                torch_input_compatible_types = _find_onnx_data_type(torch_input)
                allowed_types = self.type_constraints[schema_input.type_str]
                if not allowed_types.intersection(torch_input_compatible_types) and (not any((fx_type_utils.is_optional_onnx_dtype_str(onnx_type_str) for onnx_type_str in allowed_types))):
                    with diagnostic.log_section(logging.INFO, "Failed: input type mismatch for input '%s'!", schema_input.name):
                        diagnostic.info('Actual %s vs\nExpected %s', torch_input_compatible_types, allowed_types)
                    is_perfect_match = False
            for (attribute_name, attribute) in function_attributes.items():
                if not self._match_onnx_attribute_type(attribute_name, attribute):
                    with diagnostic.log_section(logging.INFO, "Failed: attribute '%s' type mismatch!", attribute_name):
                        diagnostic.info('Actual %s vs\nExpected %s', type(attribute), self.attributes[attribute_name].type)
                    is_perfect_match = False
            self._record_matching_score(function_inputs, function_attributes)
            diagnostic.info('match score: %d', self.match_score)
            return is_perfect_match

    @_beartype.beartype
    def _match_onnx_attribute_type(self, attribute_name: str, attribute: Union[fx_type_utils.Argument, onnxscript_graph_building.TorchScriptTensor], is_sequence: bool=False) -> bool:
        if False:
            return 10
        if isinstance(attribute, (int, float, bool, str)):
            attribute_onnx_type = fx_type_utils.from_python_type_to_onnx_attribute_type(type(attribute), is_sequence=is_sequence)
            if attribute_onnx_type != self.attributes[attribute_name].type:
                return False
        elif isinstance(attribute, (list, tuple)) and attribute:
            return self._match_onnx_attribute_type(attribute_name, attribute[0], is_sequence=True)
        else:
            return False
        return True

    @_beartype.beartype
    def _record_matching_score(self, inputs: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], attributes: Dict[str, fx_type_utils.Argument]):
        if False:
            i = 10
            return i + 15
        "Calculate the inputs matching score of the OpSchema requirements to find the nearest match.\n\n        Only the functions which have the same number of inputs and attributes as the\n        OpSchema are eligible to be a nearest match candidate. Thus, we don't need to\n        check the length of inputs and attributes here, and only check the types of\n        inputs and attributes.\n\n        How the matchsing score is calculated:\n            score += 1 if one input/attribute type is in the type constraints.\n\n        Limitations:\n            None/NoeType/[] could result in zero matches, and the same score of overloads,\n            which will be recorded in SARIF.\n\n        Args:\n            inputs: The input arguments.\n            attributes: The input keyword arguments.\n\n        Returns:\n            True if the inputs match the requirements, False otherwise.\n        "
        self._matching_score = 0
        for (schema_input, torch_input) in zip(self.op_schema.inputs, inputs):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if allowed_types.intersection(torch_input_compatible_types):
                self._matching_score += 1
        for (attribute_name, attribute_proto) in self.attributes.items():
            attribute = attributes[attribute_name]
            attribute_onnx_type = fx_type_utils.from_python_type_to_onnx_attribute_type(type(attribute))
            if attribute_onnx_type != attribute_proto.type:
                self._matching_score -= 1

    @_beartype.beartype
    def _separate_input_attributes_from_arguments(self, param_schemas: Sequence['onnxscript.values.ParamSchema'], args: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], kwargs: Dict[str, fx_type_utils.Argument], fill_defaults: bool=True) -> Tuple[List[Any], Dict[str, Any]]:
        if False:
            return 10
        'Separate Python args and kwargs into ONNX inputs and attributes.\n\n        Extra_kwargs are ignored if their values are None. For example, if the\n        OpSchema has an attribute "rounding_mode" and the caller provides\n        "rounding_mode=None", the attribute "rounding_mode" will not be included\n        in the returned attributes when the OnnxFunction signature doesn\'t have\n        "rounding_mode" as an attribute.\n\n        Args:\n            param_schemas: The parameter schemas of an Op or a OnnxFunction.\n            args: The Python positional arguments supplied by the caller.\n            kwargs: The Python keyword arguments supplied by the caller.\n            fill_defaults: Whether to fill the default values for attributes.\n\n        Returns:\n            A tuple of two elements:\n            - A list of ONNX inputs.\n            - An dictionary of ONNX attribute names and values.\n\n        Raises:\n            TypeError: When allow_extra_kwargs is False and there are unknown kwargs.\n            TypeError: When a required input is not provided.\n        '
        import onnx
        onnx_inputs: List[Any] = []
        onnx_attributes: Dict[str, Any] = dict()
        copy_kwargs = kwargs.copy()
        for (i, param) in enumerate(param_schemas):
            if param.is_variadic_input:
                onnx_inputs.extend(args[i:])
                args = []
                continue
            if i < len(args):
                if param.is_input:
                    onnx_inputs.append(args[i])
                else:
                    onnx_attributes[param.name] = args[i]
            elif param.name in copy_kwargs:
                if param.is_input:
                    onnx_inputs.append(copy_kwargs[param.name])
                    copy_kwargs.pop(param.name)
                else:
                    onnx_attributes[param.name] = copy_kwargs[param.name]
            elif param.is_attribute and self.attributes[param.name].default_value.type != onnx.AttributeProto.UNDEFINED:
                if fill_defaults:
                    onnx_attributes[param.name] = param.default
            elif param.is_input:
                if fill_defaults:
                    onnx_inputs.append(None)
        for (k, v) in copy_kwargs.items():
            if k not in onnx_attributes and v is not None:
                onnx_attributes[k] = v
        return (onnx_inputs, onnx_attributes)

@_beartype.beartype
def _find_onnx_data_type(torch_input: Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list, tuple]]) -> Set[str]:
    if False:
        return 10
    'Convert inputs data type from torch acceptable dtype to the compatible onnx dtype string.'
    if isinstance(torch_input, fx_type_utils.TensorLike) and torch_input.dtype is not None:
        return fx_type_utils.from_torch_dtype_to_onnx_dtype_str(torch_input.dtype)
    if isinstance(torch_input, (int, float, bool, str)):
        return fx_type_utils.from_torch_dtype_to_onnx_dtype_str(type(torch_input))
    if isinstance(torch_input, (list, tuple)) and torch_input:
        set_dtype = _find_onnx_data_type(torch_input[0])
        if any((isinstance(input, fx_type_utils.TensorLike) for input in torch_input)):
            return {f'seq({dtype})' for dtype in set_dtype}
        else:
            return set_dtype
    if torch_input is None or (isinstance(torch_input, fx_type_utils.TensorLike) and torch_input.dtype is None) or (isinstance(torch_input, (list, tuple)) and (not torch_input)):
        return set()
    raise RuntimeError(f'Unknown input type from input: {torch_input}')