from inspect import Parameter, Signature
import ast
import os
from pathlib import Path
from contextlib import closing
from typing import Union, Optional
from typing import Sequence, List, Any
from nvidia.dali import backend as _b
from nvidia.dali import types as _types
from nvidia.dali.ops import _registry, _names, _docs
from nvidia.dali import types
from nvidia.dali import ops, fn

def _create_annotation_placeholder(typename):
    if False:
        return 10

    class _AnnotationPlaceholderMeta(type):

        def __repr__(cls):
            if False:
                for i in range(10):
                    print('nop')
            return typename

    class _AnnotationPlaceholder(metaclass=_AnnotationPlaceholderMeta):
        __name__ = typename
        __qualname__ = typename
        __module__ = 'typing'

        def __init__(self, val):
            if False:
                i = 10
                return i + 15
            self._val = val

        def __repr__(self):
            if False:
                print('Hello World!')
            return str(self._val)
    return _AnnotationPlaceholder
_DataNode = _create_annotation_placeholder('DataNode')
_DALIDataType = _create_annotation_placeholder('DALIDataType')
_DALIImageType = _create_annotation_placeholder('DALIImageType')
_DALIInterpType = _create_annotation_placeholder('DALIInterpType')
_enum_mapping = {types.DALIDataType: _DALIDataType, types.DALIImageType: _DALIImageType, types.DALIInterpType: _DALIInterpType}
_MAX_INPUT_SPELLED_OUT = 10

def _scalar_element_annotation(scalar_dtype):
    if False:
        while True:
            i = 10
    conv_fn = _types._known_types[scalar_dtype][1]
    try:
        dummy_val = conv_fn(0)
        t = type(dummy_val)
        if t in _enum_mapping:
            return _enum_mapping[t]
        return t
    except NotImplementedError:
        return Any
    except TypeError:
        return Any

def _arg_type_annotation(arg_dtype):
    if False:
        i = 10
        return i + 15
    'Convert regular key-word argument type to annotation. Handles Lists and scalars.\n\n    Parameters\n    ----------\n    arg_dtype : _type_\n        _description_\n    '
    if arg_dtype in _types._vector_types:
        scalar_dtype = _types._vector_types[arg_dtype]
        scalar_annotation = _scalar_element_annotation(scalar_dtype)
        return Union[Sequence[scalar_annotation], scalar_annotation]
    return _scalar_element_annotation(arg_dtype)

def _get_positional_input_param(schema, idx):
    if False:
        return 10
    'Get the Parameter representing positional inputs at `idx`. Automatically mark it as\n    optional. The DataNode annotation currently hides the possibility of MIS.\n\n    The double underscore `__` prefix for argument name is an additional way to indicate\n    positional only arguments, as per MyPy docs. It is obeyed by the VSCode.\n\n    TODO(klecki): Constant promotions - ArrayLike? Also: Multiple Input Sets.\n    '
    default = Parameter.empty if idx < schema.MinNumInput() else None
    annotation = _DataNode if idx < schema.MinNumInput() else Optional[_DataNode]
    if schema.HasInputDox():
        return Parameter(f'__{schema.GetInputName(idx)}', kind=Parameter.POSITIONAL_ONLY, default=default, annotation=annotation)
    else:
        return Parameter(f'__input_{idx}', kind=Parameter.POSITIONAL_ONLY, default=default, annotation=annotation)

def _get_positional_input_params(schema):
    if False:
        i = 10
        return i + 15
    'Get the list of positional only inputs to the operator.\n    '
    param_list = []
    if not schema.HasInputDox() and schema.MaxNumInput() > _MAX_INPUT_SPELLED_OUT:
        param_list.append(Parameter('input', Parameter.VAR_POSITIONAL, annotation=_DataNode))
    else:
        for i in range(schema.MaxNumInput()):
            param_list.append(_get_positional_input_param(schema, i))
    return param_list

def _get_keyword_params(schema, all_args_optional=False):
    if False:
        while True:
            i = 10
    'Get the list of annotated keyword Parameters to the operator.\n    '
    param_list = []
    for arg in schema.GetArgumentNames():
        if schema.IsDeprecatedArg(arg):
            continue
        arg_dtype = schema.GetArgumentType(arg)
        kw_annotation = _arg_type_annotation(arg_dtype)
        is_arg_input = schema.IsTensorArgument(arg)
        annotation = Union[_DataNode, kw_annotation] if is_arg_input else kw_annotation
        if schema.IsArgumentOptional(arg):
            annotation = Optional[annotation]
        default = Parameter.empty
        if schema.HasArgumentDefaultValue(arg):
            default_value_string = schema.GetArgumentDefaultValueString(arg)
            default_value = ast.literal_eval(default_value_string)
            default = types._type_convert_value(arg_dtype, default_value)
            if type(default) in _enum_mapping:
                default = _enum_mapping[type(default)](default)
        elif schema.IsArgumentOptional(arg):
            default = None
        if all_args_optional:
            annotation = Optional[annotation]
            if default == Parameter.empty:
                default = None
        param_list.append(Parameter(name=arg, kind=Parameter.KEYWORD_ONLY, default=default, annotation=annotation))
    return param_list

def _get_implicit_keyword_params(schema, all_args_optional=False):
    if False:
        for i in range(10):
            print('nop')
    'All operators have some additional kwargs, that are not listed in schema, but are\n    implicitly used by DALI.\n    '
    _ = all_args_optional
    return [Parameter(name='device', kind=Parameter.KEYWORD_ONLY, default=None, annotation=Optional[str]), Parameter(name='name', kind=Parameter.KEYWORD_ONLY, default=None, annotation=Optional[str])]

def _call_signature(schema, include_inputs=True, include_kwargs=True, include_self=False, data_node_return=True, all_args_optional=False, filter_annotations=False) -> Signature:
    if False:
        print('Hello World!')
    'Generate a Signature for given schema.\n\n    Parameters\n    ----------\n    schema : OpSchema\n        Schema for the operator.\n    include_inputs : bool, optional\n        If positional inputs should be included in the signature, by default True\n    include_kwargs : bool, optional\n        If keyword arguments should be included in the signature, by default True\n    include_self : bool, optional\n        Prepend `self` as first positional argument in the signature, by default False\n    data_node_return : bool, optional\n        If the signature should have a return annotation or return None (for ops class __init__),\n        by default True\n    all_args_optional : bool, optional\n        Make all keyword arguments optional, even if they are not - needed by the ops API, where\n        the argument can be specified in either __init__ or __call__, by default False\n    '
    param_list = []
    if include_self:
        param_list.append(Parameter('self', kind=Parameter.POSITIONAL_ONLY))
    if include_inputs:
        param_list.extend(_get_positional_input_params(schema))
    if include_kwargs:
        param_list.extend(_get_keyword_params(schema, all_args_optional=all_args_optional))
        param_list.extend(_get_implicit_keyword_params(schema, all_args_optional=all_args_optional))
    if data_node_return:
        if schema.HasOutputFn():
            return_annotation = Union[_DataNode, Sequence[_DataNode], None]
        else:
            num_regular_output = schema.CalculateOutputs(_b.OpSpec(''))
            if num_regular_output == 0:
                return_annotation = None
            elif num_regular_output == 1:
                return_annotation = _DataNode
            else:
                return_annotation = Sequence[_DataNode]
    else:
        return_annotation = None
    if filter_annotations:
        param_list = [Parameter(name=p.name, kind=p.kind, default=p.default) for p in param_list]
        return_annotation = Signature.empty
    return Signature(param_list, return_annotation=return_annotation)

def inspect_repr_fixups(signature: str) -> str:
    if False:
        i = 10
        return i + 15
    "Replace the weird quirks of printing the repr of signature.\n    We use signature object for type safety and additional validation, but the printing rules\n    are questionable in some cases. Python type hints advocate the usage of `None` instead of its\n    type, but printing a signature would insert NoneType (specifically replacing\n    Optional[Union[...]] with Union[..., None] and printing it as Union[..., NoneType]).\n    The NoneType doesn't exist as a `types` definition in some Pythons.\n    "
    return signature.replace('NoneType', 'None')

def _gen_fn_signature(schema, schema_name, fn_name):
    if False:
        for i in range(10):
            print('nop')
    'Write the stub of the fn API function with the docstring, for given operator.\n    '
    return inspect_repr_fixups(f'\ndef {fn_name}{_call_signature(schema, include_inputs=True, include_kwargs=True)}:\n    """{_docs._docstring_generator_fn(schema_name)}\n    """\n    ...\n')

def _gen_ops_signature(schema, schema_name, cls_name):
    if False:
        while True:
            i = 10
    'Write the stub of the fn API class with the docstring, __init__ and __call__ for given\n    operator.\n    '
    return inspect_repr_fixups(f'\nclass {cls_name}:\n    """{_docs._docstring_generator(schema_name)}\n    """\n    def __init__{_call_signature(schema, include_inputs=False, include_kwargs=True, include_self=True, data_node_return=False, all_args_optional=True)}:\n        ...\n\n    def __call__{_call_signature(schema, include_inputs=True, include_kwargs=True, include_self=True, all_args_optional=True)}:\n        """{_docs._docstring_generator_call(schema_name)}\n        """\n        ...\n')
_HEADER = '\n# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n#\n# Licensed under the Apache License, Version 2.0 (the "License");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n#     http://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an "AS IS" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License.\n\nfrom typing import Union, Optional\nfrom typing import Sequence, Any\n\nfrom nvidia.dali.data_node import DataNode\n\nfrom nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType\n\n'

def _build_module_tree():
    if False:
        while True:
            i = 10
    'Build a tree of DALI submodules, starting with empty string as a root one, like:\n    {\n        "" : {\n            "decoders" : {},\n            "experimental": {\n                "readers": {}\n            }\n            "readers" : {},\n        }\n    }\n    '
    module_tree = {}
    processed = set()
    for schema_name in _registry._all_registered_ops():
        schema = _b.TryGetSchema(schema_name)
        if schema is None:
            continue
        if schema.IsDocHidden() or schema.IsInternal():
            continue
        (dotted_name, module_nesting, op_name) = _names._process_op_name(schema_name)
        if dotted_name not in processed:
            module_nesting.insert(0, '')
            curr_dict = module_tree
            for curr_module in module_nesting:
                if curr_module not in curr_dict:
                    curr_dict[curr_module] = dict()
                curr_dict = curr_dict[curr_module]
    return module_tree

def _get_op(api_module, full_qualified_name: List[str]):
    if False:
        i = 10
        return i + 15
    'Resolve the operator function/class from the api_module: ops or fn,\n    by accessing the fully qualified name.\n\n    Parameters\n    ----------\n    api_module : module\n        fn or orps\n    full_qualified_name : List[str]\n        For example ["readers", "File"]\n    '
    op = api_module
    for elem in full_qualified_name:
        op = getattr(op, elem, None)
    return op

def _group_signatures(api: str):
    if False:
        return 10
    'Divide all operators registered into the "ops" or "fn" api into 4 categories and return them\n    as a dictionary:\n    * python_only - there is just the Python definition\n    * hidden_or_internal - op is hidden or internal, defined in backend\n    * python_wrapper - op defined in backend, has a hand-written wrapper (op._generated = False)\n    * generated - op was generated automatically from backend definition (op._generated = True)\n\n    Each entry in the dict contains a list of: `(schema_name : str, op : Callable or Class)`\n    depending on the api type.\n\n    '
    sig_groups = {'python_only': [], 'hidden_or_internal': [], 'python_wrapper': [], 'generated': []}
    api_module = fn if api == 'fn' else ops
    for schema_name in sorted(_registry._all_registered_ops()):
        schema = _b.TryGetSchema(schema_name)
        (_, module_nesting, op_name) = _names._process_op_name(schema_name, api=api)
        op = _get_op(api_module, module_nesting + [op_name])
        if schema is None:
            if op is not None:
                sig_groups['python_only'].append((schema_name, op))
            continue
        if schema.IsDocHidden() or schema.IsInternal():
            sig_groups['hidden_or_internal'].append((schema_name, op))
            continue
        if not getattr(op, '_generated', False):
            sig_groups['python_wrapper'].append((schema_name, op))
            continue
        sig_groups['generated'].append((schema_name, op))
    return sig_groups

class StubFileManager:

    def __init__(self, nvidia_dali_path: Path, api: str):
        if False:
            return 10
        self._module_to_file = {}
        self._nvidia_dali_path = nvidia_dali_path
        self._api = api
        self._module_tree = _build_module_tree()

    def get(self, module_nesting: List[str]):
        if False:
            return 10
        "Get the file representing the given submodule nesting.\n        List may be empty for top-level api module.\n\n        When the file is accessed the first time, it's header and submodule imports are\n        written.\n        "
        module_path = Path('/'.join(module_nesting))
        if module_path not in self._module_to_file:
            file_path = self._nvidia_dali_path / self._api / module_path / '__init__.pyi'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            open(file_path, 'w').close()
            f = open(file_path, 'a')
            self._module_to_file[module_path] = f
            f.write(_HEADER)
            full_module_nesting = [''] + module_nesting
            submodules_dict = self._module_tree
            for submodule in full_module_nesting:
                submodules_dict = submodules_dict[submodule]
            direct_submodules = submodules_dict.keys()
            for direct_submodule in direct_submodules:
                f.write(f'from . import {direct_submodule}\n')
            f.write('\n\n')
        return self._module_to_file[module_path]

    def close(self):
        if False:
            i = 10
            return i + 15
        for (_, f) in self._module_to_file.items():
            f.close()

def gen_all_signatures(nvidia_dali_path, api):
    if False:
        for i in range(10):
            print('nop')
    'Generate the signatures for "fn" or "ops" api.\n\n    Parameters\n    ----------\n    nvidia_dali_path : Path\n        The path to the wheel pre-packaging to the nvidia/dali directory.\n    api : str\n        "fn" or "ops"\n    '
    nvidia_dali_path = Path(nvidia_dali_path)
    with closing(StubFileManager(nvidia_dali_path, api)) as stub_manager:
        sig_groups = _group_signatures(api)
        for (schema_name, op) in sig_groups['python_only'] + sig_groups['python_wrapper']:
            (_, module_nesting, op_name) = _names._process_op_name(schema_name, api=api)
            stub_manager.get(module_nesting).write(f'\n\nfrom {op._impl_module} import ({op.__name__} as {op.__name__})\n\n')
        for (schema_name, op) in sig_groups['generated']:
            (_, module_nesting, op_name) = _names._process_op_name(schema_name, api=api)
            schema = _b.TryGetSchema(schema_name)
            if api == 'fn':
                stub_manager.get(module_nesting).write(_gen_fn_signature(schema, schema_name, op_name))
            else:
                stub_manager.get(module_nesting).write(_gen_ops_signature(schema, schema_name, op_name))