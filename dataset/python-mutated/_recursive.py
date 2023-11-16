import collections
import functools
import inspect
import sys
import textwrap
import types
import warnings
from typing import Dict, List, Set, Type
import torch
import torch._jit_internal as _jit_internal
from torch._sources import fake_range
from torch.jit._builtins import _find_builtin
from torch.jit._check import AttributeTypeIsSupportedChecker
from torch.jit._state import _add_script_class, _get_script_class, _python_cu
from torch.jit.frontend import get_class_properties, get_default_args, get_jit_class_def, get_jit_def
from torch.nn import Module
ScriptMethodStub = collections.namedtuple('ScriptMethodStub', ('resolution_callback', 'def_', 'original_method'))
PropertyStub = collections.namedtuple('PropertyStub', ('resolution_callback', 'def_'))
ignored_attributes = ['_version', '_parameters', '_buffers', '_non_persistent_buffers_set', '_backward_hooks', '_backward_pre_hooks', '_forward_hooks', '_forward_hooks_with_kwargs', '_forward_pre_hooks', '_forward_pre_hooks_with_kwargs', '_forward_hooks_always_called', '_state_dict_hooks', '_state_dict_pre_hooks', '_load_state_dict_pre_hooks', '_load_state_dict_post_hooks', '_modules', '_initializing', 'dump_patches']

def _compile_and_register_class(obj, rcb, qualified_name):
    if False:
        print('Hello World!')
    script_class = _get_script_class(obj)
    if not script_class:
        ast = get_jit_class_def(obj, obj.__name__)
        defaults = torch.jit.frontend.get_default_args_for_class(obj)
        script_class = torch._C._jit_script_class_compile(qualified_name, ast, defaults, rcb)
        _add_script_class(obj, script_class)
    return script_class

def make_stub(func, name):
    if False:
        i = 10
        return i + 15
    rcb = _jit_internal.createResolutionCallbackFromClosure(func)
    ast = get_jit_def(func, name, self_name='RecursiveScriptModule')
    return ScriptMethodStub(rcb, ast, func)

def make_stub_from_method(nn_module, method_name):
    if False:
        while True:
            i = 10
    func = getattr(nn_module, method_name)
    if isinstance(func, ScriptMethodStub):
        return func
    return make_stub(func, method_name)

def make_stubs_from_exported_methods(mod):
    if False:
        return 10
    stubs = []
    for name in dir(mod):
        item = getattr(mod, name, None)
        if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.EXPORT:
            stubs.append(make_stub_from_method(mod, name))
    return stubs

def jit_ignored_properties(module):
    if False:
        print('Hello World!')
    user_annotated_ignored_attributes = getattr(module, '__jit_ignored_attributes__', list())

    def get_properties_names(module):
        if False:
            print('Hello World!')
        return {k for (k, v) in vars(module).items() if isinstance(v, property)}
    properties = get_properties_names(type(module))
    user_annoted_ignored_properties = set()
    for ignored_attr in user_annotated_ignored_attributes:
        if ignored_attr in properties:
            user_annoted_ignored_properties.add(ignored_attr)
    return user_annoted_ignored_properties
_constant_types = (bool, float, int, str, type(None), torch.device, torch.layout, torch.dtype)

def _get_valid_constant(attr, v, owner_type):
    if False:
        while True:
            i = 10
    if isinstance(v, _constant_types):
        return v
    elif isinstance(v, (tuple, list)):
        return tuple((_get_valid_constant(attr, x, owner_type) for x in v))
    constants = ', '.join((torch.typename(typ) for typ in _constant_types))
    raise TypeError(textwrap.dedent(f"\n        '{torch.typename(type(v))}' object in attribute '{owner_type}.{attr}' is not a valid constant.\n        Valid constants are:\n        1. a nn.ModuleList\n        2. a value of type {{{constants}}}\n        3. a list or tuple of (2)\n        "))

class SourceContext(torch._C._jit_tree_views.SourceRangeFactory):

    def __init__(self, source, filename, file_lineno, leading_whitespace_len):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(source, filename, file_lineno, leading_whitespace_len)

def get_annotations(obj):
    if False:
        while True:
            i = 10
    if sys.version_info < (3, 10):
        return getattr(obj, '__annotations__', {})
    annotations = inspect.get_annotations(obj)
    if annotations:
        return annotations

    def get_cls_annotations(cls):
        if False:
            while True:
                i = 10
        cls_annotations = inspect.get_annotations(cls)
        if cls_annotations:
            return cls_annotations
        for base in cls.__bases__:
            cls_annotations = get_cls_annotations(base)
            if cls_annotations:
                return cls_annotations
        return {}
    cls = obj if isinstance(obj, type) else type(obj)
    return get_cls_annotations(cls)

def infer_concrete_type_builder(nn_module, share_types=True):
    if False:
        while True:
            i = 10
    "\n    Build a ConcreteModuleTypeBuilder from an nn.Module.\n\n    This ConcreteModuleType doesn't have a JIT type associated with it yet, it\n    must be filled in by the caller.\n    "
    concrete_type_builder = torch._C.ConcreteModuleTypeBuilder(type(nn_module))
    if isinstance(nn_module, torch.nn.ModuleDict):
        concrete_type_builder.set_module_dict()
    if isinstance(nn_module, (torch.nn.ModuleList, torch.nn.Sequential)):
        concrete_type_builder.set_module_list()
    if isinstance(nn_module, torch.nn.ParameterList):
        concrete_type_builder.set_parameter_list()
    if isinstance(nn_module, torch.nn.ParameterDict):
        concrete_type_builder.set_parameter_dict()
    class_annotations = get_annotations(nn_module)
    if isinstance(nn_module, torch.ao.quantization.QuantWrapper):
        class_annotations = {}
    user_annotated_ignored_attributes = getattr(nn_module, '__jit_ignored_attributes__', list())
    concrete_type_builder.add_ignored_attributes(user_annotated_ignored_attributes)
    ignored_properties = jit_ignored_properties(nn_module)

    def infer_type(name, item):
        if False:
            print('Hello World!')
        inferred = False
        try:
            if name in class_annotations and class_annotations[name] != torch.nn.Module.__annotations__['forward']:
                ann_to_type = torch.jit.annotations.ann_to_type(class_annotations[name], fake_range())
                attr_type = torch._C.InferredType(ann_to_type)
            elif isinstance(item, torch.jit.Attribute):
                ann_to_type = torch.jit.annotations.ann_to_type(item.type, fake_range())
                attr_type = torch._C.InferredType(ann_to_type)
            else:
                attr_type = torch._C._jit_try_infer_type(item)
                inferred = True
        except RuntimeError as re:
            raise RuntimeError(f'Error inferring type for {name}: {item}: {re}') from re
        return (attr_type, inferred)
    added_names = set()
    for (name, item) in nn_module._parameters.items():
        if name in user_annotated_ignored_attributes:
            continue
        assert item is None or isinstance(item, torch.Tensor)
        (attr_type, _) = infer_type(name, item)
        concrete_type_builder.add_attribute(name, attr_type.type(), True, False)
        added_names.add(name)
    for (name, item) in nn_module._buffers.items():
        if name in user_annotated_ignored_attributes:
            continue
        assert item is None or isinstance(item, torch.Tensor)
        (attr_type, _) = infer_type(name, item)
        concrete_type_builder.add_attribute(name, attr_type.type(), False, True)
        added_names.add(name)
    for (name, item) in nn_module._modules.items():
        if name in user_annotated_ignored_attributes:
            continue
        (attr_type, _) = infer_type(name, item)
        if item is None:
            concrete_type_builder.add_attribute(name, attr_type.type(), False, False)
            continue
        if attr_type.success():
            assert attr_type.type().is_interface_type()
            sub_concrete_type = torch._C.ConcreteModuleType.from_jit_type(attr_type.type())
        else:
            sub_concrete_type = get_module_concrete_type(item, share_types)
        concrete_type_builder.add_module(name, sub_concrete_type)
        added_names.add(name)
    constants_set = set(getattr(nn_module, '__constants__', ()))
    for (name, ann) in class_annotations.items():
        if torch._jit_internal.is_final(ann):
            constants_set.add(name)
    for name in constants_set:
        if name in added_names:
            if name in nn_module._modules:
                hint = 'submodule'
            elif name in nn_module._buffers:
                hint = 'buffer'
            elif name in nn_module._parameters:
                hint = 'parameter'
            else:
                raise AssertionError('added_names must be submodule, parameter, or buffer')
            warnings.warn(f"'{name}' was found in ScriptModule constants,  but it is a non-constant {hint}. Consider removing it.")
            continue
        if not hasattr(nn_module, name):
            warnings.warn(f"'{name}' was found in ScriptModule constants, but was not actually set in __init__. Consider removing it.")
            continue
        value = getattr(nn_module, name)
        concrete_type_builder.add_constant(name, _get_valid_constant(name, value, type(nn_module).__name__))
        added_names.add(name)
    overloads = getattr(nn_module, '__overloads__', {})
    overloads.update(get_overload_name_mapping(get_overload_annotations(nn_module, ignored_properties)))
    for (name, overloaded_names) in overloads.items():
        concrete_type_builder.add_overload(name, overloaded_names)
    for (name, value) in nn_module.__dict__.items():
        if name in ignored_attributes or name.startswith('__'):
            continue
        if name in user_annotated_ignored_attributes:
            continue
        if name in added_names:
            continue
        isoverloadpacket = isinstance(value, torch._ops.OpOverloadPacket)
        if isoverloadpacket:
            value = value.op
        if inspect.isfunction(value):
            try:
                scripted_fn = torch.jit.script(value)
                concrete_type_builder.add_function_attribute(name, torch._C._jit_try_infer_type(scripted_fn).type(), value)
            except Exception as e:
                hint = f'(This function exists as an attribute on the Python module, but we failed to compile it to a TorchScript function. \nThe error stack is reproduced here:\n{e}'
                concrete_type_builder.add_failed_attribute(name, hint)
                pass
            continue
        builtin_symbol_name = _find_builtin(value)
        if builtin_symbol_name:
            concrete_type_builder.add_builtin_function(name, builtin_symbol_name)
            continue
        if isinstance(value, torch.jit.ScriptFunction):
            concrete_type_builder.add_function_attribute(name, torch._C._jit_try_infer_type(value).type(), value)
            continue
        (attr_type, inferred) = infer_type(name, value)
        if attr_type.success():
            concrete_type_builder.add_attribute(name, attr_type.type(), False, False)
        else:
            inferred_msg = 'Its type was inferred; try adding a type annotation for the attribute.' if inferred else ''
            additional_info = f'{attr_type.reason()}. {inferred_msg}'
            hint = f"(This attribute exists on the Python module, but we failed to convert Python type: '{torch.typename(type(value))}' to a TorchScript type. {additional_info})"
            concrete_type_builder.add_failed_attribute(name, hint)
    for hook in nn_module._forward_hooks.values():
        concrete_type_builder.add_forward_hook(hook)
    for pre_hook in nn_module._forward_pre_hooks.values():
        concrete_type_builder.add_forward_pre_hook(pre_hook)
    return concrete_type_builder

class ConcreteTypeStore:
    type_store: Dict[Type[Module], List[torch._C.ConcreteModuleType]]
    methods_compiled: Set[torch._C.ConcreteModuleType]

    def __init__(self):
        if False:
            return 10
        self.type_store = {}
        self.methods_compiled = set()

    def get_or_create_concrete_type(self, nn_module):
        if False:
            print('Hello World!')
        'Infer a ConcreteType from this `nn.Module` instance. Underlying JIT types are re-used if possible.'
        concrete_type_builder = infer_concrete_type_builder(nn_module)
        nn_module_type = type(nn_module)
        if nn_module_type not in self.type_store:
            self.type_store[nn_module_type] = []
        known_types = self.type_store[nn_module_type]
        for known_type in known_types:
            if known_type.equals(concrete_type_builder):
                return known_type
        concrete_type = concrete_type_builder.build()
        self.type_store[nn_module_type].append(concrete_type)
        return concrete_type
concrete_type_store = ConcreteTypeStore()

def create_methods_and_properties_from_stubs(concrete_type, method_stubs, property_stubs):
    if False:
        for i in range(10):
            print('nop')
    method_defs = [m.def_ for m in method_stubs]
    method_rcbs = [m.resolution_callback for m in method_stubs]
    method_defaults = [get_default_args(m.original_method) for m in method_stubs]
    property_defs = [p.def_ for p in property_stubs]
    property_rcbs = [p.resolution_callback for p in property_stubs]
    concrete_type._create_methods_and_properties(property_defs, property_rcbs, method_defs, method_rcbs, method_defaults)

def create_hooks_from_stubs(concrete_type, hook_stubs, pre_hook_stubs):
    if False:
        while True:
            i = 10
    hook_defs = [h.def_ for h in hook_stubs]
    hook_rcbs = [h.resolution_callback for h in hook_stubs]
    pre_hook_defs = [h.def_ for h in pre_hook_stubs]
    pre_hook_rcbs = [h.resolution_callback for h in pre_hook_stubs]
    concrete_type._create_hooks(hook_defs, hook_rcbs, pre_hook_defs, pre_hook_rcbs)

def get_module_concrete_type(nn_module, share_types=True):
    if False:
        return 10
    '\n    Get a concrete type for nn_modules.\n\n    If share_types is True, the concrete type is fetched from concrete_type_store.\n    If it is False, a new concrete type is created without first searching concrete_type_store.\n\n    Args:\n        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.\n        share_types = Whether to share underlying JIT types between modules (if possible).\n\n    Returns:\n        A concrete type for nn_module.\n    '
    assert isinstance(nn_module, Module)
    if isinstance(nn_module, torch.jit.ScriptModule) and hasattr(nn_module, '_concrete_type'):
        return nn_module._concrete_type
    if share_types:
        concrete_type = concrete_type_store.get_or_create_concrete_type(nn_module)
    else:
        concrete_type_builder = infer_concrete_type_builder(nn_module, share_types)
        concrete_type_builder.set_poisoned()
        concrete_type = concrete_type_builder.build()
    return concrete_type

def create_script_class(obj):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create and return a RecursiveScriptClass instance from a Python object.\n\n    Arguments:\n        obj: A Python object.\n    '
    qualified_class_name = _jit_internal._qualified_name(type(obj))
    rcb = _jit_internal.createResolutionCallbackForClassMethods(type(obj))
    _compile_and_register_class(type(obj), rcb, qualified_class_name)
    class_ty = _python_cu.get_class(qualified_class_name)
    cpp_object = torch._C._create_object_with_type(class_ty)
    for (name, value) in obj.__dict__.items():
        cpp_object.setattr(name, value)
    return wrap_cpp_class(cpp_object)

def create_script_module(nn_module, stubs_fn, share_types=True, is_tracing=False):
    if False:
        return 10
    "\n    Create a new ScriptModule from an nn.Module.\n\n    Args:\n        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.\n        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.\n        share_types:  Whether to share underlying JIT types between modules (if possible).\n            NOTE: Only set to False this when we cannot guarantee type sharing will work\n                correctly. This only happens today for traced modules, where the same\n                module can produce different traced methods depending on the inputs.\n        is_tracing: Whether this function is called during tracing or scripting. If tracing,\n                we don't need to do AttributeTypeIsSupportedChecker because all the unsupported\n                attributes will be baked as constant in the tracing graph. In addition,\n                this check significantly slows down the traced modules when the module size is big.\n    "
    assert not isinstance(nn_module, torch.jit.RecursiveScriptModule)
    check_module_initialized(nn_module)
    concrete_type = get_module_concrete_type(nn_module, share_types)
    if not is_tracing:
        AttributeTypeIsSupportedChecker().check(nn_module)
    return create_script_module_impl(nn_module, concrete_type, stubs_fn)

def create_script_module_impl(nn_module, concrete_type, stubs_fn):
    if False:
        i = 10
        return i + 15
    '\n    Convert an nn.Module to a RecursiveScriptModule.\n\n    Args:\n        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.\n        concrete_type:  The fully initialized ConcreteType of the module.\n        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.\n    '
    cpp_module = torch._C._create_module_with_type(concrete_type.jit_type)
    method_stubs = stubs_fn(nn_module)
    property_stubs = get_property_stubs(nn_module)
    (hook_stubs, pre_hook_stubs) = get_hook_stubs(nn_module)
    user_annotated_ignored_attributes = getattr(nn_module, '__jit_ignored_attributes__', list())
    ignored_properties = jit_ignored_properties(nn_module)

    def init_fn(script_module):
        if False:
            return 10
        for name in concrete_type.get_attributes().keys():
            orig_value = getattr(nn_module, name)
            orig_value = orig_value.value if isinstance(orig_value, torch.jit.Attribute) else orig_value
            cpp_module.setattr(name, orig_value)
        for (name, sub_concrete_type) in concrete_type.get_modules():
            orig_value = getattr(nn_module, name)
            assert isinstance(orig_value, Module), f'Expected Module but got {type(orig_value)}'
            module_type = sub_concrete_type.jit_type
            if isinstance(module_type, torch._C.InterfaceType):
                scripted = interface_script(module_type, orig_value)
            elif isinstance(orig_value, torch.jit.ScriptModule):
                scripted = orig_value
            else:
                scripted = create_script_module_impl(orig_value, sub_concrete_type, stubs_fn)
            cpp_module.setattr(name, scripted)
            script_module._modules[name] = scripted
        for name in dir(nn_module):
            if name in ignored_properties:
                continue
            item = getattr(nn_module, name, None)
            if inspect.ismethod(item) and _jit_internal.is_ignored_fn(item):
                unbound_function = getattr(nn_module, name).__func__
                bound_method = unbound_function.__get__(script_module)
                setattr(script_module, name, bound_method)
            elif concrete_type.is_ignored_attribute(name):
                setattr(script_module, name, item)
        script_module._concrete_type = concrete_type
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
    if concrete_type not in concrete_type_store.methods_compiled:
        create_methods_and_properties_from_stubs(concrete_type, method_stubs, property_stubs)
        create_hooks_from_stubs(concrete_type, hook_stubs, pre_hook_stubs)
        torch._C._run_emit_module_hook(cpp_module)
        concrete_type_store.methods_compiled.add(concrete_type)
    for (idx, fn) in enumerate(script_module._c._get_forward_pre_hooks()):
        script_module._forward_pre_hooks[idx] = fn
    for (idx, fn) in enumerate(script_module._c._get_forward_hooks()):
        script_module._forward_hooks[idx] = fn
    if isinstance(nn_module, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)) and '__len__' not in cpp_module._method_names():
        script_module.define(f'def __len__(self):\n   return {len(nn_module)}\n')
    if isinstance(nn_module, torch.nn.ModuleDict) and '__contains__' not in cpp_module._method_names():
        if len(nn_module.keys()):
            keys = repr(list(nn_module.keys()))
            script_module.define(f'def __contains__(self, key: str):\n   return key in {keys}\n')
        else:
            script_module.define('def __contains__(self, key: str):\n   return False\n')
    for method_stub in method_stubs:
        if method_stub.original_method is None:
            continue
        name = method_stub.original_method.__name__
        if name != method_stub.def_.name().name:
            continue
        script_method = cpp_module._get_method(name)
        wrapped_script_method = functools.wraps(method_stub.original_method)(script_method)
        script_module.__dict__[name] = wrapped_script_method
    for property_stub in property_stubs:
        property_name = property_stub.def_.name().name
        fget = cpp_module._get_method(property_stub.def_.getter_name().name)
        setter_name = property_stub.def_.setter_name()
        fset = cpp_module._get_method(setter_name.name) if setter_name else None
        script_module.__dict__[property_name] = property(property_name, fget, fset)
    for name in dir(nn_module):
        if name in ignored_properties:
            continue
        item = getattr(nn_module, name, None)
        if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.COPY_TO_SCRIPT_WRAPPER:
            add_python_attr_to_scripted_model(script_module, nn_module, name)
    return script_module

def script_model_defines_attr(script_model, attr):
    if False:
        i = 10
        return i + 15
    script_attr = getattr(script_model, attr, None)
    if script_attr is None:
        return False
    default_attr = getattr(torch.jit.RecursiveScriptModule, attr, None)
    if default_attr is None:
        return False
    return script_attr != default_attr

def add_python_attr_to_scripted_model(script_model, orig, attr):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(orig, attr) and script_model_defines_attr(script_model, attr):
        setattr(script_model, attr, getattr(orig, attr))

def get_overload_annotations(mod, jit_ignored_properties):
    if False:
        for i in range(10):
            print('nop')
    overloads = {}
    for name in dir(type(mod)):
        if name in jit_ignored_properties:
            continue
        item = getattr(mod, name, None)
        if not callable(item):
            continue
        if hasattr(item, '__module__') and item.__module__ is not None:
            method_overloads = _jit_internal._get_overloaded_methods(item, mod.__class__)
            if method_overloads is None:
                continue
            if item.__func__ in method_overloads:
                raise RuntimeError(_jit_internal.get_overload_no_implementation_error_message('method', item.__func__))
            names = [name + '__' + str(i) for i in range(len(method_overloads))]
            overloads[item] = list(zip(names, method_overloads))
    return overloads

def get_overload_name_mapping(overload_info):
    if False:
        for i in range(10):
            print('nop')
    overload_name_mappings: Dict[str, List[str]] = {}
    for (orig_fn, overloads) in overload_info.items():
        original_name = orig_fn.__name__
        if original_name not in overload_name_mappings:
            overload_name_mappings[original_name] = []
        for (overload_name, _) in overloads:
            overload_name_mappings[original_name].append(overload_name)
    return overload_name_mappings

def _check_no_signature(func):
    if False:
        print('Hello World!')
    signature = torch.jit.annotations.get_signature(func, None, fake_range(), inspect.ismethod(func))
    if signature is None:
        qual_name = _jit_internal._qualified_name(func)
        raise RuntimeError(f'Must explicitly add type annotations to overloaded functions: {qual_name}')

def make_stubs_for_overloads(overload_info):
    if False:
        while True:
            i = 10
    overload_stubs = []
    for (orig_fn, overloads) in overload_info.items():
        orig_ast = get_jit_def(orig_fn, orig_fn.__name__, self_name='RecursiveScriptModule')
        for (overload_name, overload_fn) in overloads:
            _check_no_signature(overload_fn)
            over_ast = get_jit_def(overload_fn, overload_fn.__name__, self_name='RecursiveScriptModule')
            new_ast = torch._C._replace_overloaded_method_decl(over_ast.decl(), orig_ast, overload_name)
            _rcb = _jit_internal.createResolutionCallbackFromClosure(orig_fn)
            overload_stubs.append(ScriptMethodStub(_rcb, new_ast, overload_fn))
    return overload_stubs

def check_module_initialized(mod):
    if False:
        return 10
    assert isinstance(mod, torch.nn.Module)
    if not hasattr(mod, '_parameters'):
        raise RuntimeError(f"'{torch.typename(type(mod))}' has not been initialized, did you forget to call 'super()'?")
    if not hasattr(mod, 'remote_parameters'):
        for (name, param) in mod._parameters.items():
            if param is not None and torch.nn.parameter.is_lazy(param):
                raise RuntimeError("'{}' has uninitialized parameters {}. Did you forget to run a forward pass?".format(torch.typename(type(mod)), name))
        for (name, buf) in mod._buffers.items():
            if buf is not None and torch.nn.parameter.is_lazy(buf):
                raise RuntimeError("'{}' has uninitialized buffers {}. Did you forget to run a forward pass?".format(torch.typename(type(mod)), name))

def infer_methods_to_compile(nn_module):
    if False:
        for i in range(10):
            print('nop')
    'Implement the default rules for which methods should act as starting points for compilation.\n\n    (TODO add a link when the rules are published).\n    '
    check_module_initialized(nn_module)
    user_annotated_ignored_attributes = getattr(nn_module, '__jit_ignored_attributes__', list())
    ignored_properties = jit_ignored_properties(nn_module)
    methods: List[str] = []
    if hasattr(nn_module, 'forward') and (not _jit_internal.is_ignored_fn(nn_module.forward)):
        forward_func = getattr(nn_module.forward, '__func__', None)
        module_forward = getattr(torch.nn.Module, 'forward', None)
        if forward_func != module_forward:
            methods = ['forward']
    exported = []
    for name in dir(nn_module):
        if name in ignored_properties:
            continue
        item = getattr(nn_module, name, None)
        if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.EXPORT:
            exported.append(name)
    methods = methods + exported
    overload_name_mappings = dict(getattr(nn_module, '__overloads__', {}))
    overload_info = get_overload_annotations(nn_module, ignored_properties)
    overload_name_mappings.update(get_overload_name_mapping(overload_info))
    overload_stubs = make_stubs_for_overloads(overload_info)
    nn_module.__overloads__ = overload_name_mappings

    def ignore_overloaded(method_name):
        if False:
            while True:
                i = 10
        return method_name not in overload_name_mappings
    filtered_methods = filter(ignore_overloaded, methods)
    uniquer: Set[str] = set()
    uniqued_methods = []
    for name in filtered_methods:
        if name in uniquer:
            continue
        uniqued_methods.append(name)
        uniquer.add(name)
    stubs = []
    for method in uniqued_methods:
        stubs.append(make_stub_from_method(nn_module, method))
    return overload_stubs + stubs

def get_hook_stubs(nn_module):
    if False:
        i = 10
        return i + 15
    'Return forward hook and pre_hook ScriptModuleStubs.'
    check_module_initialized(nn_module)
    hook_map: Dict = {}
    hook_stubs = []
    for hook in nn_module._forward_hooks.values():
        if hook.__name__ in hook_map:
            if id(hook) != id(hook_map[hook.__name__]):
                raise RuntimeError(f"Hook '{hook.__name__}' on {type(nn_module).__name__} has at least two different python definitions. Please use unique names for all hooks.")
        else:
            hook_map[hook.__name__] = hook
        hook_stubs.append(make_stub(hook, hook.__name__))
    pre_hook_stubs = []
    for pre_hook in nn_module._forward_pre_hooks.values():
        if pre_hook.__name__ in hook_map:
            if id(pre_hook) != id(hook_map[pre_hook.__name__]):
                raise RuntimeError(f"Pre-hook '{pre_hook.__name__}' on {type(nn_module).__name__} has at least two different python definitions. Please use unique names for all hooks.")
        else:
            hook_map[pre_hook.__name__] = pre_hook
        pre_hook_stubs.append(make_stub(pre_hook, pre_hook.__name__))
    return (hook_stubs, pre_hook_stubs)

def get_property_stubs(nn_module):
    if False:
        i = 10
        return i + 15
    'Create property stubs for the properties of the module by creating method stubs for the getter and setter.'
    module_ty = type(nn_module)
    properties_asts = get_class_properties(module_ty, self_name='RecursiveScriptModule')
    rcbs = {}
    for name in dir(module_ty):
        item = getattr(module_ty, name, None)
        if isinstance(item, property):
            if not item.fget:
                raise RuntimeError(f'Property {name} of {nn_module.__name__} must have a getter')
            rcbs[name] = _jit_internal.createResolutionCallbackFromClosure(item.fget)
    stubs = [PropertyStub(rcbs[ast.name().name], ast) for ast in properties_asts]
    return stubs

def interface_script(mod_interface, nn_module):
    if False:
        i = 10
        return i + 15
    '\n    Make a ScriptModule from an nn.Module, using the interface methods rule for determining which methods to compile.\n\n    Args:\n        mod_interface: the interface type that the module have\n        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.\n    '
    if isinstance(nn_module, torch.jit.ScriptModule):
        return nn_module
    check_module_initialized(nn_module)

    def infer_interface_methods_to_compile(nn_module):
        if False:
            return 10
        'Rule to infer the methods from the interface type.\n\n        It is used to know which methods need to act as starting points for compilation.\n        '
        stubs = []
        for method in mod_interface.getMethodNames():
            stubs.append(make_stub_from_method(nn_module, method))
        return stubs
    return create_script_module(nn_module, infer_interface_methods_to_compile)

def try_compile_fn(fn, loc):
    if False:
        print('Hello World!')
    if _jit_internal.is_ignored_fn(fn):
        return None
    if isinstance(fn, torch.nn.Module):
        return None
    if not inspect.isfunction(fn) and (not inspect.ismethod(fn)):
        raise RuntimeError(f'`{fn}` is not a function. Recursive scripting only supports Python functions or methods currently.\nConsider manually annotating `{fn}` with @torch.jit.script.')
    rcb = _jit_internal.createResolutionCallbackFromClosure(fn)
    return torch.jit.script(fn, _rcb=rcb)

def wrap_cpp_class(cpp_class):
    if False:
        i = 10
        return i + 15
    'Wrap this torch._C.Object in a Python RecursiveScriptClass.'
    return torch.jit.RecursiveScriptClass(cpp_class)

def wrap_cpp_module(cpp_module):
    if False:
        return 10
    'Wrap this torch._C.ScriptModule in a Python ScriptModule, recursively for all submodules.'

    def init_fn(script_module):
        if False:
            i = 10
            return i + 15
        for (name, cpp_module) in torch._C.ModuleDict(script_module._c).items():
            setattr(script_module, name, wrap_cpp_module(cpp_module))
        script_module._concrete_type = torch._C.ConcreteModuleType.from_jit_type(script_module._c._type())
        for (idx, fn) in enumerate(script_module._c._get_forward_pre_hooks()):
            script_module._forward_pre_hooks[idx] = fn
        for (idx, fn) in enumerate(script_module._c._get_forward_hooks()):
            script_module._forward_hooks[idx] = fn
    return torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)

def compile_unbound_method(concrete_type, fn):
    if False:
        print('Hello World!')
    if _jit_internal.is_ignored_fn(fn):
        return None
    stub = make_stub(fn, fn.__name__)
    with torch._jit_internal._disable_emit_hooks():
        create_methods_and_properties_from_stubs(concrete_type, (stub,), ())
    return stub

def lazy_bind(concrete_type, unbound_method):
    if False:
        print('Hello World!')
    '\n    Return a function that lazily binds `unbound_method` to a provided Module IValue, then invokes the method.\n\n    We do this so that any Python shenanigans that\n    will poison type sharing are impossible at compile time.\n    '

    def lazy_binding_method(cpp_module, *args):
        if False:
            print('Hello World!')

        def init_fn(script_module):
            if False:
                while True:
                    i = 10
            orig_class = concrete_type.py_class
            for name in dir(orig_class):
                item = getattr(orig_class, name, None)
                if _jit_internal.is_ignored_fn(item):
                    setattr(script_module, name, item)
            for (name, value) in concrete_type.get_constants().items():
                setattr(script_module, name, value)
        script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
        method = types.MethodType(unbound_method, script_module)
        return method(*args)
    lazy_binding_method.original_fn = unbound_method
    lazy_binding_method.__name__ = unbound_method.__name__
    torch._jit_internal.copy_torchscript_modifier(unbound_method, lazy_binding_method)
    return lazy_binding_method