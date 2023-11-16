"""Utilities for abstract.py."""
import collections
import dataclasses
import logging
from typing import Any, Collection, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple, Union
from pytype import datatypes
from pytype.pyc import opcodes
from pytype.pyc import pyc
from pytype.pytd import pytd
from pytype.pytd import pytd_utils
from pytype.typegraph import cfg
from pytype.typegraph import cfg_utils
log = logging.getLogger(__name__)
_ArgsDictType = Dict[str, cfg.Variable]
_ContextType = Any
_BaseValueType = Any
_ParameterizedClassType = Any
_TypeParamType = Any
T = '_T'
T2 = '_T2'
K = '_K'
V = '_V'
ARGS = '_ARGS'
RET = '_RET'
DYNAMIC_ATTRIBUTE_MARKERS = ['HAS_DYNAMIC_ATTRIBUTES', '_HAS_DYNAMIC_ATTRIBUTES', 'has_dynamic_attributes']
TOP_LEVEL_IGNORE = frozenset({'__builtins__', '__doc__', '__file__', '__future__', '__module__', '__name__', '__annotations__'})
CLASS_LEVEL_IGNORE = frozenset({'__builtins__', '__class__', '__module__', '__name__', '__qualname__', '__slots__', '__annotations__'})

class DummyContainer:

    def __init__(self, container):
        if False:
            print('Hello World!')
        self.container = container
DUMMY_CONTAINER = DummyContainer(None)

class ConversionError(ValueError):
    pass

class EvaluationError(Exception):
    """Used to signal an errorlog error during type name evaluation."""

    @property
    def errors(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args

    @property
    def details(self):
        if False:
            i = 10
            return i + 15
        return '\n'.join((error.message for error in self.errors))

class GenericTypeError(Exception):
    """The error for user-defined generic types."""

    def __init__(self, annot, error):
        if False:
            i = 10
            return i + 15
        super().__init__(annot, error)
        self.annot = annot
        self.error = error

class ModuleLoadError(Exception):
    """Signal an error when trying to lazily load a submodule."""

class AsInstance:
    """Wrapper, used for marking things that we want to convert to an instance."""

    def __init__(self, cls):
        if False:
            print('Hello World!')
        self.cls = cls

class AsReturnValue(AsInstance):
    """Specially mark return values, to handle Never properly."""

@dataclasses.dataclass(eq=True, frozen=True)
class LazyFormalTypeParameters:
    template: Sequence[Any]
    parameters: Sequence[pytd.Node]
    subst: Dict[str, cfg.Variable]

class Local:
    """A possibly annotated local variable."""

    def __init__(self, node: cfg.CFGNode, op: Optional[opcodes.Opcode], typ: Optional[_BaseValueType], orig: Optional[cfg.Variable], ctx: _ContextType):
        if False:
            while True:
                i = 10
        self._ops = [op]
        self.final = False
        if typ:
            self.typ = ctx.program.NewVariable([typ], [], node)
        else:
            self.typ = None
        self.orig = orig
        self.ctx = ctx

    @classmethod
    def merge(cls, node, op, local1, local2):
        if False:
            i = 10
            return i + 15
        'Merges two locals.'
        ctx = local1.ctx
        typ_values = set()
        for typ in [local1.typ, local2.typ]:
            if typ:
                typ_values.update(typ.Data(node))
        typ = ctx.convert.merge_values(typ_values) if typ_values else None
        if local1.orig and local2.orig:
            orig = ctx.program.NewVariable()
            orig.PasteVariable(local1.orig, node)
            orig.PasteVariable(local2.orig, node)
        else:
            orig = local1.orig or local2.orig
        return cls(node, op, typ, orig, ctx)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'Local(typ={self.typ}, orig={self.orig}, final={self.final})'

    @property
    def stack(self):
        if False:
            i = 10
            return i + 15
        return self.ctx.vm.simple_stack(self._ops[-1])

    @property
    def last_update_op(self):
        if False:
            print('Hello World!')
        return self._ops[-1]

    def update(self, node, op, typ, orig, final=False):
        if False:
            return 10
        "Update this variable's annotation and/or value."
        if op in self._ops:
            return
        self._ops.append(op)
        self.final = final
        if typ:
            if self.typ:
                self.typ.AddBinding(typ, [], node)
            else:
                self.typ = self.ctx.program.NewVariable([typ], [], node)
        if orig:
            self.orig = orig

    def get_type(self, node, name):
        if False:
            print('Hello World!')
        "Gets the variable's annotation."
        if not self.typ:
            return None
        values = self.typ.Data(node)
        if len(values) > 1:
            self.ctx.errorlog.ambiguous_annotation(self.stack, values, name)
            return self.ctx.convert.unsolvable
        elif values:
            return values[0]
        else:
            return None

@dataclasses.dataclass(eq=True, frozen=True)
class BadType:
    name: Optional[str]
    typ: _BaseValueType
    error_details: Optional[Any] = None
_ISINSTANCE_CACHE = {}

def _isinstance(obj, name_or_names):
    if False:
        while True:
            i = 10
    'Do an isinstance() call for a class defined in pytype.abstract.\n\n  Args:\n    obj: An instance.\n    name_or_names: A name or tuple of names of classes in pytype.abstract.\n\n  Returns:\n    Whether obj is an instance of name_or_names.\n  '
    if not _ISINSTANCE_CACHE:
        from pytype.abstract import abstract
        for attr in dir(abstract):
            if attr[0].isupper():
                _ISINSTANCE_CACHE[attr] = getattr(abstract, attr)
    if name_or_names.__class__ == tuple:
        class_or_classes = tuple((_ISINSTANCE_CACHE[name] for name in name_or_names))
    else:
        class_or_classes = _ISINSTANCE_CACHE[name_or_names]
    return isinstance(obj, class_or_classes)

def _make(cls_name, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Make an instance of cls_name with the given arguments.'
    from pytype.abstract import abstract
    return getattr(abstract, cls_name)(*args, **kwargs)

class _None:
    pass

def get_atomic_value(variable, constant_type=None, default=_None()):
    if False:
        print('Hello World!')
    'Get the atomic value stored in this variable.'
    if len(variable.bindings) == 1:
        (v,) = variable.bindings
        if isinstance(v.data, constant_type or object):
            return v.data
    if not isinstance(default, _None):
        return default
    if not variable.bindings:
        raise ConversionError('Cannot get atomic value from empty variable.')
    bindings = variable.bindings
    name = bindings[0].data.ctx.convert.constant_name(constant_type)
    raise ConversionError('Cannot get atomic value %s from variable. %s %s' % (name, variable, [b.data for b in bindings]))

def match_atomic_value(variable, typ=None):
    if False:
        for i in range(10):
            print('nop')
    try:
        get_atomic_value(variable, typ)
    except ConversionError:
        return False
    return True

def get_atomic_python_constant(variable, constant_type=None):
    if False:
        while True:
            i = 10
    "Get the concrete atomic Python value stored in this variable.\n\n  This is used for things that are stored in cfg.Variable, but we\n  need the actual data in order to proceed. E.g. function / class definitions.\n\n  Args:\n    variable: A cfg.Variable. It can only have one possible value.\n    constant_type: Optionally, the required type of the constant.\n  Returns:\n    A Python constant. (Typically, a string, a tuple, or a code object.)\n  Raises:\n    ConversionError: If the value in this Variable is purely abstract, i.e.\n      doesn't store a Python value, or if it has more than one possible value.\n  "
    atomic = get_atomic_value(variable)
    return atomic.ctx.convert.value_to_constant(atomic, constant_type)

def match_atomic_python_constant(variable, typ=None):
    if False:
        return 10
    try:
        get_atomic_python_constant(variable, typ)
    except ConversionError:
        return False
    return True

def get_views(variables, node):
    if False:
        i = 10
        return i + 15
    'Get all possible views of the given variables at a particular node.\n\n  For performance reasons, this method uses node.CanHaveCombination for\n  filtering. For a more precise check, you can call\n  node.HasCombination(list(view.values())). Do so judiciously, as the latter\n  method can be very slow.\n\n  This function can be used either as a regular generator or in an optimized way\n  to yield only functionally unique views:\n    views = get_views(...)\n    skip_future = None\n    while True:\n      try:\n        view = views.send(skip_future)\n      except StopIteration:\n        break\n      ...\n    The caller should set `skip_future` to True when it is safe to skip\n    equivalent future views and False otherwise.\n\n  Args:\n    variables: The variables.\n    node: The node.\n\n  Yields:\n    A datatypes.AcessTrackingDict mapping variables to bindings.\n  '
    try:
        combinations = cfg_utils.deep_variable_product(variables)
    except cfg_utils.TooComplexError:
        log.info('get_views: too many binding combinations to generate accurate views, falling back to unsolvable')
        combinations = ((var.AddBinding(node.program.default_data, [], node) for var in variables),)
    seen = []
    for combination in combinations:
        view = {value.variable: value for value in combination}
        if any((subset <= view.items() for subset in seen)):
            log.info('Skipping view (already seen): %r', view)
            continue
        combination = list(view.values())
        if not node.CanHaveCombination(combination):
            log.info('Skipping combination (unreachable): %r', combination)
            continue
        view = datatypes.AccessTrackingDict(view)
        skip_future = (yield view)
        if skip_future:
            seen.append(view.accessed_subset.items())

def equivalent_to(binding, cls):
    if False:
        while True:
            i = 10
    'Whether binding.data is equivalent to cls, modulo parameterization.'
    return _isinstance(binding.data, 'Class') and binding.data.full_name == cls.full_name

def is_subclass(value, cls):
    if False:
        return 10
    'Whether value is a subclass of cls, modulo parameterization.'
    if _isinstance(value, 'Union'):
        return any((is_subclass(v, cls) for v in value.options))
    return _isinstance(value, 'Class') and any((value_cls.full_name == cls.full_name for value_cls in value.mro))

def apply_mutations(node, get_mutations):
    if False:
        for i in range(10):
            print('nop')
    'Apply mutations yielded from a get_mutations function.'
    log.info('Applying mutations')
    num_mutations = 0
    for mut in get_mutations():
        if not num_mutations:
            node = node.ConnectNew('ApplyMutations')
        num_mutations += 1
        mut.instance.merge_instance_type_parameter(node, mut.name, mut.value)
    log.info('Applied %d mutations', num_mutations)
    return node

def get_mro_bases(bases):
    if False:
        while True:
            i = 10
    'Get bases for MRO computation.'
    mro_bases = []
    has_user_generic = False
    for base_var in bases:
        if not base_var.data:
            continue
        base = base_var.data[0]
        mro_bases.append(base)
        if _isinstance(base, 'ParameterizedClass') and base.full_name != 'typing.Generic':
            has_user_generic = True
    if has_user_generic:
        return [b for b in mro_bases if b.full_name != 'typing.Generic']
    else:
        return mro_bases

def _merge_type(t0, t1, name, cls):
    if False:
        while True:
            i = 10
    "Merge two types.\n\n  Rules: Type `Any` can match any type, we will return the other type if one\n  of them is `Any`. Return the sub-class if the types have inheritance\n  relationship.\n\n  Args:\n    t0: The first type.\n    t1: The second type.\n    name: Type parameter name.\n    cls: The class_mixin.Class on which any error should be reported.\n  Returns:\n    A type.\n  Raises:\n    GenericTypeError: if the types don't match.\n  "
    if t0 is None or _isinstance(t0, 'Unsolvable'):
        return t1
    if t1 is None or _isinstance(t1, 'Unsolvable'):
        return t0
    if t0 in t1.mro:
        return t1
    if t1 in t0.mro:
        return t0
    raise GenericTypeError(cls, f'Conflicting value for TypeVar {name}')

def parse_formal_type_parameters(base, prefix, formal_type_parameters, container=None):
    if False:
        return 10
    "Parse type parameters from base class.\n\n  Args:\n    base: base class.\n    prefix: the full name of subclass of base class.\n    formal_type_parameters: the mapping of type parameter name to its type.\n    container: An abstract value whose class template is used when prefix=None\n      to decide how to handle type parameters that are aliased to other type\n      parameters. Values that are in the class template are kept, while all\n      others are ignored.\n\n  Raises:\n    GenericTypeError: If the lazy types of type parameter don't match\n  "

    def merge(t0, t1, name):
        if False:
            for i in range(10):
                print('nop')
        return _merge_type(t0, t1, name, base)
    if _isinstance(base, 'ParameterizedClass'):
        if base.full_name == 'typing.Generic':
            return
        if _isinstance(base.base_cls, ('InterpreterClass', 'PyTDClass')):
            formal_type_parameters.merge_from(base.base_cls.all_formal_type_parameters, merge)
        params = base.get_formal_type_parameters()
        if hasattr(container, 'cls'):
            container_template = container.cls.template
        else:
            container_template = ()
        for (name, param) in params.items():
            if _isinstance(param, 'TypeParameter'):
                if prefix:
                    formal_type_parameters.add_alias(name, prefix + '.' + param.name, merge)
                elif param in container_template:
                    formal_type_parameters[name] = param
            elif name not in formal_type_parameters:
                formal_type_parameters[name] = param
            else:
                last_type = formal_type_parameters[name]
                formal_type_parameters[name] = merge(last_type, param, name)
    else:
        if _isinstance(base, ('InterpreterClass', 'PyTDClass')):
            formal_type_parameters.merge_from(base.all_formal_type_parameters, merge)
        if base.template:
            for item in base.template:
                if _isinstance(item, 'TypeParameter'):
                    name = full_type_name(base, item.name)
                    if name not in formal_type_parameters:
                        formal_type_parameters[name] = None

def full_type_name(val, name):
    if False:
        i = 10
        return i + 15
    'Compute complete type parameter name with scope.\n\n  Args:\n    val: The object with type parameters.\n    name: The short type parameter name (e.g., T).\n\n  Returns:\n    The full type parameter name (e.g., List.T).\n  '
    if _isinstance(val, 'Instance'):
        return full_type_name(val.cls, name)
    for t in val.template:
        if name in (t.name, t.full_name):
            return t.full_name
    for t in val.all_template_names:
        if t.split('.')[-1] == name or t == name:
            return t
    return name

def maybe_extract_tuple(t):
    if False:
        while True:
            i = 10
    'Returns a tuple of Variables.'
    values = t.data
    if len(values) > 1:
        return (t,)
    (v,) = values
    if not _isinstance(v, 'Tuple'):
        return (t,)
    return v.pyval

def eval_expr(ctx, node, f_globals, f_locals, expr):
    if False:
        while True:
            i = 10
    'Evaluate an expression with the given node and globals.'
    log.info('Evaluating expr: %r', expr)
    with ctx.errorlog.checkpoint() as record:
        try:
            code = ctx.vm.compile_src(expr, mode='eval')
        except pyc.CompileError as e:
            ctx.errorlog.python_compiler_error(None, 0, e.error)
            ret = ctx.new_unsolvable(node)
        else:
            (_, _, _, ret) = ctx.vm.run_bytecode(node, code, f_globals, f_locals)
    log.info('Finished evaluating expr: %r', expr)
    if record.errors:
        e = EvaluationError(*(error.drop_traceback() for error in record.errors))
    else:
        e = None
    return (ret, e)

def match_type_container(typ, container_type_name: Union[str, Tuple[str, ...]]):
    if False:
        return 10
    'Unpack the type parameter from ContainerType[T].'
    if typ is None:
        return None
    if isinstance(container_type_name, str):
        container_type_name = (container_type_name,)
    if not (_isinstance(typ, 'ParameterizedClass') and typ.full_name in container_type_name):
        return None
    param = typ.get_formal_type_parameter(T)
    return param

def get_annotations_dict(members):
    if False:
        print('Hello World!')
    "Get __annotations__ from a members map.\n\n  Returns None rather than {} if the dict does not exist so that callers always\n  have a reference to the actual dictionary, and can mutate it if needed.\n\n  Args:\n    members: A dict of member name to variable\n\n  Returns:\n    members['__annotations__'] unpacked as a python dict, or None\n  "
    if '__annotations__' not in members:
        return None
    annots_var = members['__annotations__']
    try:
        annots = get_atomic_value(annots_var)
    except ConversionError:
        return None
    return annots if _isinstance(annots, 'AnnotationsDict') else None

def is_concrete_dict(val: _BaseValueType) -> bool:
    if False:
        while True:
            i = 10
    return val.is_concrete and _isinstance(val, 'Dict')

def is_concrete_list(val: _BaseValueType) -> bool:
    if False:
        print('Hello World!')
    return val.is_concrete and _isinstance(val, 'List')

def is_indefinite_iterable(val: _BaseValueType) -> bool:
    if False:
        i = 10
        return i + 15
    'True if val is a non-concrete instance of typing.Iterable.'
    instance = _isinstance(val, 'Instance')
    cls_instance = _isinstance(val.cls, 'Class')
    if not (instance and cls_instance and (not val.is_concrete)):
        return False
    for cls in val.cls.mro:
        if cls.full_name == 'builtins.str':
            return False
        elif cls.full_name == 'builtins.tuple':
            return _isinstance(cls, 'PyTDClass')
        elif cls.full_name == 'typing.Iterable':
            return True
    return False

def is_var_indefinite_iterable(var):
    if False:
        while True:
            i = 10
    'True if all bindings of var are indefinite sequences.'
    return all((is_indefinite_iterable(x) for x in var.data))

def is_dataclass(val: _BaseValueType) -> bool:
    if False:
        print('Hello World!')
    return _isinstance(val, 'Class') and '__dataclass_fields__' in val.metadata

def is_attrs(val: _BaseValueType) -> bool:
    if False:
        print('Hello World!')
    return _isinstance(val, 'Class') and '__attrs_attrs__' in val.metadata

def merged_type_parameter(node, var, param):
    if False:
        return 10
    if not var.bindings:
        return node.program.NewVariable()
    if is_var_splat(var):
        var = unwrap_splat(var)
    params = [v.get_instance_type_parameter(param) for v in var.data]
    return var.data[0].ctx.join_variables(node, params)

def is_var_splat(var):
    if False:
        for i in range(10):
            print('nop')
    if var.data and _isinstance(var.data[0], 'Splat'):
        assert len(var.bindings) == 1
        return True
    return False

def unwrap_splat(var):
    if False:
        print('Hello World!')
    return var.data[0].iterable

def is_callable(value: _BaseValueType) -> bool:
    if False:
        while True:
            i = 10
    "Returns whether 'value' is a callable."
    if _isinstance(value, ('Function', 'BoundFunction', 'ClassMethod', 'StaticMethod')):
        return True
    if not _isinstance(value.cls, 'Class'):
        return False
    (_, attr) = value.ctx.attribute_handler.get_attribute(value.ctx.root_node, value.cls, '__call__')
    return attr is not None

def expand_type_parameter_instances(bindings: Iterable[cfg.Binding]):
    if False:
        print('Hello World!')
    'Expands any TypeParameterInstance values in `bindings`.'
    bindings = list(bindings)
    seen = set()
    while bindings:
        b = bindings.pop(0)
        if _isinstance(b.data, 'TypeParameterInstance'):
            if b.data in seen:
                continue
            seen.add(b.data)
            param_value = b.data.instance.get_instance_type_parameter(b.data.name)
            if param_value.bindings:
                bindings = param_value.bindings + bindings
                continue
        yield b

def get_type_parameter_substitutions(val: _BaseValueType, type_params: Iterable[_TypeParamType]) -> Mapping[str, cfg.Variable]:
    if False:
        for i in range(10):
            print('nop')
    "Get values for type_params from val's type parameters."
    subst = {}
    for p in type_params:
        if _isinstance(val, 'Class'):
            param_value = val.get_formal_type_parameter(p.name).instantiate(val.ctx.root_node)
        else:
            param_value = val.get_instance_type_parameter(p.name)
        subst[p.full_name] = param_value
    return subst

def build_generic_template(type_params: Sequence[_BaseValueType], base_type: _BaseValueType) -> Tuple[Sequence[str], Sequence[_TypeParamType]]:
    if False:
        print('Hello World!')
    'Build a typing.Generic template from a sequence of type parameters.'
    if not all((_isinstance(item, 'TypeParameter') for item in type_params)):
        base_type.ctx.errorlog.invalid_annotation(base_type.ctx.vm.frames, base_type, 'Parameters to Generic[...] must all be type variables')
        type_params = [item for item in type_params if _isinstance(item, 'TypeParameter')]
    template = [item.name for item in type_params]
    if len(set(template)) != len(template):
        base_type.ctx.errorlog.invalid_annotation(base_type.ctx.vm.frames, base_type, 'Parameters to Generic[...] must all be unique')
    return (template, type_params)

def is_generic_protocol(val: _BaseValueType) -> bool:
    if False:
        print('Hello World!')
    return _isinstance(val, 'ParameterizedClass') and val.full_name == 'typing.Protocol'

def combine_substs(substs1: Optional[Collection[Dict[str, cfg.Variable]]], substs2: Optional[Collection[Dict[str, cfg.Variable]]]) -> Collection[Dict[str, cfg.Variable]]:
    if False:
        return 10
    'Combines the two collections of type parameter substitutions.'
    if substs1 and substs2:
        return tuple(({**sub1, **sub2} for sub1 in substs1 for sub2 in substs2))
    elif substs1:
        return substs1
    elif substs2:
        return substs2
    else:
        return ()

def flatten(value, classes):
    if False:
        while True:
            i = 10
    'Flatten the contents of value into classes.\n\n  If value is a Class, it is appended to classes.\n  If value is a PythonConstant of type tuple, then each element of the tuple\n  that has a single binding is also flattened.\n  Any other type of value, or tuple elements that have multiple bindings are\n  ignored.\n\n  Args:\n    value: An abstract value.\n    classes: A list to be modified.\n\n  Returns:\n    True iff a value was ignored during flattening.\n  '
    if _isinstance(value, 'AnnotationClass'):
        value = value.base_cls
    if _isinstance(value, 'Class'):
        classes.append(value)
        return False
    elif _isinstance(value, 'Tuple'):
        ambiguous = False
        for var in value.pyval:
            if len(var.bindings) != 1 or flatten(var.bindings[0].data, classes):
                ambiguous = True
        return ambiguous
    elif _isinstance(value, 'Union'):
        ambiguous = False
        for val in value.options:
            if flatten(val, classes):
                ambiguous = True
        return ambiguous
    else:
        return True

def check_against_mro(ctx, target, class_spec):
    if False:
        return 10
    "Check if any of the classes are in the target's MRO.\n\n  Args:\n    ctx: The abstract context.\n    target: A BaseValue whose MRO will be checked.\n    class_spec: A Class or PythonConstant tuple of classes (i.e. the second\n      argument to isinstance or issubclass).\n\n  Returns:\n    True if any class in classes is found in the target's MRO,\n    False if no match is found and None if it's ambiguous.\n  "
    classes = []
    ambiguous = flatten(class_spec, classes)
    for c in classes:
        if ctx.matcher(None).match_from_mro(target, c, allow_compat_builtins=False):
            return True
    return None if ambiguous else False

def maybe_unwrap_decorated_function(func):
    if False:
        while True:
            i = 10
    try:
        func.func.data
    except AttributeError:
        return None
    return func.func

def unwrap_final(val):
    if False:
        for i in range(10):
            print('nop')
    'Unwrap Final[T] -> T.'
    if _isinstance(val, 'FinalAnnotation'):
        return val.annotation
    elif _isinstance(val, 'Instance') and val.cls.full_name == 'typing.Final':
        return get_atomic_value(val.get_instance_type_parameter(T))
    return val

def is_recursive_annotation(annot):
    if False:
        while True:
            i = 10
    return annot.is_late_annotation() and annot.is_recursive()

def is_ellipsis(val):
    if False:
        while True:
            i = 10
    return val == val.ctx.convert.ellipsis or (val.is_concrete and val.pyval == '...')

def update_args_dict(args: _ArgsDictType, update: _ArgsDictType, node: cfg.CFGNode) -> None:
    if False:
        return 10
    'Update a {str: Variable} dict by merging bindings.'
    for (k, v) in update.items():
        if k in args:
            args[k].PasteVariable(v, node)
        else:
            args[k] = v

def show_constant(val: _BaseValueType) -> str:
    if False:
        while True:
            i = 10
    'Pretty-print a value if it is a constant.\n\n  Recurses into a constant, printing the underlying Python value for constants\n  and just using "..." for everything else (e.g., Variables). This is useful for\n  generating clear error messages that show the exact values related to an error\n  while preventing implementation details from leaking into the message.\n\n  Args:\n    val: an abstract value.\n\n  Returns:\n    A string of the pretty-printed constant.\n  '

    def _ellipsis_printer(v):
        if False:
            return 10
        if _isinstance(v, 'PythonConstant'):
            return v.str_of_constant(_ellipsis_printer)
        return '...'
    return _ellipsis_printer(val)

def get_generic_type(val: _BaseValueType) -> Optional[_ParameterizedClassType]:
    if False:
        i = 10
        return i + 15
    'Gets the generic type of an abstract value.\n\n  Args:\n    val: The abstract value.\n\n  Returns:\n    The type of the value, with concrete type parameters replaced by TypeVars.\n    For example, the generic type of `[0]` is `List[T]`.\n  '
    is_class = _isinstance(val, 'Class')
    if is_class:
        cls = val
    elif _isinstance(val.cls, 'Class'):
        cls = val.cls
    else:
        return None
    for parent_cls in cls.mro:
        if _isinstance(parent_cls, 'ParameterizedClass'):
            base_cls = parent_cls.base_cls
        else:
            base_cls = parent_cls
        if _isinstance(base_cls, 'Class') and base_cls.template:
            ctx = base_cls.ctx
            params = {item.name: item for item in base_cls.template}
            generic_cls = _make('ParameterizedClass', base_cls, params, ctx)
            if is_class:
                return _make('ParameterizedClass', ctx.convert.type_type, {T: generic_cls}, ctx)
            else:
                return generic_cls
    return None

def with_empty_substitutions(subst, pytd_type, node, ctx):
    if False:
        for i in range(10):
            print('nop')
    new_subst = {t.full_name: ctx.convert.empty.to_variable(node) for t in pytd_utils.GetTypeParameters(pytd_type) if t.full_name not in subst}
    return subst.copy(**new_subst)

def get_var_fullhash_component(var: cfg.Variable, seen: Optional[Set[_BaseValueType]]=None) -> Tuple[Any, ...]:
    if False:
        i = 10
        return i + 15
    return tuple(sorted((v.get_fullhash(seen) for v in var.data)))

def get_dict_fullhash_component(vardict: Dict[str, cfg.Variable], *, names: Optional[Set[str]]=None, seen: Optional[Set[_BaseValueType]]=None) -> Tuple[Any, ...]:
    if False:
        for i in range(10):
            print('nop')
    'Hash a dictionary.\n\n  This contains the keys and the full hashes of the data in the values.\n\n  Arguments:\n    vardict: A dictionary mapping str to Variable.\n    names: If this is non-None, the snapshot will include only those\n      dictionary entries whose keys appear in names.\n    seen: Optionally, a set of seen values for recursion detection.\n\n  Returns:\n    A hashable tuple of the dictionary.\n  '
    if names is not None:
        vardict = {name: vardict[name] for name in names.intersection(vardict)}
    return tuple(sorted(((k, get_var_fullhash_component(v, seen)) for (k, v) in vardict.items())))

def simplify_variable(var, node, ctx):
    if False:
        for i in range(10):
            print('nop')
    'Deduplicates identical data in `var`.'
    if not var:
        return var
    bindings_by_hash = collections.defaultdict(list)
    for b in var.bindings:
        bindings_by_hash[b.data.get_fullhash()].append(b)
    if len(bindings_by_hash) == len(var.bindings):
        return var
    new_var = ctx.program.NewVariable()
    for bindings in bindings_by_hash.values():
        new_var.AddBinding(bindings[0].data, bindings, node)
    return new_var

def _abstractify_value(val, ctx, seen=None):
    if False:
        print('Hello World!')
    'Converts a maybe-abstract value to a concrete one.\n\n  Args:\n    val: A value.\n    ctx: The context.\n    seen: Optionally, a seen values set.\n\n  Unlike ctx.convert.get_maybe_abstract_instance, this method recursively\n  descends into lists and tuples.\n\n  Returns:\n    A concrete value.\n  '
    if seen is None:
        seen = set()
    if not val.is_concrete or val in seen:
        return val
    seen = seen | {val}
    if not isinstance(val.pyval, (list, tuple)):
        return ctx.convert.get_maybe_abstract_instance(val)
    new_content = []
    for elem in val.pyval:
        new_elem_data = [_abstractify_value(v, ctx, seen) for v in elem.data]
        if any((v != new_v for (v, new_v) in zip(elem.data, new_elem_data))):
            new_elem = ctx.program.NewVariable()
            for (b, new_data) in zip(elem.bindings, new_elem_data):
                new_elem.PasteBindingWithNewData(b, new_data)
            new_content.append(new_elem)
        else:
            new_content.append(elem)
    if any((elem != new_elem for (elem, new_elem) in zip(val.pyval, new_content))):
        return type(val)(type(val.pyval)(new_content), ctx)
    else:
        return val

def abstractify_variable(var, ctx):
    if False:
        while True:
            i = 10
    if not any((v.is_concrete for v in var.data)):
        return var
    new_var = ctx.program.NewVariable()
    for b in var.bindings:
        new_var.PasteBindingWithNewData(b, _abstractify_value(b.data, ctx))
    return new_var