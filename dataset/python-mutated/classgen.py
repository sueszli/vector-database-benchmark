"""Base support for generating classes from data declarations.

Contains common functionality used by dataclasses, attrs and namedtuples.
"""
import abc
import collections
import dataclasses
import logging
from typing import Any, ClassVar, Dict, List
from pytype.abstract import abstract
from pytype.abstract import abstract_utils
from pytype.abstract import class_mixin
from pytype.overlays import overlay_utils
from pytype.overlays import special_builtins
log = logging.getLogger(__name__)
Param = overlay_utils.Param
Attribute = class_mixin.Attribute
AttributeKinds = class_mixin.AttributeKinds

class Ordering:
    """Possible orderings for get_class_locals."""
    FIRST_ANNOTATE = object()
    LAST_ASSIGN = object()

class Decorator(abstract.PyTDFunction, metaclass=abc.ABCMeta):
    """Base class for decorators that generate classes from data declarations."""
    DEFAULT_ARGS: ClassVar[Dict[str, Any]] = {'init': True, 'kw_only': False, 'auto_attribs': False}

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self._current_args = None
        self.partial_args = {}
        self.args = {}

    @abc.abstractmethod
    def decorate(self, node, cls):
        if False:
            i = 10
            return i + 15
        'Apply the decorator to cls.'

    def get_initial_args(self):
        if False:
            for i in range(10):
                print('nop')
        ret = self.DEFAULT_ARGS.copy()
        ret.update(self.partial_args)
        return ret

    def update_kwargs(self, args):
        if False:
            while True:
                i = 10
        'Update current_args with the Args passed to the decorator.'
        self._current_args = self.get_initial_args()
        for (k, v) in args.namedargs.items():
            if k in self._current_args:
                try:
                    self._current_args[k] = abstract_utils.get_atomic_python_constant(v)
                except abstract_utils.ConversionError:
                    self.ctx.errorlog.not_supported_yet(self.ctx.vm.frames, f'Non-constant argument to decorator: {k!r}')

    def set_current_args(self, kwargs):
        if False:
            while True:
                i = 10
        'Set current_args when constructing a class directly.'
        self._current_args = self.get_initial_args()
        self._current_args.update(kwargs)

    def init_name(self, attr):
        if False:
            return 10
        'Attribute name as an __init__ keyword, could differ from attr.name.'
        return attr.name

    def make_init(self, node, cls, attrs, init_method_name='__init__'):
        if False:
            return 10
        pos_params = []
        kwonly_params = []
        all_kwonly = self.args[cls]['kw_only']
        for attr in attrs:
            if not attr.init:
                continue
            typ = attr.init_type or attr.typ
            param = Param(name=self.init_name(attr), typ=typ, default=attr.default)
            if all_kwonly or attr.kw_only:
                kwonly_params.append(param)
            else:
                pos_params.append(param)
        return overlay_utils.make_method(self.ctx, node, init_method_name, pos_params, 0, kwonly_params)

    def call(self, node, func, args, alias_map=None):
        if False:
            for i in range(10):
                print('nop')
        'Construct a decorator, and call it on the class.'
        args = args.simplify(node, self.ctx)
        self.match_args(node, args)
        if not self._current_args:
            self.update_kwargs(args)
        if not args.posargs:
            return (node, self.to_variable(node))
        cls_var = args.posargs[0]
        (cls,) = cls_var.data
        if not isinstance(cls, abstract.Class):
            return (node, cls_var)
        self.args[cls] = self._current_args
        self._current_args = None
        self.decorate(node, cls)
        return (node, cls_var)

class FieldConstructor(abstract.PyTDFunction):
    """Implements constructors for fields."""

    def get_kwarg(self, args, name, default):
        if False:
            i = 10
            return i + 15
        if name not in args.namedargs:
            return default
        try:
            return abstract_utils.get_atomic_python_constant(args.namedargs[name])
        except abstract_utils.ConversionError:
            self.ctx.errorlog.not_supported_yet(self.ctx.vm.frames, f'Non-constant argument {name!r}')

    def get_positional_names(self):
        if False:
            print('Hello World!')
        return []

def is_method(var):
    if False:
        print('Hello World!')
    if var is None:
        return False
    return isinstance(var.data[0], (abstract.INTERPRETER_FUNCTION_TYPES, special_builtins.ClassMethodInstance, special_builtins.PropertyInstance, special_builtins.StaticMethodInstance))

def is_dunder(name):
    if False:
        return 10
    return name.startswith('__') and name.endswith('__')

def add_member(node, cls, name, typ):
    if False:
        while True:
            i = 10
    if typ.formal:
        instance = typ.ctx.convert.empty.to_variable(node)
    else:
        instance = typ.ctx.vm.init_class(node, typ, extra_key=name)
    cls.members[name] = instance

def is_relevant_class_local(class_local: abstract_utils.Local, class_local_name: str, allow_methods: bool):
    if False:
        print('Hello World!')
    "Tests whether the current class local could be relevant for type checking.\n\n  For example, this doesn't match __dunder__ class locals.\n\n  To get an abstract_utils.Local from a vm.LocalOps, you can use,\n  'vm_instance.annotated_locals[cls_name][op.name]'.\n\n  Args:\n    class_local: the local to query\n    class_local_name: the name of the class local (because abstract_utils.Local\n      does not hold this information).\n    allow_methods: whether to allow methods class locals to match\n\n  Returns:\n    Whether this class local could possibly be relevant for type checking.\n      Callers will usually want to filter even further.\n  "
    if is_dunder(class_local_name):
        return False
    if not allow_methods and (not class_local.typ) and is_method(class_local.orig):
        return False
    return True

def get_class_locals(cls_name: str, allow_methods: bool, ordering, ctx):
    if False:
        while True:
            i = 10
    "Gets a dictionary of the class's local variables.\n\n  Args:\n    cls_name: The name of an abstract.InterpreterClass.\n    allow_methods: A bool, whether to allow methods as variables.\n    ordering: A classgen.Ordering describing the order in which the variables\n      should appear.\n    ctx: The abstract context.\n\n  Returns:\n    A collections.OrderedDict of the locals.\n  "
    out = collections.OrderedDict()
    if cls_name not in ctx.vm.local_ops:
        return out
    for op in ctx.vm.local_ops[cls_name]:
        local = ctx.vm.annotated_locals[cls_name][op.name]
        if not is_relevant_class_local(local, op.name, allow_methods):
            continue
        if ordering is Ordering.FIRST_ANNOTATE:
            if not op.is_annotate() or op.name in out:
                continue
        else:
            assert ordering is Ordering.LAST_ASSIGN
            if not op.is_assign():
                continue
            elif op.name in out:
                out.move_to_end(op.name)
        out[op.name] = local
    return out

def make_replace_method(ctx, node, cls, *, kwargs_name='kwargs'):
    if False:
        for i in range(10):
            print('nop')
    'Create a replace() method for a dataclass.'
    typevar = abstract.TypeParameter(abstract_utils.T + cls.name, ctx, bound=cls)
    return overlay_utils.make_method(ctx=ctx, node=node, name='replace', return_type=typevar, self_param=overlay_utils.Param('self', typevar), kwargs=overlay_utils.Param(kwargs_name))

def get_or_create_annotations_dict(members, ctx):
    if False:
        for i in range(10):
            print('nop')
    "Get __annotations__ from members map, create and attach it if not present.\n\n  The returned dict is also referenced by members, so it is safe to mutate.\n\n  Args:\n    members: A dict of member name to variable.\n    ctx: context.Context instance.\n\n  Returns:\n    members['__annotations__'] unpacked as a python dict\n  "
    annotations_dict = abstract_utils.get_annotations_dict(members)
    if annotations_dict is None:
        annotations_dict = abstract.AnnotationsDict({}, ctx)
        members['__annotations__'] = annotations_dict.to_variable(ctx.root_node)
    return annotations_dict

@dataclasses.dataclass
class Field:
    """A class member variable."""
    name: str
    typ: Any
    default: Any = None

@dataclasses.dataclass
class ClassProperties:
    """Properties needed to construct a class."""
    name: str
    fields: List[Field]
    bases: List[Any]

    @classmethod
    def from_field_names(cls, name, field_names, ctx):
        if False:
            for i in range(10):
                print('nop')
        'Make a ClassProperties from field names with no types.'
        fields = [Field(n, ctx.convert.unsolvable, None) for n in field_names]
        return cls(name, fields, [])

def make_annotations_dict(fields, node, ctx):
    if False:
        return 10
    locals_ = {f.name: abstract_utils.Local(node, None, f.typ, None, ctx) for f in fields}
    return abstract.AnnotationsDict(locals_, ctx).to_variable(node)