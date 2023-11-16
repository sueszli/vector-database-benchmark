"""Implementation of types from the fiddle library."""
import re
from typing import Any, Dict, Tuple
from pytype.abstract import abstract
from pytype.abstract import abstract_utils
from pytype.abstract import function
from pytype.abstract import mixin
from pytype.overlays import classgen
from pytype.overlays import overlay
from pytype.pytd import pytd
Node = Any
Variable = Any
_INSTANCE_CACHE: Dict[Tuple[Node, abstract.Class, str], abstract.Instance] = {}

class FiddleOverlay(overlay.Overlay):
    """A custom overlay for the 'fiddle' module."""

    def __init__(self, ctx):
        if False:
            while True:
                i = 10
        "Initializes the FiddleOverlay.\n\n    This function loads the AST for the fiddle module, which is used to\n    access type information for any members that are not explicitly provided by\n    the overlay. See get_attribute in attribute.py for how it's used.\n\n    Args:\n      ctx: An instance of context.Context.\n    "
        if ctx.options.use_fiddle_overlay:
            member_map = {'Config': overlay.add_name('Config', BuildableBuilder), 'Partial': overlay.add_name('Partial', BuildableBuilder)}
        else:
            member_map = {}
        ast = ctx.loader.import_name('fiddle')
        super().__init__(ctx, 'fiddle', member_map, ast)

class BuildableBuilder(abstract.PyTDClass, mixin.HasSlots):
    """Factory for creating fiddle.Config classes."""

    def __init__(self, name, ctx, module):
        if False:
            for i in range(10):
                print('nop')
        pytd_cls = ctx.loader.lookup_pytd(module, name)
        if isinstance(pytd_cls, pytd.Constant):
            pytd_cls = ctx.convert.constant_to_value(pytd_cls).pytd_cls
        super().__init__(name, pytd_cls, ctx)
        mixin.HasSlots.init_mixin(self)
        self.set_native_slot('__getitem__', self.getitem_slot)
        self.fiddle_type_name = name
        self.module = module

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'Fiddle{self.name}'

    def _match_pytd_init(self, node, init_var, args):
        if False:
            print('Hello World!')
        init = init_var.data[0]
        old_pytd_sigs = []
        for signature in init.signatures:
            old_pytd_sig = signature.pytd_sig
            signature.pytd_sig = old_pytd_sig.Replace(params=tuple((p.Replace(optional=True) for p in old_pytd_sig.params)))
            old_pytd_sigs.append(old_pytd_sig)
        try:
            init.match_args(node, args)
        finally:
            for (signature, old_pytd_sig) in zip(init.signatures, old_pytd_sigs):
                signature.pytd_sig = old_pytd_sig

    def _match_interpreter_init(self, node, init_var, args):
        if False:
            while True:
                i = 10
        init = init_var.data[0]
        old_defaults = {}
        for k in init.signature.param_names:
            old_defaults[k] = init.signature.defaults.get(k)
            init.signature.defaults[k] = self.ctx.new_unsolvable(node)
        try:
            function.call_function(self.ctx, node, init_var, args)
        finally:
            for (k, default) in old_defaults.items():
                if default:
                    init.signature.defaults[k] = default
                else:
                    del init.signature.defaults[k]

    def _make_init_args(self, node, underlying, args, kwargs):
        if False:
            print('Hello World!')
        'Unwrap Config instances for arg matching.'

        def unwrap(arg_var):
            if False:
                i = 10
                return i + 15
            for d in arg_var.data:
                if isinstance(d, Buildable):
                    if isinstance(d.underlying, abstract.Function):
                        return self.ctx.new_unsolvable(node)
                    else:
                        return d.underlying.instantiate(node)
            return arg_var
        new_args = (underlying.instantiate(node),)
        new_args += tuple((unwrap(arg) for arg in args[1:]))
        new_kwargs = {k: unwrap(arg) for (k, arg) in kwargs.items()}
        return function.Args(posargs=new_args, namedargs=new_kwargs)

    def _check_init_args(self, node, underlying, args, kwargs):
        if False:
            i = 10
            return i + 15
        if len(args) > 1 or kwargs:
            (_, init_var) = self.ctx.attribute_handler.get_attribute(node, underlying, '__init__')
            if abstract_utils.is_dataclass(underlying):
                args = self._make_init_args(node, underlying, args, kwargs)
                init = init_var.data[0]
                if isinstance(init, abstract.PyTDFunction):
                    self._match_pytd_init(node, init_var, args)
                else:
                    self._match_interpreter_init(node, init_var, args)

    def new_slot(self, node, unused_cls, *args, **kwargs) -> Tuple[Node, abstract.Instance]:
        if False:
            return 10
        'Create a Config or Partial instance from args.'
        underlying = args[0].data[0]
        self._check_init_args(node, underlying, args, kwargs)
        (node, ret) = make_instance(self.name, underlying, node, self.ctx)
        return (node, ret.to_variable(node))

    def getitem_slot(self, node, index_var) -> Tuple[Node, abstract.Instance]:
        if False:
            while True:
                i = 10
        'Specialize the generic class with the value of index_var.'
        underlying = index_var.data[0]
        ret = BuildableType(self.name, underlying, self.ctx, module=self.module)
        return (node, ret.to_variable(node))

    def get_own_new(self, node, value) -> Tuple[Node, Variable]:
        if False:
            while True:
                i = 10
        new = abstract.NativeFunction('__new__', self.new_slot, self.ctx)
        return (node, new.to_variable(node))

class BuildableType(abstract.ParameterizedClass):
    """Base generic class for fiddle.Config and fiddle.Partial."""

    def __init__(self, fiddle_type_name, underlying, ctx, template=None, module='fiddle'):
        if False:
            i = 10
            return i + 15
        base_cls = BuildableBuilder(fiddle_type_name, ctx, module)
        if isinstance(underlying, abstract.Function):
            formal_type_parameters = {abstract_utils.T: ctx.convert.unsolvable}
        else:
            formal_type_parameters = {abstract_utils.T: underlying}
        super().__init__(base_cls, formal_type_parameters, ctx, template)
        self.fiddle_type_name = fiddle_type_name
        self.underlying = underlying

    def replace(self, inner_types):
        if False:
            print('Hello World!')
        inner_types = dict(inner_types)
        new_underlying = inner_types[abstract_utils.T]
        typ = self.__class__
        return typ(self.fiddle_type_name, new_underlying, self.ctx, self.template)

    def instantiate(self, node, container=None):
        if False:
            return 10
        (_, ret) = make_instance(self.fiddle_type_name, self.underlying, node, self.ctx)
        return ret.to_variable(node)

    def __repr__(self):
        if False:
            return 10
        return f'{self.fiddle_type_name}Type[{self.underlying}]'

class Buildable(abstract.Instance):

    def __init__(self, fiddle_type_name, cls, ctx, container=None):
        if False:
            i = 10
            return i + 15
        super().__init__(cls, ctx, container)
        self.fiddle_type_name = fiddle_type_name
        self.underlying = None

class Config(Buildable):
    """An instantiation of a fiddle.Config with a particular template."""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__('Config', *args, **kwargs)

class Partial(Buildable):
    """An instantiation of a fiddle.Partial with a particular template."""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__('Partial', *args, **kwargs)

def _convert_type(typ, subst, ctx):
    if False:
        print('Hello World!')
    'Helper function for recursive type conversion of fields.'
    if isinstance(typ, abstract.TypeParameter) and typ.name in subst:
        typ = subst[typ.name]
    new_typ = BuildableType('Config', typ, ctx, module='fiddle')
    return abstract.Union([new_typ, typ], ctx)

def _make_fields(typ, ctx):
    if False:
        return 10
    'Helper function for recursive type conversion of fields.'
    if isinstance(typ, abstract.ParameterizedClass):
        subst = typ.formal_type_parameters
        typ = typ.base_cls
    else:
        subst = {}
    if abstract_utils.is_dataclass(typ):
        fields = [classgen.Field(x.name, _convert_type(x.typ, subst, ctx), x.default) for x in typ.metadata['__dataclass_fields__']]
        return fields
    return []

def make_instance(subclass_name: str, underlying: abstract.Class, node, ctx) -> Tuple[Node, abstract.BaseValue]:
    if False:
        i = 10
        return i + 15
    'Generate a Buildable instance from an underlying template class.'
    if subclass_name not in ('Config', 'Partial'):
        raise ValueError(f'Unexpected instance class: {subclass_name}')
    cache_key = (ctx.root_node, underlying, subclass_name)
    if cache_key in _INSTANCE_CACHE:
        return (node, _INSTANCE_CACHE[cache_key])
    _INSTANCE_CACHE[cache_key] = ctx.convert.unsolvable
    instance_class = {'Config': Config, 'Partial': Partial}[subclass_name]
    try:
        cls = BuildableType(subclass_name, underlying, ctx, module='fiddle')
    except KeyError:
        return (node, ctx.convert.unsolvable)
    obj = instance_class(cls, ctx)
    obj.underlying = underlying
    fields = _make_fields(underlying, ctx)
    for f in fields:
        obj.members[f.name] = f.typ.instantiate(node)
    obj.members['__annotations__'] = classgen.make_annotations_dict(fields, node, ctx)
    _INSTANCE_CACHE[cache_key] = obj
    return (node, obj)

def is_fiddle_buildable_pytd(cls: pytd.Class) -> bool:
    if False:
        while True:
            i = 10
    fiddle = re.fullmatch('fiddle\\.(.+\\.)?(Config|Partial)', cls.name)
    pax = re.fullmatch('(.+\\.)?pax_fiddle.(Pax)?(Config|Partial)', cls.name)
    return bool(fiddle or pax)

def get_fiddle_buildable_subclass(cls: pytd.Class) -> str:
    if False:
        for i in range(10):
            print('nop')
    if re.search('\\.(Pax)?Config$', cls.name):
        return 'Config'
    if re.search('\\.(Pax)?Partial$', cls.name):
        return 'Partial'
    raise ValueError(f'Unexpected {cls.name} when computing fiddle Buildable subclass; allowed suffixes are `.Config`, and `.Partial`.')