"""Overlay for the enum standard library.

For InterpreterClass enums, i.e. ones in the file being analyzed, the overlay
is accessed by:
1. abstract.BuildClass sees a class with enum.Enum as its base, and calls
EnumBuilder.make_class.
2. EnumBuilder.make_class does some validation, then passes along the actual
creation to ctx.make_class. Notably, EnumBuilder passes in EnumInstance to
ctx.make_class, which provides enum-specific behavior.
3. ctx.make_class does its usual, then calls call_metaclass_init on the newly
created EnumInstance. This bounces back into the overlay, namely EnumMetaInit.
4. EnumMetaInit does the actual transformation of members into proper enum
members.

The transformation into an enum happens so late because enum members are
instances of the enums, which is easier to accomplish when the enum class has
already been created.

PytdClass enums, i.e. those loaded from type stubs, enter the overlay when the
pytd.Class is wrapped with an abstract.PyTDClass in convert.py. After wrapping,
call_metaclass_init is called, allowing EnumMetaInit to transform the PyTDClass
into a proper enum.
"""
import collections
import contextlib
import logging
from typing import Any, Dict, Optional, Union
from pytype.abstract import abstract
from pytype.abstract import abstract_utils
from pytype.abstract import class_mixin
from pytype.abstract import function
from pytype.overlays import classgen
from pytype.overlays import overlay
from pytype.overlays import overlay_utils
from pytype.overlays import special_builtins
from pytype.pytd import pytd
from pytype.pytd import pytd_utils
from pytype.typegraph import cfg
log = logging.getLogger(__name__)
_unsupported = ('ReprEnum', 'EnumCheck', 'FlagBoundary', 'verify', 'property', 'member', 'nonmember', 'global_enum', 'show_flag_values')

class EnumOverlay(overlay.Overlay):
    """An overlay for the enum std lib module."""

    def __init__(self, ctx):
        if False:
            while True:
                i = 10
        if ctx.options.use_enum_overlay:
            member_map = {'Enum': overlay.add_name('Enum', EnumBuilder), 'EnumMeta': EnumMeta, 'EnumType': EnumMeta, 'IntEnum': overlay.add_name('IntEnum', EnumBuilder), 'StrEnum': overlay.add_name('StrEnum', EnumBuilder), **{name: overlay.add_name(name, overlay_utils.not_supported_yet) for name in _unsupported}}
        else:
            member_map = {}
        super().__init__(ctx, 'enum', member_map, ctx.loader.import_name('enum'))

class _DelGetAttributeMixin:
    _member_map: Dict[str, Any]

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        if '__getattribute__' in self._member_map:
            del self._member_map['__getattribute__']

class EnumBuilder(_DelGetAttributeMixin, abstract.PyTDClass):
    """Overlays enum.Enum."""

    def __init__(self, name, ctx, module):
        if False:
            return 10
        super().__init__(name, ctx.loader.lookup_pytd(module, name), ctx)

    def make_class(self, node, props):
        if False:
            while True:
                i = 10
        'Check the members for errors, then create the enum class.'
        props.bases = props.bases or [self.to_variable(node)]
        last_base = props.bases[-1]
        if not any((b.is_enum for b in last_base.data)):
            msg = 'The last base class for an enum must be enum.Enum or a subclass of enum.Enum'
            self.ctx.errorlog.base_class_error(self.ctx.vm.frames, last_base, details=msg)
            return (node, self.ctx.new_unsolvable(node))
        props.metaclass_var = props.metaclass_var or self.ctx.vm.loaded_overlays['enum'].members['EnumMeta']
        props.class_type = EnumInstance
        return self.ctx.make_class(node, props)

    def call(self, node, func, args, alias_map=None):
        if False:
            while True:
                i = 10
        'Implements the behavior of the enum functional API.'
        args = args.simplify(node, self.ctx)
        args = args.replace(posargs=(self.ctx.new_unsolvable(node),) + args.posargs)
        (node, pytd_new_var) = self.ctx.attribute_handler.get_attribute(node, self, '__new__', self.to_binding(node))
        pytd_new = abstract_utils.get_atomic_value(pytd_new_var)
        (lookup_sig, api_sig) = sorted((s.signature for s in pytd_new.signatures), key=lambda s: s.maximum_param_count())
        lookup_new = abstract.SimpleFunction(lookup_sig, self.ctx)
        try:
            return lookup_new.call(node, None, args, alias_map)
        except function.FailedFunctionCall as e:
            log.info('Called Enum.__new__ as lookup, but failed:\n%s', e)
        api_new = abstract.SimpleFunction(api_sig, self.ctx)
        api_new.call(node, None, args, alias_map)
        argmap = {name: var for (name, var, _) in api_sig.iter_args(args)}
        cls_name_var = argmap['value']
        try:
            names = abstract_utils.get_atomic_python_constant(argmap['names'])
        except abstract_utils.ConversionError as e:
            log.info('Failed to unwrap values in enum functional interface:\n%s', e)
            return (node, self.ctx.new_unsolvable(node))
        if isinstance(names, str):
            names = names.replace(',', ' ').split()
            fields = {name: self.ctx.convert.build_int(node) for name in names}
        elif isinstance(names, dict):
            fields = names
        else:
            try:
                possible_pairs = [abstract_utils.get_atomic_python_constant(p) for p in names]
            except abstract_utils.ConversionError as e:
                log.debug('Failed to unwrap possible enum field pairs:\n  %s', e)
                return (node, self.ctx.new_unsolvable(node))
            if not possible_pairs:
                fields = {}
            elif isinstance(possible_pairs[0], str):
                fields = {name: self.ctx.convert.build_int(node) for name in possible_pairs}
            else:
                try:
                    fields = {abstract_utils.get_atomic_python_constant(name): value for (name, value) in possible_pairs}
                except abstract_utils.ConversionError as e:
                    log.debug('Failed to unwrap field names for enum:\n  %s', e)
                    return (node, self.ctx.new_unsolvable(node))
        cls_dict = abstract.Dict(self.ctx)
        cls_dict.update(node, fields)
        metaclass = self.ctx.vm.loaded_overlays['enum'].members['EnumMeta']
        props = class_mixin.ClassBuilderProperties(name_var=cls_name_var, bases=[self.to_variable(node)], class_dict_var=cls_dict.to_variable(node), metaclass_var=metaclass, class_type=EnumInstance)
        return self.ctx.make_class(node, props)

class EnumInstance(abstract.InterpreterClass):
    """A wrapper for classes that subclass enum.Enum."""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.member_type = None
        self.member_attrs = {}
        self._instantiating = False

    @contextlib.contextmanager
    def _is_instantiating(self):
        if False:
            i = 10
            return i + 15
        old_instantiating = self._instantiating
        self._instantiating = True
        try:
            yield
        finally:
            self._instantiating = old_instantiating

    def instantiate(self, node, container=None):
        if False:
            i = 10
            return i + 15
        del container
        instance = abstract.Instance(self, self.ctx)
        instance.members['name'] = self.ctx.convert.build_nonatomic_string(node)
        if self.member_type:
            value = self.member_type.instantiate(node)
        else:
            value = self.ctx.new_unsolvable(node)
        instance.members['value'] = value
        for (attr_name, attr_type) in self.member_attrs.items():
            if self._instantiating:
                instance.members[attr_name] = self.ctx.new_unsolvable(node)
            else:
                with self._is_instantiating():
                    instance.members[attr_name] = attr_type.instantiate(node)
        return instance.to_variable(node)

    def is_empty_enum(self):
        if False:
            print('Hello World!')
        for member in self.members.values():
            for b in member.data:
                if b.cls == self:
                    return False
        return True

    def get_enum_members(self, qualified=False):
        if False:
            i = 10
            return i + 15
        ret = {k: v for (k, v) in self.members.items() if all((d.cls == self for d in v.data))}
        if qualified:
            return {f'{self.name}.{k}': v for (k, v) in ret.items()}
        else:
            return ret

class EnumCmpEQ(abstract.SimpleFunction):
    """Implements the functionality of __eq__ for an enum."""

    def __init__(self, ctx):
        if False:
            return 10
        sig = function.Signature(name='__eq__', param_names=('self', 'other'), posonly_count=0, varargs_name=None, kwonly_params=(), kwargs_name=None, defaults={}, annotations={'return': ctx.convert.bool_type})
        super().__init__(sig, ctx)

    def call(self, node, func, args, alias_map=None):
        if False:
            print('Hello World!')
        (_, argmap) = self.match_and_map_args(node, args, alias_map)
        this_var = argmap['self']
        other_var = argmap['other']
        try:
            this = abstract_utils.get_atomic_value(this_var)
            other = abstract_utils.get_atomic_value(other_var)
        except abstract_utils.ConversionError:
            return (node, self.ctx.convert.build_bool(node))
        return (node, self.ctx.convert.build_bool(node, this.cls == other.cls))

class EnumMeta(_DelGetAttributeMixin, abstract.PyTDClass):
    """Wrapper for enum.EnumMeta.

  EnumMeta is essentially a container for the functions that drive a lot of the
  enum behavior: EnumMetaInit for modifying enum classes, for example.
  """

    def __init__(self, ctx, module):
        if False:
            while True:
                i = 10
        pytd_cls = ctx.loader.lookup_pytd(module, 'EnumMeta')
        super().__init__('EnumMeta', pytd_cls, ctx)
        init = EnumMetaInit(ctx)
        self._member_map['__init__'] = init
        self.members['__init__'] = init.to_variable(ctx.root_node)
        getitem = EnumMetaGetItem(ctx)
        self._member_map['__getitem__'] = getitem
        self.members['__getitem__'] = getitem.to_variable(ctx.root_node)

class EnumMetaInit(abstract.SimpleFunction):
    """Implements the functionality of EnumMeta.__init__.

  Overlaying this function is necessary in order to hook into pytype's metaclass
  handling and set up the Enum classes correctly.
  """

    def __init__(self, ctx):
        if False:
            print('Hello World!')
        sig = function.Signature(name='__init__', param_names=('cls', 'name', 'bases', 'namespace'), posonly_count=0, varargs_name=None, kwonly_params=(), kwargs_name=None, defaults={}, annotations={})
        super().__init__(sig, ctx)
        self._str_pytd = ctx.loader.lookup_pytd('builtins', 'str')

    def _get_class_locals(self, node, cls_name, cls_dict):
        if False:
            for i in range(10):
                print('nop')
        if cls_name in self.ctx.vm.local_ops:
            ret = classgen.get_class_locals(cls_name, False, classgen.Ordering.LAST_ASSIGN, self.ctx).items()
            return ret
        ret = {name: abstract_utils.Local(node, None, None, value, self.ctx) for (name, value) in cls_dict.items()}
        return ret.items()

    def _make_new(self, node, member_type, cls):
        if False:
            return 10
        return overlay_utils.make_method(ctx=self.ctx, node=node, name='__new__', params=[overlay_utils.Param('value', abstract.Union([member_type, cls], self.ctx))], return_type=cls)

    def _get_base_type(self, bases):
        if False:
            for i in range(10):
                print('nop')
        if len(bases) > 1:
            base_type_var = bases[-2]
            base_type = abstract_utils.get_atomic_value(base_type_var, default=None)
            if not base_type:
                return None
            elif '__new__' in base_type or base_type.full_name.startswith('builtins'):
                return base_type
            else:
                return None
        elif bases and len(bases[0].data) == 1:
            base_type_cls = abstract_utils.get_atomic_value(bases[0])
            if isinstance(base_type_cls, EnumInstance):
                if base_type_cls.member_type == self.ctx.convert.unsolvable and base_type_cls.is_empty_enum():
                    return None
                else:
                    return base_type_cls.member_type
            elif base_type_cls.is_enum:
                return self._get_base_type(base_type_cls.bases())
        return None

    def _get_member_new(self, node, cls, base_type):
        if False:
            for i in range(10):
                print('nop')
        if '__new__' in cls:
            return cls.get_own_new(node, cls.to_binding(node))
        if base_type and '__new__' in base_type:
            return base_type.get_own_new(node, base_type.to_binding(node))
        enum_base = abstract_utils.get_atomic_value(cls.bases()[-1])
        if enum_base.full_name != 'enum.Enum' and '__new_member__' in enum_base:
            (node, new) = self.ctx.attribute_handler.get_attribute(node, enum_base, '__new_member__')
            new = abstract_utils.get_atomic_value(new)
            if isinstance(new, abstract.BoundFunction):
                new = new.underlying
            return (node, new.to_variable(node))
        return (node, None)

    def _invalid_name(self, name) -> bool:
        if False:
            i = 10
            return i + 15
        if name in abstract_utils.DYNAMIC_ATTRIBUTE_MARKERS:
            return True
        if name.startswith('__') and name.endswith('__'):
            return True
        return False

    def _is_descriptor(self, node, local) -> bool:
        if False:
            return 10

        def _check(value):
            if False:
                print('Hello World!')
            if isinstance(value, (abstract.Function, abstract.BoundFunction, abstract.ClassMethod, abstract.StaticMethod)):
                return True
            for attr_name in ('__get__', '__set__', '__delete__'):
                (_, attr) = self.ctx.attribute_handler.get_attribute(node, value, attr_name)
                if attr is not None:
                    return True
            return False
        return any((_check(value) for value in local.orig.data))

    def _not_valid_member(self, node, name, local) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if self._invalid_name(name):
            return True
        if not local.orig:
            return True
        return self._is_descriptor(node, local)

    def _is_orig_auto(self, orig):
        if False:
            return 10
        try:
            data = abstract_utils.get_atomic_value(orig)
        except abstract_utils.ConversionError as e:
            log.info('Failed to extract atomic enum value for auto() check: %s', e)
            return False
        return isinstance(data, abstract.Instance) and data.cls.full_name == 'enum.auto'

    def _call_generate_next_value(self, node, cls, name):
        if False:
            while True:
                i = 10
        (node, method) = self.ctx.attribute_handler.get_attribute(node, cls, '_generate_next_value_', cls.to_binding(node))
        if method and all((abstract_utils.is_callable(m) for m in method.data)):
            args = function.Args(posargs=(self.ctx.convert.build_string(node, name), self.ctx.convert.build_int(node), self.ctx.convert.build_int(node), self.ctx.convert.build_list(node, [])))
            return function.call_function(self.ctx, node, method, args)
        else:
            return (node, self.ctx.convert.build_int(node))

    def _value_to_starargs(self, node, value_var, base_type):
        if False:
            return 10
        if len(value_var.data) > 1:
            return self.ctx.convert.build_tuple(node, [value_var])
        value = abstract_utils.get_atomic_value(value_var)
        if self.ctx.matcher(node).match_from_mro(value.cls, self.ctx.convert.tuple_type):
            args = value_var
        else:
            args = self.ctx.convert.build_tuple(node, [value_var])
        if base_type and base_type.full_name == 'builtins.tuple':
            args = self.ctx.convert.build_tuple(node, [args])
        return args

    def _mark_dynamic_enum(self, cls):
        if False:
            while True:
                i = 10
        if cls.maybe_missing_members:
            return
        if cls.cls.full_name != 'enum.EnumMeta':
            cls.maybe_missing_members = True
            return
        for base_var in cls.bases():
            for base in base_var.data:
                if not base.is_enum:
                    continue
                if base.cls.full_name != 'enum.EnumMeta' or base.maybe_missing_members or base.has_dynamic_attributes():
                    cls.maybe_missing_members = True
                    return

    def _setup_interpreterclass(self, node, cls):
        if False:
            i = 10
            return i + 15
        member_types = []
        member_attrs = collections.defaultdict(list)
        base_type = self._get_base_type(cls.bases())
        (node, enum_new) = self._get_member_new(node, cls, base_type)
        for (name, local) in self._get_class_locals(node, cls.name, cls.members):
            if self._not_valid_member(node, name, local):
                continue
            assert local.orig, 'A local with no assigned value was passed to the enum overlay.'
            value = local.orig
            if self._is_orig_auto(value):
                (node, value) = self._call_generate_next_value(node, cls, name)
            if enum_new:
                new_args = function.Args(posargs=(cls.to_variable(node),), starargs=self._value_to_starargs(node, value, base_type))
                (node, member_var) = function.call_function(self.ctx, node, enum_new, new_args, fallback_to_unsolvable=False)
                try:
                    member = abstract_utils.get_atomic_value(member_var)
                except abstract_utils.ConversionError:
                    if member_var.data and all((m.cls == member_var.data[0].cls for m in member_var.data)):
                        member = member_var.data[0]
                    else:
                        member_var = self.ctx.vm.convert.create_new_unknown(node)
                        member = abstract_utils.get_atomic_value(member_var)
            else:
                member = abstract.Instance(cls, self.ctx)
                member_var = member.to_variable(node)
            member.name = f'{cls.full_name}.{name}'
            if '_value_' not in member.members:
                if base_type:
                    args = function.Args(posargs=(), starargs=self._value_to_starargs(node, value, base_type))
                    (node, value) = base_type.call(node, base_type.to_binding(node), args)
                member.members['_value_'] = value
            if '__init__' in cls:
                init_args = function.Args(posargs=(member_var,), starargs=self._value_to_starargs(node, value, base_type))
                (node, init) = self.ctx.attribute_handler.get_attribute(node, cls, '__init__', cls.to_binding(node))
                (node, _) = function.call_function(self.ctx, node, init, init_args)
            member.members['value'] = member.members['_value_']
            member.members['name'] = self.ctx.convert.build_string(node, name)
            for attr_name in member.members:
                if attr_name in ('name', 'value'):
                    continue
                member_attrs[attr_name].extend(member.members[attr_name].data)
            cls.members[name] = member.to_variable(node)
            member_types.extend(value.data)
        if '__new__' in cls:
            saved_new = cls.members['__new__']
            if not any((isinstance(x, special_builtins.ClassMethodInstance) for x in saved_new.data)):
                args = function.Args(posargs=(saved_new,))
                (node, saved_new) = self.ctx.vm.load_special_builtin('classmethod').call(node, None, args)
            cls.members['__new_member__'] = saved_new
        self._mark_dynamic_enum(cls)
        if base_type:
            member_type = base_type
        elif member_types:
            member_type = self.ctx.convert.merge_classes(member_types)
        else:
            member_type = self.ctx.convert.unsolvable
        if member_types:
            cls.members['__new__'] = self._make_new(node, member_type, cls)
        cls.member_type = member_type
        member_attrs = {n: self.ctx.convert.merge_classes(ts) for (n, ts) in member_attrs.items()}
        cls.member_attrs = member_attrs
        if '_generate_next_value_' in cls.members:
            gnv = cls.members['_generate_next_value_']
            if not any((isinstance(x, special_builtins.StaticMethodInstance) for x in gnv.data)):
                args = function.Args(posargs=(gnv,))
                (node, new_gnv) = self.ctx.vm.load_special_builtin('staticmethod').call(node, None, args)
                cls.members['_generate_next_value_'] = new_gnv
        return node

    def _setup_pytdclass(self, node, cls):
        if False:
            return 10
        member_types = []
        for pytd_val in cls.pytd_cls.constants:
            if self._invalid_name(pytd_val.name):
                continue
            if isinstance(pytd_val.type, pytd.Annotated) and "'property'" in pytd_val.type.annotations:
                continue
            if isinstance(pytd_val.type, pytd.GenericType) and pytd_val.type.base_type.name == 'typing.ClassVar':
                continue
            member = abstract.Instance(cls, self.ctx)
            member.name = f'{cls.full_name}.{pytd_val.name}'
            member.members['name'] = self.ctx.convert.constant_to_var(pyval=pytd.Constant(name='name', type=self._str_pytd, value=pytd_val.name), node=node)
            if pytd_val.type.name == cls.pytd_cls.name:
                value_type = pytd.AnythingType()
            else:
                value_type = pytd_val.type
            member.members['value'] = self.ctx.convert.constant_to_var(pyval=pytd.Constant(name='value', type=value_type), node=node)
            member.members['_value_'] = member.members['value']
            cls._member_map[pytd_val.name] = member
            cls.members[pytd_val.name] = member.to_variable(node)
            member_types.append(value_type)
        self._mark_dynamic_enum(cls)
        if member_types:
            member_type = self.ctx.convert.constant_to_value(pytd_utils.JoinTypes(member_types))
            cls.members['__new__'] = self._make_new(node, member_type, cls)
        return node

    def call(self, node, func, args, alias_map=None):
        if False:
            while True:
                i = 10
        (node, ret) = super().call(node, func, args, alias_map)
        argmap = self._map_args(node, args)
        cls_var = argmap['cls']
        (cls,) = cls_var.data
        if isinstance(cls, abstract.PyTDClass) and cls.full_name.startswith('enum.'):
            return (node, ret)
        if isinstance(cls, abstract.InterpreterClass):
            node = self._setup_interpreterclass(node, cls)
        elif isinstance(cls, abstract.PyTDClass):
            node = self._setup_pytdclass(node, cls)
        else:
            raise ValueError(f'Expected an InterpreterClass or PyTDClass, but got {type(cls)}')
        return (node, ret)

class EnumMetaGetItem(abstract.SimpleFunction):
    """Implements the functionality of __getitem__ for enums."""

    def __init__(self, ctx):
        if False:
            return 10
        sig = function.Signature(name='__getitem__', param_names=('cls', 'name'), posonly_count=0, varargs_name=None, kwonly_params=(), kwargs_name=None, defaults={}, annotations={'name': ctx.convert.str_type})
        super().__init__(sig, ctx)

    def _get_member_by_name(self, enum: Union[EnumInstance, abstract.PyTDClass], name: str) -> Optional[cfg.Variable]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(enum, EnumInstance):
            return enum.members.get(name)
        elif name in enum:
            enum.load_lazy_attribute(name)
            return enum.members[name]

    def call(self, node, func, args, alias_map=None):
        if False:
            while True:
                i = 10
        (_, argmap) = self.match_and_map_args(node, args, alias_map)
        cls_var = argmap['cls']
        name_var = argmap['name']
        try:
            cls = abstract_utils.get_atomic_value(cls_var)
        except abstract_utils.ConversionError:
            return (node, self.ctx.new_unsolvable(node))
        if isinstance(cls, abstract.Instance):
            cls = cls.cls
        try:
            name = abstract_utils.get_atomic_python_constant(name_var, str)
        except abstract_utils.ConversionError:
            return (node, cls.instantiate(node))
        inst = self._get_member_by_name(cls, name)
        if inst:
            return (node, inst)
        else:
            self.ctx.errorlog.attribute_error(self.ctx.vm.frames, cls_var.bindings[0], name)
            return (node, self.ctx.new_unsolvable(node))