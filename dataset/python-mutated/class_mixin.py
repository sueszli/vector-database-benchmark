"""Mixin for all class-like abstract classes."""
import dataclasses
import logging
from typing import Any, List, Mapping, Optional, Sequence, Type
from pytype import datatypes
from pytype.abstract import abstract_utils
from pytype.abstract import function
from pytype.abstract import mixin
from pytype.pytd import mro
from pytype.pytd import pytd
from pytype.typegraph import cfg
log = logging.getLogger(__name__)
_isinstance = abstract_utils._isinstance
_make = abstract_utils._make
_InterpreterFunction = Any
FunctionMapType = Mapping[str, Sequence[_InterpreterFunction]]
_METADATA_KEYS = {'dataclasses.dataclass': '__dataclass_fields__', 'attr.s': '__attrs_attrs__', 'attr.attrs': '__attrs_attrs__', 'attr._make.attrs': '__attrs_attrs__', 'attr.define': '__attrs_attrs__', 'attr.mutable': '__attrs_attrs__', 'attr.frozen': '__attrs_attrs__', 'attr._next_gen.define': '__attrs_attrs__', 'attr._next_gen.mutable': '__attrs_attrs__', 'attr._next_gen.frozen': '__attrs_attrs__', 'typing.dataclass_transform': '__dataclass_transform__', 'typing_extensions.dataclass_transform': '__dataclass_transform__'}

def get_metadata_key(decorator):
    if False:
        while True:
            i = 10
    return _METADATA_KEYS.get(decorator)

class AttributeKinds:
    CLASSVAR = 'classvar'
    INITVAR = 'initvar'

@dataclasses.dataclass
class Attribute:
    """Represents a class member variable.

  Members:
    name: field name
    typ: field python type
    init: Whether the field should be included in the generated __init__
    kw_only: Whether the field is kw_only in the generated __init__
    default: Default value
    kind: Kind of attribute (see the AttributeKinds enum)

  Used in metadata (see Class.metadata below).
  """
    name: str
    typ: Any
    init: bool
    kw_only: bool
    default: Any
    kind: str = ''
    init_type: Any = None
    pytd_const: Any = None

    @classmethod
    def from_pytd_constant(cls, const, ctx, *, kw_only=False):
        if False:
            return 10
        'Generate an Attribute from a pytd.Constant.'
        typ = ctx.convert.constant_to_value(const.type)
        val = const.value and typ.instantiate(ctx.root_node)
        return cls(name=const.name, typ=typ, init=True, kw_only=kw_only, default=val, pytd_const=const)

    @classmethod
    def from_param(cls, param, ctx):
        if False:
            print('Hello World!')
        const = pytd.Constant(param.name, param.type, param.optional)
        return cls.from_pytd_constant(const, ctx, kw_only=param.kind == pytd.ParameterKind.KWONLY)

    def to_pytd_constant(self):
        if False:
            return 10
        return self.pytd_const

    def __repr__(self):
        if False:
            while True:
                i = 10
        return str({'name': self.name, 'typ': self.typ, 'init': self.init, 'default': self.default})

@dataclasses.dataclass
class ClassBuilderProperties:
    """Inputs to ctx.make_class.

  Members:
    name_var: Class name.
    bases: Base classes.
    class_dict_var: Members of the class, as a Variable containing an
        abstract.Dict value.
    metaclass_var: The class's metaclass, if any.
    new_class_var: If not None, make_class() will return new_class_var with
        the newly constructed class added as a binding. Otherwise, a new
        variable if returned.
    class_type: The internal type to build an instance of. Defaults to
        abstract.InterpreterClass. If set, must be a subclass of
        abstract.InterpreterClass.
    decorators: Decorators applied to this class.
    undecorated_methods: All methods defined in this class, without any
        decorators applied. For example, if we have the following class:
            class C:
              @add_x_parameter  # decorator that adds a `x` parameter
              def f(self):
                pass
        then class_dict_var contains function f with signature (self, x),
        while undecorated_methods contains f with signature (self).
  """
    name_var: cfg.Variable
    bases: List[Any]
    class_dict_var: cfg.Variable
    metaclass_var: Optional[cfg.Variable] = None
    new_class_var: Optional[cfg.Variable] = None
    class_type: Optional[Type['Class']] = None
    decorators: Optional[List[str]] = None
    undecorated_methods: Optional[FunctionMapType] = None

class Class(metaclass=mixin.MixinMeta):
    """Mix-in to mark all class-like values."""
    overloads = ('_get_class', 'call', 'compute_mro', 'get_own_new', 'get_special_attribute', 'update_official_name')

    def __new__(cls, *unused_args, **unused_kwds):
        if False:
            for i in range(10):
                print('nop')
        'Prevent direct instantiation.'
        assert cls is not Class, 'Cannot instantiate Class'
        return object.__new__(cls)

    def init_mixin(self, metaclass):
        if False:
            print('Hello World!')
        'Mix-in equivalent of __init__.'
        if metaclass is None:
            metaclass = self._get_inherited_metaclass()
        if metaclass:
            self.cls = metaclass
        self.metadata = {}
        self.decorators = []
        self._instance_cache = {}
        self._init_abstract_methods()
        self._init_protocol_attributes()
        self._init_overrides_bool()
        self._all_formal_type_parameters = datatypes.AliasingDict()
        self._all_formal_type_parameters_loaded = False
        self.additional_init_methods = []
        if self.is_test_class():
            self.additional_init_methods.append('setUp')

    def _get_class(self):
        if False:
            i = 10
            return i + 15
        return self.ctx.convert.type_type

    def bases(self):
        if False:
            print('Hello World!')
        return []

    @property
    def all_formal_type_parameters(self):
        if False:
            print('Hello World!')
        self._load_all_formal_type_parameters()
        return self._all_formal_type_parameters

    def _load_all_formal_type_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        'Load _all_formal_type_parameters.'
        if self._all_formal_type_parameters_loaded:
            return
        bases = [abstract_utils.get_atomic_value(base, default=self.ctx.convert.unsolvable) for base in self.bases()]
        for base in bases:
            abstract_utils.parse_formal_type_parameters(base, self.full_name, self._all_formal_type_parameters)
        self._all_formal_type_parameters_loaded = True

    def get_own_attributes(self):
        if False:
            while True:
                i = 10
        'Get the attributes defined by this class.'
        raise NotImplementedError(self.__class__.__name__)

    def has_protocol_base(self):
        if False:
            print('Hello World!')
        'Returns whether this class inherits directly from typing.Protocol.\n\n    Subclasses that may inherit from Protocol should override this method.\n    '
        return False

    def _init_protocol_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        "Compute this class's protocol attributes."
        if _isinstance(self, 'ParameterizedClass'):
            self.protocol_attributes = self.base_cls.protocol_attributes
            return
        if not self.has_protocol_base():
            self.protocol_attributes = set()
            return
        if _isinstance(self, 'PyTDClass') and self.pytd_cls.name.startswith('typing.'):
            protocol_attributes = set()
            if self.pytd_cls.name == 'typing.Mapping':
                mapping_attrs = {'__contains__', 'keys', 'items', 'values', 'get', '__eq__', '__ne__'}
                protocol_attributes |= mapping_attrs
            protocol_attributes |= self.abstract_methods
            self.protocol_attributes = protocol_attributes
            return
        self.protocol_attributes = self.get_own_attributes()
        protocol_attributes = set()
        for cls in reversed(self.mro):
            if not isinstance(cls, Class):
                continue
            if cls.is_protocol:
                protocol_attributes |= {a for a in cls.protocol_attributes if a in cls}
            else:
                protocol_attributes = {a for a in protocol_attributes if a not in cls}
        self.protocol_attributes = protocol_attributes

    def _init_overrides_bool(self):
        if False:
            while True:
                i = 10
        'Compute and cache whether the class sets its own boolean value.'
        if _isinstance(self, 'ParameterizedClass'):
            self.overrides_bool = self.base_cls.overrides_bool
            return
        for cls in self.mro:
            if isinstance(cls, Class):
                if any((x in cls.get_own_attributes() for x in ('__bool__', '__len__'))):
                    self.overrides_bool = True
                    return
        self.overrides_bool = False

    def get_own_abstract_methods(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the abstract methods defined by this class.'
        raise NotImplementedError(self.__class__.__name__)

    def _init_abstract_methods(self):
        if False:
            while True:
                i = 10
        "Compute this class's abstract methods."
        self.abstract_methods = self.get_own_abstract_methods()
        abstract_methods = set()
        for cls in reversed(self.mro):
            if not isinstance(cls, Class):
                continue
            abstract_methods = {m for m in abstract_methods if m not in cls or m in cls.abstract_methods}
            abstract_methods |= {m for m in cls.abstract_methods if m in cls}
        self.abstract_methods = abstract_methods

    def _has_explicit_abcmeta(self):
        if False:
            for i in range(10):
                print('nop')
        return any((base.full_name == 'abc.ABCMeta' for base in self.cls.mro))

    def _has_implicit_abcmeta(self):
        if False:
            while True:
                i = 10
        'Whether the class should be considered implicitly abstract.'
        if not _isinstance(self, 'InterpreterClass'):
            return False
        for var in self._bases:
            if any((base.full_name == 'typing.Protocol' or (isinstance(base, Class) and base.is_protocol) for base in var.data)):
                return True
        return False

    @property
    def is_abstract(self):
        if False:
            while True:
                i = 10
        return (self._has_explicit_abcmeta() or self._has_implicit_abcmeta()) and bool(self.abstract_methods)

    def is_test_class(self):
        if False:
            for i in range(10):
                print('nop')
        return any((base.full_name in ('unittest.TestCase', 'unittest.case.TestCase') for base in self.mro))

    @property
    def is_enum(self):
        if False:
            while True:
                i = 10
        return any((cls.full_name == 'enum.EnumMeta' for cls in self.cls.mro))

    @property
    def is_protocol(self):
        if False:
            print('Hello World!')
        return bool(self.protocol_attributes)

    @property
    def is_typed_dict_class(self):
        if False:
            for i in range(10):
                print('nop')
        return self.full_name == 'typing.TypedDict' or self.__class__.__name__ == 'TypedDictClass'

    def get_annotated_local(self, name):
        if False:
            i = 10
            return i + 15
        ann = abstract_utils.get_annotations_dict(self.members)
        return ann and ann.annotated_locals.get(name)

    def _get_inherited_metaclass(self):
        if False:
            return 10
        for base in self.mro[1:]:
            if isinstance(base, Class) and base.cls != self.ctx.convert.unsolvable and (base.cls.full_name != 'builtins.type'):
                return base.cls
        return None

    def call_metaclass_init(self, node):
        if False:
            return 10
        "Call the metaclass's __init__ method if it does anything interesting."
        if self.cls.full_name == 'builtins.type':
            return node
        elif isinstance(self.cls, Class) and '__dataclass_transform__' in self.cls.metadata:
            self.metadata['__dataclass_transform__'] = True
            return node
        (node, init) = self.ctx.attribute_handler.get_attribute(node, self.cls, '__init__')
        if not init or not any((_isinstance(f, 'SignedFunction') for f in init.data)):
            return node
        args = function.Args(posargs=(self.to_variable(node), self.ctx.convert.build_string(node, self.name), self.ctx.convert.build_tuple(node, self.bases()), self.ctx.new_unsolvable(node)))
        log.debug('Calling __init__ on metaclass %s of class %s', self.cls.name, self.name)
        (node, _) = function.call_function(self.ctx, node, init, args)
        return node

    def call_init_subclass(self, node):
        if False:
            print('Hello World!')
        'Call init_subclass(cls) for all base classes.'
        for cls in self.mro:
            node = cls.init_subclass(node, self)
        return node

    def get_own_new(self, node, value):
        if False:
            print('Hello World!')
        "Get this value's __new__ method, if it isn't object.__new__.\n\n    Args:\n      node: The current node.\n      value: A cfg.Binding containing this value.\n\n    Returns:\n      A tuple of (1) a node and (2) either a cfg.Variable of the special\n      __new__ method, or None.\n    "
        (node, new) = self.ctx.attribute_handler.get_attribute(node, value.data, '__new__')
        if new is None:
            return (node, None)
        if len(new.bindings) == 1:
            f = new.bindings[0].data
            if _isinstance(f, 'AMBIGUOUS_OR_EMPTY') or self.ctx.convert.object_type.is_object_new(f):
                return (node, None)
        return (node, new)

    def _call_new_and_init(self, node, value, args):
        if False:
            i = 10
            return i + 15
        'Call __new__ if it has been overridden on the given value.'
        (node, new) = self.get_own_new(node, value)
        if new is None:
            return (node, None)
        cls = value.AssignToNewVariable(node)
        new_args = args.replace(posargs=(cls,) + args.posargs)
        (node, variable) = function.call_function(self.ctx, node, new, new_args)
        for val in variable.bindings:
            if not isinstance(val.data, Class) and self == val.data.cls:
                node = self.call_init(node, val, args)
        return (node, variable)

    def _call_method(self, node, value, method_name, args):
        if False:
            while True:
                i = 10
        (node, bound_method) = self.ctx.vm.get_bound_method(node, value.data, method_name, value)
        if bound_method:
            call_repr = f'{self.name}.{method_name}(..._)'
            log.debug('calling %s', call_repr)
            (node, ret) = function.call_function(self.ctx, node, bound_method, args)
            log.debug('%s returned %r', call_repr, ret)
        return node

    def call_init(self, node, value, args):
        if False:
            return 10
        node = self._call_method(node, value, '__init__', args)
        for method in self.additional_init_methods:
            node = self._call_method(node, value, method, function.Args(()))
        return node

    def _new_instance(self, container, node, args):
        if False:
            print('Hello World!')
        "Returns a (possibly cached) instance of 'self'."
        del args
        key = self.ctx.vm.current_opcode or node
        assert key
        if key not in self._instance_cache:
            self._instance_cache[key] = _make('Instance', self, self.ctx, container)
        return self._instance_cache[key]

    def _check_not_instantiable(self):
        if False:
            while True:
                i = 10
        'Report [not-instantiable] if the class cannot be instantiated.'
        if not self.is_abstract or self.from_annotation:
            return
        if self.ctx.vm.frame and self.ctx.vm.frame.func:
            calling_func = self.ctx.vm.frame.func.data
            if _isinstance(calling_func, 'InterpreterFunction') and calling_func.name.startswith(f'{self.name}.'):
                return
        self.ctx.errorlog.not_instantiable(self.ctx.vm.frames, self)

    def call(self, node, func, args, alias_map=None):
        if False:
            return 10
        del alias_map
        self._check_not_instantiable()
        (node, variable) = self._call_new_and_init(node, func, args)
        if variable is None:
            value = self._new_instance(None, node, args)
            variable = self.ctx.program.NewVariable()
            val = variable.AddBinding(value, [func], node)
            node = self.call_init(node, val, args)
        return (node, variable)

    def get_special_attribute(self, node, name, valself):
        if False:
            i = 10
            return i + 15
        'Fetch a special attribute.'
        if name == '__getitem__' and valself is None:
            if self.cls.full_name not in ('builtins.type', 'dataclasses._InitVarMeta'):
                (_, att) = self.ctx.attribute_handler.get_attribute(node, self.cls, name, self.to_binding(node))
                if att:
                    return att
            container = self.to_annotation_container()
            return container.get_special_attribute(node, name, valself)
        return Class.super(self.get_special_attribute)(node, name, valself)

    def has_dynamic_attributes(self):
        if False:
            while True:
                i = 10
        return any((a in self for a in abstract_utils.DYNAMIC_ATTRIBUTE_MARKERS))

    def compute_is_dynamic(self):
        if False:
            return 10
        return any((c.has_dynamic_attributes() for c in self.mro if isinstance(c, Class)))

    def compute_mro(self):
        if False:
            print('Hello World!')
        'Compute the class precedence list (mro) according to C3.'
        bases = abstract_utils.get_mro_bases(self.bases())
        bases = [[self]] + [list(base.mro) for base in bases] + [list(bases)]
        base2cls = {}
        newbases = []
        for row in bases:
            baselist = []
            for base in row:
                if _isinstance(base, 'ParameterizedClass'):
                    base2cls[base.base_cls] = base
                    baselist.append(base.base_cls)
                else:
                    base2cls[base] = base
                    baselist.append(base)
            newbases.append(baselist)
        return tuple((base2cls[base] for base in mro.MROMerge(newbases)))

    def _get_mro_attrs_for_attrs(self, cls_attrs, metadata_key):
        if False:
            for i in range(10):
                print('nop')
        'Traverse the MRO and collect base class attributes for metadata_key.'
        base_attrs = []
        taken_attr_names = {a.name for a in cls_attrs}
        for base_cls in self.mro[1:]:
            if not isinstance(base_cls, Class):
                continue
            sub_attrs = base_cls.metadata.get(metadata_key, None)
            if sub_attrs is None:
                continue
            for a in sub_attrs:
                if a.name not in taken_attr_names:
                    taken_attr_names.add(a.name)
                    base_attrs.append(a)
        return base_attrs + cls_attrs

    def _recompute_attrs_type_from_mro(self, all_attrs, type_params):
        if False:
            print('Hello World!')
        'Traverse the MRO and apply Generic type params to class attributes.\n\n    This IS REQUIRED for dataclass instances that inherits from a Generic.\n\n    Args:\n      all_attrs: All __init__ attributes of a class.\n      type_params: List of ParameterizedClass instances that will override\n        TypeVar attributes in all_attrs.\n    '
        for (typ_name, typ_obj) in type_params.items():
            for attr in all_attrs.values():
                if typ_name == attr.typ.cls.name:
                    attr.typ = typ_obj

    def _get_attrs_from_mro(self, cls_attrs, metadata_key):
        if False:
            return 10
        'Traverse the MRO and collect base class attributes for metadata_key.'
        if metadata_key == '__attrs_attrs__':
            return self._get_mro_attrs_for_attrs(cls_attrs, metadata_key)
        all_attrs = {}
        sub_attrs = []
        type_params = {}
        attributes_to_ignore = set()
        for base_cls in reversed(self.mro[1:]):
            if not isinstance(base_cls, Class):
                continue
            attributes_to_ignore.update(getattr(base_cls, 'IMPLICIT_FIELDS', ()))
            if _isinstance(base_cls, 'ParameterizedClass'):
                type_params = base_cls.formal_type_parameters
                base_cls = base_cls.base_cls
            if metadata_key in base_cls.metadata:
                sub_attrs.append([a for a in base_cls.metadata[metadata_key] if a.name not in attributes_to_ignore])
        sub_attrs.append(cls_attrs)
        for attrs in sub_attrs:
            for a in attrs:
                all_attrs[a.name] = a
        self._recompute_attrs_type_from_mro(all_attrs, type_params)
        return list(all_attrs.values())

    def record_attr_ordering(self, own_attrs):
        if False:
            for i in range(10):
                print('nop')
        'Records the order of attrs to write in the output pyi.'
        self.metadata['attr_order'] = own_attrs

    def compute_attr_metadata(self, own_attrs, decorator):
        if False:
            i = 10
            return i + 15
        'Sets combined metadata based on inherited and own attrs.\n\n    Args:\n      own_attrs: The attrs defined explicitly in this class\n      decorator: The fully qualified decorator name\n\n    Returns:\n      The list of combined attrs.\n    '
        assert decorator in _METADATA_KEYS, f'No metadata key for {decorator}'
        key = _METADATA_KEYS[decorator]
        attrs = self._get_attrs_from_mro(own_attrs, key)
        self.metadata[key] = attrs
        return attrs

    def update_official_name(self, name: str) -> None:
        if False:
            return 10
        'Update the official name.'
        if self._official_name is None or name == self.name or (self._official_name != self.name and name < self._official_name):
            self._official_name = name
            for member_var in self.members.values():
                for member in member_var.data:
                    if isinstance(member, Class):
                        member.update_official_name(f'{name}.{member.name}')