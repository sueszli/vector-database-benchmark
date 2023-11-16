""" Provide a base class for objects that can have declarative, typed,
serializable properties.

.. note::
    These classes form part of the very low-level machinery that implements
    the Bokeh model and property system. It is unlikely that any of these
    classes or their methods will be applicable to any standard usage or to
    anyone who is not directly developing on Bokeh's own infrastructure.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import difflib
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, Literal, NoReturn, TypedDict, TypeVar, Union, overload
from weakref import WeakSet
if TYPE_CHECKING:
    F = TypeVar('F', bound=Callable[..., Any])

    def lru_cache(arg: int | None) -> Callable[[F], F]:
        if False:
            for i in range(10):
                print('nop')
        ...
else:
    from functools import lru_cache
if TYPE_CHECKING:
    from typing_extensions import Self
from ..util.strings import append_docstring, nice_join
from ..util.warnings import warn
from .property.descriptor_factory import PropertyDescriptorFactory
from .property.descriptors import PropertyDescriptor, UnsetValueError
from .property.override import Override
from .property.singletons import Intrinsic, Undefined
from .property.wrappers import PropertyValueContainer
from .serialization import ObjectRep, Ref, Serializable, Serializer
from .types import ID
if TYPE_CHECKING:
    from typing_extensions import NotRequired, TypeAlias
    from ..client.session import ClientSession
    from ..server.session import ServerSession
    from .property.bases import Property
    from .property.dataspec import DataSpec
__all__ = ('abstract', 'HasProps', 'MetaHasProps', 'NonQualified', 'Qualified')
if TYPE_CHECKING:
    Setter: TypeAlias = Union[ClientSession, ServerSession]
C = TypeVar('C', bound=type['HasProps'])
_abstract_classes: WeakSet[type[HasProps]] = WeakSet()

def abstract(cls: C) -> C:
    if False:
        for i in range(10):
            print('nop')
    ' A decorator to mark abstract base classes derived from |HasProps|.\n\n    '
    if not issubclass(cls, HasProps):
        raise TypeError(f'{cls.__name__} is not a subclass of HasProps')
    _abstract_classes.add(cls)
    cls.__doc__ = append_docstring(cls.__doc__, _ABSTRACT_ADMONITION)
    return cls

def is_abstract(cls: type[HasProps]) -> bool:
    if False:
        print('Hello World!')
    return cls in _abstract_classes

def is_DataModel(cls: type[HasProps]) -> bool:
    if False:
        print('Hello World!')
    from ..model import DataModel
    return issubclass(cls, HasProps) and getattr(cls, '__data_model__', False) and (cls != DataModel)

def _overridden_defaults(class_dict: dict[str, Any]) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    overridden_defaults: dict[str, Any] = {}
    for (name, prop) in tuple(class_dict.items()):
        if isinstance(prop, Override):
            del class_dict[name]
            if prop.default_overridden:
                overridden_defaults[name] = prop.default
    return overridden_defaults

def _generators(class_dict: dict[str, Any]):
    if False:
        while True:
            i = 10
    generators: dict[str, PropertyDescriptorFactory[Any]] = {}
    for (name, generator) in tuple(class_dict.items()):
        if isinstance(generator, PropertyDescriptorFactory):
            del class_dict[name]
            generators[name] = generator
    return generators

class _ModelResolver:
    """ """
    _known_models: dict[str, type[HasProps]]

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self._known_models = {}

    def add(self, cls: type[HasProps]) -> None:
        if False:
            while True:
                i = 10
        if not (issubclass(cls, Local) or cls.__name__.startswith('_')):
            previous = self._known_models.get(cls.__qualified_model__, None)
            if previous is not None and (not hasattr(cls, '__implementation__')):
                raise Warning(f"Duplicate qualified model declaration of '{cls.__qualified_model__}'. Previous definition: {previous}")
            self._known_models[cls.__qualified_model__] = cls

    def remove(self, cls: type[HasProps]) -> None:
        if False:
            for i in range(10):
                print('nop')
        del self._known_models[cls.__qualified_model__]

    @property
    def known_models(self) -> dict[str, type[HasProps]]:
        if False:
            i = 10
            return i + 15
        return dict(self._known_models)

    def clear_extensions(self) -> None:
        if False:
            i = 10
            return i + 15

        def is_extension(obj: type[HasProps]) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return getattr(obj, '__implementation__', None) is not None or getattr(obj, '__javascript__', None) is not None or getattr(obj, '__css__', None) is not None
        self._known_models = {key: val for (key, val) in self._known_models.items() if not is_extension(val)}
_default_resolver = _ModelResolver()

class MetaHasProps(type):
    """ Specialize the construction of |HasProps| classes.

    This class is a `metaclass`_ for |HasProps| that is responsible for
    creating and adding the ``PropertyDescriptor`` instances that delegate
    validation and serialization to |Property| attributes.

    .. _metaclass: https://docs.python.org/3/reference/datamodel.html#metaclasses

    """
    __properties__: dict[str, Property[Any]]
    __overridden_defaults__: dict[str, Any]
    __themed_values__: dict[str, Any]

    def __new__(cls, class_name: str, bases: tuple[type, ...], class_dict: dict[str, Any]):
        if False:
            i = 10
            return i + 15
        '\n\n        '
        overridden_defaults = _overridden_defaults(class_dict)
        generators = _generators(class_dict)
        properties = {}
        for (name, generator) in generators.items():
            descriptors = generator.make_descriptors(name)
            for descriptor in descriptors:
                name = descriptor.name
                if name in class_dict:
                    raise RuntimeError(f'Two property generators both created {class_name}.{name}')
                class_dict[name] = descriptor
                properties[name] = descriptor.property
        class_dict['__properties__'] = properties
        class_dict['__overridden_defaults__'] = overridden_defaults
        return super().__new__(cls, class_name, bases, class_dict)

    def __init__(cls, class_name: str, bases: tuple[type, ...], _) -> None:
        if False:
            for i in range(10):
                print('nop')
        if class_name == 'HasProps':
            return
        base_properties: dict[str, Any] = {}
        for base in (x for x in bases if issubclass(x, HasProps)):
            base_properties.update(base.properties(_with_props=True))
        own_properties = {k: v for (k, v) in cls.__dict__.items() if isinstance(v, PropertyDescriptor)}
        redeclared = own_properties.keys() & base_properties.keys()
        if redeclared:
            warn(f'Properties {redeclared!r} in class {cls.__name__} were previously declared on a parent class. It never makes sense to do this. Redundant properties should be deleted here, or on the parent class. Override() can be used to change a default value of a base class property.', RuntimeWarning)
        unused_overrides = cls.__overridden_defaults__.keys() - cls.properties(_with_props=True).keys()
        if unused_overrides:
            warn(f'Overrides of {unused_overrides} in class {cls.__name__} does not override anything.', RuntimeWarning)

    @property
    def model_class_reverse_map(cls) -> dict[str, type[HasProps]]:
        if False:
            for i in range(10):
                print('nop')
        return _default_resolver.known_models

class Local:
    """Don't register this class in model registry. """

class Qualified:
    """Resolve this class by a fully qualified name. """

class NonQualified:
    """Resolve this class by a non-qualified name. """

class HasProps(Serializable, metaclass=MetaHasProps):
    """ Base class for all class types that have Bokeh properties.

    """
    _initialized: bool = False
    _property_values: dict[str, Any]
    _unstable_default_values: dict[str, Any]
    _unstable_themed_values: dict[str, Any]
    __view_model__: ClassVar[str]
    __view_module__: ClassVar[str]
    __qualified_model__: ClassVar[str]
    __implementation__: ClassVar[Any]
    __data_model__: ClassVar[bool]

    @classmethod
    def __init_subclass__(cls):
        if False:
            print('Hello World!')
        super().__init_subclass__()
        if '__view_model__' not in cls.__dict__:
            cls.__view_model__ = cls.__qualname__.replace('<locals>.', '')
        if '__view_module__' not in cls.__dict__:
            cls.__view_module__ = cls.__module__
        if '__qualified_model__' not in cls.__dict__:

            def qualified():
                if False:
                    for i in range(10):
                        print('nop')
                module = cls.__view_module__
                model = cls.__view_model__
                if issubclass(cls, NonQualified):
                    return model
                if not issubclass(cls, Qualified):
                    head = module.split('.')[0]
                    if head == 'bokeh' or head == '__main__' or '__implementation__' in cls.__dict__:
                        return model
                return f'{module}.{model}'
            cls.__qualified_model__ = qualified()
        _default_resolver.add(cls)

    def __init__(self, **properties: Any) -> None:
        if False:
            while True:
                i = 10
        '\n\n        '
        super().__init__()
        self._property_values = {}
        self._unstable_default_values = {}
        self._unstable_themed_values = {}
        for (name, value) in properties.items():
            if value is Undefined or value is Intrinsic:
                continue
            setattr(self, name, value)
        initialized = set(properties.keys())
        for name in self.properties(_with_props=True):
            if name in initialized:
                continue
            desc = self.lookup(name)
            if desc.has_unstable_default(self):
                desc._get(self)
        self._initialized = True

    def __setattr__(self, name: str, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Intercept attribute setting on HasProps in order to special case\n        a few situations:\n\n        * short circuit all property machinery for ``_private`` attributes\n        * suggest similar attribute names on attribute errors\n\n        Args:\n            name (str) : the name of the attribute to set on this object\n            value (obj) : the value to set\n\n        Returns:\n            None\n\n        '
        if name.startswith('_'):
            return super().__setattr__(name, value)
        properties = self.properties(_with_props=True)
        if name in properties:
            return super().__setattr__(name, value)
        descriptor = getattr(self.__class__, name, None)
        if isinstance(descriptor, property):
            return super().__setattr__(name, value)
        self._raise_attribute_error_with_matches(name, properties)

    def __getattr__(self, name: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        ' Intercept attribute setting on HasProps in order to special case\n        a few situations:\n\n        * short circuit all property machinery for ``_private`` attributes\n        * suggest similar attribute names on attribute errors\n\n        Args:\n            name (str) : the name of the attribute to set on this object\n\n        Returns:\n            Any\n\n        '
        if name.startswith('_'):
            return super().__getattribute__(name)
        properties = self.properties(_with_props=True)
        if name in properties:
            return super().__getattribute__(name)
        descriptor = getattr(self.__class__, name, None)
        if isinstance(descriptor, property):
            return super().__getattribute__(name)
        self._raise_attribute_error_with_matches(name, properties)

    def _raise_attribute_error_with_matches(self, name: str, properties: Iterable[str]) -> NoReturn:
        if False:
            while True:
                i = 10
        (matches, text) = (difflib.get_close_matches(name.lower(), properties), 'similar')
        if not matches:
            (matches, text) = (sorted(properties), 'possible')
        raise AttributeError(f'unexpected attribute {name!r} to {self.__class__.__name__}, {text} attributes are {nice_join(matches)}')

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        name = self.__class__.__name__
        return f'{name}(...)'
    __repr__ = __str__

    def equals(self, other: HasProps) -> bool:
        if False:
            print('Hello World!')
        ' Structural equality of models.\n\n        Args:\n            other (HasProps) : the other instance to compare to\n\n        Returns:\n            True, if properties are structurally equal, otherwise False\n\n        '
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.properties_with_values() == other.properties_with_values()

    def to_serializable(self, serializer: Serializer) -> ObjectRep:
        if False:
            for i in range(10):
                print('nop')
        rep = ObjectRep(type='object', name=self.__qualified_model__)
        properties = self.properties_with_values(include_defaults=False)
        attributes = {key: serializer.encode(val) for (key, val) in properties.items()}
        if attributes:
            rep['attributes'] = attributes
        return rep

    def set_from_json(self, name: str, value: Any, *, setter: Setter | None=None) -> None:
        if False:
            return 10
        ' Set a property value on this object from JSON.\n\n        Args:\n            name: (str) : name of the attribute to set\n\n            json: (JSON-value) : value to set to the attribute to\n\n            models (dict or None, optional) :\n                Mapping of model ids to models (default: None)\n\n                This is needed in cases where the attributes to update also\n                have values that have references.\n\n            setter(ClientSession or ServerSession or None, optional) :\n                This is used to prevent "boomerang" updates to Bokeh apps.\n\n                In the context of a Bokeh server application, incoming updates\n                to properties will be annotated with the session that is\n                doing the updating. This value is propagated through any\n                subsequent change notifications that the update triggers.\n                The session can compare the event setter to itself, and\n                suppress any updates that originate from itself.\n\n        Returns:\n            None\n\n        '
        if name in self.properties(_with_props=True):
            log.trace(f'Patching attribute {name!r} of {self!r} with {value!r}')
            descriptor = self.lookup(name)
            descriptor.set_from_json(self, value, setter=setter)
        else:
            log.warning("JSON had attr %r on obj %r, which is a client-only or invalid attribute that shouldn't have been sent", name, self)

    def update(self, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        " Updates the object's properties from the given keyword arguments.\n\n        Returns:\n            None\n\n        Examples:\n\n            The following are equivalent:\n\n            .. code-block:: python\n\n                from bokeh.models import Range1d\n\n                r = Range1d\n\n                # set properties individually:\n                r.start = 10\n                r.end = 20\n\n                # update properties together:\n                r.update(start=10, end=20)\n\n        "
        for (k, v) in kwargs.items():
            setattr(self, k, v)

    @overload
    @classmethod
    def lookup(cls, name: str, *, raises: Literal[True]=True) -> PropertyDescriptor[Any]:
        if False:
            return 10
        ...

    @overload
    @classmethod
    def lookup(cls, name: str, *, raises: Literal[False]=False) -> PropertyDescriptor[Any] | None:
        if False:
            return 10
        ...

    @classmethod
    def lookup(cls, name: str, *, raises: bool=True) -> PropertyDescriptor[Any] | None:
        if False:
            i = 10
            return i + 15
        ' Find the ``PropertyDescriptor`` for a Bokeh property on a class,\n        given the property name.\n\n        Args:\n            name (str) : name of the property to search for\n            raises (bool) : whether to raise or return None if missing\n\n        Returns:\n            PropertyDescriptor : descriptor for property named ``name``\n\n        '
        attr = getattr(cls, name, None)
        if attr is not None or (attr is None and (not raises)):
            return attr
        raise AttributeError(f'{cls.__name__}.{name} property descriptor does not exist')

    @overload
    @classmethod
    @lru_cache(None)
    def properties(cls, *, _with_props: Literal[False]=False) -> set[str]:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    @classmethod
    @lru_cache(None)
    def properties(cls, *, _with_props: Literal[True]=True) -> dict[str, Property[Any]]:
        if False:
            for i in range(10):
                print('nop')
        ...

    @classmethod
    @lru_cache(None)
    def properties(cls, *, _with_props: bool=False) -> set[str] | dict[str, Property[Any]]:
        if False:
            i = 10
            return i + 15
        ' Collect the names of properties on this class.\n\n        .. warning::\n            In a future version of Bokeh, this method will return a dictionary\n            mapping property names to property objects. To future-proof this\n            current usage of this method, wrap the return value in ``list``.\n\n        Returns:\n            property names\n\n        '
        props: dict[str, Property[Any]] = {}
        for c in reversed(cls.__mro__):
            props.update(getattr(c, '__properties__', {}))
        if not _with_props:
            return set(props)
        return props

    @classmethod
    @lru_cache(None)
    def descriptors(cls) -> list[PropertyDescriptor[Any]]:
        if False:
            return 10
        ' List of property descriptors in the order of definition. '
        return [cls.lookup(name) for (name, _) in cls.properties(_with_props=True).items()]

    @classmethod
    @lru_cache(None)
    def properties_with_refs(cls) -> dict[str, Property[Any]]:
        if False:
            print('Hello World!')
        ' Collect the names of all properties on this class that also have\n        references.\n\n        This method *always* traverses the class hierarchy and includes\n        properties defined on any parent classes.\n\n        Returns:\n            set[str] : names of properties that have references\n\n        '
        return {k: v for (k, v) in cls.properties(_with_props=True).items() if v.has_ref}

    @classmethod
    @lru_cache(None)
    def dataspecs(cls) -> dict[str, DataSpec]:
        if False:
            print('Hello World!')
        ' Collect the names of all ``DataSpec`` properties on this class.\n\n        This method *always* traverses the class hierarchy and includes\n        properties defined on any parent classes.\n\n        Returns:\n            set[str] : names of ``DataSpec`` properties\n\n        '
        from .property.dataspec import DataSpec
        return {k: v for (k, v) in cls.properties(_with_props=True).items() if isinstance(v, DataSpec)}

    def properties_with_values(self, *, include_defaults: bool=True, include_undefined: bool=False) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        ' Collect a dict mapping property names to their values.\n\n        This method *always* traverses the class hierarchy and includes\n        properties defined on any parent classes.\n\n        Non-serializable properties are skipped and property values are in\n        "serialized" format which may be slightly different from the values\n        you would normally read from the properties; the intent of this method\n        is to return the information needed to losslessly reconstitute the\n        object instance.\n\n        Args:\n            include_defaults (bool, optional) :\n                Whether to include properties that haven\'t been explicitly set\n                since the object was created. (default: True)\n\n        Returns:\n           dict : mapping from property names to their values\n\n        '
        return self.query_properties_with_values(lambda prop: prop.serialized, include_defaults=include_defaults, include_undefined=include_undefined)

    @classmethod
    def _overridden_defaults(cls) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        ' Returns a dictionary of defaults that have been overridden.\n\n        .. note::\n            This is an implementation detail of ``Property``.\n\n        '
        defaults: dict[str, Any] = {}
        for c in reversed(cls.__mro__):
            defaults.update(getattr(c, '__overridden_defaults__', {}))
        return defaults

    def query_properties_with_values(self, query: Callable[[PropertyDescriptor[Any]], bool], *, include_defaults: bool=True, include_undefined: bool=False) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        ' Query the properties values of |HasProps| instances with a\n        predicate.\n\n        Args:\n            query (callable) :\n                A callable that accepts property descriptors and returns True\n                or False\n\n            include_defaults (bool, optional) :\n                Whether to include properties that have not been explicitly\n                set by a user (default: True)\n\n        Returns:\n            dict : mapping of property names and values for matching properties\n\n        '
        themed_keys: set[str] = set()
        result: dict[str, Any] = {}
        keys = self.properties(_with_props=True)
        if include_defaults:
            selected_keys = set(keys)
        else:
            selected_keys = set(self._property_values.keys()) | set(self._unstable_default_values.keys())
            themed_values = self.themed_values()
            if themed_values is not None:
                themed_keys = set(themed_values.keys())
                selected_keys |= themed_keys
        for key in keys:
            descriptor = self.lookup(key)
            if not query(descriptor):
                continue
            try:
                value = descriptor.get_value(self)
            except UnsetValueError:
                if include_undefined:
                    value = Undefined
                else:
                    raise
            else:
                if key not in selected_keys:
                    continue
                if not include_defaults and key not in themed_keys:
                    if isinstance(value, PropertyValueContainer) and key in self._unstable_default_values:
                        continue
            result[key] = value
        return result

    def themed_values(self) -> dict[str, Any] | None:
        if False:
            print('Hello World!')
        ' Get any theme-provided overrides.\n\n        Results are returned as a dict from property name to value, or\n        ``None`` if no theme overrides any values for this instance.\n\n        Returns:\n            dict or None\n\n        '
        return getattr(self, '__themed_values__', None)

    def apply_theme(self, property_values: dict[str, Any]) -> None:
        if False:
            return 10
        ' Apply a set of theme values which will be used rather than\n        defaults, but will not override application-set values.\n\n        The passed-in dictionary may be kept around as-is and shared with\n        other instances to save memory (so neither the caller nor the\n        |HasProps| instance should modify it).\n\n        Args:\n            property_values (dict) : theme values to use in place of defaults\n\n        Returns:\n            None\n\n        '
        old_dict = self.themed_values()
        if old_dict is property_values:
            return
        removed: set[str] = set()
        if old_dict is not None:
            removed.update(set(old_dict.keys()))
        added = set(property_values.keys())
        old_values: dict[str, Any] = {}
        for k in added.union(removed):
            old_values[k] = getattr(self, k)
        if len(property_values) > 0:
            setattr(self, '__themed_values__', property_values)
        elif hasattr(self, '__themed_values__'):
            delattr(self, '__themed_values__')
        for (k, v) in old_values.items():
            if k in self._unstable_themed_values:
                del self._unstable_themed_values[k]
        for (k, v) in old_values.items():
            descriptor = self.lookup(k)
            if isinstance(descriptor, PropertyDescriptor):
                descriptor.trigger_if_changed(self, v)

    def unapply_theme(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Remove any themed values and restore defaults.\n\n        Returns:\n            None\n\n        '
        self.apply_theme(property_values={})

    def clone(self) -> Self:
        if False:
            print('Hello World!')
        ' Duplicate a HasProps object.\n\n        This creates a shallow clone of the original model, i.e. any\n        mutable containers or child models will not be duplicated.\n\n        '
        attrs = self.properties_with_values(include_defaults=False, include_undefined=True)
        return self.__class__(**{key: val for (key, val) in attrs.items() if val is not Undefined})
KindRef = Any

class PropertyDef(TypedDict):
    name: str
    kind: KindRef
    default: NotRequired[Any]

class OverrideDef(TypedDict):
    name: str
    default: Any

class ModelDef(TypedDict):
    type: Literal['model']
    name: str
    extends: NotRequired[Ref | None]
    properties: NotRequired[list[PropertyDef]]
    overrides: NotRequired[list[OverrideDef]]

def _HasProps_to_serializable(cls: type[HasProps], serializer: Serializer) -> Ref | ModelDef:
    if False:
        return 10
    from ..model import DataModel, Model
    ref = Ref(id=ID(cls.__qualified_model__))
    serializer.add_ref(cls, ref)
    if not is_DataModel(cls):
        return ref
    bases: list[type[HasProps]] = [base for base in cls.__bases__ if issubclass(base, Model) and base != DataModel]
    if len(bases) == 0:
        extends = None
    elif len(bases) == 1:
        [base] = bases
        extends = serializer.encode(base)
    else:
        serializer.error('multiple bases are not supported')
    properties: list[PropertyDef] = []
    overrides: list[OverrideDef] = []
    for prop_name in cls.__properties__:
        descriptor = cls.lookup(prop_name)
        kind = 'Any'
        default = descriptor.property._default
        if default is Undefined:
            prop_def = PropertyDef(name=prop_name, kind=kind)
        else:
            if descriptor.is_unstable(default):
                default = default()
            prop_def = PropertyDef(name=prop_name, kind=kind, default=serializer.encode(default))
        properties.append(prop_def)
    for (prop_name, default) in getattr(cls, '__overridden_defaults__', {}).items():
        overrides.append(OverrideDef(name=prop_name, default=serializer.encode(default)))
    modeldef = ModelDef(type='model', name=cls.__qualified_model__)
    if extends is not None:
        modeldef['extends'] = extends
    if properties:
        modeldef['properties'] = properties
    if overrides:
        modeldef['overrides'] = overrides
    return modeldef
Serializer.register(MetaHasProps, _HasProps_to_serializable)
_ABSTRACT_ADMONITION = '\n    .. note::\n        This is an abstract base class used to help organize the hierarchy of Bokeh\n        model types. **It is not useful to instantiate on its own.**\n\n'