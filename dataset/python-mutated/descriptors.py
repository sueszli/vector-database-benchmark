""" Provide Python descriptors for delegating to Bokeh properties.

The Python `descriptor protocol`_ allows fine-grained control over all
attribute access on instances ("You control the dot"). Bokeh uses the
descriptor protocol to provide easy-to-use, declarative, type-based

class properties that can automatically validate and serialize their
values, as well as help provide sophisticated documentation.

A Bokeh property really consist of two parts: a familiar "property"
portion, such as ``Int``, ``String``, etc., as well as an associated
Python descriptor that delegates attribute access to the property instance.

For example, a very simplified definition of a range-like object might
be:

.. code-block:: python

    from bokeh.model import Model
    from bokeh.core.properties import Float

    class Range(Model):
        start = Float(help="start point")
        end   = Float(help="end point")

When this class is created, the ``MetaHasProps`` metaclass wires up both
the ``start`` and ``end`` attributes to a ``Float`` property. Then, when
a user accesses those attributes, the descriptor delegates all get and
set operations to the ``Float`` property.

.. code-block:: python

    rng = Range()

    # The descriptor __set__ method delegates to Float, which can validate
    # the value 10.3 as a valid floating point value
    rng.start = 10.3

    # But can raise a validation exception if an attempt to set to a list
    # is made
    rng.end = [1,2,3]   # ValueError !

More sophisticated properties such as ``DataSpec`` and its subclasses can
exert control over how values are serialized. Consider this example with
the ``Circle`` glyph and its ``x`` attribute that is a ``NumberSpec``:

.. code-block:: python

    from bokeh.models import Circle

    c = Circle()

    c.x = 10      # serializes to {'value': 10}

    c.x = 'foo'   # serializes to {'field': 'foo'}

There are many other examples like this throughout Bokeh. In this way users
may operate simply and naturally, and not be concerned with the low-level
details around validation, serialization, and documentation.

This module provides the class ``PropertyDescriptor`` and various subclasses
that can be used to attach Bokeh properties to Bokeh models.

.. note::
    These classes form part of the very low-level machinery that implements
    the Bokeh model and property system. It is unlikely that any of these
    classes or their methods will be applicable to any standard usage or to
    anyone who is not directly developing on Bokeh's own infrastructure.

.. _descriptor protocol: https://docs.python.org/3/howto/descriptor.html

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from copy import copy
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
if TYPE_CHECKING:
    from typing_extensions import TypeGuard
    from ...document.events import DocumentPatchedEvent
    from ..has_props import HasProps, Setter
    from .alias import Alias, DeprecatedAlias
    from .bases import Property
__all__ = ('AliasPropertyDescriptor', 'ColumnDataPropertyDescriptor', 'DataSpecPropertyDescriptor', 'DeprecatedAliasPropertyDescriptor', 'PropertyDescriptor', 'UnitsSpecPropertyDescriptor', 'UnsetValueError')
T = TypeVar('T')

class UnsetValueError(ValueError):
    """ Represents state in which descriptor without value was accessed. """

class AliasPropertyDescriptor(Generic[T]):
    """

    """
    serialized: bool = False

    @property
    def aliased_name(self) -> str:
        if False:
            print('Hello World!')
        return self.alias.aliased_name

    def __init__(self, name: str, alias: Alias[T]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.alias = alias
        self.property = alias
        self.__doc__ = f'This is a compatibility alias for the {self.aliased_name!r} property.'

    def __get__(self, obj: HasProps | None, owner: type[HasProps] | None) -> T:
        if False:
            for i in range(10):
                print('nop')
        if obj is not None:
            return getattr(obj, self.aliased_name)
        elif owner is not None:
            return self
        raise ValueError("both 'obj' and 'owner' are None, don't know what to do")

    def __set__(self, obj: HasProps | None, value: T) -> None:
        if False:
            return 10
        setattr(obj, self.aliased_name, value)

    @property
    def readonly(self) -> bool:
        if False:
            print('Hello World!')
        return self.alias.readonly

    def has_unstable_default(self, obj: HasProps) -> bool:
        if False:
            while True:
                i = 10
        return obj.lookup(self.aliased_name).has_unstable_default(obj)

    def class_default(self, cls: type[HasProps], *, no_eval: bool=False):
        if False:
            i = 10
            return i + 15
        return cls.lookup(self.aliased_name).class_default(cls, no_eval=no_eval)

class DeprecatedAliasPropertyDescriptor(AliasPropertyDescriptor[T]):
    """

    """
    alias: DeprecatedAlias[T]

    def __init__(self, name: str, alias: DeprecatedAlias[T]) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name, alias)
        (major, minor, patch) = self.alias.since
        since = f'{major}.{minor}.{patch}'
        self.__doc__ = f'This is a backwards compatibility alias for the {self.aliased_name!r} property.\n\n.. note::\n    Property {self.name!r} was deprecated in Bokeh {since} and will be removed\n    in the future. Update your code to use {self.aliased_name!r} instead.\n'

    def _warn(self) -> None:
        if False:
            i = 10
            return i + 15
        deprecated(self.alias.since, self.name, self.aliased_name, self.alias.extra)

    def __get__(self, obj: HasProps | None, owner: type[HasProps] | None) -> T:
        if False:
            while True:
                i = 10
        if obj is not None:
            self._warn()
        return super().__get__(obj, owner)

    def __set__(self, obj: HasProps | None, value: T) -> None:
        if False:
            return 10
        self._warn()
        super().__set__(obj, value)

class PropertyDescriptor(Generic[T]):
    """ A base class for Bokeh properties with simple get/set and serialization
    behavior.

    """
    name: str
    __doc__: str | None

    def __init__(self, name: str, property: Property[T]) -> None:
        if False:
            i = 10
            return i + 15
        ' Create a PropertyDescriptor for basic Bokeh properties.\n\n        Args:\n            name (str) : The attribute name that this property is for\n            property (Property) : A basic property to create a descriptor for\n\n        '
        self.name = name
        self.property = property
        self.__doc__ = self.property.__doc__

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        ' Basic string representation of ``PropertyDescriptor``.\n\n        Delegates to ``self.property.__str__``\n\n        '
        return f'{self.property}'

    def __get__(self, obj: HasProps | None, owner: type[HasProps] | None) -> T:
        if False:
            for i in range(10):
                print('nop')
        ' Implement the getter for the Python `descriptor protocol`_.\n\n        For instance attribute access, we delegate to the |Property|. For\n        class attribute access, we return ourself.\n\n        Args:\n            obj (HasProps or None) :\n                The instance to set a new property value on (for instance\n                attribute access), or None (for class attribute access)\n\n            owner (obj) :\n                The new value to set the property to\n\n        Returns:\n            None\n\n        Examples:\n\n            .. code-block:: python\n\n                >>> from bokeh.models import Range1d\n\n                >>> r = Range1d(start=10, end=20)\n\n                # instance attribute access, returns the property value\n                >>> r.start\n                10\n\n                # class attribute access, returns the property descriptor\n                >>> Range1d.start\n                <bokeh.core.property.descriptors.PropertyDescriptor at 0x1148b3390>\n\n        '
        if obj is not None:
            value = self._get(obj)
            if value is Undefined:
                raise UnsetValueError(f"{obj}.{self.name} doesn't have a value set")
            return value
        elif owner is not None:
            return self
        raise ValueError("both 'obj' and 'owner' are None, don't know what to do")

    def __set__(self, obj: HasProps, value: T, *, setter: Setter | None=None) -> None:
        if False:
            print('Hello World!')
        ' Implement the setter for the Python `descriptor protocol`_.\n\n        .. note::\n            An optional argument ``setter`` has been added to the standard\n            setter arguments. When needed, this value should be provided by\n            explicitly invoking ``__set__``. See below for more information.\n\n        Args:\n            obj (HasProps) :\n                The instance to set a new property value on\n\n            value (obj) :\n                The new value to set the property to\n\n            setter (ClientSession or ServerSession or None, optional) :\n                This is used to prevent "boomerang" updates to Bokeh apps.\n                (default: None)\n\n                In the context of a Bokeh server application, incoming updates\n                to properties will be annotated with the session that is\n                doing the updating. This value is propagated through any\n                subsequent change notifications that the update triggers.\n                The session can compare the event setter to itself, and\n                suppress any updates that originate from itself.\n\n        Returns:\n            None\n\n        '
        if not hasattr(obj, '_property_values'):
            class_name = obj.__class__.__name__
            raise RuntimeError(f'Cannot set a property value {self.name!r} on a {class_name} instance before HasProps.__init__')
        if self.property.readonly and obj._initialized:
            class_name = obj.__class__.__name__
            raise RuntimeError(f'{class_name}.{self.name} is a readonly property')
        value = self.property.prepare_value(obj, self.name, value)
        old = self._get(obj)
        self._set(obj, old, value, setter=setter)

    def __delete__(self, obj: HasProps) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Implement the deleter for the Python `descriptor protocol`_.\n\n        Args:\n            obj (HasProps) : An instance to delete this property from\n\n        '
        if self.name in obj._property_values:
            old_value = obj._property_values[self.name]
            del obj._property_values[self.name]
            self.trigger_if_changed(obj, old_value)
        if self.name in obj._unstable_default_values:
            del obj._unstable_default_values[self.name]

    def class_default(self, cls: type[HasProps], *, no_eval: bool=False):
        if False:
            for i in range(10):
                print('nop')
        ' Get the default value for a specific subtype of ``HasProps``,\n        which may not be used for an individual instance.\n\n        Args:\n            cls (class) : The class to get the default value for.\n\n            no_eval (bool, optional) :\n                Whether to evaluate callables for defaults (default: False)\n\n        Returns:\n            object\n\n\n        '
        return self.property.themed_default(cls, self.name, None, no_eval=no_eval)

    def instance_default(self, obj: HasProps) -> T:
        if False:
            return 10
        ' Get the default value that will be used for a specific instance.\n\n        Args:\n            obj (HasProps) : The instance to get the default value for.\n\n        Returns:\n            object\n\n        '
        return self.property.themed_default(obj.__class__, self.name, obj.themed_values())

    def get_value(self, obj: HasProps) -> Any:
        if False:
            print('Hello World!')
        ' Produce the value used for serialization.\n\n        Sometimes it is desirable for the serialized value to differ from\n        the ``__get__`` in order for the ``__get__`` value to appear simpler\n        for user or developer convenience.\n\n        Args:\n            obj (HasProps) : the object to get the serialized attribute for\n\n        Returns:\n            Any\n\n        '
        return self.__get__(obj, obj.__class__)

    def set_from_json(self, obj: HasProps, value: Any, *, setter: Setter | None=None):
        if False:
            while True:
                i = 10
        'Sets the value of this property from a JSON value.\n\n        Args:\n            obj: (HasProps) : instance to set the property value on\n\n            json: (JSON-value) : value to set to the attribute to\n\n            models (dict or None, optional) :\n                Mapping of model ids to models (default: None)\n\n                This is needed in cases where the attributes to update also\n                have values that have references.\n\n            setter (ClientSession or ServerSession or None, optional) :\n                This is used to prevent "boomerang" updates to Bokeh apps.\n                (default: None)\n\n                In the context of a Bokeh server application, incoming updates\n                to properties will be annotated with the session that is\n                doing the updating. This value is propagated through any\n                subsequent change notifications that the update triggers.\n                The session can compare the event setter to itself, and\n                suppress any updates that originate from itself.\n\n        Returns:\n            None\n\n        '
        value = self.property.prepare_value(obj, self.name, value)
        old = self._get(obj)
        self._set(obj, old, value, setter=setter)

    def trigger_if_changed(self, obj: HasProps, old: Any) -> None:
        if False:
            while True:
                i = 10
        ' Send a change event notification if the property is set to a\n        value is not equal to ``old``.\n\n        Args:\n            obj (HasProps)\n                The object the property is being set on.\n\n            old (obj) :\n                The previous value of the property to compare\n\n        Returns:\n            None\n\n        '
        new_value = self.__get__(obj, obj.__class__)
        if not self.property.matches(old, new_value):
            self._trigger(obj, old, new_value)

    @property
    def has_ref(self) -> bool:
        if False:
            while True:
                i = 10
        ' Whether the property can refer to another ``HasProps`` instance.\n\n        For basic properties, delegate to the ``has_ref`` attribute on the\n        |Property|.\n\n        '
        return self.property.has_ref

    @property
    def readonly(self) -> bool:
        if False:
            return 10
        ' Whether this property is read-only.\n\n        Read-only properties may only be modified by the client (i.e., by BokehJS\n        in the browser).\n\n        '
        return self.property.readonly

    @property
    def serialized(self) -> bool:
        if False:
            print('Hello World!')
        ' Whether the property should be serialized when serializing an\n        object.\n\n        This would be False for a "virtual" or "convenience" property that\n        duplicates information already available in other properties, for\n        example.\n\n        '
        return self.property.serialized

    def has_unstable_default(self, obj: HasProps) -> bool:
        if False:
            print('Hello World!')
        return self.property._may_have_unstable_default() or self.is_unstable(obj.__overridden_defaults__.get(self.name, None))

    @classmethod
    def is_unstable(cls, value: Any) -> TypeGuard[Callable[[], Any]]:
        if False:
            i = 10
            return i + 15
        from .instance import InstanceDefault
        return isinstance(value, (FunctionType, InstanceDefault))

    def _get(self, obj: HasProps) -> T:
        if False:
            i = 10
            return i + 15
        ' Internal implementation of instance attribute access for the\n        ``PropertyDescriptor`` getter.\n\n        If the value has not been explicitly set by a user, return that\n        value. Otherwise, return the default.\n\n        Args:\n            obj (HasProps) : the instance to get a value of this property for\n\n        Returns:\n            object\n\n        Raises:\n            RuntimeError\n                If the |HasProps| instance has not yet been initialized, or if\n                this descriptor is on a class that is not a |HasProps|.\n\n        '
        if not hasattr(obj, '_property_values'):
            class_name = obj.__class__.__name__
            raise RuntimeError(f'Cannot get a property value {self.name!r} from a {class_name} instance before HasProps.__init__')
        if self.name not in obj._property_values:
            return self._get_default(obj)
        else:
            return obj._property_values[self.name]

    def _get_default(self, obj: HasProps) -> T:
        if False:
            return 10
        ' Internal implementation of instance attribute access for default\n        values.\n\n        Handles bookeeping around ``PropertyContainer`` value, etc.\n\n        '
        if self.name in obj._property_values:
            raise RuntimeError('Bokeh internal error, does not handle the case of self.name already in _property_values')
        themed_values = obj.themed_values()
        is_themed = themed_values is not None and self.name in themed_values
        unstable_dict = obj._unstable_themed_values if is_themed else obj._unstable_default_values
        if self.name in unstable_dict:
            return unstable_dict[self.name]
        default = self.instance_default(obj)
        if self.has_unstable_default(obj):
            if isinstance(default, PropertyValueContainer):
                default._register_owner(obj, self)
            unstable_dict[self.name] = default
        return default

    def _set_value(self, obj: HasProps, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Actual descriptor value assignment. '
        if isinstance(value, PropertyValueContainer):
            value._register_owner(obj, self)
        if self.name in obj._unstable_themed_values:
            del obj._unstable_themed_values[self.name]
        if self.name in obj._unstable_default_values:
            del obj._unstable_default_values[self.name]
        obj._property_values[self.name] = value

    def _set(self, obj: HasProps, old: Any, value: Any, *, hint: DocumentPatchedEvent | None=None, setter: Setter | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Internal implementation helper to set property values.\n\n        This function handles bookkeeping around noting whether values have\n        been explicitly set, etc.\n\n        Args:\n            obj (HasProps)\n                The object the property is being set on.\n\n            old (obj) :\n                The previous value of the property to compare\n\n            hint (event hint or None, optional)\n                An optional update event hint, e.g. ``ColumnStreamedEvent``\n                (default: None)\n\n                Update event hints are usually used at times when better\n                update performance can be obtained by special-casing in\n                some way (e.g. streaming or patching column data sources)\n\n            setter (ClientSession or ServerSession or None, optional) :\n                This is used to prevent "boomerang" updates to Bokeh apps.\n                (default: None)\n\n                In the context of a Bokeh server application, incoming updates\n                to properties will be annotated with the session that is\n                doing the updating. This value is propagated through any\n                subsequent change notifications that the update triggers.\n                The session can compare the event setter to itself, and\n                suppress any updates that originate from itself.\n\n        Returns:\n            None\n\n        '
        if value is Undefined:
            raise RuntimeError('internal error attempting to set Undefined value')
        if self.property.matches(value, old) and hint is None:
            return
        was_set = self.name in obj._property_values
        old_attr_value = obj._property_values[self.name] if was_set else old
        if old_attr_value is not value:
            if isinstance(old_attr_value, PropertyValueContainer):
                old_attr_value._unregister_owner(obj, self)
            self._set_value(obj, value)
        self._trigger(obj, old, value, hint=hint, setter=setter)

    def _notify_mutated(self, obj: HasProps, old: Any, hint: DocumentPatchedEvent | None=None) -> None:
        if False:
            return 10
        ' A method to call when a container is mutated "behind our back"\n        and we detect it with our ``PropertyContainer`` wrappers.\n\n        Args:\n            obj (HasProps) :\n                The object who\'s container value was mutated\n\n            old (object) :\n                The "old" value of the container\n\n                In this case, somewhat weirdly, ``old`` is a copy and the\n                new value should already be set unless we change it due to\n                validation.\n\n            hint (event hint or None, optional)\n                An optional update event hint, e.g. ``ColumnStreamedEvent``\n                (default: None)\n\n                Update event hints are usually used at times when better\n                update performance can be obtained by special-casing in\n                some way (e.g. streaming or patching column data sources)\n\n        Returns:\n            None\n\n        '
        value = self.__get__(obj, obj.__class__)
        value = self.property.prepare_value(obj, self.name, value, hint=hint)
        self._set(obj, old, value, hint=hint)

    def _trigger(self, obj: HasProps, old: Any, value: Any, *, hint: DocumentPatchedEvent | None=None, setter: Setter | None=None) -> None:
        if False:
            print('Hello World!')
        ' Unconditionally send a change event notification for the property.\n\n        Args:\n            obj (HasProps)\n                The object the property is being set on.\n\n            old (obj) :\n                The previous value of the property\n\n            new (obj) :\n                The new value of the property\n\n            hint (event hint or None, optional)\n                An optional update event hint, e.g. ``ColumnStreamedEvent``\n                (default: None)\n\n                Update event hints are usually used at times when better\n                update performance can be obtained by special-casing in\n                some way (e.g. streaming or patching column data sources)\n\n            setter (ClientSession or ServerSession or None, optional) :\n                This is used to prevent "boomerang" updates to Bokeh apps.\n                (default: None)\n\n                In the context of a Bokeh server application, incoming updates\n                to properties will be annotated with the session that is\n                doing the updating. This value is propagated through any\n                subsequent change notifications that the update triggers.\n                The session can compare the event setter to itself, and\n                suppress any updates that originate from itself.\n\n\n        Returns:\n            None\n\n        '
        if hasattr(obj, 'trigger'):
            obj.trigger(self.name, old, value, hint, setter)
_CDS_SET_FROM_CDS_ERROR = '\nColumnDataSource.data properties may only be set from plain Python dicts,\nnot other ColumnDataSource.data values.\n\nIf you need to copy set from one CDS to another, make a shallow copy by\ncalling dict: s1.data = dict(s2.data)\n'

class ColumnDataPropertyDescriptor(PropertyDescriptor):
    """ A ``PropertyDescriptor`` specialized to handling ``ColumnData`` properties.

    """

    def __set__(self, obj, value, *, setter=None):
        if False:
            for i in range(10):
                print('nop')
        ' Implement the setter for the Python `descriptor protocol`_.\n\n        This method first separately extracts and removes any ``units`` field\n        in the JSON, and sets the associated units property directly. The\n        remaining value is then passed to the superclass ``__set__`` to\n        be handled.\n\n        .. note::\n            An optional argument ``setter`` has been added to the standard\n            setter arguments. When needed, this value should be provided by\n            explicitly invoking ``__set__``. See below for more information.\n\n        Args:\n            obj (HasProps) :\n                The instance to set a new property value on\n\n            value (obj) :\n                The new value to set the property to\n\n            setter (ClientSession or ServerSession or None, optional) :\n                This is used to prevent "boomerang" updates to Bokeh apps.\n                (default: None)\n\n                In the context of a Bokeh server application, incoming updates\n                to properties will be annotated with the session that is\n                doing the updating. This value is propagated through any\n                subsequent change notifications that the update triggers.\n                The session can compare the event setter to itself, and\n                suppress any updates that originate from itself.\n\n        Returns:\n            None\n\n        '
        if not hasattr(obj, '_property_values'):
            class_name = obj.__class__.__name__
            raise RuntimeError(f'Cannot set a property value {self.name!r} on a {class_name} instance before HasProps.__init__')
        if self.property.readonly and obj._initialized:
            class_name = obj.__class__.__name__
            raise RuntimeError(f'{class_name}.{self.name} is a readonly property')
        if isinstance(value, PropertyValueColumnData):
            raise ValueError(_CDS_SET_FROM_CDS_ERROR)
        from ...document.events import ColumnDataChangedEvent
        hint = ColumnDataChangedEvent(obj.document, obj, 'data', setter=setter) if obj.document else None
        value = self.property.prepare_value(obj, self.name, value)
        old = self._get(obj)
        self._set(obj, old, value, hint=hint, setter=setter)

class DataSpecPropertyDescriptor(PropertyDescriptor):
    """ A ``PropertyDescriptor`` for Bokeh |DataSpec| properties that serialize to
    field/value dictionaries.

    """

    def get_value(self, obj: HasProps) -> Any:
        if False:
            return 10
        '\n\n        '
        return self.property.to_serializable(obj, self.name, getattr(obj, self.name))

    def set_from_json(self, obj: HasProps, value: Any, *, setter: Setter | None=None):
        if False:
            while True:
                i = 10
        ' Sets the value of this property from a JSON value.\n\n        This method first\n\n        Args:\n            obj (HasProps) :\n\n            json (JSON-dict) :\n\n            models(seq[Model], optional) :\n\n            setter (ClientSession or ServerSession or None, optional) :\n                This is used to prevent "boomerang" updates to Bokeh apps.\n                (default: None)\n\n                In the context of a Bokeh server application, incoming updates\n                to properties will be annotated with the session that is\n                doing the updating. This value is propagated through any\n                subsequent change notifications that the update triggers.\n                The session can compare the event setter to itself, and\n                suppress any updates that originate from itself.\n\n        Returns:\n            None\n\n        '
        if isinstance(value, dict):
            old = getattr(obj, self.name)
            if old is not None:
                try:
                    self.property.value_type.validate(old, False)
                    if 'value' in value:
                        value = value['value']
                except ValueError:
                    if isinstance(old, str) and 'field' in value:
                        value = value['field']
        super().set_from_json(obj, value, setter=setter)

class UnitsSpecPropertyDescriptor(DataSpecPropertyDescriptor):
    """ A ``PropertyDescriptor`` for Bokeh ``UnitsSpec`` properties that
    contribute associated ``_units`` properties automatically as a side effect.

    """

    def __init__(self, name, property, units_property) -> None:
        if False:
            i = 10
            return i + 15
        '\n\n        Args:\n            name (str) :\n                The attribute name that this property is for\n\n            property (Property) :\n                A basic property to create a descriptor for\n\n            units_property (Property) :\n                An associated property to hold units information\n\n        '
        super().__init__(name, property)
        self.units_prop = units_property

    def __set__(self, obj, value, *, setter=None):
        if False:
            i = 10
            return i + 15
        ' Implement the setter for the Python `descriptor protocol`_.\n\n        This method first separately extracts and removes any ``units`` field\n        in the JSON, and sets the associated units property directly. The\n        remaining value is then passed to the superclass ``__set__`` to\n        be handled.\n\n        .. note::\n            An optional argument ``setter`` has been added to the standard\n            setter arguments. When needed, this value should be provided by\n            explicitly invoking ``__set__``. See below for more information.\n\n        Args:\n            obj (HasProps) :\n                The instance to set a new property value on\n\n            value (obj) :\n                The new value to set the property to\n\n            setter (ClientSession or ServerSession or None, optional) :\n                This is used to prevent "boomerang" updates to Bokeh apps.\n                (default: None)\n\n                In the context of a Bokeh server application, incoming updates\n                to properties will be annotated with the session that is\n                doing the updating. This value is propagated through any\n                subsequent change notifications that the update triggers.\n                The session can compare the event setter to itself, and\n                suppress any updates that originate from itself.\n\n        Returns:\n            None\n\n        '
        value = self._extract_units(obj, value)
        super().__set__(obj, value, setter=setter)

    def set_from_json(self, obj, json, *, models=None, setter=None):
        if False:
            i = 10
            return i + 15
        ' Sets the value of this property from a JSON value.\n\n        This method first separately extracts and removes any ``units`` field\n        in the JSON, and sets the associated units property directly. The\n        remaining JSON is then passed to the superclass ``set_from_json`` to\n        be handled.\n\n        Args:\n            obj: (HasProps) : instance to set the property value on\n\n            json: (JSON-value) : value to set to the attribute to\n\n            models (dict or None, optional) :\n                Mapping of model ids to models (default: None)\n\n                This is needed in cases where the attributes to update also\n                have values that have references.\n\n            setter (ClientSession or ServerSession or None, optional) :\n                This is used to prevent "boomerang" updates to Bokeh apps.\n                (default: None)\n\n                In the context of a Bokeh server application, incoming updates\n                to properties will be annotated with the session that is\n                doing the updating. This value is propagated through any\n                subsequent change notifications that the update triggers.\n                The session can compare the event setter to itself, and\n                suppress any updates that originate from itself.\n\n        Returns:\n            None\n\n        '
        json = self._extract_units(obj, json)
        super().set_from_json(obj, json, setter=setter)

    def _extract_units(self, obj, value):
        if False:
            for i in range(10):
                print('nop')
        " Internal helper for dealing with units associated units properties\n        when setting values on ``UnitsSpec`` properties.\n\n        When ``value`` is a dict, this function may mutate the value of the\n        associated units property.\n\n        Args:\n            obj (HasProps) : instance to update units spec property value for\n            value (obj) : new value to set for the property\n\n        Returns:\n            copy of ``value``, with 'units' key and value removed when\n            applicable\n\n        "
        if isinstance(value, dict):
            if 'units' in value:
                value = copy(value)
            units = value.pop('units', None)
            if units:
                self.units_prop.__set__(obj, units)
        return value