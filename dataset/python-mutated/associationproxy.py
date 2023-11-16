"""Contain the ``AssociationProxy`` class.

The ``AssociationProxy`` is a Python property object which provides
transparent proxied access to the endpoint of an association object.

See the example ``examples/association/proxied_association.py``.

"""
from __future__ import annotations
import operator
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generic
from typing import ItemsView
from typing import Iterable
from typing import Iterator
from typing import KeysView
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import MutableSet
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import ValuesView
from .. import ColumnElement
from .. import exc
from .. import inspect
from .. import orm
from .. import util
from ..orm import collections
from ..orm import InspectionAttrExtensionType
from ..orm import interfaces
from ..orm import ORMDescriptor
from ..orm.base import SQLORMOperations
from ..orm.interfaces import _AttributeOptions
from ..orm.interfaces import _DCAttributeOptions
from ..orm.interfaces import _DEFAULT_ATTRIBUTE_OPTIONS
from ..sql import operators
from ..sql import or_
from ..sql.base import _NoArg
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import SupportsIndex
from ..util.typing import SupportsKeysAndGetItem
if typing.TYPE_CHECKING:
    from ..orm.interfaces import MapperProperty
    from ..orm.interfaces import PropComparator
    from ..orm.mapper import Mapper
    from ..sql._typing import _ColumnExpressionArgument
    from ..sql._typing import _InfoType
_T = TypeVar('_T', bound=Any)
_T_co = TypeVar('_T_co', bound=Any, covariant=True)
_T_con = TypeVar('_T_con', bound=Any, contravariant=True)
_S = TypeVar('_S', bound=Any)
_KT = TypeVar('_KT', bound=Any)
_VT = TypeVar('_VT', bound=Any)

def association_proxy(target_collection: str, attr: str, *, creator: Optional[_CreatorProtocol]=None, getset_factory: Optional[_GetSetFactoryProtocol]=None, proxy_factory: Optional[_ProxyFactoryProtocol]=None, proxy_bulk_set: Optional[_ProxyBulkSetProtocol]=None, info: Optional[_InfoType]=None, cascade_scalar_deletes: bool=False, create_on_none_assignment: bool=False, init: Union[_NoArg, bool]=_NoArg.NO_ARG, repr: Union[_NoArg, bool]=_NoArg.NO_ARG, default: Optional[Any]=_NoArg.NO_ARG, default_factory: Union[_NoArg, Callable[[], _T]]=_NoArg.NO_ARG, compare: Union[_NoArg, bool]=_NoArg.NO_ARG, kw_only: Union[_NoArg, bool]=_NoArg.NO_ARG) -> AssociationProxy[Any]:
    if False:
        return 10
    "Return a Python property implementing a view of a target\n    attribute which references an attribute on members of the\n    target.\n\n    The returned value is an instance of :class:`.AssociationProxy`.\n\n    Implements a Python property representing a relationship as a collection\n    of simpler values, or a scalar value.  The proxied property will mimic\n    the collection type of the target (list, dict or set), or, in the case of\n    a one to one relationship, a simple scalar value.\n\n    :param target_collection: Name of the attribute that is the immediate\n      target.  This attribute is typically mapped by\n      :func:`~sqlalchemy.orm.relationship` to link to a target collection, but\n      can also be a many-to-one or non-scalar relationship.\n\n    :param attr: Attribute on the associated instance or instances that\n      are available on instances of the target object.\n\n    :param creator: optional.\n\n      Defines custom behavior when new items are added to the proxied\n      collection.\n\n      By default, adding new items to the collection will trigger a\n      construction of an instance of the target object, passing the given\n      item as a positional argument to the target constructor.  For cases\n      where this isn't sufficient, :paramref:`.association_proxy.creator`\n      can supply a callable that will construct the object in the\n      appropriate way, given the item that was passed.\n\n      For list- and set- oriented collections, a single argument is\n      passed to the callable. For dictionary oriented collections, two\n      arguments are passed, corresponding to the key and value.\n\n      The :paramref:`.association_proxy.creator` callable is also invoked\n      for scalar (i.e. many-to-one, one-to-one) relationships. If the\n      current value of the target relationship attribute is ``None``, the\n      callable is used to construct a new object.  If an object value already\n      exists, the given attribute value is populated onto that object.\n\n      .. seealso::\n\n        :ref:`associationproxy_creator`\n\n    :param cascade_scalar_deletes: when True, indicates that setting\n        the proxied value to ``None``, or deleting it via ``del``, should\n        also remove the source object.  Only applies to scalar attributes.\n        Normally, removing the proxied target will not remove the proxy\n        source, as this object may have other state that is still to be\n        kept.\n\n        .. versionadded:: 1.3\n\n        .. seealso::\n\n            :ref:`cascade_scalar_deletes` - complete usage example\n\n    :param create_on_none_assignment: when True, indicates that setting\n      the proxied value to ``None`` should **create** the source object\n      if it does not exist, using the creator.  Only applies to scalar\n      attributes.  This is mutually exclusive\n      vs. the :paramref:`.assocation_proxy.cascade_scalar_deletes`.\n\n      .. versionadded:: 2.0.18\n\n    :param init: Specific to :ref:`orm_declarative_native_dataclasses`,\n     specifies if the mapped attribute should be part of the ``__init__()``\n     method as generated by the dataclass process.\n\n     .. versionadded:: 2.0.0b4\n\n    :param repr: Specific to :ref:`orm_declarative_native_dataclasses`,\n     specifies if the attribute established by this :class:`.AssociationProxy`\n     should be part of the ``__repr__()`` method as generated by the dataclass\n     process.\n\n     .. versionadded:: 2.0.0b4\n\n    :param default_factory: Specific to\n     :ref:`orm_declarative_native_dataclasses`, specifies a default-value\n     generation function that will take place as part of the ``__init__()``\n     method as generated by the dataclass process.\n\n     .. versionadded:: 2.0.0b4\n\n    :param compare: Specific to\n     :ref:`orm_declarative_native_dataclasses`, indicates if this field\n     should be included in comparison operations when generating the\n     ``__eq__()`` and ``__ne__()`` methods for the mapped class.\n\n     .. versionadded:: 2.0.0b4\n\n    :param kw_only: Specific to :ref:`orm_declarative_native_dataclasses`,\n     indicates if this field should be marked as keyword-only when generating\n     the ``__init__()`` method as generated by the dataclass process.\n\n     .. versionadded:: 2.0.0b4\n\n    :param info: optional, will be assigned to\n     :attr:`.AssociationProxy.info` if present.\n\n\n    The following additional parameters involve injection of custom behaviors\n    within the :class:`.AssociationProxy` object and are for advanced use\n    only:\n\n    :param getset_factory: Optional.  Proxied attribute access is\n        automatically handled by routines that get and set values based on\n        the `attr` argument for this proxy.\n\n        If you would like to customize this behavior, you may supply a\n        `getset_factory` callable that produces a tuple of `getter` and\n        `setter` functions.  The factory is called with two arguments, the\n        abstract type of the underlying collection and this proxy instance.\n\n    :param proxy_factory: Optional.  The type of collection to emulate is\n        determined by sniffing the target collection.  If your collection\n        type can't be determined by duck typing or you'd like to use a\n        different collection implementation, you may supply a factory\n        function to produce those collections.  Only applicable to\n        non-scalar relationships.\n\n    :param proxy_bulk_set: Optional, use with proxy_factory.\n\n\n    "
    return AssociationProxy(target_collection, attr, creator=creator, getset_factory=getset_factory, proxy_factory=proxy_factory, proxy_bulk_set=proxy_bulk_set, info=info, cascade_scalar_deletes=cascade_scalar_deletes, create_on_none_assignment=create_on_none_assignment, attribute_options=_AttributeOptions(init, repr, default, default_factory, compare, kw_only))

class AssociationProxyExtensionType(InspectionAttrExtensionType):
    ASSOCIATION_PROXY = 'ASSOCIATION_PROXY'
    "Symbol indicating an :class:`.InspectionAttr` that's\n    of type :class:`.AssociationProxy`.\n\n    Is assigned to the :attr:`.InspectionAttr.extension_type`\n    attribute.\n\n    "

class _GetterProtocol(Protocol[_T_co]):

    def __call__(self, instance: Any) -> _T_co:
        if False:
            i = 10
            return i + 15
        ...

class _SetterProtocol(Protocol):
    ...

class _PlainSetterProtocol(_SetterProtocol, Protocol[_T_con]):

    def __call__(self, instance: Any, value: _T_con) -> None:
        if False:
            i = 10
            return i + 15
        ...

class _DictSetterProtocol(_SetterProtocol, Protocol[_T_con]):

    def __call__(self, instance: Any, key: Any, value: _T_con) -> None:
        if False:
            i = 10
            return i + 15
        ...

class _CreatorProtocol(Protocol):
    ...

class _PlainCreatorProtocol(_CreatorProtocol, Protocol[_T_con]):

    def __call__(self, value: _T_con) -> Any:
        if False:
            print('Hello World!')
        ...

class _KeyCreatorProtocol(_CreatorProtocol, Protocol[_T_con]):

    def __call__(self, key: Any, value: Optional[_T_con]) -> Any:
        if False:
            i = 10
            return i + 15
        ...

class _LazyCollectionProtocol(Protocol[_T]):

    def __call__(self) -> Union[MutableSet[_T], MutableMapping[Any, _T], MutableSequence[_T]]:
        if False:
            i = 10
            return i + 15
        ...

class _GetSetFactoryProtocol(Protocol):

    def __call__(self, collection_class: Optional[Type[Any]], assoc_instance: AssociationProxyInstance[Any]) -> Tuple[_GetterProtocol[Any], _SetterProtocol]:
        if False:
            i = 10
            return i + 15
        ...

class _ProxyFactoryProtocol(Protocol):

    def __call__(self, lazy_collection: _LazyCollectionProtocol[Any], creator: _CreatorProtocol, value_attr: str, parent: AssociationProxyInstance[Any]) -> Any:
        if False:
            return 10
        ...

class _ProxyBulkSetProtocol(Protocol):

    def __call__(self, proxy: _AssociationCollection[Any], collection: Iterable[Any]) -> None:
        if False:
            while True:
                i = 10
        ...

class _AssociationProxyProtocol(Protocol[_T]):
    """describes the interface of :class:`.AssociationProxy`
    without including descriptor methods in the interface."""
    creator: Optional[_CreatorProtocol]
    key: str
    target_collection: str
    value_attr: str
    cascade_scalar_deletes: bool
    create_on_none_assignment: bool
    getset_factory: Optional[_GetSetFactoryProtocol]
    proxy_factory: Optional[_ProxyFactoryProtocol]
    proxy_bulk_set: Optional[_ProxyBulkSetProtocol]

    @util.ro_memoized_property
    def info(self) -> _InfoType:
        if False:
            while True:
                i = 10
        ...

    def for_class(self, class_: Type[Any], obj: Optional[object]=None) -> AssociationProxyInstance[_T]:
        if False:
            print('Hello World!')
        ...

    def _default_getset(self, collection_class: Any) -> Tuple[_GetterProtocol[Any], _SetterProtocol]:
        if False:
            i = 10
            return i + 15
        ...

class AssociationProxy(interfaces.InspectionAttrInfo, ORMDescriptor[_T], _DCAttributeOptions, _AssociationProxyProtocol[_T]):
    """A descriptor that presents a read/write view of an object attribute."""
    is_attribute = True
    extension_type = AssociationProxyExtensionType.ASSOCIATION_PROXY

    def __init__(self, target_collection: str, attr: str, *, creator: Optional[_CreatorProtocol]=None, getset_factory: Optional[_GetSetFactoryProtocol]=None, proxy_factory: Optional[_ProxyFactoryProtocol]=None, proxy_bulk_set: Optional[_ProxyBulkSetProtocol]=None, info: Optional[_InfoType]=None, cascade_scalar_deletes: bool=False, create_on_none_assignment: bool=False, attribute_options: Optional[_AttributeOptions]=None):
        if False:
            return 10
        'Construct a new :class:`.AssociationProxy`.\n\n        The :class:`.AssociationProxy` object is typically constructed using\n        the :func:`.association_proxy` constructor function. See the\n        description of :func:`.association_proxy` for a description of all\n        parameters.\n\n\n        '
        self.target_collection = target_collection
        self.value_attr = attr
        self.creator = creator
        self.getset_factory = getset_factory
        self.proxy_factory = proxy_factory
        self.proxy_bulk_set = proxy_bulk_set
        if cascade_scalar_deletes and create_on_none_assignment:
            raise exc.ArgumentError('The cascade_scalar_deletes and create_on_none_assignment parameters are mutually exclusive.')
        self.cascade_scalar_deletes = cascade_scalar_deletes
        self.create_on_none_assignment = create_on_none_assignment
        self.key = '_%s_%s_%s' % (type(self).__name__, target_collection, id(self))
        if info:
            self.info = info
        if attribute_options and attribute_options != _DEFAULT_ATTRIBUTE_OPTIONS:
            self._has_dataclass_arguments = True
            self._attribute_options = attribute_options
        else:
            self._has_dataclass_arguments = False
            self._attribute_options = _DEFAULT_ATTRIBUTE_OPTIONS

    @overload
    def __get__(self, instance: Literal[None], owner: Literal[None]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __get__(self, instance: Literal[None], owner: Any) -> AssociationProxyInstance[_T]:
        if False:
            print('Hello World!')
        ...

    @overload
    def __get__(self, instance: object, owner: Any) -> _T:
        if False:
            print('Hello World!')
        ...

    def __get__(self, instance: object, owner: Any) -> Union[AssociationProxyInstance[_T], _T, AssociationProxy[_T]]:
        if False:
            i = 10
            return i + 15
        if owner is None:
            return self
        inst = self._as_instance(owner, instance)
        if inst:
            return inst.get(instance)
        assert instance is None
        return self

    def __set__(self, instance: object, values: _T) -> None:
        if False:
            i = 10
            return i + 15
        class_ = type(instance)
        self._as_instance(class_, instance).set(instance, values)

    def __delete__(self, instance: object) -> None:
        if False:
            while True:
                i = 10
        class_ = type(instance)
        self._as_instance(class_, instance).delete(instance)

    def for_class(self, class_: Type[Any], obj: Optional[object]=None) -> AssociationProxyInstance[_T]:
        if False:
            i = 10
            return i + 15
        'Return the internal state local to a specific mapped class.\n\n        E.g., given a class ``User``::\n\n            class User(Base):\n                # ...\n\n                keywords = association_proxy(\'kws\', \'keyword\')\n\n        If we access this :class:`.AssociationProxy` from\n        :attr:`_orm.Mapper.all_orm_descriptors`, and we want to view the\n        target class for this proxy as mapped by ``User``::\n\n            inspect(User).all_orm_descriptors["keywords"].for_class(User).target_class\n\n        This returns an instance of :class:`.AssociationProxyInstance` that\n        is specific to the ``User`` class.   The :class:`.AssociationProxy`\n        object remains agnostic of its parent class.\n\n        :param class\\_: the class that we are returning state for.\n\n        :param obj: optional, an instance of the class that is required\n         if the attribute refers to a polymorphic target, e.g. where we have\n         to look at the type of the actual destination object to get the\n         complete path.\n\n        .. versionadded:: 1.3 - :class:`.AssociationProxy` no longer stores\n           any state specific to a particular parent class; the state is now\n           stored in per-class :class:`.AssociationProxyInstance` objects.\n\n\n        '
        return self._as_instance(class_, obj)

    def _as_instance(self, class_: Any, obj: Any) -> AssociationProxyInstance[_T]:
        if False:
            for i in range(10):
                print('nop')
        try:
            inst = class_.__dict__[self.key + '_inst']
        except KeyError:
            inst = None
        if inst is None:
            owner = self._calc_owner(class_)
            if owner is not None:
                inst = AssociationProxyInstance.for_proxy(self, owner, obj)
                setattr(class_, self.key + '_inst', inst)
            else:
                inst = None
        if inst is not None and (not inst._is_canonical):
            return inst._non_canonical_get_for_object(obj)
        else:
            return inst

    def _calc_owner(self, target_cls: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        try:
            insp = inspect(target_cls)
        except exc.NoInspectionAvailable:
            return None
        else:
            return insp.mapper.class_manager.class_

    def _default_getset(self, collection_class: Any) -> Tuple[_GetterProtocol[Any], _SetterProtocol]:
        if False:
            return 10
        attr = self.value_attr
        _getter = operator.attrgetter(attr)

        def getter(instance: Any) -> Optional[Any]:
            if False:
                for i in range(10):
                    print('nop')
            return _getter(instance) if instance is not None else None
        if collection_class is dict:

            def dict_setter(instance: Any, k: Any, value: Any) -> None:
                if False:
                    return 10
                setattr(instance, attr, value)
            return (getter, dict_setter)
        else:

            def plain_setter(o: Any, v: Any) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                setattr(o, attr, v)
            return (getter, plain_setter)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return 'AssociationProxy(%r, %r)' % (self.target_collection, self.value_attr)
_Self = TypeVar('_Self', bound='AssociationProxyInstance[Any]')

class AssociationProxyInstance(SQLORMOperations[_T]):
    """A per-class object that serves class- and object-specific results.

    This is used by :class:`.AssociationProxy` when it is invoked
    in terms of a specific class or instance of a class, i.e. when it is
    used as a regular Python descriptor.

    When referring to the :class:`.AssociationProxy` as a normal Python
    descriptor, the :class:`.AssociationProxyInstance` is the object that
    actually serves the information.   Under normal circumstances, its presence
    is transparent::

        >>> User.keywords.scalar
        False

    In the special case that the :class:`.AssociationProxy` object is being
    accessed directly, in order to get an explicit handle to the
    :class:`.AssociationProxyInstance`, use the
    :meth:`.AssociationProxy.for_class` method::

        proxy_state = inspect(User).all_orm_descriptors["keywords"].for_class(User)

        # view if proxy object is scalar or not
        >>> proxy_state.scalar
        False

    .. versionadded:: 1.3

    """
    collection_class: Optional[Type[Any]]
    parent: _AssociationProxyProtocol[_T]

    def __init__(self, parent: _AssociationProxyProtocol[_T], owning_class: Type[Any], target_class: Type[Any], value_attr: str):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.key = parent.key
        self.owning_class = owning_class
        self.target_collection = parent.target_collection
        self.collection_class = None
        self.target_class = target_class
        self.value_attr = value_attr
    target_class: Type[Any]
    'The intermediary class handled by this\n    :class:`.AssociationProxyInstance`.\n\n    Intercepted append/set/assignment events will result\n    in the generation of new instances of this class.\n\n    '

    @classmethod
    def for_proxy(cls, parent: AssociationProxy[_T], owning_class: Type[Any], parent_instance: Any) -> AssociationProxyInstance[_T]:
        if False:
            print('Hello World!')
        target_collection = parent.target_collection
        value_attr = parent.value_attr
        prop = cast('orm.RelationshipProperty[_T]', orm.class_mapper(owning_class).get_property(target_collection))
        if not isinstance(prop, orm.RelationshipProperty):
            raise NotImplementedError('association proxy to a non-relationship intermediary is not supported') from None
        target_class = prop.mapper.class_
        try:
            target_assoc = cast('AssociationProxyInstance[_T]', cls._cls_unwrap_target_assoc_proxy(target_class, value_attr))
        except AttributeError:
            return AmbiguousAssociationProxyInstance(parent, owning_class, target_class, value_attr)
        except Exception as err:
            raise exc.InvalidRequestError(f'Association proxy received an unexpected error when trying to retreive attribute "{target_class.__name__}.{parent.value_attr}" from class "{target_class.__name__}": {err}') from err
        else:
            return cls._construct_for_assoc(target_assoc, parent, owning_class, target_class, value_attr)

    @classmethod
    def _construct_for_assoc(cls, target_assoc: Optional[AssociationProxyInstance[_T]], parent: _AssociationProxyProtocol[_T], owning_class: Type[Any], target_class: Type[Any], value_attr: str) -> AssociationProxyInstance[_T]:
        if False:
            for i in range(10):
                print('nop')
        if target_assoc is not None:
            return ObjectAssociationProxyInstance(parent, owning_class, target_class, value_attr)
        attr = getattr(target_class, value_attr)
        if not hasattr(attr, '_is_internal_proxy'):
            return AmbiguousAssociationProxyInstance(parent, owning_class, target_class, value_attr)
        is_object = attr._impl_uses_objects
        if is_object:
            return ObjectAssociationProxyInstance(parent, owning_class, target_class, value_attr)
        else:
            return ColumnAssociationProxyInstance(parent, owning_class, target_class, value_attr)

    def _get_property(self) -> MapperProperty[Any]:
        if False:
            while True:
                i = 10
        return orm.class_mapper(self.owning_class).get_property(self.target_collection)

    @property
    def _comparator(self) -> PropComparator[Any]:
        if False:
            print('Hello World!')
        return getattr(self.owning_class, self.target_collection).comparator

    def __clause_element__(self) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError("The association proxy can't be used as a plain column expression; it only works inside of a comparison expression")

    @classmethod
    def _cls_unwrap_target_assoc_proxy(cls, target_class: Any, value_attr: str) -> Optional[AssociationProxyInstance[_T]]:
        if False:
            i = 10
            return i + 15
        attr = getattr(target_class, value_attr)
        assert not isinstance(attr, AssociationProxy)
        if isinstance(attr, AssociationProxyInstance):
            return attr
        return None

    @util.memoized_property
    def _unwrap_target_assoc_proxy(self) -> Optional[AssociationProxyInstance[_T]]:
        if False:
            while True:
                i = 10
        return self._cls_unwrap_target_assoc_proxy(self.target_class, self.value_attr)

    @property
    def remote_attr(self) -> SQLORMOperations[_T]:
        if False:
            print('Hello World!')
        "The 'remote' class attribute referenced by this\n        :class:`.AssociationProxyInstance`.\n\n        .. seealso::\n\n            :attr:`.AssociationProxyInstance.attr`\n\n            :attr:`.AssociationProxyInstance.local_attr`\n\n        "
        return cast('SQLORMOperations[_T]', getattr(self.target_class, self.value_attr))

    @property
    def local_attr(self) -> SQLORMOperations[Any]:
        if False:
            for i in range(10):
                print('nop')
        "The 'local' class attribute referenced by this\n        :class:`.AssociationProxyInstance`.\n\n        .. seealso::\n\n            :attr:`.AssociationProxyInstance.attr`\n\n            :attr:`.AssociationProxyInstance.remote_attr`\n\n        "
        return cast('SQLORMOperations[Any]', getattr(self.owning_class, self.target_collection))

    @property
    def attr(self) -> Tuple[SQLORMOperations[Any], SQLORMOperations[_T]]:
        if False:
            print('Hello World!')
        'Return a tuple of ``(local_attr, remote_attr)``.\n\n        This attribute was originally intended to facilitate using the\n        :meth:`_query.Query.join` method to join across the two relationships\n        at once, however this makes use of a deprecated calling style.\n\n        To use :meth:`_sql.select.join` or :meth:`_orm.Query.join` with\n        an association proxy, the current method is to make use of the\n        :attr:`.AssociationProxyInstance.local_attr` and\n        :attr:`.AssociationProxyInstance.remote_attr` attributes separately::\n\n            stmt = (\n                select(Parent).\n                join(Parent.proxied.local_attr).\n                join(Parent.proxied.remote_attr)\n            )\n\n        A future release may seek to provide a more succinct join pattern\n        for association proxy attributes.\n\n        .. seealso::\n\n            :attr:`.AssociationProxyInstance.local_attr`\n\n            :attr:`.AssociationProxyInstance.remote_attr`\n\n        '
        return (self.local_attr, self.remote_attr)

    @util.memoized_property
    def scalar(self) -> bool:
        if False:
            return 10
        'Return ``True`` if this :class:`.AssociationProxyInstance`\n        proxies a scalar relationship on the local side.'
        scalar = not self._get_property().uselist
        if scalar:
            self._initialize_scalar_accessors()
        return scalar

    @util.memoized_property
    def _value_is_scalar(self) -> bool:
        if False:
            while True:
                i = 10
        return not self._get_property().mapper.get_property(self.value_attr).uselist

    @property
    def _target_is_object(self) -> bool:
        if False:
            print('Hello World!')
        raise NotImplementedError()
    _scalar_get: _GetterProtocol[_T]
    _scalar_set: _PlainSetterProtocol[_T]

    def _initialize_scalar_accessors(self) -> None:
        if False:
            while True:
                i = 10
        if self.parent.getset_factory:
            (get, set_) = self.parent.getset_factory(None, self)
        else:
            (get, set_) = self.parent._default_getset(None)
        (self._scalar_get, self._scalar_set) = (get, cast('_PlainSetterProtocol[_T]', set_))

    def _default_getset(self, collection_class: Any) -> Tuple[_GetterProtocol[Any], _SetterProtocol]:
        if False:
            print('Hello World!')
        attr = self.value_attr
        _getter = operator.attrgetter(attr)

        def getter(instance: Any) -> Optional[_T]:
            if False:
                print('Hello World!')
            return _getter(instance) if instance is not None else None
        if collection_class is dict:

            def dict_setter(instance: Any, k: Any, value: _T) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                setattr(instance, attr, value)
            return (getter, dict_setter)
        else:

            def plain_setter(o: Any, v: _T) -> None:
                if False:
                    i = 10
                    return i + 15
                setattr(o, attr, v)
            return (getter, plain_setter)

    @util.ro_non_memoized_property
    def info(self) -> _InfoType:
        if False:
            print('Hello World!')
        return self.parent.info

    @overload
    def get(self: _Self, obj: Literal[None]) -> _Self:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def get(self, obj: Any) -> _T:
        if False:
            print('Hello World!')
        ...

    def get(self, obj: Any) -> Union[Optional[_T], AssociationProxyInstance[_T]]:
        if False:
            return 10
        if obj is None:
            return self
        proxy: _T
        if self.scalar:
            target = getattr(obj, self.target_collection)
            return self._scalar_get(target)
        else:
            try:
                (creator_id, self_id, proxy) = cast('Tuple[int, int, _T]', getattr(obj, self.key))
            except AttributeError:
                pass
            else:
                if id(obj) == creator_id and id(self) == self_id:
                    assert self.collection_class is not None
                    return proxy
            (self.collection_class, proxy) = self._new(_lazy_collection(obj, self.target_collection))
            setattr(obj, self.key, (id(obj), id(self), proxy))
            return proxy

    def set(self, obj: Any, values: _T) -> None:
        if False:
            while True:
                i = 10
        if self.scalar:
            creator = cast('_PlainCreatorProtocol[_T]', self.parent.creator if self.parent.creator else self.target_class)
            target = getattr(obj, self.target_collection)
            if target is None:
                if values is None and (not self.parent.create_on_none_assignment):
                    return
                setattr(obj, self.target_collection, creator(values))
            else:
                self._scalar_set(target, values)
                if values is None and self.parent.cascade_scalar_deletes:
                    setattr(obj, self.target_collection, None)
        else:
            proxy = self.get(obj)
            assert self.collection_class is not None
            if proxy is not values:
                proxy._bulk_replace(self, values)

    def delete(self, obj: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.owning_class is None:
            self._calc_owner(obj, None)
        if self.scalar:
            target = getattr(obj, self.target_collection)
            if target is not None:
                delattr(target, self.value_attr)
        delattr(obj, self.target_collection)

    def _new(self, lazy_collection: _LazyCollectionProtocol[_T]) -> Tuple[Type[Any], _T]:
        if False:
            for i in range(10):
                print('nop')
        creator = self.parent.creator if self.parent.creator is not None else cast('_CreatorProtocol', self.target_class)
        collection_class = util.duck_type_collection(lazy_collection())
        if collection_class is None:
            raise exc.InvalidRequestError(f'lazy collection factory did not return a valid collection type, got {collection_class}')
        if self.parent.proxy_factory:
            return (collection_class, self.parent.proxy_factory(lazy_collection, creator, self.value_attr, self))
        if self.parent.getset_factory:
            (getter, setter) = self.parent.getset_factory(collection_class, self)
        else:
            (getter, setter) = self.parent._default_getset(collection_class)
        if collection_class is list:
            return (collection_class, cast(_T, _AssociationList(lazy_collection, creator, getter, setter, self)))
        elif collection_class is dict:
            return (collection_class, cast(_T, _AssociationDict(lazy_collection, creator, getter, setter, self)))
        elif collection_class is set:
            return (collection_class, cast(_T, _AssociationSet(lazy_collection, creator, getter, setter, self)))
        else:
            raise exc.ArgumentError('could not guess which interface to use for collection_class "%s" backing "%s"; specify a proxy_factory and proxy_bulk_set manually' % (self.collection_class, self.target_collection))

    def _set(self, proxy: _AssociationCollection[Any], values: Iterable[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.parent.proxy_bulk_set:
            self.parent.proxy_bulk_set(proxy, values)
        elif self.collection_class is list:
            cast('_AssociationList[Any]', proxy).extend(values)
        elif self.collection_class is dict:
            cast('_AssociationDict[Any, Any]', proxy).update(values)
        elif self.collection_class is set:
            cast('_AssociationSet[Any]', proxy).update(values)
        else:
            raise exc.ArgumentError('no proxy_bulk_set supplied for custom collection_class implementation')

    def _inflate(self, proxy: _AssociationCollection[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        creator = self.parent.creator and self.parent.creator or cast(_CreatorProtocol, self.target_class)
        if self.parent.getset_factory:
            (getter, setter) = self.parent.getset_factory(self.collection_class, self)
        else:
            (getter, setter) = self.parent._default_getset(self.collection_class)
        proxy.creator = creator
        proxy.getter = getter
        proxy.setter = setter

    def _criterion_exists(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> ColumnElement[bool]:
        if False:
            for i in range(10):
                print('nop')
        is_has = kwargs.pop('is_has', None)
        target_assoc = self._unwrap_target_assoc_proxy
        if target_assoc is not None:
            inner = target_assoc._criterion_exists(criterion=criterion, **kwargs)
            return self._comparator._criterion_exists(inner)
        if self._target_is_object:
            attr = getattr(self.target_class, self.value_attr)
            value_expr = attr.comparator._criterion_exists(criterion, **kwargs)
        else:
            if kwargs:
                raise exc.ArgumentError("Can't apply keyword arguments to column-targeted association proxy; use ==")
            elif is_has and criterion is not None:
                raise exc.ArgumentError('Non-empty has() not allowed for column-targeted association proxy; use ==')
            value_expr = criterion
        return self._comparator._criterion_exists(value_expr)

    def any(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> ColumnElement[bool]:
        if False:
            for i in range(10):
                print('nop')
        "Produce a proxied 'any' expression using EXISTS.\n\n        This expression will be a composed product\n        using the :meth:`.Relationship.Comparator.any`\n        and/or :meth:`.Relationship.Comparator.has`\n        operators of the underlying proxied attributes.\n\n        "
        if self._unwrap_target_assoc_proxy is None and (self.scalar and (not self._target_is_object or self._value_is_scalar)):
            raise exc.InvalidRequestError("'any()' not implemented for scalar attributes. Use has().")
        return self._criterion_exists(criterion=criterion, is_has=False, **kwargs)

    def has(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> ColumnElement[bool]:
        if False:
            return 10
        "Produce a proxied 'has' expression using EXISTS.\n\n        This expression will be a composed product\n        using the :meth:`.Relationship.Comparator.any`\n        and/or :meth:`.Relationship.Comparator.has`\n        operators of the underlying proxied attributes.\n\n        "
        if self._unwrap_target_assoc_proxy is None and (not self.scalar or (self._target_is_object and (not self._value_is_scalar))):
            raise exc.InvalidRequestError("'has()' not implemented for collections.  Use any().")
        return self._criterion_exists(criterion=criterion, is_has=True, **kwargs)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return '%s(%r)' % (self.__class__.__name__, self.parent)

class AmbiguousAssociationProxyInstance(AssociationProxyInstance[_T]):
    """an :class:`.AssociationProxyInstance` where we cannot determine
    the type of target object.
    """
    _is_canonical = False

    def _ambiguous(self) -> NoReturn:
        if False:
            i = 10
            return i + 15
        raise AttributeError("Association proxy %s.%s refers to an attribute '%s' that is not directly mapped on class %s; therefore this operation cannot proceed since we don't know what type of object is referred towards" % (self.owning_class.__name__, self.target_collection, self.value_attr, self.target_class))

    def get(self, obj: Any) -> Any:
        if False:
            print('Hello World!')
        if obj is None:
            return self
        else:
            return super().get(obj)

    def __eq__(self, obj: object) -> NoReturn:
        if False:
            i = 10
            return i + 15
        self._ambiguous()

    def __ne__(self, obj: object) -> NoReturn:
        if False:
            print('Hello World!')
        self._ambiguous()

    def any(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> NoReturn:
        if False:
            while True:
                i = 10
        self._ambiguous()

    def has(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> NoReturn:
        if False:
            while True:
                i = 10
        self._ambiguous()

    @util.memoized_property
    def _lookup_cache(self) -> Dict[Type[Any], AssociationProxyInstance[_T]]:
        if False:
            i = 10
            return i + 15
        return {}

    def _non_canonical_get_for_object(self, parent_instance: Any) -> AssociationProxyInstance[_T]:
        if False:
            print('Hello World!')
        if parent_instance is not None:
            actual_obj = getattr(parent_instance, self.target_collection)
            if actual_obj is not None:
                try:
                    insp = inspect(actual_obj)
                except exc.NoInspectionAvailable:
                    pass
                else:
                    mapper = insp.mapper
                    instance_class = mapper.class_
                    if instance_class not in self._lookup_cache:
                        self._populate_cache(instance_class, mapper)
                    try:
                        return self._lookup_cache[instance_class]
                    except KeyError:
                        pass
        return self

    def _populate_cache(self, instance_class: Any, mapper: Mapper[Any]) -> None:
        if False:
            while True:
                i = 10
        prop = orm.class_mapper(self.owning_class).get_property(self.target_collection)
        if mapper.isa(prop.mapper):
            target_class = instance_class
            try:
                target_assoc = self._cls_unwrap_target_assoc_proxy(target_class, self.value_attr)
            except AttributeError:
                pass
            else:
                self._lookup_cache[instance_class] = self._construct_for_assoc(cast('AssociationProxyInstance[_T]', target_assoc), self.parent, self.owning_class, target_class, self.value_attr)

class ObjectAssociationProxyInstance(AssociationProxyInstance[_T]):
    """an :class:`.AssociationProxyInstance` that has an object as a target."""
    _target_is_object: bool = True
    _is_canonical = True

    def contains(self, other: Any, **kw: Any) -> ColumnElement[bool]:
        if False:
            for i in range(10):
                print('nop')
        "Produce a proxied 'contains' expression using EXISTS.\n\n        This expression will be a composed product\n        using the :meth:`.Relationship.Comparator.any`,\n        :meth:`.Relationship.Comparator.has`,\n        and/or :meth:`.Relationship.Comparator.contains`\n        operators of the underlying proxied attributes.\n        "
        target_assoc = self._unwrap_target_assoc_proxy
        if target_assoc is not None:
            return self._comparator._criterion_exists(target_assoc.contains(other) if not target_assoc.scalar else target_assoc == other)
        elif self._target_is_object and self.scalar and (not self._value_is_scalar):
            return self._comparator.has(getattr(self.target_class, self.value_attr).contains(other))
        elif self._target_is_object and self.scalar and self._value_is_scalar:
            raise exc.InvalidRequestError("contains() doesn't apply to a scalar object endpoint; use ==")
        else:
            return self._comparator._criterion_exists(**{self.value_attr: other})

    def __eq__(self, obj: Any) -> ColumnElement[bool]:
        if False:
            while True:
                i = 10
        if obj is None:
            return or_(self._comparator.has(**{self.value_attr: obj}), self._comparator == None)
        else:
            return self._comparator.has(**{self.value_attr: obj})

    def __ne__(self, obj: Any) -> ColumnElement[bool]:
        if False:
            return 10
        return self._comparator.has(getattr(self.target_class, self.value_attr) != obj)

class ColumnAssociationProxyInstance(AssociationProxyInstance[_T]):
    """an :class:`.AssociationProxyInstance` that has a database column as a
    target.
    """
    _target_is_object: bool = False
    _is_canonical = True

    def __eq__(self, other: Any) -> ColumnElement[bool]:
        if False:
            while True:
                i = 10
        expr = self._criterion_exists(self.remote_attr.operate(operators.eq, other))
        if other is None:
            return or_(expr, self._comparator == None)
        else:
            return expr

    def operate(self, op: operators.OperatorType, *other: Any, **kwargs: Any) -> ColumnElement[Any]:
        if False:
            print('Hello World!')
        return self._criterion_exists(self.remote_attr.operate(op, *other, **kwargs))

class _lazy_collection(_LazyCollectionProtocol[_T]):

    def __init__(self, obj: Any, target: str):
        if False:
            print('Hello World!')
        self.parent = obj
        self.target = target

    def __call__(self) -> Union[MutableSet[_T], MutableMapping[Any, _T], MutableSequence[_T]]:
        if False:
            while True:
                i = 10
        return getattr(self.parent, self.target)

    def __getstate__(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return {'obj': self.parent, 'target': self.target}

    def __setstate__(self, state: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.parent = state['obj']
        self.target = state['target']
_IT = TypeVar('_IT', bound='Any')
'instance type - this is the type of object inside a collection.\n\nthis is not the same as the _T of AssociationProxy and\nAssociationProxyInstance itself, which will often refer to the\ncollection[_IT] type.\n\n'

class _AssociationCollection(Generic[_IT]):
    getter: _GetterProtocol[_IT]
    "A function.  Given an associated object, return the 'value'."
    creator: _CreatorProtocol
    '\n    A function that creates new target entities.  Given one parameter:\n    value.  This assertion is assumed::\n\n    obj = creator(somevalue)\n    assert getter(obj) == somevalue\n    '
    parent: AssociationProxyInstance[_IT]
    setter: _SetterProtocol
    'A function.  Given an associated object and a value, store that\n        value on the object.\n    '
    lazy_collection: _LazyCollectionProtocol[_IT]
    'A callable returning a list-based collection of entities (usually an\n          object attribute managed by a SQLAlchemy relationship())'

    def __init__(self, lazy_collection: _LazyCollectionProtocol[_IT], creator: _CreatorProtocol, getter: _GetterProtocol[_IT], setter: _SetterProtocol, parent: AssociationProxyInstance[_IT]):
        if False:
            return 10
        'Constructs an _AssociationCollection.\n\n        This will always be a subclass of either _AssociationList,\n        _AssociationSet, or _AssociationDict.\n\n        '
        self.lazy_collection = lazy_collection
        self.creator = creator
        self.getter = getter
        self.setter = setter
        self.parent = parent
    if typing.TYPE_CHECKING:
        col: Collection[_IT]
    else:
        col = property(lambda self: self.lazy_collection())

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self.col)

    def __bool__(self) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(self.col)

    def __getstate__(self) -> Any:
        if False:
            i = 10
            return i + 15
        return {'parent': self.parent, 'lazy_collection': self.lazy_collection}

    def __setstate__(self, state: Any) -> None:
        if False:
            i = 10
            return i + 15
        self.parent = state['parent']
        self.lazy_collection = state['lazy_collection']
        self.parent._inflate(self)

    def clear(self) -> None:
        if False:
            return 10
        raise NotImplementedError()

class _AssociationSingleItem(_AssociationCollection[_T]):
    setter: _PlainSetterProtocol[_T]
    creator: _PlainCreatorProtocol[_T]

    def _create(self, value: _T) -> Any:
        if False:
            return 10
        return self.creator(value)

    def _get(self, object_: Any) -> _T:
        if False:
            i = 10
            return i + 15
        return self.getter(object_)

    def _bulk_replace(self, assoc_proxy: AssociationProxyInstance[Any], values: Iterable[_IT]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.clear()
        assoc_proxy._set(self, values)

class _AssociationList(_AssociationSingleItem[_T], MutableSequence[_T]):
    """Generic, converting, list-to-list proxy."""
    col: MutableSequence[_T]

    def _set(self, object_: Any, value: _T) -> None:
        if False:
            i = 10
            return i + 15
        self.setter(object_, value)

    @overload
    def __getitem__(self, index: int) -> _T:
        if False:
            return 10
        ...

    @overload
    def __getitem__(self, index: slice) -> MutableSequence[_T]:
        if False:
            print('Hello World!')
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[_T, MutableSequence[_T]]:
        if False:
            print('Hello World!')
        if not isinstance(index, slice):
            return self._get(self.col[index])
        else:
            return [self._get(member) for member in self.col[index]]

    @overload
    def __setitem__(self, index: int, value: _T) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[_T]) -> None:
        if False:
            print('Hello World!')
        ...

    def __setitem__(self, index: Union[int, slice], value: Union[_T, Iterable[_T]]) -> None:
        if False:
            return 10
        if not isinstance(index, slice):
            self._set(self.col[index], cast('_T', value))
        else:
            if index.stop is None:
                stop = len(self)
            elif index.stop < 0:
                stop = len(self) + index.stop
            else:
                stop = index.stop
            step = index.step or 1
            start = index.start or 0
            rng = list(range(index.start or 0, stop, step))
            sized_value = list(value)
            if step == 1:
                for i in rng:
                    del self[start]
                i = start
                for item in sized_value:
                    self.insert(i, item)
                    i += 1
            else:
                if len(sized_value) != len(rng):
                    raise ValueError('attempt to assign sequence of size %s to extended slice of size %s' % (len(sized_value), len(rng)))
                for (i, item) in zip(rng, value):
                    self._set(self.col[i], item)

    @overload
    def __delitem__(self, index: int) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __delitem__(self, index: slice) -> None:
        if False:
            return 10
        ...

    def __delitem__(self, index: Union[slice, int]) -> None:
        if False:
            for i in range(10):
                print('nop')
        del self.col[index]

    def __contains__(self, value: object) -> bool:
        if False:
            i = 10
            return i + 15
        for member in self.col:
            if self._get(member) == value:
                return True
        return False

    def __iter__(self) -> Iterator[_T]:
        if False:
            print('Hello World!')
        'Iterate over proxied values.\n\n        For the actual domain objects, iterate over .col instead or\n        just use the underlying collection directly from its property\n        on the parent.\n        '
        for member in self.col:
            yield self._get(member)
        return

    def append(self, value: _T) -> None:
        if False:
            print('Hello World!')
        col = self.col
        item = self._create(value)
        col.append(item)

    def count(self, value: Any) -> int:
        if False:
            for i in range(10):
                print('nop')
        count = 0
        for v in self:
            if v == value:
                count += 1
        return count

    def extend(self, values: Iterable[_T]) -> None:
        if False:
            print('Hello World!')
        for v in values:
            self.append(v)

    def insert(self, index: int, value: _T) -> None:
        if False:
            while True:
                i = 10
        self.col[index:index] = [self._create(value)]

    def pop(self, index: int=-1) -> _T:
        if False:
            i = 10
            return i + 15
        return self.getter(self.col.pop(index))

    def remove(self, value: _T) -> None:
        if False:
            i = 10
            return i + 15
        for (i, val) in enumerate(self):
            if val == value:
                del self.col[i]
                return
        raise ValueError('value not in list')

    def reverse(self) -> NoReturn:
        if False:
            i = 10
            return i + 15
        'Not supported, use reversed(mylist)'
        raise NotImplementedError()

    def sort(self) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        'Not supported, use sorted(mylist)'
        raise NotImplementedError()

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        del self.col[0:len(self.col)]

    def __eq__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        return list(self) == other

    def __ne__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return list(self) != other

    def __lt__(self, other: List[_T]) -> bool:
        if False:
            return 10
        return list(self) < other

    def __le__(self, other: List[_T]) -> bool:
        if False:
            i = 10
            return i + 15
        return list(self) <= other

    def __gt__(self, other: List[_T]) -> bool:
        if False:
            i = 10
            return i + 15
        return list(self) > other

    def __ge__(self, other: List[_T]) -> bool:
        if False:
            i = 10
            return i + 15
        return list(self) >= other

    def __add__(self, other: List[_T]) -> List[_T]:
        if False:
            print('Hello World!')
        try:
            other = list(other)
        except TypeError:
            return NotImplemented
        return list(self) + other

    def __radd__(self, other: List[_T]) -> List[_T]:
        if False:
            while True:
                i = 10
        try:
            other = list(other)
        except TypeError:
            return NotImplemented
        return other + list(self)

    def __mul__(self, n: SupportsIndex) -> List[_T]:
        if False:
            while True:
                i = 10
        if not isinstance(n, int):
            return NotImplemented
        return list(self) * n

    def __rmul__(self, n: SupportsIndex) -> List[_T]:
        if False:
            i = 10
            return i + 15
        if not isinstance(n, int):
            return NotImplemented
        return n * list(self)

    def __iadd__(self, iterable: Iterable[_T]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        self.extend(iterable)
        return self

    def __imul__(self, n: SupportsIndex) -> Self:
        if False:
            i = 10
            return i + 15
        if not isinstance(n, int):
            raise NotImplementedError()
        if n == 0:
            self.clear()
        elif n > 1:
            self.extend(list(self) * (n - 1))
        return self
    if typing.TYPE_CHECKING:

        def index(self, value: Any, start: int=..., stop: int=...) -> int:
            if False:
                return 10
            ...
    else:

        def index(self, value: Any, *arg) -> int:
            if False:
                print('Hello World!')
            ls = list(self)
            return ls.index(value, *arg)

    def copy(self) -> List[_T]:
        if False:
            return 10
        return list(self)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return repr(list(self))

    def __hash__(self) -> NoReturn:
        if False:
            return 10
        raise TypeError('%s objects are unhashable' % type(self).__name__)
    if not typing.TYPE_CHECKING:
        for (func_name, func) in list(locals().items()):
            if callable(func) and func.__name__ == func_name and (not func.__doc__) and hasattr(list, func_name):
                func.__doc__ = getattr(list, func_name).__doc__
        del func_name, func

class _AssociationDict(_AssociationCollection[_VT], MutableMapping[_KT, _VT]):
    """Generic, converting, dict-to-dict proxy."""
    setter: _DictSetterProtocol[_VT]
    creator: _KeyCreatorProtocol[_VT]
    col: MutableMapping[_KT, Optional[_VT]]

    def _create(self, key: _KT, value: Optional[_VT]) -> Any:
        if False:
            print('Hello World!')
        return self.creator(key, value)

    def _get(self, object_: Any) -> _VT:
        if False:
            for i in range(10):
                print('nop')
        return self.getter(object_)

    def _set(self, object_: Any, key: _KT, value: _VT) -> None:
        if False:
            while True:
                i = 10
        return self.setter(object_, key, value)

    def __getitem__(self, key: _KT) -> _VT:
        if False:
            i = 10
            return i + 15
        return self._get(self.col[key])

    def __setitem__(self, key: _KT, value: _VT) -> None:
        if False:
            for i in range(10):
                print('nop')
        if key in self.col:
            self._set(self.col[key], key, value)
        else:
            self.col[key] = self._create(key, value)

    def __delitem__(self, key: _KT) -> None:
        if False:
            for i in range(10):
                print('nop')
        del self.col[key]

    def __contains__(self, key: object) -> bool:
        if False:
            i = 10
            return i + 15
        return key in self.col

    def __iter__(self) -> Iterator[_KT]:
        if False:
            while True:
                i = 10
        return iter(self.col.keys())

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.col.clear()

    def __eq__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        return dict(self) == other

    def __ne__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        return dict(self) != other

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return repr(dict(self))

    @overload
    def get(self, __key: _KT, /) -> Optional[_VT]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def get(self, __key: _KT, /, default: Union[_VT, _T]) -> Union[_VT, _T]:
        if False:
            while True:
                i = 10
        ...

    def get(self, __key: _KT, /, default: Optional[Union[_VT, _T]]=None) -> Union[_VT, _T, None]:
        if False:
            while True:
                i = 10
        try:
            return self[__key]
        except KeyError:
            return default

    def setdefault(self, key: _KT, default: Optional[_VT]=None) -> _VT:
        if False:
            i = 10
            return i + 15
        if key not in self.col:
            self.col[key] = self._create(key, default)
            return default
        else:
            return self[key]

    def keys(self) -> KeysView[_KT]:
        if False:
            print('Hello World!')
        return self.col.keys()

    def items(self) -> ItemsView[_KT, _VT]:
        if False:
            while True:
                i = 10
        return ItemsView(self)

    def values(self) -> ValuesView[_VT]:
        if False:
            i = 10
            return i + 15
        return ValuesView(self)

    @overload
    def pop(self, __key: _KT, /) -> _VT:
        if False:
            print('Hello World!')
        ...

    @overload
    def pop(self, __key: _KT, /, default: Union[_VT, _T]=...) -> Union[_VT, _T]:
        if False:
            for i in range(10):
                print('nop')
        ...

    def pop(self, __key: _KT, /, *arg: Any, **kw: Any) -> Union[_VT, _T]:
        if False:
            while True:
                i = 10
        member = self.col.pop(__key, *arg, **kw)
        return self._get(member)

    def popitem(self) -> Tuple[_KT, _VT]:
        if False:
            print('Hello World!')
        item = self.col.popitem()
        return (item[0], self._get(item[1]))

    @overload
    def update(self, __m: SupportsKeysAndGetItem[_KT, _VT], **kwargs: _VT) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def update(self, __m: Iterable[tuple[_KT, _VT]], **kwargs: _VT) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def update(self, **kwargs: _VT) -> None:
        if False:
            i = 10
            return i + 15
        ...

    def update(self, *a: Any, **kw: Any) -> None:
        if False:
            i = 10
            return i + 15
        up: Dict[_KT, _VT] = {}
        up.update(*a, **kw)
        for (key, value) in up.items():
            self[key] = value

    def _bulk_replace(self, assoc_proxy: AssociationProxyInstance[Any], values: Mapping[_KT, _VT]) -> None:
        if False:
            i = 10
            return i + 15
        existing = set(self)
        constants = existing.intersection(values or ())
        additions = set(values or ()).difference(constants)
        removals = existing.difference(constants)
        for (key, member) in values.items() or ():
            if key in additions:
                self[key] = member
            elif key in constants:
                self[key] = member
        for key in removals:
            del self[key]

    def copy(self) -> Dict[_KT, _VT]:
        if False:
            print('Hello World!')
        return dict(self.items())

    def __hash__(self) -> NoReturn:
        if False:
            while True:
                i = 10
        raise TypeError('%s objects are unhashable' % type(self).__name__)
    if not typing.TYPE_CHECKING:
        for (func_name, func) in list(locals().items()):
            if callable(func) and func.__name__ == func_name and (not func.__doc__) and hasattr(dict, func_name):
                func.__doc__ = getattr(dict, func_name).__doc__
        del func_name, func

class _AssociationSet(_AssociationSingleItem[_T], MutableSet[_T]):
    """Generic, converting, set-to-set proxy."""
    col: MutableSet[_T]

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self.col)

    def __bool__(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if self.col:
            return True
        else:
            return False

    def __contains__(self, __o: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        for member in self.col:
            if self._get(member) == __o:
                return True
        return False

    def __iter__(self) -> Iterator[_T]:
        if False:
            print('Hello World!')
        'Iterate over proxied values.\n\n        For the actual domain objects, iterate over .col instead or just use\n        the underlying collection directly from its property on the parent.\n\n        '
        for member in self.col:
            yield self._get(member)
        return

    def add(self, __element: _T, /) -> None:
        if False:
            while True:
                i = 10
        if __element not in self:
            self.col.add(self._create(__element))

    def discard(self, __element: _T, /) -> None:
        if False:
            for i in range(10):
                print('nop')
        for member in self.col:
            if self._get(member) == __element:
                self.col.discard(member)
                break

    def remove(self, __element: _T, /) -> None:
        if False:
            i = 10
            return i + 15
        for member in self.col:
            if self._get(member) == __element:
                self.col.discard(member)
                return
        raise KeyError(__element)

    def pop(self) -> _T:
        if False:
            return 10
        if not self.col:
            raise KeyError('pop from an empty set')
        member = self.col.pop()
        return self._get(member)

    def update(self, *s: Iterable[_T]) -> None:
        if False:
            return 10
        for iterable in s:
            for value in iterable:
                self.add(value)

    def _bulk_replace(self, assoc_proxy: Any, values: Iterable[_T]) -> None:
        if False:
            i = 10
            return i + 15
        existing = set(self)
        constants = existing.intersection(values or ())
        additions = set(values or ()).difference(constants)
        removals = existing.difference(constants)
        appender = self.add
        remover = self.remove
        for member in values or ():
            if member in additions:
                appender(member)
            elif member in constants:
                appender(member)
        for member in removals:
            remover(member)

    def __ior__(self, other: AbstractSet[_S]) -> MutableSet[Union[_T, _S]]:
        if False:
            return 10
        if not collections._set_binops_check_strict(self, other):
            raise NotImplementedError()
        for value in other:
            self.add(value)
        return self

    def _set(self) -> Set[_T]:
        if False:
            i = 10
            return i + 15
        return set(iter(self))

    def union(self, *s: Iterable[_S]) -> MutableSet[Union[_T, _S]]:
        if False:
            while True:
                i = 10
        return set(self).union(*s)

    def __or__(self, __s: AbstractSet[_S]) -> MutableSet[Union[_T, _S]]:
        if False:
            print('Hello World!')
        return self.union(__s)

    def difference(self, *s: Iterable[Any]) -> MutableSet[_T]:
        if False:
            i = 10
            return i + 15
        return set(self).difference(*s)

    def __sub__(self, s: AbstractSet[Any]) -> MutableSet[_T]:
        if False:
            while True:
                i = 10
        return self.difference(s)

    def difference_update(self, *s: Iterable[Any]) -> None:
        if False:
            while True:
                i = 10
        for other in s:
            for value in other:
                self.discard(value)

    def __isub__(self, s: AbstractSet[Any]) -> Self:
        if False:
            while True:
                i = 10
        if not collections._set_binops_check_strict(self, s):
            raise NotImplementedError()
        for value in s:
            self.discard(value)
        return self

    def intersection(self, *s: Iterable[Any]) -> MutableSet[_T]:
        if False:
            print('Hello World!')
        return set(self).intersection(*s)

    def __and__(self, s: AbstractSet[Any]) -> MutableSet[_T]:
        if False:
            return 10
        return self.intersection(s)

    def intersection_update(self, *s: Iterable[Any]) -> None:
        if False:
            print('Hello World!')
        for other in s:
            (want, have) = (self.intersection(other), set(self))
            (remove, add) = (have - want, want - have)
            for value in remove:
                self.remove(value)
            for value in add:
                self.add(value)

    def __iand__(self, s: AbstractSet[Any]) -> Self:
        if False:
            print('Hello World!')
        if not collections._set_binops_check_strict(self, s):
            raise NotImplementedError()
        want = self.intersection(s)
        have: Set[_T] = set(self)
        (remove, add) = (have - want, want - have)
        for value in remove:
            self.remove(value)
        for value in add:
            self.add(value)
        return self

    def symmetric_difference(self, __s: Iterable[_T]) -> MutableSet[_T]:
        if False:
            return 10
        return set(self).symmetric_difference(__s)

    def __xor__(self, s: AbstractSet[_S]) -> MutableSet[Union[_T, _S]]:
        if False:
            return 10
        return self.symmetric_difference(s)

    def symmetric_difference_update(self, other: Iterable[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        (want, have) = (self.symmetric_difference(other), set(self))
        (remove, add) = (have - want, want - have)
        for value in remove:
            self.remove(value)
        for value in add:
            self.add(value)

    def __ixor__(self, other: AbstractSet[_S]) -> MutableSet[Union[_T, _S]]:
        if False:
            i = 10
            return i + 15
        if not collections._set_binops_check_strict(self, other):
            raise NotImplementedError()
        self.symmetric_difference_update(other)
        return self

    def issubset(self, __s: Iterable[Any]) -> bool:
        if False:
            while True:
                i = 10
        return set(self).issubset(__s)

    def issuperset(self, __s: Iterable[Any]) -> bool:
        if False:
            return 10
        return set(self).issuperset(__s)

    def clear(self) -> None:
        if False:
            return 10
        self.col.clear()

    def copy(self) -> AbstractSet[_T]:
        if False:
            while True:
                i = 10
        return set(self)

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        return set(self) == other

    def __ne__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return set(self) != other

    def __lt__(self, other: AbstractSet[Any]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return set(self) < other

    def __le__(self, other: AbstractSet[Any]) -> bool:
        if False:
            while True:
                i = 10
        return set(self) <= other

    def __gt__(self, other: AbstractSet[Any]) -> bool:
        if False:
            return 10
        return set(self) > other

    def __ge__(self, other: AbstractSet[Any]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return set(self) >= other

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return repr(set(self))

    def __hash__(self) -> NoReturn:
        if False:
            while True:
                i = 10
        raise TypeError('%s objects are unhashable' % type(self).__name__)
    if not typing.TYPE_CHECKING:
        for (func_name, func) in list(locals().items()):
            if callable(func) and func.__name__ == func_name and (not func.__doc__) and hasattr(set, func_name):
                func.__doc__ = getattr(set, func_name).__doc__
        del func_name, func