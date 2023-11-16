"""Public API functions and helpers for declarative."""
from __future__ import annotations
import itertools
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import clsregistry
from . import instrumentation
from . import interfaces
from . import mapperlib
from ._orm_constructors import composite
from ._orm_constructors import deferred
from ._orm_constructors import mapped_column
from ._orm_constructors import relationship
from ._orm_constructors import synonym
from .attributes import InstrumentedAttribute
from .base import _inspect_mapped_class
from .base import _is_mapped_class
from .base import Mapped
from .base import ORMDescriptor
from .decl_base import _add_attribute
from .decl_base import _as_declarative
from .decl_base import _ClassScanMapperConfig
from .decl_base import _declarative_constructor
from .decl_base import _DeferredMapperConfig
from .decl_base import _del_attribute
from .decl_base import _mapper
from .descriptor_props import Composite
from .descriptor_props import Synonym
from .descriptor_props import Synonym as _orm_synonym
from .mapper import Mapper
from .properties import MappedColumn
from .relationships import RelationshipProperty
from .state import InstanceState
from .. import exc
from .. import inspection
from .. import util
from ..sql import sqltypes
from ..sql.base import _NoArg
from ..sql.elements import SQLCoreOperations
from ..sql.schema import MetaData
from ..sql.selectable import FromClause
from ..util import hybridmethod
from ..util import hybridproperty
from ..util import typing as compat_typing
from ..util.typing import CallableReference
from ..util.typing import flatten_newtype
from ..util.typing import is_generic
from ..util.typing import is_literal
from ..util.typing import is_newtype
from ..util.typing import Literal
from ..util.typing import Self
if TYPE_CHECKING:
    from ._typing import _O
    from ._typing import _RegistryType
    from .decl_base import _DataclassArguments
    from .instrumentation import ClassManager
    from .interfaces import MapperProperty
    from .state import InstanceState
    from ..sql._typing import _TypeEngineArgument
    from ..sql.type_api import _MatchedOnType
_T = TypeVar('_T', bound=Any)
_TT = TypeVar('_TT', bound=Any)
_TypeAnnotationMapType = Mapping[Any, '_TypeEngineArgument[Any]']
_MutableTypeAnnotationMapType = Dict[Any, '_TypeEngineArgument[Any]']
_DeclaredAttrDecorated = Callable[..., Union[Mapped[_T], ORMDescriptor[_T], SQLCoreOperations[_T]]]

def has_inherited_table(cls: Type[_O]) -> bool:
    if False:
        print('Hello World!')
    'Given a class, return True if any of the classes it inherits from has a\n    mapped table, otherwise return False.\n\n    This is used in declarative mixins to build attributes that behave\n    differently for the base class vs. a subclass in an inheritance\n    hierarchy.\n\n    .. seealso::\n\n        :ref:`decl_mixin_inheritance`\n\n    '
    for class_ in cls.__mro__[1:]:
        if getattr(class_, '__table__', None) is not None:
            return True
    return False

class _DynamicAttributesType(type):

    def __setattr__(cls, key: str, value: Any) -> None:
        if False:
            while True:
                i = 10
        if '__mapper__' in cls.__dict__:
            _add_attribute(cls, key, value)
        else:
            type.__setattr__(cls, key, value)

    def __delattr__(cls, key: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if '__mapper__' in cls.__dict__:
            _del_attribute(cls, key)
        else:
            type.__delattr__(cls, key)

class DeclarativeAttributeIntercept(_DynamicAttributesType, inspection.Inspectable[Mapper[Any]]):
    """Metaclass that may be used in conjunction with the
    :class:`_orm.DeclarativeBase` class to support addition of class
    attributes dynamically.

    """

@compat_typing.dataclass_transform(field_specifiers=(MappedColumn, RelationshipProperty, Composite, Synonym, mapped_column, relationship, composite, synonym, deferred))
class DCTransformDeclarative(DeclarativeAttributeIntercept):
    """metaclass that includes @dataclass_transforms"""

class DeclarativeMeta(DeclarativeAttributeIntercept):
    metadata: MetaData
    registry: RegistryType

    def __init__(cls, classname: Any, bases: Any, dict_: Any, **kw: Any) -> None:
        if False:
            while True:
                i = 10
        dict_ = cls.__dict__
        reg = getattr(cls, '_sa_registry', None)
        if reg is None:
            reg = dict_.get('registry', None)
            if not isinstance(reg, registry):
                raise exc.InvalidRequestError("Declarative base class has no 'registry' attribute, or registry is not a sqlalchemy.orm.registry() object")
            else:
                cls._sa_registry = reg
        if not cls.__dict__.get('__abstract__', False):
            _as_declarative(reg, cls, dict_)
        type.__init__(cls, classname, bases, dict_)

def synonym_for(name: str, map_column: bool=False) -> Callable[[Callable[..., Any]], Synonym[Any]]:
    if False:
        return 10
    'Decorator that produces an :func:`_orm.synonym`\n    attribute in conjunction with a Python descriptor.\n\n    The function being decorated is passed to :func:`_orm.synonym` as the\n    :paramref:`.orm.synonym.descriptor` parameter::\n\n        class MyClass(Base):\n            __tablename__ = \'my_table\'\n\n            id = Column(Integer, primary_key=True)\n            _job_status = Column("job_status", String(50))\n\n            @synonym_for("job_status")\n            @property\n            def job_status(self):\n                return "Status: %s" % self._job_status\n\n    The :ref:`hybrid properties <mapper_hybrids>` feature of SQLAlchemy\n    is typically preferred instead of synonyms, which is a more legacy\n    feature.\n\n    .. seealso::\n\n        :ref:`synonyms` - Overview of synonyms\n\n        :func:`_orm.synonym` - the mapper-level function\n\n        :ref:`mapper_hybrids` - The Hybrid Attribute extension provides an\n        updated approach to augmenting attribute behavior more flexibly than\n        can be achieved with synonyms.\n\n    '

    def decorate(fn: Callable[..., Any]) -> Synonym[Any]:
        if False:
            for i in range(10):
                print('nop')
        return _orm_synonym(name, map_column=map_column, descriptor=fn)
    return decorate

class _declared_attr_common:

    def __init__(self, fn: Callable[..., Any], cascading: bool=False, quiet: bool=False):
        if False:
            while True:
                i = 10
        if isinstance(fn, classmethod):
            fn = fn.__func__
        self.fget = fn
        self._cascading = cascading
        self._quiet = quiet
        self.__doc__ = fn.__doc__

    def _collect_return_annotation(self) -> Optional[Type[Any]]:
        if False:
            return 10
        return util.get_annotations(self.fget).get('return')

    def __get__(self, instance: Optional[object], owner: Any) -> Any:
        if False:
            while True:
                i = 10
        cls = owner
        manager = attributes.opt_manager_of_class(cls)
        if manager is None:
            if not re.match('^__.+__$', self.fget.__name__):
                util.warn('Unmanaged access of declarative attribute %s from non-mapped class %s' % (self.fget.__name__, cls.__name__))
            return self.fget(cls)
        elif manager.is_mapped:
            return self.fget(cls)
        declarative_scan = manager.declarative_scan()
        assert declarative_scan is not None
        reg = declarative_scan.declared_attr_reg
        if self in reg:
            return reg[self]
        else:
            reg[self] = obj = self.fget(cls)
            return obj

class _declared_directive(_declared_attr_common, Generic[_T]):
    if typing.TYPE_CHECKING:

        def __init__(self, fn: Callable[..., _T], cascading: bool=False):
            if False:
                return 10
            ...

        def __get__(self, instance: Optional[object], owner: Any) -> _T:
            if False:
                for i in range(10):
                    print('nop')
            ...

        def __set__(self, instance: Any, value: Any) -> None:
            if False:
                while True:
                    i = 10
            ...

        def __delete__(self, instance: Any) -> None:
            if False:
                while True:
                    i = 10
            ...

        def __call__(self, fn: Callable[..., _TT]) -> _declared_directive[_TT]:
            if False:
                for i in range(10):
                    print('nop')
            ...

class declared_attr(interfaces._MappedAttribute[_T], _declared_attr_common):
    """Mark a class-level method as representing the definition of
    a mapped property or Declarative directive.

    :class:`_orm.declared_attr` is typically applied as a decorator to a class
    level method, turning the attribute into a scalar-like property that can be
    invoked from the uninstantiated class. The Declarative mapping process
    looks for these :class:`_orm.declared_attr` callables as it scans classes,
    and assumes any attribute marked with :class:`_orm.declared_attr` will be a
    callable that will produce an object specific to the Declarative mapping or
    table configuration.

    :class:`_orm.declared_attr` is usually applicable to
    :ref:`mixins <orm_mixins_toplevel>`, to define relationships that are to be
    applied to different implementors of the class. It may also be used to
    define dynamically generated column expressions and other Declarative
    attributes.

    Example::

        class ProvidesUserMixin:
            "A mixin that adds a 'user' relationship to classes."

            user_id: Mapped[int] = mapped_column(ForeignKey("user_table.id"))

            @declared_attr
            def user(cls) -> Mapped["User"]:
                return relationship("User")

    When used with Declarative directives such as ``__tablename__``, the
    :meth:`_orm.declared_attr.directive` modifier may be used which indicates
    to :pep:`484` typing tools that the given method is not dealing with
    :class:`_orm.Mapped` attributes::

        class CreateTableName:
            @declared_attr.directive
            def __tablename__(cls) -> str:
                return cls.__name__.lower()

    :class:`_orm.declared_attr` can also be applied directly to mapped
    classes, to allow for attributes that dynamically configure themselves
    on subclasses when using mapped inheritance schemes.   Below
    illustrates :class:`_orm.declared_attr` to create a dynamic scheme
    for generating the :paramref:`_orm.Mapper.polymorphic_identity` parameter
    for subclasses::

        class Employee(Base):
            __tablename__ = 'employee'

            id: Mapped[int] = mapped_column(primary_key=True)
            type: Mapped[str] = mapped_column(String(50))

            @declared_attr.directive
            def __mapper_args__(cls) -> Dict[str, Any]:
                if cls.__name__ == 'Employee':
                    return {
                            "polymorphic_on":cls.type,
                            "polymorphic_identity":"Employee"
                    }
                else:
                    return {"polymorphic_identity":cls.__name__}

        class Engineer(Employee):
            pass

    :class:`_orm.declared_attr` supports decorating functions that are
    explicitly decorated with ``@classmethod``. This is never necessary from a
    runtime perspective, however may be needed in order to support :pep:`484`
    typing tools that don't otherwise recognize the decorated function as
    having class-level behaviors for the ``cls`` parameter::

        class SomethingMixin:
            x: Mapped[int]
            y: Mapped[int]

            @declared_attr
            @classmethod
            def x_plus_y(cls) -> Mapped[int]:
                return column_property(cls.x + cls.y)

    .. versionadded:: 2.0 - :class:`_orm.declared_attr` can accommodate a
       function decorated with ``@classmethod`` to help with :pep:`484`
       integration where needed.


    .. seealso::

        :ref:`orm_mixins_toplevel` - Declarative Mixin documentation with
        background on use patterns for :class:`_orm.declared_attr`.

    """
    if typing.TYPE_CHECKING:

        def __init__(self, fn: _DeclaredAttrDecorated[_T], cascading: bool=False):
            if False:
                return 10
            ...

        def __set__(self, instance: Any, value: Any) -> None:
            if False:
                for i in range(10):
                    print('nop')
            ...

        def __delete__(self, instance: Any) -> None:
            if False:
                while True:
                    i = 10
            ...

        @overload
        def __get__(self, instance: None, owner: Any) -> InstrumentedAttribute[_T]:
            if False:
                while True:
                    i = 10
            ...

        @overload
        def __get__(self, instance: object, owner: Any) -> _T:
            if False:
                return 10
            ...

        def __get__(self, instance: Optional[object], owner: Any) -> Union[InstrumentedAttribute[_T], _T]:
            if False:
                return 10
            ...

    @hybridmethod
    def _stateful(cls, **kw: Any) -> _stateful_declared_attr[_T]:
        if False:
            while True:
                i = 10
        return _stateful_declared_attr(**kw)

    @hybridproperty
    def directive(cls) -> _declared_directive[Any]:
        if False:
            return 10
        return _declared_directive

    @hybridproperty
    def cascading(cls) -> _stateful_declared_attr[_T]:
        if False:
            print('Hello World!')
        return cls._stateful(cascading=True)

class _stateful_declared_attr(declared_attr[_T]):
    kw: Dict[str, Any]

    def __init__(self, **kw: Any):
        if False:
            i = 10
            return i + 15
        self.kw = kw

    @hybridmethod
    def _stateful(self, **kw: Any) -> _stateful_declared_attr[_T]:
        if False:
            for i in range(10):
                print('nop')
        new_kw = self.kw.copy()
        new_kw.update(kw)
        return _stateful_declared_attr(**new_kw)

    def __call__(self, fn: _DeclaredAttrDecorated[_T]) -> declared_attr[_T]:
        if False:
            i = 10
            return i + 15
        return declared_attr(fn, **self.kw)

def declarative_mixin(cls: Type[_T]) -> Type[_T]:
    if False:
        print('Hello World!')
    'Mark a class as providing the feature of "declarative mixin".\n\n    E.g.::\n\n        from sqlalchemy.orm import declared_attr\n        from sqlalchemy.orm import declarative_mixin\n\n        @declarative_mixin\n        class MyMixin:\n\n            @declared_attr\n            def __tablename__(cls):\n                return cls.__name__.lower()\n\n            __table_args__ = {\'mysql_engine\': \'InnoDB\'}\n            __mapper_args__= {\'always_refresh\': True}\n\n            id =  Column(Integer, primary_key=True)\n\n        class MyModel(MyMixin, Base):\n            name = Column(String(1000))\n\n    The :func:`_orm.declarative_mixin` decorator currently does not modify\n    the given class in any way; it\'s current purpose is strictly to assist\n    the :ref:`Mypy plugin <mypy_toplevel>` in being able to identify\n    SQLAlchemy declarative mixin classes when no other context is present.\n\n    .. versionadded:: 1.4.6\n\n    .. seealso::\n\n        :ref:`orm_mixins_toplevel`\n\n        :ref:`mypy_declarative_mixins` - in the\n        :ref:`Mypy plugin documentation <mypy_toplevel>`\n\n    '
    return cls

def _setup_declarative_base(cls: Type[Any]) -> None:
    if False:
        print('Hello World!')
    if 'metadata' in cls.__dict__:
        metadata = cls.__dict__['metadata']
    else:
        metadata = None
    if 'type_annotation_map' in cls.__dict__:
        type_annotation_map = cls.__dict__['type_annotation_map']
    else:
        type_annotation_map = None
    reg = cls.__dict__.get('registry', None)
    if reg is not None:
        if not isinstance(reg, registry):
            raise exc.InvalidRequestError("Declarative base class has a 'registry' attribute that is not an instance of sqlalchemy.orm.registry()")
        elif type_annotation_map is not None:
            raise exc.InvalidRequestError("Declarative base class has both a 'registry' attribute and a type_annotation_map entry.  Per-base type_annotation_maps are not supported.  Please apply the type_annotation_map to this registry directly.")
    else:
        reg = registry(metadata=metadata, type_annotation_map=type_annotation_map)
        cls.registry = reg
    cls._sa_registry = reg
    if 'metadata' not in cls.__dict__:
        cls.metadata = cls.registry.metadata
    if getattr(cls, '__init__', object.__init__) is object.__init__:
        cls.__init__ = cls.registry.constructor

class MappedAsDataclass(metaclass=DCTransformDeclarative):
    """Mixin class to indicate when mapping this class, also convert it to be
    a dataclass.

    .. seealso::

        :ref:`orm_declarative_native_dataclasses` - complete background
        on SQLAlchemy native dataclass mapping

    .. versionadded:: 2.0

    """

    def __init_subclass__(cls, init: Union[_NoArg, bool]=_NoArg.NO_ARG, repr: Union[_NoArg, bool]=_NoArg.NO_ARG, eq: Union[_NoArg, bool]=_NoArg.NO_ARG, order: Union[_NoArg, bool]=_NoArg.NO_ARG, unsafe_hash: Union[_NoArg, bool]=_NoArg.NO_ARG, match_args: Union[_NoArg, bool]=_NoArg.NO_ARG, kw_only: Union[_NoArg, bool]=_NoArg.NO_ARG, dataclass_callable: Union[_NoArg, Callable[..., Type[Any]]]=_NoArg.NO_ARG) -> None:
        if False:
            return 10
        apply_dc_transforms: _DataclassArguments = {'init': init, 'repr': repr, 'eq': eq, 'order': order, 'unsafe_hash': unsafe_hash, 'match_args': match_args, 'kw_only': kw_only, 'dataclass_callable': dataclass_callable}
        current_transforms: _DataclassArguments
        if hasattr(cls, '_sa_apply_dc_transforms'):
            current = cls._sa_apply_dc_transforms
            _ClassScanMapperConfig._assert_dc_arguments(current)
            cls._sa_apply_dc_transforms = current_transforms = {k: current.get(k, _NoArg.NO_ARG) if v is _NoArg.NO_ARG else v for (k, v) in apply_dc_transforms.items()}
        else:
            cls._sa_apply_dc_transforms = current_transforms = apply_dc_transforms
        super().__init_subclass__()
        if not _is_mapped_class(cls):
            new_anno = _ClassScanMapperConfig._update_annotations_for_non_mapped_class(cls)
            _ClassScanMapperConfig._apply_dataclasses_to_any_class(current_transforms, cls, new_anno)

class DeclarativeBase(inspection.Inspectable[InstanceState[Any]], metaclass=DeclarativeAttributeIntercept):
    """Base class used for declarative class definitions.

    The :class:`_orm.DeclarativeBase` allows for the creation of new
    declarative bases in such a way that is compatible with type checkers::


        from sqlalchemy.orm import DeclarativeBase

        class Base(DeclarativeBase):
            pass


    The above ``Base`` class is now usable as the base for new declarative
    mappings.  The superclass makes use of the ``__init_subclass__()``
    method to set up new classes and metaclasses aren't used.

    When first used, the :class:`_orm.DeclarativeBase` class instantiates a new
    :class:`_orm.registry` to be used with the base, assuming one was not
    provided explicitly. The :class:`_orm.DeclarativeBase` class supports
    class-level attributes which act as parameters for the construction of this
    registry; such as to indicate a specific :class:`_schema.MetaData`
    collection as well as a specific value for
    :paramref:`_orm.registry.type_annotation_map`::

        from typing_extensions import Annotated

        from sqlalchemy import BigInteger
        from sqlalchemy import MetaData
        from sqlalchemy import String
        from sqlalchemy.orm import DeclarativeBase

        bigint = Annotated[int, "bigint"]
        my_metadata = MetaData()

        class Base(DeclarativeBase):
            metadata = my_metadata
            type_annotation_map = {
                str: String().with_variant(String(255), "mysql", "mariadb"),
                bigint: BigInteger()
            }

    Class-level attributes which may be specified include:

    :param metadata: optional :class:`_schema.MetaData` collection.
     If a :class:`_orm.registry` is constructed automatically, this
     :class:`_schema.MetaData` collection will be used to construct it.
     Otherwise, the local :class:`_schema.MetaData` collection will supercede
     that used by an existing :class:`_orm.registry` passed using the
     :paramref:`_orm.DeclarativeBase.registry` parameter.
    :param type_annotation_map: optional type annotation map that will be
     passed to the :class:`_orm.registry` as
     :paramref:`_orm.registry.type_annotation_map`.
    :param registry: supply a pre-existing :class:`_orm.registry` directly.

    .. versionadded:: 2.0  Added :class:`.DeclarativeBase`, so that declarative
       base classes may be constructed in such a way that is also recognized
       by :pep:`484` type checkers.   As a result, :class:`.DeclarativeBase`
       and other subclassing-oriented APIs should be seen as
       superseding previous "class returned by a function" APIs, namely
       :func:`_orm.declarative_base` and :meth:`_orm.registry.generate_base`,
       where the base class returned cannot be recognized by type checkers
       without using plugins.

    **__init__ behavior**

    In a plain Python class, the base-most ``__init__()`` method in the class
    hierarchy is ``object.__init__()``, which accepts no arguments. However,
    when the :class:`_orm.DeclarativeBase` subclass is first declared, the
    class is given an ``__init__()`` method that links to the
    :paramref:`_orm.registry.constructor` constructor function, if no
    ``__init__()`` method is already present; this is the usual declarative
    constructor that will assign keyword arguments as attributes on the
    instance, assuming those attributes are established at the class level
    (i.e. are mapped, or are linked to a descriptor). This constructor is
    **never accessed by a mapped class without being called explicitly via
    super()**, as mapped classes are themselves given an ``__init__()`` method
    directly which calls :paramref:`_orm.registry.constructor`, so in the
    default case works independently of what the base-most ``__init__()``
    method does.

    .. versionchanged:: 2.0.1  :class:`_orm.DeclarativeBase` has a default
       constructor that links to :paramref:`_orm.registry.constructor` by
       default, so that calls to ``super().__init__()`` can access this
       constructor. Previously, due to an implementation mistake, this default
       constructor was missing, and calling ``super().__init__()`` would invoke
       ``object.__init__()``.

    The :class:`_orm.DeclarativeBase` subclass may also declare an explicit
    ``__init__()`` method which will replace the use of the
    :paramref:`_orm.registry.constructor` function at this level::

        class Base(DeclarativeBase):
            def __init__(self, id=None):
                self.id = id

    Mapped classes still will not invoke this constructor implicitly; it
    remains only accessible by calling ``super().__init__()``::

        class MyClass(Base):
            def __init__(self, id=None, name=None):
                self.name = name
                super().__init__(id=id)

    Note that this is a different behavior from what functions like the legacy
    :func:`_orm.declarative_base` would do; the base created by those functions
    would always install :paramref:`_orm.registry.constructor` for
    ``__init__()``.


    """
    if typing.TYPE_CHECKING:

        def _sa_inspect_type(self) -> Mapper[Self]:
            if False:
                return 10
            ...

        def _sa_inspect_instance(self) -> InstanceState[Self]:
            if False:
                while True:
                    i = 10
            ...
        _sa_registry: ClassVar[_RegistryType]
        registry: ClassVar[_RegistryType]
        'Refers to the :class:`_orm.registry` in use where new\n        :class:`_orm.Mapper` objects will be associated.'
        metadata: ClassVar[MetaData]
        'Refers to the :class:`_schema.MetaData` collection that will be used\n        for new :class:`_schema.Table` objects.\n\n        .. seealso::\n\n            :ref:`orm_declarative_metadata`\n\n        '
        __name__: ClassVar[str]
        __mapper__: ClassVar[Mapper[Any]]
        'The :class:`_orm.Mapper` object to which a particular class is\n        mapped.\n\n        May also be acquired using :func:`_sa.inspect`, e.g.\n        ``inspect(klass)``.\n\n        '
        __table__: ClassVar[FromClause]
        'The :class:`_sql.FromClause` to which a particular subclass is\n        mapped.\n\n        This is usually an instance of :class:`_schema.Table` but may also\n        refer to other kinds of :class:`_sql.FromClause` such as\n        :class:`_sql.Subquery`, depending on how the class is mapped.\n\n        .. seealso::\n\n            :ref:`orm_declarative_metadata`\n\n        '
        __tablename__: Any
        'String name to assign to the generated\n        :class:`_schema.Table` object, if not specified directly via\n        :attr:`_orm.DeclarativeBase.__table__`.\n\n        .. seealso::\n\n            :ref:`orm_declarative_table`\n\n        '
        __mapper_args__: Any
        'Dictionary of arguments which will be passed to the\n        :class:`_orm.Mapper` constructor.\n\n        .. seealso::\n\n            :ref:`orm_declarative_mapper_options`\n\n        '
        __table_args__: Any
        'A dictionary or tuple of arguments that will be passed to the\n        :class:`_schema.Table` constructor.  See\n        :ref:`orm_declarative_table_configuration`\n        for background on the specific structure of this collection.\n\n        .. seealso::\n\n            :ref:`orm_declarative_table_configuration`\n\n        '

        def __init__(self, **kw: Any):
            if False:
                return 10
            ...

    def __init_subclass__(cls) -> None:
        if False:
            while True:
                i = 10
        if DeclarativeBase in cls.__bases__:
            _check_not_declarative(cls, DeclarativeBase)
            _setup_declarative_base(cls)
        else:
            _as_declarative(cls._sa_registry, cls, cls.__dict__)
        super().__init_subclass__()

def _check_not_declarative(cls: Type[Any], base: Type[Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    cls_dict = cls.__dict__
    if '__table__' in cls_dict and (not (callable(cls_dict['__table__']) or hasattr(cls_dict['__table__'], '__get__'))) or isinstance(cls_dict.get('__tablename__', None), str):
        raise exc.InvalidRequestError(f'Cannot use {base.__name__!r} directly as a declarative base class. Create a Base by creating a subclass of it.')

class DeclarativeBaseNoMeta(inspection.Inspectable[InstanceState[Any]]):
    """Same as :class:`_orm.DeclarativeBase`, but does not use a metaclass
    to intercept new attributes.

    The :class:`_orm.DeclarativeBaseNoMeta` base may be used when use of
    custom metaclasses is desirable.

    .. versionadded:: 2.0


    """
    _sa_registry: ClassVar[_RegistryType]
    registry: ClassVar[_RegistryType]
    'Refers to the :class:`_orm.registry` in use where new\n    :class:`_orm.Mapper` objects will be associated.'
    metadata: ClassVar[MetaData]
    'Refers to the :class:`_schema.MetaData` collection that will be used\n    for new :class:`_schema.Table` objects.\n\n    .. seealso::\n\n        :ref:`orm_declarative_metadata`\n\n    '
    __mapper__: ClassVar[Mapper[Any]]
    'The :class:`_orm.Mapper` object to which a particular class is\n    mapped.\n\n    May also be acquired using :func:`_sa.inspect`, e.g.\n    ``inspect(klass)``.\n\n    '
    __table__: Optional[FromClause]
    'The :class:`_sql.FromClause` to which a particular subclass is\n    mapped.\n\n    This is usually an instance of :class:`_schema.Table` but may also\n    refer to other kinds of :class:`_sql.FromClause` such as\n    :class:`_sql.Subquery`, depending on how the class is mapped.\n\n    .. seealso::\n\n        :ref:`orm_declarative_metadata`\n\n    '
    if typing.TYPE_CHECKING:

        def _sa_inspect_type(self) -> Mapper[Self]:
            if False:
                i = 10
                return i + 15
            ...

        def _sa_inspect_instance(self) -> InstanceState[Self]:
            if False:
                return 10
            ...
        __tablename__: Any
        'String name to assign to the generated\n        :class:`_schema.Table` object, if not specified directly via\n        :attr:`_orm.DeclarativeBase.__table__`.\n\n        .. seealso::\n\n            :ref:`orm_declarative_table`\n\n        '
        __mapper_args__: Any
        'Dictionary of arguments which will be passed to the\n        :class:`_orm.Mapper` constructor.\n\n        .. seealso::\n\n            :ref:`orm_declarative_mapper_options`\n\n        '
        __table_args__: Any
        'A dictionary or tuple of arguments that will be passed to the\n        :class:`_schema.Table` constructor.  See\n        :ref:`orm_declarative_table_configuration`\n        for background on the specific structure of this collection.\n\n        .. seealso::\n\n            :ref:`orm_declarative_table_configuration`\n\n        '

        def __init__(self, **kw: Any):
            if False:
                for i in range(10):
                    print('nop')
            ...

    def __init_subclass__(cls) -> None:
        if False:
            while True:
                i = 10
        if DeclarativeBaseNoMeta in cls.__bases__:
            _check_not_declarative(cls, DeclarativeBaseNoMeta)
            _setup_declarative_base(cls)
        else:
            _as_declarative(cls._sa_registry, cls, cls.__dict__)

def add_mapped_attribute(target: Type[_O], key: str, attr: MapperProperty[Any]) -> None:
    if False:
        i = 10
        return i + 15
    'Add a new mapped attribute to an ORM mapped class.\n\n    E.g.::\n\n        add_mapped_attribute(User, "addresses", relationship(Address))\n\n    This may be used for ORM mappings that aren\'t using a declarative\n    metaclass that intercepts attribute set operations.\n\n    .. versionadded:: 2.0\n\n\n    '
    _add_attribute(target, key, attr)

def declarative_base(*, metadata: Optional[MetaData]=None, mapper: Optional[Callable[..., Mapper[Any]]]=None, cls: Type[Any]=object, name: str='Base', class_registry: Optional[clsregistry._ClsRegistryType]=None, type_annotation_map: Optional[_TypeAnnotationMapType]=None, constructor: Callable[..., None]=_declarative_constructor, metaclass: Type[Any]=DeclarativeMeta) -> Any:
    if False:
        while True:
            i = 10
    'Construct a base class for declarative class definitions.\n\n    The new base class will be given a metaclass that produces\n    appropriate :class:`~sqlalchemy.schema.Table` objects and makes\n    the appropriate :class:`_orm.Mapper` calls based on the\n    information provided declaratively in the class and any subclasses\n    of the class.\n\n    .. versionchanged:: 2.0 Note that the :func:`_orm.declarative_base`\n       function is superseded by the new :class:`_orm.DeclarativeBase` class,\n       which generates a new "base" class using subclassing, rather than\n       return value of a function.  This allows an approach that is compatible\n       with :pep:`484` typing tools.\n\n    The :func:`_orm.declarative_base` function is a shorthand version\n    of using the :meth:`_orm.registry.generate_base`\n    method.  That is, the following::\n\n        from sqlalchemy.orm import declarative_base\n\n        Base = declarative_base()\n\n    Is equivalent to::\n\n        from sqlalchemy.orm import registry\n\n        mapper_registry = registry()\n        Base = mapper_registry.generate_base()\n\n    See the docstring for :class:`_orm.registry`\n    and :meth:`_orm.registry.generate_base`\n    for more details.\n\n    .. versionchanged:: 1.4  The :func:`_orm.declarative_base`\n       function is now a specialization of the more generic\n       :class:`_orm.registry` class.  The function also moves to the\n       ``sqlalchemy.orm`` package from the ``declarative.ext`` package.\n\n\n    :param metadata:\n      An optional :class:`~sqlalchemy.schema.MetaData` instance.  All\n      :class:`~sqlalchemy.schema.Table` objects implicitly declared by\n      subclasses of the base will share this MetaData.  A MetaData instance\n      will be created if none is provided.  The\n      :class:`~sqlalchemy.schema.MetaData` instance will be available via the\n      ``metadata`` attribute of the generated declarative base class.\n\n    :param mapper:\n      An optional callable, defaults to :class:`_orm.Mapper`. Will\n      be used to map subclasses to their Tables.\n\n    :param cls:\n      Defaults to :class:`object`. A type to use as the base for the generated\n      declarative base class. May be a class or tuple of classes.\n\n    :param name:\n      Defaults to ``Base``.  The display name for the generated\n      class.  Customizing this is not required, but can improve clarity in\n      tracebacks and debugging.\n\n    :param constructor:\n      Specify the implementation for the ``__init__`` function on a mapped\n      class that has no ``__init__`` of its own.  Defaults to an\n      implementation that assigns \\**kwargs for declared\n      fields and relationships to an instance.  If ``None`` is supplied,\n      no __init__ will be provided and construction will fall back to\n      cls.__init__ by way of the normal Python semantics.\n\n    :param class_registry: optional dictionary that will serve as the\n      registry of class names-> mapped classes when string names\n      are used to identify classes inside of :func:`_orm.relationship`\n      and others.  Allows two or more declarative base classes\n      to share the same registry of class names for simplified\n      inter-base relationships.\n\n    :param type_annotation_map: optional dictionary of Python types to\n        SQLAlchemy :class:`_types.TypeEngine` classes or instances.  This\n        is used exclusively by the :class:`_orm.MappedColumn` construct\n        to produce column types based on annotations within the\n        :class:`_orm.Mapped` type.\n\n\n        .. versionadded:: 2.0\n\n        .. seealso::\n\n            :ref:`orm_declarative_mapped_column_type_map`\n\n    :param metaclass:\n      Defaults to :class:`.DeclarativeMeta`.  A metaclass or __metaclass__\n      compatible callable to use as the meta type of the generated\n      declarative base class.\n\n    .. seealso::\n\n        :class:`_orm.registry`\n\n    '
    return registry(metadata=metadata, class_registry=class_registry, constructor=constructor, type_annotation_map=type_annotation_map).generate_base(mapper=mapper, cls=cls, name=name, metaclass=metaclass)

class registry:
    """Generalized registry for mapping classes.

    The :class:`_orm.registry` serves as the basis for maintaining a collection
    of mappings, and provides configurational hooks used to map classes.

    The three general kinds of mappings supported are Declarative Base,
    Declarative Decorator, and Imperative Mapping.   All of these mapping
    styles may be used interchangeably:

    * :meth:`_orm.registry.generate_base` returns a new declarative base
      class, and is the underlying implementation of the
      :func:`_orm.declarative_base` function.

    * :meth:`_orm.registry.mapped` provides a class decorator that will
      apply declarative mapping to a class without the use of a declarative
      base class.

    * :meth:`_orm.registry.map_imperatively` will produce a
      :class:`_orm.Mapper` for a class without scanning the class for
      declarative class attributes. This method suits the use case historically
      provided by the ``sqlalchemy.orm.mapper()`` classical mapping function,
      which is removed as of SQLAlchemy 2.0.

    .. versionadded:: 1.4

    .. seealso::

        :ref:`orm_mapping_classes_toplevel` - overview of class mapping
        styles.

    """
    _class_registry: clsregistry._ClsRegistryType
    _managers: weakref.WeakKeyDictionary[ClassManager[Any], Literal[True]]
    _non_primary_mappers: weakref.WeakKeyDictionary[Mapper[Any], Literal[True]]
    metadata: MetaData
    constructor: CallableReference[Callable[..., None]]
    type_annotation_map: _MutableTypeAnnotationMapType
    _dependents: Set[_RegistryType]
    _dependencies: Set[_RegistryType]
    _new_mappers: bool

    def __init__(self, *, metadata: Optional[MetaData]=None, class_registry: Optional[clsregistry._ClsRegistryType]=None, type_annotation_map: Optional[_TypeAnnotationMapType]=None, constructor: Callable[..., None]=_declarative_constructor):
        if False:
            for i in range(10):
                print('nop')
        'Construct a new :class:`_orm.registry`\n\n        :param metadata:\n          An optional :class:`_schema.MetaData` instance.  All\n          :class:`_schema.Table` objects generated using declarative\n          table mapping will make use of this :class:`_schema.MetaData`\n          collection.  If this argument is left at its default of ``None``,\n          a blank :class:`_schema.MetaData` collection is created.\n\n        :param constructor:\n          Specify the implementation for the ``__init__`` function on a mapped\n          class that has no ``__init__`` of its own.  Defaults to an\n          implementation that assigns \\**kwargs for declared\n          fields and relationships to an instance.  If ``None`` is supplied,\n          no __init__ will be provided and construction will fall back to\n          cls.__init__ by way of the normal Python semantics.\n\n        :param class_registry: optional dictionary that will serve as the\n          registry of class names-> mapped classes when string names\n          are used to identify classes inside of :func:`_orm.relationship`\n          and others.  Allows two or more declarative base classes\n          to share the same registry of class names for simplified\n          inter-base relationships.\n\n        :param type_annotation_map: optional dictionary of Python types to\n          SQLAlchemy :class:`_types.TypeEngine` classes or instances.\n          The provided dict will update the default type mapping.  This\n          is used exclusively by the :class:`_orm.MappedColumn` construct\n          to produce column types based on annotations within the\n          :class:`_orm.Mapped` type.\n\n          .. versionadded:: 2.0\n\n          .. seealso::\n\n              :ref:`orm_declarative_mapped_column_type_map`\n\n\n        '
        lcl_metadata = metadata or MetaData()
        if class_registry is None:
            class_registry = weakref.WeakValueDictionary()
        self._class_registry = class_registry
        self._managers = weakref.WeakKeyDictionary()
        self._non_primary_mappers = weakref.WeakKeyDictionary()
        self.metadata = lcl_metadata
        self.constructor = constructor
        self.type_annotation_map = {}
        if type_annotation_map is not None:
            self.update_type_annotation_map(type_annotation_map)
        self._dependents = set()
        self._dependencies = set()
        self._new_mappers = False
        with mapperlib._CONFIGURE_MUTEX:
            mapperlib._mapper_registries[self] = True

    def update_type_annotation_map(self, type_annotation_map: _TypeAnnotationMapType) -> None:
        if False:
            return 10
        'update the :paramref:`_orm.registry.type_annotation_map` with new\n        values.'
        self.type_annotation_map.update({sub_type: sqltype for (typ, sqltype) in type_annotation_map.items() for sub_type in compat_typing.expand_unions(typ, include_union=True, discard_none=True)})

    def _resolve_type(self, python_type: _MatchedOnType) -> Optional[sqltypes.TypeEngine[Any]]:
        if False:
            print('Hello World!')
        search: Iterable[Tuple[_MatchedOnType, Type[Any]]]
        python_type_type: Type[Any]
        if is_generic(python_type):
            if is_literal(python_type):
                python_type_type = cast('Type[Any]', python_type)
                search = ((python_type, python_type_type), (Literal, python_type_type))
            else:
                python_type_type = python_type.__origin__
                search = ((python_type, python_type_type),)
        elif is_newtype(python_type):
            python_type_type = flatten_newtype(python_type)
            search = ((python_type, python_type_type),)
        else:
            python_type_type = cast('Type[Any]', python_type)
            flattened = None
            search = ((pt, pt) for pt in python_type_type.__mro__)
        for (pt, flattened) in search:
            sql_type = self.type_annotation_map.get(pt)
            if sql_type is None:
                sql_type = sqltypes._type_map_get(pt)
            if sql_type is not None:
                sql_type_inst = sqltypes.to_instance(sql_type)
                resolved_sql_type = sql_type_inst._resolve_for_python_type(python_type_type, pt, flattened)
                if resolved_sql_type is not None:
                    return resolved_sql_type
        return None

    @property
    def mappers(self) -> FrozenSet[Mapper[Any]]:
        if False:
            for i in range(10):
                print('nop')
        'read only collection of all :class:`_orm.Mapper` objects.'
        return frozenset((manager.mapper for manager in self._managers)).union(self._non_primary_mappers)

    def _set_depends_on(self, registry: RegistryType) -> None:
        if False:
            for i in range(10):
                print('nop')
        if registry is self:
            return
        registry._dependents.add(self)
        self._dependencies.add(registry)

    def _flag_new_mapper(self, mapper: Mapper[Any]) -> None:
        if False:
            i = 10
            return i + 15
        mapper._ready_for_configure = True
        if self._new_mappers:
            return
        for reg in self._recurse_with_dependents({self}):
            reg._new_mappers = True

    @classmethod
    def _recurse_with_dependents(cls, registries: Set[RegistryType]) -> Iterator[RegistryType]:
        if False:
            print('Hello World!')
        todo = registries
        done = set()
        while todo:
            reg = todo.pop()
            done.add(reg)
            todo.update(reg._dependents.difference(done))
            yield reg
            todo.update(reg._dependents.difference(done))

    @classmethod
    def _recurse_with_dependencies(cls, registries: Set[RegistryType]) -> Iterator[RegistryType]:
        if False:
            i = 10
            return i + 15
        todo = registries
        done = set()
        while todo:
            reg = todo.pop()
            done.add(reg)
            todo.update(reg._dependencies.difference(done))
            yield reg
            todo.update(reg._dependencies.difference(done))

    def _mappers_to_configure(self) -> Iterator[Mapper[Any]]:
        if False:
            print('Hello World!')
        return itertools.chain((manager.mapper for manager in list(self._managers) if manager.is_mapped and (not manager.mapper.configured) and manager.mapper._ready_for_configure), (npm for npm in list(self._non_primary_mappers) if not npm.configured and npm._ready_for_configure))

    def _add_non_primary_mapper(self, np_mapper: Mapper[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._non_primary_mappers[np_mapper] = True

    def _dispose_cls(self, cls: Type[_O]) -> None:
        if False:
            return 10
        clsregistry.remove_class(cls.__name__, cls, self._class_registry)

    def _add_manager(self, manager: ClassManager[Any]) -> None:
        if False:
            print('Hello World!')
        self._managers[manager] = True
        if manager.is_mapped:
            raise exc.ArgumentError("Class '%s' already has a primary mapper defined. " % manager.class_)
        assert manager.registry is None
        manager.registry = self

    def configure(self, cascade: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Configure all as-yet unconfigured mappers in this\n        :class:`_orm.registry`.\n\n        The configure step is used to reconcile and initialize the\n        :func:`_orm.relationship` linkages between mapped classes, as well as\n        to invoke configuration events such as the\n        :meth:`_orm.MapperEvents.before_configured` and\n        :meth:`_orm.MapperEvents.after_configured`, which may be used by ORM\n        extensions or user-defined extension hooks.\n\n        If one or more mappers in this registry contain\n        :func:`_orm.relationship` constructs that refer to mapped classes in\n        other registries, this registry is said to be *dependent* on those\n        registries. In order to configure those dependent registries\n        automatically, the :paramref:`_orm.registry.configure.cascade` flag\n        should be set to ``True``. Otherwise, if they are not configured, an\n        exception will be raised.  The rationale behind this behavior is to\n        allow an application to programmatically invoke configuration of\n        registries while controlling whether or not the process implicitly\n        reaches other registries.\n\n        As an alternative to invoking :meth:`_orm.registry.configure`, the ORM\n        function :func:`_orm.configure_mappers` function may be used to ensure\n        configuration is complete for all :class:`_orm.registry` objects in\n        memory. This is generally simpler to use and also predates the usage of\n        :class:`_orm.registry` objects overall. However, this function will\n        impact all mappings throughout the running Python process and may be\n        more memory/time consuming for an application that has many registries\n        in use for different purposes that may not be needed immediately.\n\n        .. seealso::\n\n            :func:`_orm.configure_mappers`\n\n\n        .. versionadded:: 1.4.0b2\n\n        '
        mapperlib._configure_registries({self}, cascade=cascade)

    def dispose(self, cascade: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Dispose of all mappers in this :class:`_orm.registry`.\n\n        After invocation, all the classes that were mapped within this registry\n        will no longer have class instrumentation associated with them. This\n        method is the per-:class:`_orm.registry` analogue to the\n        application-wide :func:`_orm.clear_mappers` function.\n\n        If this registry contains mappers that are dependencies of other\n        registries, typically via :func:`_orm.relationship` links, then those\n        registries must be disposed as well. When such registries exist in\n        relation to this one, their :meth:`_orm.registry.dispose` method will\n        also be called, if the :paramref:`_orm.registry.dispose.cascade` flag\n        is set to ``True``; otherwise, an error is raised if those registries\n        were not already disposed.\n\n        .. versionadded:: 1.4.0b2\n\n        .. seealso::\n\n            :func:`_orm.clear_mappers`\n\n        '
        mapperlib._dispose_registries({self}, cascade=cascade)

    def _dispose_manager_and_mapper(self, manager: ClassManager[Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'mapper' in manager.__dict__:
            mapper = manager.mapper
            mapper._set_dispose_flags()
        class_ = manager.class_
        self._dispose_cls(class_)
        instrumentation._instrumentation_factory.unregister(class_)

    def generate_base(self, mapper: Optional[Callable[..., Mapper[Any]]]=None, cls: Type[Any]=object, name: str='Base', metaclass: Type[Any]=DeclarativeMeta) -> Any:
        if False:
            i = 10
            return i + 15
        'Generate a declarative base class.\n\n        Classes that inherit from the returned class object will be\n        automatically mapped using declarative mapping.\n\n        E.g.::\n\n            from sqlalchemy.orm import registry\n\n            mapper_registry = registry()\n\n            Base = mapper_registry.generate_base()\n\n            class MyClass(Base):\n                __tablename__ = "my_table"\n                id = Column(Integer, primary_key=True)\n\n        The above dynamically generated class is equivalent to the\n        non-dynamic example below::\n\n            from sqlalchemy.orm import registry\n            from sqlalchemy.orm.decl_api import DeclarativeMeta\n\n            mapper_registry = registry()\n\n            class Base(metaclass=DeclarativeMeta):\n                __abstract__ = True\n                registry = mapper_registry\n                metadata = mapper_registry.metadata\n\n                __init__ = mapper_registry.constructor\n\n        .. versionchanged:: 2.0 Note that the\n           :meth:`_orm.registry.generate_base` method is superseded by the new\n           :class:`_orm.DeclarativeBase` class, which generates a new "base"\n           class using subclassing, rather than return value of a function.\n           This allows an approach that is compatible with :pep:`484` typing\n           tools.\n\n        The :meth:`_orm.registry.generate_base` method provides the\n        implementation for the :func:`_orm.declarative_base` function, which\n        creates the :class:`_orm.registry` and base class all at once.\n\n        See the section :ref:`orm_declarative_mapping` for background and\n        examples.\n\n        :param mapper:\n          An optional callable, defaults to :class:`_orm.Mapper`.\n          This function is used to generate new :class:`_orm.Mapper` objects.\n\n        :param cls:\n          Defaults to :class:`object`. A type to use as the base for the\n          generated declarative base class. May be a class or tuple of classes.\n\n        :param name:\n          Defaults to ``Base``.  The display name for the generated\n          class.  Customizing this is not required, but can improve clarity in\n          tracebacks and debugging.\n\n        :param metaclass:\n          Defaults to :class:`.DeclarativeMeta`.  A metaclass or __metaclass__\n          compatible callable to use as the meta type of the generated\n          declarative base class.\n\n        .. seealso::\n\n            :ref:`orm_declarative_mapping`\n\n            :func:`_orm.declarative_base`\n\n        '
        metadata = self.metadata
        bases = not isinstance(cls, tuple) and (cls,) or cls
        class_dict: Dict[str, Any] = dict(registry=self, metadata=metadata)
        if isinstance(cls, type):
            class_dict['__doc__'] = cls.__doc__
        if self.constructor is not None:
            class_dict['__init__'] = self.constructor
        class_dict['__abstract__'] = True
        if mapper:
            class_dict['__mapper_cls__'] = mapper
        if hasattr(cls, '__class_getitem__'):

            def __class_getitem__(cls: Type[_T], key: Any) -> Type[_T]:
                if False:
                    i = 10
                    return i + 15
                return cls
            class_dict['__class_getitem__'] = __class_getitem__
        return metaclass(name, bases, class_dict)

    @compat_typing.dataclass_transform(field_specifiers=(MappedColumn, RelationshipProperty, Composite, Synonym, mapped_column, relationship, composite, synonym, deferred))
    @overload
    def mapped_as_dataclass(self, __cls: Type[_O], /) -> Type[_O]:
        if False:
            return 10
        ...

    @overload
    def mapped_as_dataclass(self, __cls: Literal[None]=..., /, *, init: Union[_NoArg, bool]=..., repr: Union[_NoArg, bool]=..., eq: Union[_NoArg, bool]=..., order: Union[_NoArg, bool]=..., unsafe_hash: Union[_NoArg, bool]=..., match_args: Union[_NoArg, bool]=..., kw_only: Union[_NoArg, bool]=..., dataclass_callable: Union[_NoArg, Callable[..., Type[Any]]]=...) -> Callable[[Type[_O]], Type[_O]]:
        if False:
            while True:
                i = 10
        ...

    def mapped_as_dataclass(self, __cls: Optional[Type[_O]]=None, /, *, init: Union[_NoArg, bool]=_NoArg.NO_ARG, repr: Union[_NoArg, bool]=_NoArg.NO_ARG, eq: Union[_NoArg, bool]=_NoArg.NO_ARG, order: Union[_NoArg, bool]=_NoArg.NO_ARG, unsafe_hash: Union[_NoArg, bool]=_NoArg.NO_ARG, match_args: Union[_NoArg, bool]=_NoArg.NO_ARG, kw_only: Union[_NoArg, bool]=_NoArg.NO_ARG, dataclass_callable: Union[_NoArg, Callable[..., Type[Any]]]=_NoArg.NO_ARG) -> Union[Type[_O], Callable[[Type[_O]], Type[_O]]]:
        if False:
            for i in range(10):
                print('nop')
        'Class decorator that will apply the Declarative mapping process\n        to a given class, and additionally convert the class to be a\n        Python dataclass.\n\n        .. seealso::\n\n            :ref:`orm_declarative_native_dataclasses` - complete background\n            on SQLAlchemy native dataclass mapping\n\n\n        .. versionadded:: 2.0\n\n\n        '

        def decorate(cls: Type[_O]) -> Type[_O]:
            if False:
                for i in range(10):
                    print('nop')
            setattr(cls, '_sa_apply_dc_transforms', {'init': init, 'repr': repr, 'eq': eq, 'order': order, 'unsafe_hash': unsafe_hash, 'match_args': match_args, 'kw_only': kw_only, 'dataclass_callable': dataclass_callable})
            _as_declarative(self, cls, cls.__dict__)
            return cls
        if __cls:
            return decorate(__cls)
        else:
            return decorate

    def mapped(self, cls: Type[_O]) -> Type[_O]:
        if False:
            return 10
        "Class decorator that will apply the Declarative mapping process\n        to a given class.\n\n        E.g.::\n\n            from sqlalchemy.orm import registry\n\n            mapper_registry = registry()\n\n            @mapper_registry.mapped\n            class Foo:\n                __tablename__ = 'some_table'\n\n                id = Column(Integer, primary_key=True)\n                name = Column(String)\n\n        See the section :ref:`orm_declarative_mapping` for complete\n        details and examples.\n\n        :param cls: class to be mapped.\n\n        :return: the class that was passed.\n\n        .. seealso::\n\n            :ref:`orm_declarative_mapping`\n\n            :meth:`_orm.registry.generate_base` - generates a base class\n            that will apply Declarative mapping to subclasses automatically\n            using a Python metaclass.\n\n        .. seealso::\n\n            :meth:`_orm.registry.mapped_as_dataclass`\n\n        "
        _as_declarative(self, cls, cls.__dict__)
        return cls

    def as_declarative_base(self, **kw: Any) -> Callable[[Type[_T]], Type[_T]]:
        if False:
            i = 10
            return i + 15
        '\n        Class decorator which will invoke\n        :meth:`_orm.registry.generate_base`\n        for a given base class.\n\n        E.g.::\n\n            from sqlalchemy.orm import registry\n\n            mapper_registry = registry()\n\n            @mapper_registry.as_declarative_base()\n            class Base:\n                @declared_attr\n                def __tablename__(cls):\n                    return cls.__name__.lower()\n                id = Column(Integer, primary_key=True)\n\n            class MyMappedClass(Base):\n                # ...\n\n        All keyword arguments passed to\n        :meth:`_orm.registry.as_declarative_base` are passed\n        along to :meth:`_orm.registry.generate_base`.\n\n        '

        def decorate(cls: Type[_T]) -> Type[_T]:
            if False:
                while True:
                    i = 10
            kw['cls'] = cls
            kw['name'] = cls.__name__
            return self.generate_base(**kw)
        return decorate

    def map_declaratively(self, cls: Type[_O]) -> Mapper[_O]:
        if False:
            print('Hello World!')
        "Map a class declaratively.\n\n        In this form of mapping, the class is scanned for mapping information,\n        including for columns to be associated with a table, and/or an\n        actual table object.\n\n        Returns the :class:`_orm.Mapper` object.\n\n        E.g.::\n\n            from sqlalchemy.orm import registry\n\n            mapper_registry = registry()\n\n            class Foo:\n                __tablename__ = 'some_table'\n\n                id = Column(Integer, primary_key=True)\n                name = Column(String)\n\n            mapper = mapper_registry.map_declaratively(Foo)\n\n        This function is more conveniently invoked indirectly via either the\n        :meth:`_orm.registry.mapped` class decorator or by subclassing a\n        declarative metaclass generated from\n        :meth:`_orm.registry.generate_base`.\n\n        See the section :ref:`orm_declarative_mapping` for complete\n        details and examples.\n\n        :param cls: class to be mapped.\n\n        :return: a :class:`_orm.Mapper` object.\n\n        .. seealso::\n\n            :ref:`orm_declarative_mapping`\n\n            :meth:`_orm.registry.mapped` - more common decorator interface\n            to this function.\n\n            :meth:`_orm.registry.map_imperatively`\n\n        "
        _as_declarative(self, cls, cls.__dict__)
        return cls.__mapper__

    def map_imperatively(self, class_: Type[_O], local_table: Optional[FromClause]=None, **kw: Any) -> Mapper[_O]:
        if False:
            return 10
        'Map a class imperatively.\n\n        In this form of mapping, the class is not scanned for any mapping\n        information.  Instead, all mapping constructs are passed as\n        arguments.\n\n        This method is intended to be fully equivalent to the now-removed\n        SQLAlchemy ``mapper()`` function, except that it\'s in terms of\n        a particular registry.\n\n        E.g.::\n\n            from sqlalchemy.orm import registry\n\n            mapper_registry = registry()\n\n            my_table = Table(\n                "my_table",\n                mapper_registry.metadata,\n                Column(\'id\', Integer, primary_key=True)\n            )\n\n            class MyClass:\n                pass\n\n            mapper_registry.map_imperatively(MyClass, my_table)\n\n        See the section :ref:`orm_imperative_mapping` for complete background\n        and usage examples.\n\n        :param class\\_: The class to be mapped.  Corresponds to the\n         :paramref:`_orm.Mapper.class_` parameter.\n\n        :param local_table: the :class:`_schema.Table` or other\n         :class:`_sql.FromClause` object that is the subject of the mapping.\n         Corresponds to the\n         :paramref:`_orm.Mapper.local_table` parameter.\n\n        :param \\**kw: all other keyword arguments are passed to the\n         :class:`_orm.Mapper` constructor directly.\n\n        .. seealso::\n\n            :ref:`orm_imperative_mapping`\n\n            :ref:`orm_declarative_mapping`\n\n        '
        return _mapper(self, class_, local_table, kw)
RegistryType = registry
if not TYPE_CHECKING:
    _RegistryType = registry

def as_declarative(**kw: Any) -> Callable[[Type[_T]], Type[_T]]:
    if False:
        i = 10
        return i + 15
    '\n    Class decorator which will adapt a given class into a\n    :func:`_orm.declarative_base`.\n\n    This function makes use of the :meth:`_orm.registry.as_declarative_base`\n    method, by first creating a :class:`_orm.registry` automatically\n    and then invoking the decorator.\n\n    E.g.::\n\n        from sqlalchemy.orm import as_declarative\n\n        @as_declarative()\n        class Base:\n            @declared_attr\n            def __tablename__(cls):\n                return cls.__name__.lower()\n            id = Column(Integer, primary_key=True)\n\n        class MyMappedClass(Base):\n            # ...\n\n    .. seealso::\n\n        :meth:`_orm.registry.as_declarative_base`\n\n    '
    (metadata, class_registry) = (kw.pop('metadata', None), kw.pop('class_registry', None))
    return registry(metadata=metadata, class_registry=class_registry).as_declarative_base(**kw)

@inspection._inspects(DeclarativeMeta, DeclarativeBase, DeclarativeAttributeIntercept)
def _inspect_decl_meta(cls: Type[Any]) -> Optional[Mapper[Any]]:
    if False:
        while True:
            i = 10
    mp: Optional[Mapper[Any]] = _inspect_mapped_class(cls)
    if mp is None:
        if _DeferredMapperConfig.has_cls(cls):
            _DeferredMapperConfig.raise_unmapped_for_cls(cls)
    return mp