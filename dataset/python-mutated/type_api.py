"""Base types API.

"""
from __future__ import annotations
from enum import Enum
from types import ModuleType
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Mapping
from typing import NewType
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import SchemaEventTarget
from .cache_key import CacheConst
from .cache_key import NO_CACHE
from .operators import ColumnOperators
from .visitors import Visitable
from .. import exc
from .. import util
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
if typing.TYPE_CHECKING:
    from ._typing import _TypeEngineArgument
    from .elements import BindParameter
    from .elements import ColumnElement
    from .operators import OperatorType
    from .sqltypes import _resolve_value_to_type as _resolve_value_to_type
    from .sqltypes import BOOLEANTYPE as BOOLEANTYPE
    from .sqltypes import INDEXABLE as INDEXABLE
    from .sqltypes import INTEGERTYPE as INTEGERTYPE
    from .sqltypes import MATCHTYPE as MATCHTYPE
    from .sqltypes import NULLTYPE as NULLTYPE
    from .sqltypes import NUMERICTYPE as NUMERICTYPE
    from .sqltypes import STRINGTYPE as STRINGTYPE
    from .sqltypes import TABLEVALUE as TABLEVALUE
    from ..engine.interfaces import Dialect
    from ..util.typing import GenericProtocol
_T = TypeVar('_T', bound=Any)
_T_co = TypeVar('_T_co', bound=Any, covariant=True)
_T_con = TypeVar('_T_con', bound=Any, contravariant=True)
_O = TypeVar('_O', bound=object)
_TE = TypeVar('_TE', bound='TypeEngine[Any]')
_CT = TypeVar('_CT', bound=Any)
_MatchedOnType = Union['GenericProtocol[Any]', NewType, Type[Any]]

class _NoValueInList(Enum):
    NO_VALUE_IN_LIST = 0
    'indicates we are trying to determine the type of an expression\n    against an empty list.'
_NO_VALUE_IN_LIST = _NoValueInList.NO_VALUE_IN_LIST

class _LiteralProcessorType(Protocol[_T_co]):

    def __call__(self, value: Any) -> str:
        if False:
            i = 10
            return i + 15
        ...

class _BindProcessorType(Protocol[_T_con]):

    def __call__(self, value: Optional[_T_con]) -> Any:
        if False:
            for i in range(10):
                print('nop')
        ...

class _ResultProcessorType(Protocol[_T_co]):

    def __call__(self, value: Any) -> Optional[_T_co]:
        if False:
            while True:
                i = 10
        ...

class _SentinelProcessorType(Protocol[_T_co]):

    def __call__(self, value: Any) -> Optional[_T_co]:
        if False:
            for i in range(10):
                print('nop')
        ...

class _BaseTypeMemoDict(TypedDict):
    impl: TypeEngine[Any]
    result: Dict[Any, Optional[_ResultProcessorType[Any]]]

class _TypeMemoDict(_BaseTypeMemoDict, total=False):
    literal: Optional[_LiteralProcessorType[Any]]
    bind: Optional[_BindProcessorType[Any]]
    sentinel: Optional[_SentinelProcessorType[Any]]
    custom: Dict[Any, object]

class _ComparatorFactory(Protocol[_T]):

    def __call__(self, expr: ColumnElement[_T]) -> TypeEngine.Comparator[_T]:
        if False:
            while True:
                i = 10
        ...

class TypeEngine(Visitable, Generic[_T]):
    """The ultimate base class for all SQL datatypes.

    Common subclasses of :class:`.TypeEngine` include
    :class:`.String`, :class:`.Integer`, and :class:`.Boolean`.

    For an overview of the SQLAlchemy typing system, see
    :ref:`types_toplevel`.

    .. seealso::

        :ref:`types_toplevel`

    """
    _sqla_type = True
    _isnull = False
    _is_tuple_type = False
    _is_table_value = False
    _is_array = False
    _is_type_decorator = False
    render_bind_cast = False
    'Render bind casts for :attr:`.BindTyping.RENDER_CASTS` mode.\n\n    If True, this type (usually a dialect level impl type) signals\n    to the compiler that a cast should be rendered around a bound parameter\n    for this type.\n\n    .. versionadded:: 2.0\n\n    .. seealso::\n\n        :class:`.BindTyping`\n\n    '
    render_literal_cast = False
    'render casts when rendering a value as an inline literal,\n    e.g. with :meth:`.TypeEngine.literal_processor`.\n\n    .. versionadded:: 2.0\n\n    '

    class Comparator(ColumnOperators, Generic[_CT]):
        """Base class for custom comparison operations defined at the
        type level.  See :attr:`.TypeEngine.comparator_factory`.


        """
        __slots__ = ('expr', 'type')
        expr: ColumnElement[_CT]
        type: TypeEngine[_CT]

        def __clause_element__(self) -> ColumnElement[_CT]:
            if False:
                for i in range(10):
                    print('nop')
            return self.expr

        def __init__(self, expr: ColumnElement[_CT]):
            if False:
                print('Hello World!')
            self.expr = expr
            self.type = expr.type

        @util.preload_module('sqlalchemy.sql.default_comparator')
        def operate(self, op: OperatorType, *other: Any, **kwargs: Any) -> ColumnElement[_CT]:
            if False:
                for i in range(10):
                    print('nop')
            default_comparator = util.preloaded.sql_default_comparator
            (op_fn, addtl_kw) = default_comparator.operator_lookup[op.__name__]
            if kwargs:
                addtl_kw = addtl_kw.union(kwargs)
            return op_fn(self.expr, op, *other, **addtl_kw)

        @util.preload_module('sqlalchemy.sql.default_comparator')
        def reverse_operate(self, op: OperatorType, other: Any, **kwargs: Any) -> ColumnElement[_CT]:
            if False:
                while True:
                    i = 10
            default_comparator = util.preloaded.sql_default_comparator
            (op_fn, addtl_kw) = default_comparator.operator_lookup[op.__name__]
            if kwargs:
                addtl_kw = addtl_kw.union(kwargs)
            return op_fn(self.expr, op, other, reverse=True, **addtl_kw)

        def _adapt_expression(self, op: OperatorType, other_comparator: TypeEngine.Comparator[Any]) -> Tuple[OperatorType, TypeEngine[Any]]:
            if False:
                while True:
                    i = 10
            'evaluate the return type of <self> <op> <othertype>,\n            and apply any adaptations to the given operator.\n\n            This method determines the type of a resulting binary expression\n            given two source types and an operator.   For example, two\n            :class:`_schema.Column` objects, both of the type\n            :class:`.Integer`, will\n            produce a :class:`.BinaryExpression` that also has the type\n            :class:`.Integer` when compared via the addition (``+``) operator.\n            However, using the addition operator with an :class:`.Integer`\n            and a :class:`.Date` object will produce a :class:`.Date`, assuming\n            "days delta" behavior by the database (in reality, most databases\n            other than PostgreSQL don\'t accept this particular operation).\n\n            The method returns a tuple of the form <operator>, <type>.\n            The resulting operator and type will be those applied to the\n            resulting :class:`.BinaryExpression` as the final operator and the\n            right-hand side of the expression.\n\n            Note that only a subset of operators make usage of\n            :meth:`._adapt_expression`,\n            including math operators and user-defined operators, but not\n            boolean comparison or special SQL keywords like MATCH or BETWEEN.\n\n            '
            return (op, self.type)
    hashable = True
    "Flag, if False, means values from this type aren't hashable.\n\n    Used by the ORM when uniquing result lists.\n\n    "
    comparator_factory: _ComparatorFactory[Any] = Comparator
    'A :class:`.TypeEngine.Comparator` class which will apply\n    to operations performed by owning :class:`_expression.ColumnElement`\n    objects.\n\n    The :attr:`.comparator_factory` attribute is a hook consulted by\n    the core expression system when column and SQL expression operations\n    are performed.   When a :class:`.TypeEngine.Comparator` class is\n    associated with this attribute, it allows custom re-definition of\n    all existing operators, as well as definition of new operators.\n    Existing operators include those provided by Python operator overloading\n    such as :meth:`.operators.ColumnOperators.__add__` and\n    :meth:`.operators.ColumnOperators.__eq__`,\n    those provided as standard\n    attributes of :class:`.operators.ColumnOperators` such as\n    :meth:`.operators.ColumnOperators.like`\n    and :meth:`.operators.ColumnOperators.in_`.\n\n    Rudimentary usage of this hook is allowed through simple subclassing\n    of existing types, or alternatively by using :class:`.TypeDecorator`.\n    See the documentation section :ref:`types_operators` for examples.\n\n    '
    sort_key_function: Optional[Callable[[Any], Any]] = None
    'A sorting function that can be passed as the key to sorted.\n\n    The default value of ``None`` indicates that the values stored by\n    this type are self-sorting.\n\n    .. versionadded:: 1.3.8\n\n    '
    should_evaluate_none: bool = False
    "If True, the Python constant ``None`` is considered to be handled\n    explicitly by this type.\n\n    The ORM uses this flag to indicate that a positive value of ``None``\n    is passed to the column in an INSERT statement, rather than omitting\n    the column from the INSERT statement which has the effect of firing\n    off column-level defaults.   It also allows types which have special\n    behavior for Python None, such as a JSON type, to indicate that\n    they'd like to handle the None value explicitly.\n\n    To set this flag on an existing type, use the\n    :meth:`.TypeEngine.evaluates_none` method.\n\n    .. seealso::\n\n        :meth:`.TypeEngine.evaluates_none`\n\n    "
    _variant_mapping: util.immutabledict[str, TypeEngine[Any]] = util.EMPTY_DICT

    def evaluates_none(self) -> Self:
        if False:
            return 10
        'Return a copy of this type which has the\n        :attr:`.should_evaluate_none` flag set to True.\n\n        E.g.::\n\n                Table(\n                    \'some_table\', metadata,\n                    Column(\n                        String(50).evaluates_none(),\n                        nullable=True,\n                        server_default=\'no value\')\n                )\n\n        The ORM uses this flag to indicate that a positive value of ``None``\n        is passed to the column in an INSERT statement, rather than omitting\n        the column from the INSERT statement which has the effect of firing\n        off column-level defaults.   It also allows for types which have\n        special behavior associated with the Python None value to indicate\n        that the value doesn\'t necessarily translate into SQL NULL; a\n        prime example of this is a JSON type which may wish to persist the\n        JSON value ``\'null\'``.\n\n        In all cases, the actual NULL SQL value can be always be\n        persisted in any column by using\n        the :obj:`_expression.null` SQL construct in an INSERT statement\n        or associated with an ORM-mapped attribute.\n\n        .. note::\n\n            The "evaluates none" flag does **not** apply to a value\n            of ``None`` passed to :paramref:`_schema.Column.default` or\n            :paramref:`_schema.Column.server_default`; in these cases,\n            ``None``\n            still means "no default".\n\n        .. seealso::\n\n            :ref:`session_forcing_null` - in the ORM documentation\n\n            :paramref:`.postgresql.JSON.none_as_null` - PostgreSQL JSON\n            interaction with this flag.\n\n            :attr:`.TypeEngine.should_evaluate_none` - class-level flag\n\n        '
        typ = self.copy()
        typ.should_evaluate_none = True
        return typ

    def copy(self, **kw: Any) -> Self:
        if False:
            return 10
        return self.adapt(self.__class__)

    def copy_value(self, value: Any) -> Any:
        if False:
            print('Hello World!')
        return value

    def literal_processor(self, dialect: Dialect) -> Optional[_LiteralProcessorType[_T]]:
        if False:
            while True:
                i = 10
        'Return a conversion function for processing literal values that are\n        to be rendered directly without using binds.\n\n        This function is used when the compiler makes use of the\n        "literal_binds" flag, typically used in DDL generation as well\n        as in certain scenarios where backends don\'t accept bound parameters.\n\n        Returns a callable which will receive a literal Python value\n        as the sole positional argument and will return a string representation\n        to be rendered in a SQL statement.\n\n        .. note::\n\n            This method is only called relative to a **dialect specific type\n            object**, which is often **private to a dialect in use** and is not\n            the same type object as the public facing one, which means it\'s not\n            feasible to subclass a :class:`.types.TypeEngine` class in order to\n            provide an alternate :meth:`_types.TypeEngine.literal_processor`\n            method, unless subclassing the :class:`_types.UserDefinedType`\n            class explicitly.\n\n            To provide alternate behavior for\n            :meth:`_types.TypeEngine.literal_processor`, implement a\n            :class:`_types.TypeDecorator` class and provide an implementation\n            of :meth:`_types.TypeDecorator.process_literal_param`.\n\n            .. seealso::\n\n                :ref:`types_typedecorator`\n\n\n        '
        return None

    def bind_processor(self, dialect: Dialect) -> Optional[_BindProcessorType[_T]]:
        if False:
            print('Hello World!')
        "Return a conversion function for processing bind values.\n\n        Returns a callable which will receive a bind parameter value\n        as the sole positional argument and will return a value to\n        send to the DB-API.\n\n        If processing is not necessary, the method should return ``None``.\n\n        .. note::\n\n            This method is only called relative to a **dialect specific type\n            object**, which is often **private to a dialect in use** and is not\n            the same type object as the public facing one, which means it's not\n            feasible to subclass a :class:`.types.TypeEngine` class in order to\n            provide an alternate :meth:`_types.TypeEngine.bind_processor`\n            method, unless subclassing the :class:`_types.UserDefinedType`\n            class explicitly.\n\n            To provide alternate behavior for\n            :meth:`_types.TypeEngine.bind_processor`, implement a\n            :class:`_types.TypeDecorator` class and provide an implementation\n            of :meth:`_types.TypeDecorator.process_bind_param`.\n\n            .. seealso::\n\n                :ref:`types_typedecorator`\n\n\n        :param dialect: Dialect instance in use.\n\n        "
        return None

    def result_processor(self, dialect: Dialect, coltype: object) -> Optional[_ResultProcessorType[_T]]:
        if False:
            i = 10
            return i + 15
        "Return a conversion function for processing result row values.\n\n        Returns a callable which will receive a result row column\n        value as the sole positional argument and will return a value\n        to return to the user.\n\n        If processing is not necessary, the method should return ``None``.\n\n        .. note::\n\n            This method is only called relative to a **dialect specific type\n            object**, which is often **private to a dialect in use** and is not\n            the same type object as the public facing one, which means it's not\n            feasible to subclass a :class:`.types.TypeEngine` class in order to\n            provide an alternate :meth:`_types.TypeEngine.result_processor`\n            method, unless subclassing the :class:`_types.UserDefinedType`\n            class explicitly.\n\n            To provide alternate behavior for\n            :meth:`_types.TypeEngine.result_processor`, implement a\n            :class:`_types.TypeDecorator` class and provide an implementation\n            of :meth:`_types.TypeDecorator.process_result_value`.\n\n            .. seealso::\n\n                :ref:`types_typedecorator`\n\n        :param dialect: Dialect instance in use.\n\n        :param coltype: DBAPI coltype argument received in cursor.description.\n\n        "
        return None

    def column_expression(self, colexpr: ColumnElement[_T]) -> Optional[ColumnElement[_T]]:
        if False:
            print('Hello World!')
        "Given a SELECT column expression, return a wrapping SQL expression.\n\n        This is typically a SQL function that wraps a column expression\n        as rendered in the columns clause of a SELECT statement.\n        It is used for special data types that require\n        columns to be wrapped in some special database function in order\n        to coerce the value before being sent back to the application.\n        It is the SQL analogue of the :meth:`.TypeEngine.result_processor`\n        method.\n\n        This method is called during the **SQL compilation** phase of a\n        statement, when rendering a SQL string. It is **not** called\n        against specific values.\n\n        .. note::\n\n            This method is only called relative to a **dialect specific type\n            object**, which is often **private to a dialect in use** and is not\n            the same type object as the public facing one, which means it's not\n            feasible to subclass a :class:`.types.TypeEngine` class in order to\n            provide an alternate :meth:`_types.TypeEngine.column_expression`\n            method, unless subclassing the :class:`_types.UserDefinedType`\n            class explicitly.\n\n            To provide alternate behavior for\n            :meth:`_types.TypeEngine.column_expression`, implement a\n            :class:`_types.TypeDecorator` class and provide an implementation\n            of :meth:`_types.TypeDecorator.column_expression`.\n\n            .. seealso::\n\n                :ref:`types_typedecorator`\n\n\n        .. seealso::\n\n            :ref:`types_sql_value_processing`\n\n        "
        return None

    @util.memoized_property
    def _has_column_expression(self) -> bool:
        if False:
            i = 10
            return i + 15
        "memoized boolean, check if column_expression is implemented.\n\n        Allows the method to be skipped for the vast majority of expression\n        types that don't use this feature.\n\n        "
        return self.__class__.column_expression.__code__ is not TypeEngine.column_expression.__code__

    def bind_expression(self, bindvalue: BindParameter[_T]) -> Optional[ColumnElement[_T]]:
        if False:
            i = 10
            return i + 15
        "Given a bind value (i.e. a :class:`.BindParameter` instance),\n        return a SQL expression in its place.\n\n        This is typically a SQL function that wraps the existing bound\n        parameter within the statement.  It is used for special data types\n        that require literals being wrapped in some special database function\n        in order to coerce an application-level value into a database-specific\n        format.  It is the SQL analogue of the\n        :meth:`.TypeEngine.bind_processor` method.\n\n        This method is called during the **SQL compilation** phase of a\n        statement, when rendering a SQL string. It is **not** called\n        against specific values.\n\n        Note that this method, when implemented, should always return\n        the exact same structure, without any conditional logic, as it\n        may be used in an executemany() call against an arbitrary number\n        of bound parameter sets.\n\n        .. note::\n\n            This method is only called relative to a **dialect specific type\n            object**, which is often **private to a dialect in use** and is not\n            the same type object as the public facing one, which means it's not\n            feasible to subclass a :class:`.types.TypeEngine` class in order to\n            provide an alternate :meth:`_types.TypeEngine.bind_expression`\n            method, unless subclassing the :class:`_types.UserDefinedType`\n            class explicitly.\n\n            To provide alternate behavior for\n            :meth:`_types.TypeEngine.bind_expression`, implement a\n            :class:`_types.TypeDecorator` class and provide an implementation\n            of :meth:`_types.TypeDecorator.bind_expression`.\n\n            .. seealso::\n\n                :ref:`types_typedecorator`\n\n        .. seealso::\n\n            :ref:`types_sql_value_processing`\n\n        "
        return None

    def _sentinel_value_resolver(self, dialect: Dialect) -> Optional[_SentinelProcessorType[_T]]:
        if False:
            while True:
                i = 10
        'Return an optional callable that will match parameter values\n        (post-bind processing) to result values\n        (pre-result-processing), for use in the "sentinel" feature.\n\n        .. versionadded:: 2.0.10\n\n        '
        return None

    @util.memoized_property
    def _has_bind_expression(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "memoized boolean, check if bind_expression is implemented.\n\n        Allows the method to be skipped for the vast majority of expression\n        types that don't use this feature.\n\n        "
        return util.method_is_overridden(self, TypeEngine.bind_expression)

    @staticmethod
    def _to_instance(cls_or_self: Union[Type[_TE], _TE]) -> _TE:
        if False:
            for i in range(10):
                print('nop')
        return to_instance(cls_or_self)

    def compare_values(self, x: Any, y: Any) -> bool:
        if False:
            while True:
                i = 10
        'Compare two values for equality.'
        return x == y

    def get_dbapi_type(self, dbapi: ModuleType) -> Optional[Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return the corresponding type object from the underlying DB-API, if\n        any.\n\n        This can be useful for calling ``setinputsizes()``, for example.\n\n        '
        return None

    @property
    def python_type(self) -> Type[Any]:
        if False:
            return 10
        'Return the Python type object expected to be returned\n        by instances of this type, if known.\n\n        Basically, for those types which enforce a return type,\n        or are known across the board to do such for all common\n        DBAPIs (like ``int`` for example), will return that type.\n\n        If a return type is not defined, raises\n        ``NotImplementedError``.\n\n        Note that any type also accommodates NULL in SQL which\n        means you can also get back ``None`` from any type\n        in practice.\n\n        '
        raise NotImplementedError()

    def with_variant(self, type_: _TypeEngineArgument[Any], *dialect_names: str) -> Self:
        if False:
            while True:
                i = 10
        'Produce a copy of this type object that will utilize the given\n        type when applied to the dialect of the given name.\n\n        e.g.::\n\n            from sqlalchemy.types import String\n            from sqlalchemy.dialects import mysql\n\n            string_type = String()\n\n            string_type = string_type.with_variant(\n                mysql.VARCHAR(collation=\'foo\'), \'mysql\', \'mariadb\'\n            )\n\n        The variant mapping indicates that when this type is\n        interpreted by a specific dialect, it will instead be\n        transmuted into the given type, rather than using the\n        primary type.\n\n        .. versionchanged:: 2.0 the :meth:`_types.TypeEngine.with_variant`\n           method now works with a :class:`_types.TypeEngine` object "in\n           place", returning a copy of the original type rather than returning\n           a wrapping object; the ``Variant`` class is no longer used.\n\n        :param type\\_: a :class:`.TypeEngine` that will be selected\n         as a variant from the originating type, when a dialect\n         of the given name is in use.\n        :param \\*dialect_names: one or more base names of the dialect which\n         uses this type. (i.e. ``\'postgresql\'``, ``\'mysql\'``, etc.)\n\n         .. versionchanged:: 2.0 multiple dialect names can be specified\n            for one variant.\n\n        .. seealso::\n\n            :ref:`types_with_variant` - illustrates the use of\n            :meth:`_types.TypeEngine.with_variant`.\n\n        '
        if not dialect_names:
            raise exc.ArgumentError('At least one dialect name is required')
        for dialect_name in dialect_names:
            if dialect_name in self._variant_mapping:
                raise exc.ArgumentError(f'Dialect {dialect_name!r} is already present in the mapping for this {self!r}')
        new_type = self.copy()
        type_ = to_instance(type_)
        if type_._variant_mapping:
            raise exc.ArgumentError("can't pass a type that already has variants as a dialect-level type to with_variant()")
        new_type._variant_mapping = self._variant_mapping.union({dialect_name: type_ for dialect_name in dialect_names})
        return new_type

    def _resolve_for_literal(self, value: Any) -> Self:
        if False:
            i = 10
            return i + 15
        'adjust this type given a literal Python value that will be\n        stored in a bound parameter.\n\n        Used exclusively by _resolve_value_to_type().\n\n        .. versionadded:: 1.4.30 or 2.0\n\n        TODO: this should be part of public API\n\n        .. seealso::\n\n            :meth:`.TypeEngine._resolve_for_python_type`\n\n        '
        return self

    def _resolve_for_python_type(self, python_type: Type[Any], matched_on: _MatchedOnType, matched_on_flattened: Type[Any]) -> Optional[Self]:
        if False:
            for i in range(10):
                print('nop')
        "given a Python type (e.g. ``int``, ``str``, etc. ) return an\n        instance of this :class:`.TypeEngine` that's appropriate for this type.\n\n        An additional argument ``matched_on`` is passed, which indicates an\n        entry from the ``__mro__`` of the given ``python_type`` that more\n        specifically matches how the caller located this :class:`.TypeEngine`\n        object.   Such as, if a lookup of some kind links the ``int`` Python\n        type to the :class:`.Integer` SQL type, and the original object\n        was some custom subclass of ``int`` such as ``MyInt(int)``, the\n        arguments passed would be ``(MyInt, int)``.\n\n        If the given Python type does not correspond to this\n        :class:`.TypeEngine`, or the Python type is otherwise ambiguous, the\n        method should return None.\n\n        For simple cases, the method checks that the ``python_type``\n        and ``matched_on`` types are the same (i.e. not a subclass), and\n        returns self; for all other cases, it returns ``None``.\n\n        The initial use case here is for the ORM to link user-defined\n        Python standard library ``enum.Enum`` classes to the SQLAlchemy\n        :class:`.Enum` SQL type when constructing ORM Declarative mappings.\n\n        :param python_type: the Python type we want to use\n        :param matched_on: the Python type that led us to choose this\n         particular :class:`.TypeEngine` class, which would be a supertype\n         of ``python_type``.   By default, the request is rejected if\n         ``python_type`` doesn't match ``matched_on`` (None is returned).\n\n        .. versionadded:: 2.0.0b4\n\n        TODO: this should be part of public API\n\n        .. seealso::\n\n            :meth:`.TypeEngine._resolve_for_literal`\n\n        "
        if python_type is not matched_on_flattened:
            return None
        return self

    @util.ro_memoized_property
    def _type_affinity(self) -> Optional[Type[TypeEngine[_T]]]:
        if False:
            while True:
                i = 10
        "Return a rudimental 'affinity' value expressing the general class\n        of type."
        typ = None
        for t in self.__class__.__mro__:
            if t is TypeEngine or TypeEngineMixin in t.__bases__:
                return typ
            elif issubclass(t, TypeEngine):
                typ = t
        else:
            return self.__class__

    @util.ro_memoized_property
    def _generic_type_affinity(self) -> Type[TypeEngine[_T]]:
        if False:
            print('Hello World!')
        best_camelcase = None
        best_uppercase = None
        if not isinstance(self, TypeEngine):
            return self.__class__
        for t in self.__class__.__mro__:
            if t.__module__ in ('sqlalchemy.sql.sqltypes', 'sqlalchemy.sql.type_api') and issubclass(t, TypeEngine) and (TypeEngineMixin not in t.__bases__) and (t not in (TypeEngine, TypeEngineMixin)) and (t.__name__[0] != '_'):
                if t.__name__.isupper() and (not best_uppercase):
                    best_uppercase = t
                elif not t.__name__.isupper() and (not best_camelcase):
                    best_camelcase = t
        return best_camelcase or best_uppercase or cast('Type[TypeEngine[_T]]', NULLTYPE.__class__)

    def as_generic(self, allow_nulltype: bool=False) -> TypeEngine[_T]:
        if False:
            while True:
                i = 10
        '\n        Return an instance of the generic type corresponding to this type\n        using heuristic rule. The method may be overridden if this\n        heuristic rule is not sufficient.\n\n        >>> from sqlalchemy.dialects.mysql import INTEGER\n        >>> INTEGER(display_width=4).as_generic()\n        Integer()\n\n        >>> from sqlalchemy.dialects.mysql import NVARCHAR\n        >>> NVARCHAR(length=100).as_generic()\n        Unicode(length=100)\n\n        .. versionadded:: 1.4.0b2\n\n\n        .. seealso::\n\n            :ref:`metadata_reflection_dbagnostic_types` - describes the\n            use of :meth:`_types.TypeEngine.as_generic` in conjunction with\n            the :meth:`_sql.DDLEvents.column_reflect` event, which is its\n            intended use.\n\n        '
        if not allow_nulltype and self._generic_type_affinity == NULLTYPE.__class__:
            raise NotImplementedError('Default TypeEngine.as_generic() heuristic method was unsuccessful for {}. A custom as_generic() method must be implemented for this type class.'.format(self.__class__.__module__ + '.' + self.__class__.__name__))
        return util.constructor_copy(self, self._generic_type_affinity)

    def dialect_impl(self, dialect: Dialect) -> TypeEngine[_T]:
        if False:
            return 10
        'Return a dialect-specific implementation for this\n        :class:`.TypeEngine`.\n\n        '
        try:
            tm = dialect._type_memos[self]
        except KeyError:
            pass
        else:
            return tm['impl']
        return self._dialect_info(dialect)['impl']

    def _unwrapped_dialect_impl(self, dialect: Dialect) -> TypeEngine[_T]:
        if False:
            while True:
                i = 10
        "Return the 'unwrapped' dialect impl for this type.\n\n        For a type that applies wrapping logic (e.g. TypeDecorator), give\n        us the real, actual dialect-level type that is used.\n\n        This is used by TypeDecorator itself as well at least one case where\n        dialects need to check that a particular specific dialect-level\n        type is in use, within the :meth:`.DefaultDialect.set_input_sizes`\n        method.\n\n        "
        return self.dialect_impl(dialect)

    def _cached_literal_processor(self, dialect: Dialect) -> Optional[_LiteralProcessorType[_T]]:
        if False:
            for i in range(10):
                print('nop')
        'Return a dialect-specific literal processor for this type.'
        try:
            return dialect._type_memos[self]['literal']
        except KeyError:
            pass
        d = self._dialect_info(dialect)
        d['literal'] = lp = d['impl'].literal_processor(dialect)
        return lp

    def _cached_bind_processor(self, dialect: Dialect) -> Optional[_BindProcessorType[_T]]:
        if False:
            while True:
                i = 10
        'Return a dialect-specific bind processor for this type.'
        try:
            return dialect._type_memos[self]['bind']
        except KeyError:
            pass
        d = self._dialect_info(dialect)
        d['bind'] = bp = d['impl'].bind_processor(dialect)
        return bp

    def _cached_result_processor(self, dialect: Dialect, coltype: Any) -> Optional[_ResultProcessorType[_T]]:
        if False:
            for i in range(10):
                print('nop')
        'Return a dialect-specific result processor for this type.'
        try:
            return dialect._type_memos[self]['result'][coltype]
        except KeyError:
            pass
        d = self._dialect_info(dialect)
        rp = d['impl'].result_processor(dialect, coltype)
        d['result'][coltype] = rp
        return rp

    def _cached_sentinel_value_processor(self, dialect: Dialect) -> Optional[_SentinelProcessorType[_T]]:
        if False:
            print('Hello World!')
        try:
            return dialect._type_memos[self]['sentinel']
        except KeyError:
            pass
        d = self._dialect_info(dialect)
        d['sentinel'] = bp = d['impl']._sentinel_value_resolver(dialect)
        return bp

    def _cached_custom_processor(self, dialect: Dialect, key: str, fn: Callable[[TypeEngine[_T]], _O]) -> _O:
        if False:
            while True:
                i = 10
        'return a dialect-specific processing object for\n        custom purposes.\n\n        The cx_Oracle dialect uses this at the moment.\n\n        '
        try:
            return cast(_O, dialect._type_memos[self]['custom'][key])
        except KeyError:
            pass
        d = self._dialect_info(dialect)
        impl = d['impl']
        custom_dict = d.setdefault('custom', {})
        custom_dict[key] = result = fn(impl)
        return result

    def _dialect_info(self, dialect: Dialect) -> _TypeMemoDict:
        if False:
            for i in range(10):
                print('nop')
        'Return a dialect-specific registry which\n        caches a dialect-specific implementation, bind processing\n        function, and one or more result processing functions.'
        if self in dialect._type_memos:
            return dialect._type_memos[self]
        else:
            impl = self._gen_dialect_impl(dialect)
            if impl is self:
                impl = self.adapt(type(self))
            assert impl is not self
            d: _TypeMemoDict = {'impl': impl, 'result': {}}
            dialect._type_memos[self] = d
            return d

    def _gen_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if False:
            return 10
        if dialect.name in self._variant_mapping:
            return self._variant_mapping[dialect.name]._gen_dialect_impl(dialect)
        else:
            return dialect.type_descriptor(self)

    @util.memoized_property
    def _static_cache_key(self) -> Union[CacheConst, Tuple[Any, ...]]:
        if False:
            i = 10
            return i + 15
        names = util.get_cls_kwargs(self.__class__)
        return (self.__class__,) + tuple(((k, self.__dict__[k]._static_cache_key if isinstance(self.__dict__[k], TypeEngine) else self.__dict__[k]) for k in names if k in self.__dict__ and (not k.startswith('_')) and (self.__dict__[k] is not None)))

    @overload
    def adapt(self, cls: Type[_TE], **kw: Any) -> _TE:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def adapt(self, cls: Type[TypeEngineMixin], **kw: Any) -> TypeEngine[Any]:
        if False:
            return 10
        ...

    def adapt(self, cls: Type[Union[TypeEngine[Any], TypeEngineMixin]], **kw: Any) -> TypeEngine[Any]:
        if False:
            return 10
        'Produce an "adapted" form of this type, given an "impl" class\n        to work with.\n\n        This method is used internally to associate generic\n        types with "implementation" types that are specific to a particular\n        dialect.\n        '
        return util.constructor_copy(self, cast(Type[TypeEngine[Any]], cls), **kw)

    def coerce_compared_value(self, op: Optional[OperatorType], value: Any) -> TypeEngine[Any]:
        if False:
            return 10
        "Suggest a type for a 'coerced' Python value in an expression.\n\n        Given an operator and value, gives the type a chance\n        to return a type which the value should be coerced into.\n\n        The default behavior here is conservative; if the right-hand\n        side is already coerced into a SQL type based on its\n        Python type, it is usually left alone.\n\n        End-user functionality extension here should generally be via\n        :class:`.TypeDecorator`, which provides more liberal behavior in that\n        it defaults to coercing the other side of the expression into this\n        type, thus applying special Python conversions above and beyond those\n        needed by the DBAPI to both ides. It also provides the public method\n        :meth:`.TypeDecorator.coerce_compared_value` which is intended for\n        end-user customization of this behavior.\n\n        "
        _coerced_type = _resolve_value_to_type(value)
        if _coerced_type is NULLTYPE or _coerced_type._type_affinity is self._type_affinity:
            return self
        else:
            return _coerced_type

    def _compare_type_affinity(self, other: TypeEngine[Any]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._type_affinity is other._type_affinity

    def compile(self, dialect: Optional[Dialect]=None) -> str:
        if False:
            print('Hello World!')
        'Produce a string-compiled form of this :class:`.TypeEngine`.\n\n        When called with no arguments, uses a "default" dialect\n        to produce a string result.\n\n        :param dialect: a :class:`.Dialect` instance.\n\n        '
        if dialect is None:
            dialect = self._default_dialect()
        return dialect.type_compiler_instance.process(self)

    @util.preload_module('sqlalchemy.engine.default')
    def _default_dialect(self) -> Dialect:
        if False:
            for i in range(10):
                print('nop')
        default = util.preloaded.engine_default
        return default.StrCompileDialect()

    def __str__(self) -> str:
        if False:
            return 10
        return str(self.compile())

    def __repr__(self) -> str:
        if False:
            return 10
        return util.generic_repr(self)

class TypeEngineMixin:
    """classes which subclass this can act as "mixin" classes for
    TypeEngine."""
    __slots__ = ()
    if TYPE_CHECKING:

        @util.memoized_property
        def _static_cache_key(self) -> Union[CacheConst, Tuple[Any, ...]]:
            if False:
                for i in range(10):
                    print('nop')
            ...

        @overload
        def adapt(self, cls: Type[_TE], **kw: Any) -> _TE:
            if False:
                for i in range(10):
                    print('nop')
            ...

        @overload
        def adapt(self, cls: Type[TypeEngineMixin], **kw: Any) -> TypeEngine[Any]:
            if False:
                while True:
                    i = 10
            ...

        def adapt(self, cls: Type[Union[TypeEngine[Any], TypeEngineMixin]], **kw: Any) -> TypeEngine[Any]:
            if False:
                i = 10
                return i + 15
            ...

        def dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
            if False:
                return 10
            ...

class ExternalType(TypeEngineMixin):
    """mixin that defines attributes and behaviors specific to third-party
    datatypes.

    "Third party" refers to datatypes that are defined outside the scope
    of SQLAlchemy within either end-user application code or within
    external extensions to SQLAlchemy.

    Subclasses currently include :class:`.TypeDecorator` and
    :class:`.UserDefinedType`.

    .. versionadded:: 1.4.28

    """
    cache_ok: Optional[bool] = None
    'Indicate if statements using this :class:`.ExternalType` are "safe to\n    cache".\n\n    The default value ``None`` will emit a warning and then not allow caching\n    of a statement which includes this type.   Set to ``False`` to disable\n    statements using this type from being cached at all without a warning.\n    When set to ``True``, the object\'s class and selected elements from its\n    state will be used as part of the cache key.  For example, using a\n    :class:`.TypeDecorator`::\n\n        class MyType(TypeDecorator):\n            impl = String\n\n            cache_ok = True\n\n            def __init__(self, choices):\n                self.choices = tuple(choices)\n                self.internal_only = True\n\n    The cache key for the above type would be equivalent to::\n\n        >>> MyType(["a", "b", "c"])._static_cache_key\n        (<class \'__main__.MyType\'>, (\'choices\', (\'a\', \'b\', \'c\')))\n\n    The caching scheme will extract attributes from the type that correspond\n    to the names of parameters in the ``__init__()`` method.  Above, the\n    "choices" attribute becomes part of the cache key but "internal_only"\n    does not, because there is no parameter named "internal_only".\n\n    The requirements for cacheable elements is that they are hashable\n    and also that they indicate the same SQL rendered for expressions using\n    this type every time for a given cache value.\n\n    To accommodate for datatypes that refer to unhashable structures such\n    as dictionaries, sets and lists, these objects can be made "cacheable"\n    by assigning hashable structures to the attributes whose names\n    correspond with the names of the arguments.  For example, a datatype\n    which accepts a dictionary of lookup values may publish this as a sorted\n    series of tuples.   Given a previously un-cacheable type as::\n\n        class LookupType(UserDefinedType):\n            \'\'\'a custom type that accepts a dictionary as a parameter.\n\n            this is the non-cacheable version, as "self.lookup" is not\n            hashable.\n\n            \'\'\'\n\n            def __init__(self, lookup):\n                self.lookup = lookup\n\n            def get_col_spec(self, **kw):\n                return "VARCHAR(255)"\n\n            def bind_processor(self, dialect):\n                # ...  works with "self.lookup" ...\n\n    Where "lookup" is a dictionary.  The type will not be able to generate\n    a cache key::\n\n        >>> type_ = LookupType({"a": 10, "b": 20})\n        >>> type_._static_cache_key\n        <stdin>:1: SAWarning: UserDefinedType LookupType({\'a\': 10, \'b\': 20}) will not\n        produce a cache key because the ``cache_ok`` flag is not set to True.\n        Set this flag to True if this type object\'s state is safe to use\n        in a cache key, or False to disable this warning.\n        symbol(\'no_cache\')\n\n    If we **did** set up such a cache key, it wouldn\'t be usable. We would\n    get a tuple structure that contains a dictionary inside of it, which\n    cannot itself be used as a key in a "cache dictionary" such as SQLAlchemy\'s\n    statement cache, since Python dictionaries aren\'t hashable::\n\n        >>> # set cache_ok = True\n        >>> type_.cache_ok = True\n\n        >>> # this is the cache key it would generate\n        >>> key = type_._static_cache_key\n        >>> key\n        (<class \'__main__.LookupType\'>, (\'lookup\', {\'a\': 10, \'b\': 20}))\n\n        >>> # however this key is not hashable, will fail when used with\n        >>> # SQLAlchemy statement cache\n        >>> some_cache = {key: "some sql value"}\n        Traceback (most recent call last): File "<stdin>", line 1,\n        in <module> TypeError: unhashable type: \'dict\'\n\n    The type may be made cacheable by assigning a sorted tuple of tuples\n    to the ".lookup" attribute::\n\n        class LookupType(UserDefinedType):\n            \'\'\'a custom type that accepts a dictionary as a parameter.\n\n            The dictionary is stored both as itself in a private variable,\n            and published in a public variable as a sorted tuple of tuples,\n            which is hashable and will also return the same value for any\n            two equivalent dictionaries.  Note it assumes the keys and\n            values of the dictionary are themselves hashable.\n\n            \'\'\'\n\n            cache_ok = True\n\n            def __init__(self, lookup):\n                self._lookup = lookup\n\n                # assume keys/values of "lookup" are hashable; otherwise\n                # they would also need to be converted in some way here\n                self.lookup = tuple(\n                    (key, lookup[key]) for key in sorted(lookup)\n                )\n\n            def get_col_spec(self, **kw):\n                return "VARCHAR(255)"\n\n            def bind_processor(self, dialect):\n                # ...  works with "self._lookup" ...\n\n    Where above, the cache key for ``LookupType({"a": 10, "b": 20})`` will be::\n\n        >>> LookupType({"a": 10, "b": 20})._static_cache_key\n        (<class \'__main__.LookupType\'>, (\'lookup\', ((\'a\', 10), (\'b\', 20))))\n\n    .. versionadded:: 1.4.14 - added the ``cache_ok`` flag to allow\n       some configurability of caching for :class:`.TypeDecorator` classes.\n\n    .. versionadded:: 1.4.28 - added the :class:`.ExternalType` mixin which\n       generalizes the ``cache_ok`` flag to both the :class:`.TypeDecorator`\n       and :class:`.UserDefinedType` classes.\n\n    .. seealso::\n\n        :ref:`sql_caching`\n\n    '

    @util.non_memoized_property
    def _static_cache_key(self) -> Union[CacheConst, Tuple[Any, ...]]:
        if False:
            for i in range(10):
                print('nop')
        cache_ok = self.__class__.__dict__.get('cache_ok', None)
        if cache_ok is None:
            for subtype in self.__class__.__mro__:
                if ExternalType in subtype.__bases__:
                    break
            else:
                subtype = self.__class__.__mro__[1]
            util.warn("%s %r will not produce a cache key because the ``cache_ok`` attribute is not set to True.  This can have significant performance implications including some performance degradations in comparison to prior SQLAlchemy versions.  Set this attribute to True if this type object's state is safe to use in a cache key, or False to disable this warning." % (subtype.__name__, self), code='cprf')
        elif cache_ok is True:
            return super()._static_cache_key
        return NO_CACHE

class UserDefinedType(ExternalType, TypeEngineMixin, TypeEngine[_T], util.EnsureKWArg):
    """Base for user defined types.

    This should be the base of new types.  Note that
    for most cases, :class:`.TypeDecorator` is probably
    more appropriate::

      import sqlalchemy.types as types

      class MyType(types.UserDefinedType):
          cache_ok = True

          def __init__(self, precision = 8):
              self.precision = precision

          def get_col_spec(self, **kw):
              return "MYTYPE(%s)" % self.precision

          def bind_processor(self, dialect):
              def process(value):
                  return value
              return process

          def result_processor(self, dialect, coltype):
              def process(value):
                  return value
              return process

    Once the type is made, it's immediately usable::

      table = Table('foo', metadata_obj,
          Column('id', Integer, primary_key=True),
          Column('data', MyType(16))
          )

    The ``get_col_spec()`` method will in most cases receive a keyword
    argument ``type_expression`` which refers to the owning expression
    of the type as being compiled, such as a :class:`_schema.Column` or
    :func:`.cast` construct.  This keyword is only sent if the method
    accepts keyword arguments (e.g. ``**kw``) in its argument signature;
    introspection is used to check for this in order to support legacy
    forms of this function.

    The :attr:`.UserDefinedType.cache_ok` class-level flag indicates if this
    custom :class:`.UserDefinedType` is safe to be used as part of a cache key.
    This flag defaults to ``None`` which will initially generate a warning
    when the SQL compiler attempts to generate a cache key for a statement
    that uses this type.  If the :class:`.UserDefinedType` is not guaranteed
    to produce the same bind/result behavior and SQL generation
    every time, this flag should be set to ``False``; otherwise if the
    class produces the same behavior each time, it may be set to ``True``.
    See :attr:`.UserDefinedType.cache_ok` for further notes on how this works.

    .. versionadded:: 1.4.28 Generalized the :attr:`.ExternalType.cache_ok`
       flag so that it is available for both :class:`.TypeDecorator` as well
       as :class:`.UserDefinedType`.

    """
    __visit_name__ = 'user_defined'
    ensure_kwarg = 'get_col_spec'

    def coerce_compared_value(self, op: Optional[OperatorType], value: Any) -> TypeEngine[Any]:
        if False:
            for i in range(10):
                print('nop')
        "Suggest a type for a 'coerced' Python value in an expression.\n\n        Default behavior for :class:`.UserDefinedType` is the\n        same as that of :class:`.TypeDecorator`; by default it returns\n        ``self``, assuming the compared value should be coerced into\n        the same type as this one.  See\n        :meth:`.TypeDecorator.coerce_compared_value` for more detail.\n\n        "
        return self

class Emulated(TypeEngineMixin):
    """Mixin for base types that emulate the behavior of a DB-native type.

    An :class:`.Emulated` type will use an available database type
    in conjunction with Python-side routines and/or database constraints
    in order to approximate the behavior of a database type that is provided
    natively by some backends.  When a native-providing backend is in
    use, the native version of the type is used.  This native version
    should include the :class:`.NativeForEmulated` mixin to allow it to be
    distinguished from :class:`.Emulated`.

    Current examples of :class:`.Emulated` are:  :class:`.Interval`,
    :class:`.Enum`, :class:`.Boolean`.

    .. versionadded:: 1.2.0b3

    """
    native: bool

    def adapt_to_emulated(self, impltype: Type[Union[TypeEngine[Any], TypeEngineMixin]], **kw: Any) -> TypeEngine[Any]:
        if False:
            for i in range(10):
                print('nop')
        'Given an impl class, adapt this type to the impl assuming\n        "emulated".\n\n        The impl should also be an "emulated" version of this type,\n        most likely the same class as this type itself.\n\n        e.g.: sqltypes.Enum adapts to the Enum class.\n\n        '
        return super().adapt(impltype, **kw)

    @overload
    def adapt(self, cls: Type[_TE], **kw: Any) -> _TE:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def adapt(self, cls: Type[TypeEngineMixin], **kw: Any) -> TypeEngine[Any]:
        if False:
            print('Hello World!')
        ...

    def adapt(self, cls: Type[Union[TypeEngine[Any], TypeEngineMixin]], **kw: Any) -> TypeEngine[Any]:
        if False:
            print('Hello World!')
        if _is_native_for_emulated(cls):
            if self.native:
                return cls.adapt_emulated_to_native(self, **kw)
            else:
                return cls.adapt_native_to_emulated(self, **kw)
        elif issubclass(cls, self.__class__):
            return self.adapt_to_emulated(cls, **kw)
        else:
            return super().adapt(cls, **kw)

def _is_native_for_emulated(typ: Type[Union[TypeEngine[Any], TypeEngineMixin]]) -> TypeGuard[Type[NativeForEmulated]]:
    if False:
        while True:
            i = 10
    return hasattr(typ, 'adapt_emulated_to_native')

class NativeForEmulated(TypeEngineMixin):
    """Indicates DB-native types supported by an :class:`.Emulated` type.

    .. versionadded:: 1.2.0b3

    """

    @classmethod
    def adapt_native_to_emulated(cls, impl: Union[TypeEngine[Any], TypeEngineMixin], **kw: Any) -> TypeEngine[Any]:
        if False:
            for i in range(10):
                print('nop')
        'Given an impl, adapt this type\'s class to the impl assuming\n        "emulated".\n\n\n        '
        impltype = impl.__class__
        return impl.adapt(impltype, **kw)

    @classmethod
    def adapt_emulated_to_native(cls, impl: Union[TypeEngine[Any], TypeEngineMixin], **kw: Any) -> TypeEngine[Any]:
        if False:
            while True:
                i = 10
        'Given an impl, adapt this type\'s class to the impl assuming\n        "native".\n\n        The impl will be an :class:`.Emulated` class but not a\n        :class:`.NativeForEmulated`.\n\n        e.g.: postgresql.ENUM produces a type given an Enum instance.\n\n        '
        return cls(**kw)

class TypeDecorator(SchemaEventTarget, ExternalType, TypeEngine[_T]):
    """Allows the creation of types which add additional functionality
    to an existing type.

    This method is preferred to direct subclassing of SQLAlchemy's
    built-in types as it ensures that all required functionality of
    the underlying type is kept in place.

    Typical usage::

      import sqlalchemy.types as types

      class MyType(types.TypeDecorator):
          '''Prefixes Unicode values with "PREFIX:" on the way in and
          strips it off on the way out.
          '''

          impl = types.Unicode

          cache_ok = True

          def process_bind_param(self, value, dialect):
              return "PREFIX:" + value

          def process_result_value(self, value, dialect):
              return value[7:]

          def copy(self, **kw):
              return MyType(self.impl.length)

    The class-level ``impl`` attribute is required, and can reference any
    :class:`.TypeEngine` class.  Alternatively, the :meth:`load_dialect_impl`
    method can be used to provide different type classes based on the dialect
    given; in this case, the ``impl`` variable can reference
    ``TypeEngine`` as a placeholder.

    The :attr:`.TypeDecorator.cache_ok` class-level flag indicates if this
    custom :class:`.TypeDecorator` is safe to be used as part of a cache key.
    This flag defaults to ``None`` which will initially generate a warning
    when the SQL compiler attempts to generate a cache key for a statement
    that uses this type.  If the :class:`.TypeDecorator` is not guaranteed
    to produce the same bind/result behavior and SQL generation
    every time, this flag should be set to ``False``; otherwise if the
    class produces the same behavior each time, it may be set to ``True``.
    See :attr:`.TypeDecorator.cache_ok` for further notes on how this works.

    Types that receive a Python type that isn't similar to the ultimate type
    used may want to define the :meth:`TypeDecorator.coerce_compared_value`
    method. This is used to give the expression system a hint when coercing
    Python objects into bind parameters within expressions. Consider this
    expression::

        mytable.c.somecol + datetime.date(2009, 5, 15)

    Above, if "somecol" is an ``Integer`` variant, it makes sense that
    we're doing date arithmetic, where above is usually interpreted
    by databases as adding a number of days to the given date.
    The expression system does the right thing by not attempting to
    coerce the "date()" value into an integer-oriented bind parameter.

    However, in the case of ``TypeDecorator``, we are usually changing an
    incoming Python type to something new - ``TypeDecorator`` by default will
    "coerce" the non-typed side to be the same type as itself. Such as below,
    we define an "epoch" type that stores a date value as an integer::

        class MyEpochType(types.TypeDecorator):
            impl = types.Integer

            epoch = datetime.date(1970, 1, 1)

            def process_bind_param(self, value, dialect):
                return (value - self.epoch).days

            def process_result_value(self, value, dialect):
                return self.epoch + timedelta(days=value)

    Our expression of ``somecol + date`` with the above type will coerce the
    "date" on the right side to also be treated as ``MyEpochType``.

    This behavior can be overridden via the
    :meth:`~TypeDecorator.coerce_compared_value` method, which returns a type
    that should be used for the value of the expression. Below we set it such
    that an integer value will be treated as an ``Integer``, and any other
    value is assumed to be a date and will be treated as a ``MyEpochType``::

        def coerce_compared_value(self, op, value):
            if isinstance(value, int):
                return Integer()
            else:
                return self

    .. warning::

       Note that the **behavior of coerce_compared_value is not inherited
       by default from that of the base type**.
       If the :class:`.TypeDecorator` is augmenting a
       type that requires special logic for certain types of operators,
       this method **must** be overridden.  A key example is when decorating
       the :class:`_postgresql.JSON` and :class:`_postgresql.JSONB` types;
       the default rules of :meth:`.TypeEngine.coerce_compared_value` should
       be used in order to deal with operators like index operations::

            from sqlalchemy import JSON
            from sqlalchemy import TypeDecorator

            class MyJsonType(TypeDecorator):
                impl = JSON

                cache_ok = True

                def coerce_compared_value(self, op, value):
                    return self.impl.coerce_compared_value(op, value)

       Without the above step, index operations such as ``mycol['foo']``
       will cause the index value ``'foo'`` to be JSON encoded.

       Similarly, when working with the :class:`.ARRAY` datatype, the
       type coercion for index operations (e.g. ``mycol[5]``) is also
       handled by :meth:`.TypeDecorator.coerce_compared_value`, where
       again a simple override is sufficient unless special rules are needed
       for particular operators::

            from sqlalchemy import ARRAY
            from sqlalchemy import TypeDecorator

            class MyArrayType(TypeDecorator):
                impl = ARRAY

                cache_ok = True

                def coerce_compared_value(self, op, value):
                    return self.impl.coerce_compared_value(op, value)


    """
    __visit_name__ = 'type_decorator'
    _is_type_decorator = True
    impl: Union[TypeEngine[Any], Type[TypeEngine[Any]]]

    @util.memoized_property
    def impl_instance(self) -> TypeEngine[Any]:
        if False:
            i = 10
            return i + 15
        return self.impl

    def __init__(self, *args: Any, **kwargs: Any):
        if False:
            return 10
        "Construct a :class:`.TypeDecorator`.\n\n        Arguments sent here are passed to the constructor\n        of the class assigned to the ``impl`` class level attribute,\n        assuming the ``impl`` is a callable, and the resulting\n        object is assigned to the ``self.impl`` instance attribute\n        (thus overriding the class attribute of the same name).\n\n        If the class level ``impl`` is not a callable (the unusual case),\n        it will be assigned to the same instance attribute 'as-is',\n        ignoring those arguments passed to the constructor.\n\n        Subclasses can override this to customize the generation\n        of ``self.impl`` entirely.\n\n        "
        if not hasattr(self.__class__, 'impl'):
            raise AssertionError("TypeDecorator implementations require a class-level variable 'impl' which refers to the class of type being decorated")
        self.impl = to_instance(self.__class__.impl, *args, **kwargs)
    coerce_to_is_types: Sequence[Type[Any]] = (type(None),)
    'Specify those Python types which should be coerced at the expression\n    level to "IS <constant>" when compared using ``==`` (and same for\n    ``IS NOT`` in conjunction with ``!=``).\n\n    For most SQLAlchemy types, this includes ``NoneType``, as well as\n    ``bool``.\n\n    :class:`.TypeDecorator` modifies this list to only include ``NoneType``,\n    as typedecorator implementations that deal with boolean types are common.\n\n    Custom :class:`.TypeDecorator` classes can override this attribute to\n    return an empty tuple, in which case no values will be coerced to\n    constants.\n\n    '

    class Comparator(TypeEngine.Comparator[_CT]):
        """A :class:`.TypeEngine.Comparator` that is specific to
        :class:`.TypeDecorator`.

        User-defined :class:`.TypeDecorator` classes should not typically
        need to modify this.


        """
        __slots__ = ()

        def operate(self, op: OperatorType, *other: Any, **kwargs: Any) -> ColumnElement[_CT]:
            if False:
                for i in range(10):
                    print('nop')
            if TYPE_CHECKING:
                assert isinstance(self.expr.type, TypeDecorator)
            kwargs['_python_is_types'] = self.expr.type.coerce_to_is_types
            return super().operate(op, *other, **kwargs)

        def reverse_operate(self, op: OperatorType, other: Any, **kwargs: Any) -> ColumnElement[_CT]:
            if False:
                print('Hello World!')
            if TYPE_CHECKING:
                assert isinstance(self.expr.type, TypeDecorator)
            kwargs['_python_is_types'] = self.expr.type.coerce_to_is_types
            return super().reverse_operate(op, other, **kwargs)

    @property
    def comparator_factory(self) -> _ComparatorFactory[Any]:
        if False:
            print('Hello World!')
        if TypeDecorator.Comparator in self.impl.comparator_factory.__mro__:
            return self.impl.comparator_factory
        else:
            return type('TDComparator', (TypeDecorator.Comparator, self.impl.comparator_factory), {})

    def _gen_dialect_impl(self, dialect: Dialect) -> TypeEngine[_T]:
        if False:
            for i in range(10):
                print('nop')
        if dialect.name in self._variant_mapping:
            adapted = dialect.type_descriptor(self._variant_mapping[dialect.name])
        else:
            adapted = dialect.type_descriptor(self)
        if adapted is not self:
            return adapted
        typedesc = self.load_dialect_impl(dialect).dialect_impl(dialect)
        tt = self.copy()
        if not isinstance(tt, self.__class__):
            raise AssertionError('Type object %s does not properly implement the copy() method, it must return an object of type %s' % (self, self.__class__))
        tt.impl = tt.impl_instance = typedesc
        return tt

    @util.ro_non_memoized_property
    def _type_affinity(self) -> Optional[Type[TypeEngine[Any]]]:
        if False:
            print('Hello World!')
        return self.impl_instance._type_affinity

    def _set_parent(self, parent: SchemaEventTarget, outer: bool=False, **kw: Any) -> None:
        if False:
            while True:
                i = 10
        'Support SchemaEventTarget'
        super()._set_parent(parent)
        if not outer and isinstance(self.impl_instance, SchemaEventTarget):
            self.impl_instance._set_parent(parent, outer=False, **kw)

    def _set_parent_with_dispatch(self, parent: SchemaEventTarget, **kw: Any) -> None:
        if False:
            print('Hello World!')
        'Support SchemaEventTarget'
        super()._set_parent_with_dispatch(parent, outer=True, **kw)
        if isinstance(self.impl_instance, SchemaEventTarget):
            self.impl_instance._set_parent_with_dispatch(parent)

    def type_engine(self, dialect: Dialect) -> TypeEngine[Any]:
        if False:
            while True:
                i = 10
        'Return a dialect-specific :class:`.TypeEngine` instance\n        for this :class:`.TypeDecorator`.\n\n        In most cases this returns a dialect-adapted form of\n        the :class:`.TypeEngine` type represented by ``self.impl``.\n        Makes usage of :meth:`dialect_impl`.\n        Behavior can be customized here by overriding\n        :meth:`load_dialect_impl`.\n\n        '
        adapted = dialect.type_descriptor(self)
        if not isinstance(adapted, type(self)):
            return adapted
        else:
            return self.load_dialect_impl(dialect)

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if False:
            i = 10
            return i + 15
        'Return a :class:`.TypeEngine` object corresponding to a dialect.\n\n        This is an end-user override hook that can be used to provide\n        differing types depending on the given dialect.  It is used\n        by the :class:`.TypeDecorator` implementation of :meth:`type_engine`\n        to help determine what type should ultimately be returned\n        for a given :class:`.TypeDecorator`.\n\n        By default returns ``self.impl``.\n\n        '
        return self.impl_instance

    def _unwrapped_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if False:
            for i in range(10):
                print('nop')
        "Return the 'unwrapped' dialect impl for this type.\n\n        This is used by the :meth:`.DefaultDialect.set_input_sizes`\n        method.\n\n        "
        typ = self.dialect_impl(dialect)
        if isinstance(typ, self.__class__):
            return typ.load_dialect_impl(dialect).dialect_impl(dialect)
        else:
            return typ

    def __getattr__(self, key: str) -> Any:
        if False:
            while True:
                i = 10
        'Proxy all other undefined accessors to the underlying\n        implementation.'
        return getattr(self.impl_instance, key)

    def process_literal_param(self, value: Optional[_T], dialect: Dialect) -> str:
        if False:
            while True:
                i = 10
        'Receive a literal parameter value to be rendered inline within\n        a statement.\n\n        .. note::\n\n            This method is called during the **SQL compilation** phase of a\n            statement, when rendering a SQL string. Unlike other SQL\n            compilation methods, it is passed a specific Python value to be\n            rendered as a string. However it should not be confused with the\n            :meth:`_types.TypeDecorator.process_bind_param` method, which is\n            the more typical method that processes the actual value passed to a\n            particular parameter at statement execution time.\n\n        Custom subclasses of :class:`_types.TypeDecorator` should override\n        this method to provide custom behaviors for incoming data values\n        that are in the special case of being rendered as literals.\n\n        The returned string will be rendered into the output string.\n\n        '
        raise NotImplementedError()

    def process_bind_param(self, value: Optional[_T], dialect: Dialect) -> Any:
        if False:
            for i in range(10):
                print('nop')
        'Receive a bound parameter value to be converted.\n\n        Custom subclasses of :class:`_types.TypeDecorator` should override\n        this method to provide custom behaviors for incoming data values.\n        This method is called at **statement execution time** and is passed\n        the literal Python data value which is to be associated with a bound\n        parameter in the statement.\n\n        The operation could be anything desired to perform custom\n        behavior, such as transforming or serializing data.\n        This could also be used as a hook for validating logic.\n\n        :param value: Data to operate upon, of any type expected by\n         this method in the subclass.  Can be ``None``.\n        :param dialect: the :class:`.Dialect` in use.\n\n        .. seealso::\n\n            :ref:`types_typedecorator`\n\n            :meth:`_types.TypeDecorator.process_result_value`\n\n        '
        raise NotImplementedError()

    def process_result_value(self, value: Optional[Any], dialect: Dialect) -> Optional[_T]:
        if False:
            i = 10
            return i + 15
        "Receive a result-row column value to be converted.\n\n        Custom subclasses of :class:`_types.TypeDecorator` should override\n        this method to provide custom behaviors for data values\n        being received in result rows coming from the database.\n        This method is called at **result fetching time** and is passed\n        the literal Python data value that's extracted from a database result\n        row.\n\n        The operation could be anything desired to perform custom\n        behavior, such as transforming or deserializing data.\n\n        :param value: Data to operate upon, of any type expected by\n         this method in the subclass.  Can be ``None``.\n        :param dialect: the :class:`.Dialect` in use.\n\n        .. seealso::\n\n            :ref:`types_typedecorator`\n\n            :meth:`_types.TypeDecorator.process_bind_param`\n\n\n        "
        raise NotImplementedError()

    @util.memoized_property
    def _has_bind_processor(self) -> bool:
        if False:
            return 10
        'memoized boolean, check if process_bind_param is implemented.\n\n        Allows the base process_bind_param to raise\n        NotImplementedError without needing to test an expensive\n        exception throw.\n\n        '
        return util.method_is_overridden(self, TypeDecorator.process_bind_param)

    @util.memoized_property
    def _has_literal_processor(self) -> bool:
        if False:
            return 10
        'memoized boolean, check if process_literal_param is implemented.'
        return util.method_is_overridden(self, TypeDecorator.process_literal_param)

    def literal_processor(self, dialect: Dialect) -> Optional[_LiteralProcessorType[_T]]:
        if False:
            return 10
        'Provide a literal processing function for the given\n        :class:`.Dialect`.\n\n        This is the method that fulfills the :class:`.TypeEngine`\n        contract for literal value conversion which normally occurs via\n        the :meth:`_types.TypeEngine.literal_processor` method.\n\n        .. note::\n\n            User-defined subclasses of :class:`_types.TypeDecorator` should\n            **not** implement this method, and should instead implement\n            :meth:`_types.TypeDecorator.process_literal_param` so that the\n            "inner" processing provided by the implementing type is maintained.\n\n        '
        if self._has_literal_processor:
            process_literal_param = self.process_literal_param
            process_bind_param = None
        elif self._has_bind_processor:
            process_literal_param = None
            process_bind_param = self.process_bind_param
        else:
            process_literal_param = None
            process_bind_param = None
        if process_literal_param is not None:
            impl_processor = self.impl_instance.literal_processor(dialect)
            if impl_processor:
                fixed_impl_processor = impl_processor
                fixed_process_literal_param = process_literal_param

                def process(value: Any) -> str:
                    if False:
                        i = 10
                        return i + 15
                    return fixed_impl_processor(fixed_process_literal_param(value, dialect))
            else:
                fixed_process_literal_param = process_literal_param

                def process(value: Any) -> str:
                    if False:
                        while True:
                            i = 10
                    return fixed_process_literal_param(value, dialect)
            return process
        elif process_bind_param is not None:
            impl_processor = self.impl_instance.literal_processor(dialect)
            if not impl_processor:
                return None
            else:
                fixed_impl_processor = impl_processor
                fixed_process_bind_param = process_bind_param

                def process(value: Any) -> str:
                    if False:
                        while True:
                            i = 10
                    return fixed_impl_processor(fixed_process_bind_param(value, dialect))
                return process
        else:
            return self.impl_instance.literal_processor(dialect)

    def bind_processor(self, dialect: Dialect) -> Optional[_BindProcessorType[_T]]:
        if False:
            i = 10
            return i + 15
        'Provide a bound value processing function for the\n        given :class:`.Dialect`.\n\n        This is the method that fulfills the :class:`.TypeEngine`\n        contract for bound value conversion which normally occurs via\n        the :meth:`_types.TypeEngine.bind_processor` method.\n\n        .. note::\n\n            User-defined subclasses of :class:`_types.TypeDecorator` should\n            **not** implement this method, and should instead implement\n            :meth:`_types.TypeDecorator.process_bind_param` so that the "inner"\n            processing provided by the implementing type is maintained.\n\n        :param dialect: Dialect instance in use.\n\n        '
        if self._has_bind_processor:
            process_param = self.process_bind_param
            impl_processor = self.impl_instance.bind_processor(dialect)
            if impl_processor:
                fixed_impl_processor = impl_processor
                fixed_process_param = process_param

                def process(value: Optional[_T]) -> Any:
                    if False:
                        while True:
                            i = 10
                    return fixed_impl_processor(fixed_process_param(value, dialect))
            else:
                fixed_process_param = process_param

                def process(value: Optional[_T]) -> Any:
                    if False:
                        while True:
                            i = 10
                    return fixed_process_param(value, dialect)
            return process
        else:
            return self.impl_instance.bind_processor(dialect)

    @util.memoized_property
    def _has_result_processor(self) -> bool:
        if False:
            while True:
                i = 10
        'memoized boolean, check if process_result_value is implemented.\n\n        Allows the base process_result_value to raise\n        NotImplementedError without needing to test an expensive\n        exception throw.\n\n        '
        return util.method_is_overridden(self, TypeDecorator.process_result_value)

    def result_processor(self, dialect: Dialect, coltype: Any) -> Optional[_ResultProcessorType[_T]]:
        if False:
            return 10
        'Provide a result value processing function for the given\n        :class:`.Dialect`.\n\n        This is the method that fulfills the :class:`.TypeEngine`\n        contract for bound value conversion which normally occurs via\n        the :meth:`_types.TypeEngine.result_processor` method.\n\n        .. note::\n\n            User-defined subclasses of :class:`_types.TypeDecorator` should\n            **not** implement this method, and should instead implement\n            :meth:`_types.TypeDecorator.process_result_value` so that the\n            "inner" processing provided by the implementing type is maintained.\n\n        :param dialect: Dialect instance in use.\n        :param coltype: A SQLAlchemy data type\n\n        '
        if self._has_result_processor:
            process_value = self.process_result_value
            impl_processor = self.impl_instance.result_processor(dialect, coltype)
            if impl_processor:
                fixed_process_value = process_value
                fixed_impl_processor = impl_processor

                def process(value: Any) -> Optional[_T]:
                    if False:
                        for i in range(10):
                            print('nop')
                    return fixed_process_value(fixed_impl_processor(value), dialect)
            else:
                fixed_process_value = process_value

                def process(value: Any) -> Optional[_T]:
                    if False:
                        return 10
                    return fixed_process_value(value, dialect)
            return process
        else:
            return self.impl_instance.result_processor(dialect, coltype)

    @util.memoized_property
    def _has_bind_expression(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return util.method_is_overridden(self, TypeDecorator.bind_expression) or self.impl_instance._has_bind_expression

    def bind_expression(self, bindparam: BindParameter[_T]) -> Optional[ColumnElement[_T]]:
        if False:
            return 10
        'Given a bind value (i.e. a :class:`.BindParameter` instance),\n        return a SQL expression which will typically wrap the given parameter.\n\n        .. note::\n\n            This method is called during the **SQL compilation** phase of a\n            statement, when rendering a SQL string. It is **not** necessarily\n            called against specific values, and should not be confused with the\n            :meth:`_types.TypeDecorator.process_bind_param` method, which is\n            the more typical method that processes the actual value passed to a\n            particular parameter at statement execution time.\n\n        Subclasses of :class:`_types.TypeDecorator` can override this method\n        to provide custom bind expression behavior for the type.  This\n        implementation will **replace** that of the underlying implementation\n        type.\n\n        '
        return self.impl_instance.bind_expression(bindparam)

    @util.memoized_property
    def _has_column_expression(self) -> bool:
        if False:
            print('Hello World!')
        "memoized boolean, check if column_expression is implemented.\n\n        Allows the method to be skipped for the vast majority of expression\n        types that don't use this feature.\n\n        "
        return util.method_is_overridden(self, TypeDecorator.column_expression) or self.impl_instance._has_column_expression

    def column_expression(self, column: ColumnElement[_T]) -> Optional[ColumnElement[_T]]:
        if False:
            i = 10
            return i + 15
        "Given a SELECT column expression, return a wrapping SQL expression.\n\n        .. note::\n\n            This method is called during the **SQL compilation** phase of a\n            statement, when rendering a SQL string. It is **not** called\n            against specific values, and should not be confused with the\n            :meth:`_types.TypeDecorator.process_result_value` method, which is\n            the more typical method that processes the actual value returned\n            in a result row subsequent to statement execution time.\n\n        Subclasses of :class:`_types.TypeDecorator` can override this method\n        to provide custom column expression behavior for the type.  This\n        implementation will **replace** that of the underlying implementation\n        type.\n\n        See the description of :meth:`_types.TypeEngine.column_expression`\n        for a complete description of the method's use.\n\n        "
        return self.impl_instance.column_expression(column)

    def coerce_compared_value(self, op: Optional[OperatorType], value: Any) -> Any:
        if False:
            print('Hello World!')
        "Suggest a type for a 'coerced' Python value in an expression.\n\n        By default, returns self.   This method is called by\n        the expression system when an object using this type is\n        on the left or right side of an expression against a plain Python\n        object which does not yet have a SQLAlchemy type assigned::\n\n            expr = table.c.somecolumn + 35\n\n        Where above, if ``somecolumn`` uses this type, this method will\n        be called with the value ``operator.add``\n        and ``35``.  The return value is whatever SQLAlchemy type should\n        be used for ``35`` for this particular operation.\n\n        "
        return self

    def copy(self, **kw: Any) -> Self:
        if False:
            print('Hello World!')
        'Produce a copy of this :class:`.TypeDecorator` instance.\n\n        This is a shallow copy and is provided to fulfill part of\n        the :class:`.TypeEngine` contract.  It usually does not\n        need to be overridden unless the user-defined :class:`.TypeDecorator`\n        has local state that should be deep-copied.\n\n        '
        instance = self.__class__.__new__(self.__class__)
        instance.__dict__.update(self.__dict__)
        return instance

    def get_dbapi_type(self, dbapi: ModuleType) -> Optional[Any]:
        if False:
            while True:
                i = 10
        'Return the DBAPI type object represented by this\n        :class:`.TypeDecorator`.\n\n        By default this calls upon :meth:`.TypeEngine.get_dbapi_type` of the\n        underlying "impl".\n        '
        return self.impl_instance.get_dbapi_type(dbapi)

    def compare_values(self, x: Any, y: Any) -> bool:
        if False:
            print('Hello World!')
        'Given two values, compare them for equality.\n\n        By default this calls upon :meth:`.TypeEngine.compare_values`\n        of the underlying "impl", which in turn usually\n        uses the Python equals operator ``==``.\n\n        This function is used by the ORM to compare\n        an original-loaded value with an intercepted\n        "changed" value, to determine if a net change\n        has occurred.\n\n        '
        return self.impl_instance.compare_values(x, y)

    @property
    def sort_key_function(self) -> Optional[Callable[[Any], Any]]:
        if False:
            i = 10
            return i + 15
        return self.impl_instance.sort_key_function

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return util.generic_repr(self, to_inspect=self.impl_instance)

class Variant(TypeDecorator[_T]):
    """deprecated.  symbol is present for backwards-compatibility with
    workaround recipes, however this actual type should not be used.

    """

    def __init__(self, *arg: Any, **kw: Any):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Variant is no longer used in SQLAlchemy; this is a placeholder symbol for backwards compatibility.')

@overload
def to_instance(typeobj: Union[Type[_TE], _TE], *arg: Any, **kw: Any) -> _TE:
    if False:
        return 10
    ...

@overload
def to_instance(typeobj: None, *arg: Any, **kw: Any) -> TypeEngine[None]:
    if False:
        i = 10
        return i + 15
    ...

def to_instance(typeobj: Union[Type[_TE], _TE, None], *arg: Any, **kw: Any) -> Union[_TE, TypeEngine[None]]:
    if False:
        print('Hello World!')
    if typeobj is None:
        return NULLTYPE
    if callable(typeobj):
        return typeobj(*arg, **kw)
    else:
        return typeobj

def adapt_type(typeobj: TypeEngine[Any], colspecs: Mapping[Type[Any], Type[TypeEngine[Any]]]) -> TypeEngine[Any]:
    if False:
        return 10
    if isinstance(typeobj, type):
        typeobj = typeobj()
    for t in typeobj.__class__.__mro__[0:-1]:
        try:
            impltype = colspecs[t]
            break
        except KeyError:
            pass
    else:
        return typeobj
    if issubclass(typeobj.__class__, impltype):
        return typeobj
    return typeobj.adapt(impltype)