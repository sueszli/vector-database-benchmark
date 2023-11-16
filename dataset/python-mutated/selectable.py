"""The :class:`_expression.FromClause` class of SQL expression elements,
representing
SQL tables and derived rowsets.

"""
from __future__ import annotations
import collections
from enum import Enum
import itertools
from typing import AbstractSet
from typing import Any as TODO_Any
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import cache_key
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from . import visitors
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from ._typing import _TP
from ._typing import is_column_element
from ._typing import is_select_statement
from ._typing import is_subquery
from ._typing import is_table
from ._typing import is_text_clause
from .annotation import Annotated
from .annotation import SupportsCloneAnnotations
from .base import _clone
from .base import _cloned_difference
from .base import _cloned_intersection
from .base import _entity_namespace_key
from .base import _EntityNamespace
from .base import _expand_cloned
from .base import _from_objects
from .base import _generative
from .base import _never_select_column
from .base import _NoArg
from .base import _select_iterables
from .base import CacheableOptions
from .base import ColumnCollection
from .base import ColumnSet
from .base import CompileState
from .base import DedupeColumnCollection
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .base import HasMemoized
from .base import Immutable
from .coercions import _document_text_coercion
from .elements import _anonymous_label
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ClauseList
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import DQLDMLClauseElement
from .elements import GroupedElement
from .elements import literal_column
from .elements import TableValuedColumn
from .elements import UnaryExpression
from .operators import OperatorType
from .sqltypes import NULLTYPE
from .visitors import _TraverseInternalsType
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import exc
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
and_ = BooleanClauseList.and_
_T = TypeVar('_T', bound=Any)
if TYPE_CHECKING:
    from ._typing import _ColumnExpressionArgument
    from ._typing import _ColumnExpressionOrStrLabelArgument
    from ._typing import _FromClauseArgument
    from ._typing import _JoinTargetArgument
    from ._typing import _LimitOffsetType
    from ._typing import _MAYBE_ENTITY
    from ._typing import _NOT_ENTITY
    from ._typing import _OnClauseArgument
    from ._typing import _SelectStatementForCompoundArgument
    from ._typing import _T0
    from ._typing import _T1
    from ._typing import _T2
    from ._typing import _T3
    from ._typing import _T4
    from ._typing import _T5
    from ._typing import _T6
    from ._typing import _T7
    from ._typing import _TextCoercedExpressionArgument
    from ._typing import _TypedColumnClauseArgument as _TCCA
    from ._typing import _TypeEngineArgument
    from .base import _AmbiguousTableNameMap
    from .base import ExecutableOption
    from .base import ReadOnlyColumnCollection
    from .cache_key import _CacheKeyTraversalType
    from .compiler import SQLCompiler
    from .dml import Delete
    from .dml import Update
    from .elements import BinaryExpression
    from .elements import KeyedColumnElement
    from .elements import Label
    from .elements import NamedColumn
    from .elements import TextClause
    from .functions import Function
    from .schema import ForeignKey
    from .schema import ForeignKeyConstraint
    from .sqltypes import TableValueType
    from .type_api import TypeEngine
    from .visitors import _CloneCallableType
_ColumnsClauseElement = Union['FromClause', ColumnElement[Any], 'TextClause']
_LabelConventionCallable = Callable[[Union['ColumnElement[Any]', 'TextClause']], Optional[str]]

class _JoinTargetProtocol(Protocol):

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        if False:
            for i in range(10):
                print('nop')
        ...

    @util.ro_non_memoized_property
    def entity_namespace(self) -> _EntityNamespace:
        if False:
            for i in range(10):
                print('nop')
        ...
_JoinTargetElement = Union['FromClause', _JoinTargetProtocol]
_OnClauseElement = Union['ColumnElement[bool]', _JoinTargetProtocol]
_ForUpdateOfArgument = Union[Union['_ColumnExpressionArgument[Any]', '_FromClauseArgument'], Sequence['_ColumnExpressionArgument[Any]']]
_SetupJoinsElement = Tuple[_JoinTargetElement, Optional[_OnClauseElement], Optional['FromClause'], Dict[str, Any]]
_SelectIterable = Iterable[Union['ColumnElement[Any]', 'TextClause']]

class _OffsetLimitParam(BindParameter[int]):
    inherit_cache = True

    @property
    def _limit_offset_value(self) -> Optional[int]:
        if False:
            return 10
        return self.effective_value

class ReturnsRows(roles.ReturnsRowsRole, DQLDMLClauseElement):
    """The base-most class for Core constructs that have some concept of
    columns that can represent rows.

    While the SELECT statement and TABLE are the primary things we think
    of in this category,  DML like INSERT, UPDATE and DELETE can also specify
    RETURNING which means they can be used in CTEs and other forms, and
    PostgreSQL has functions that return rows also.

    .. versionadded:: 1.4

    """
    _is_returns_rows = True
    _is_from_clause = False
    _is_select_base = False
    _is_select_statement = False
    _is_lateral = False

    @property
    def selectable(self) -> ReturnsRows:
        if False:
            while True:
                i = 10
        return self

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        if False:
            for i in range(10):
                print('nop')
        'A sequence of column expression objects that represents the\n        "selected" columns of this :class:`_expression.ReturnsRows`.\n\n        This is typically equivalent to .exported_columns except it is\n        delivered in the form of a straight sequence and not  keyed\n        :class:`_expression.ColumnCollection`.\n\n        '
        raise NotImplementedError()

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "Return ``True`` if this :class:`.ReturnsRows` is\n        'derived' from the given :class:`.FromClause`.\n\n        An example would be an Alias of a Table is derived from that Table.\n\n        "
        raise NotImplementedError()

    def _generate_fromclause_column_proxies(self, fromclause: FromClause) -> None:
        if False:
            while True:
                i = 10
        'Populate columns into an :class:`.AliasedReturnsRows` object.'
        raise NotImplementedError()

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        if False:
            i = 10
            return i + 15
        'reset internal collections for an incoming column being added.'
        raise NotImplementedError()

    @property
    def exported_columns(self) -> ReadOnlyColumnCollection[Any, Any]:
        if False:
            return 10
        'A :class:`_expression.ColumnCollection`\n        that represents the "exported"\n        columns of this :class:`_expression.ReturnsRows`.\n\n        The "exported" columns represent the collection of\n        :class:`_expression.ColumnElement`\n        expressions that are rendered by this SQL\n        construct.   There are primary varieties which are the\n        "FROM clause columns" of a FROM clause, such as a table, join,\n        or subquery, the "SELECTed columns", which are the columns in\n        the "columns clause" of a SELECT statement, and the RETURNING\n        columns in a DML statement..\n\n        .. versionadded:: 1.4\n\n        .. seealso::\n\n            :attr:`_expression.FromClause.exported_columns`\n\n            :attr:`_expression.SelectBase.exported_columns`\n        '
        raise NotImplementedError()

class ExecutableReturnsRows(Executable, ReturnsRows):
    """base for executable statements that return rows."""

class TypedReturnsRows(ExecutableReturnsRows, Generic[_TP]):
    """base for executable statements that return rows."""

class Selectable(ReturnsRows):
    """Mark a class as being selectable."""
    __visit_name__ = 'selectable'
    is_selectable = True

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        if False:
            return 10
        raise NotImplementedError()

    def lateral(self, name: Optional[str]=None) -> LateralFromClause:
        if False:
            return 10
        'Return a LATERAL alias of this :class:`_expression.Selectable`.\n\n        The return value is the :class:`_expression.Lateral` construct also\n        provided by the top-level :func:`_expression.lateral` function.\n\n        .. seealso::\n\n            :ref:`tutorial_lateral_correlation` -  overview of usage.\n\n        '
        return Lateral._construct(self, name=name)

    @util.deprecated('1.4', message='The :meth:`.Selectable.replace_selectable` method is deprecated, and will be removed in a future release.  Similar functionality is available via the sqlalchemy.sql.visitors module.')
    @util.preload_module('sqlalchemy.sql.util')
    def replace_selectable(self, old: FromClause, alias: Alias) -> Self:
        if False:
            i = 10
            return i + 15
        "Replace all occurrences of :class:`_expression.FromClause`\n        'old' with the given :class:`_expression.Alias`\n        object, returning a copy of this :class:`_expression.FromClause`.\n\n        "
        return util.preloaded.sql_util.ClauseAdapter(alias).traverse(self)

    def corresponding_column(self, column: KeyedColumnElement[Any], require_embedded: bool=False) -> Optional[KeyedColumnElement[Any]]:
        if False:
            return 10
        'Given a :class:`_expression.ColumnElement`, return the exported\n        :class:`_expression.ColumnElement` object from the\n        :attr:`_expression.Selectable.exported_columns`\n        collection of this :class:`_expression.Selectable`\n        which corresponds to that\n        original :class:`_expression.ColumnElement` via a common ancestor\n        column.\n\n        :param column: the target :class:`_expression.ColumnElement`\n                      to be matched.\n\n        :param require_embedded: only return corresponding columns for\n         the given :class:`_expression.ColumnElement`, if the given\n         :class:`_expression.ColumnElement`\n         is actually present within a sub-element\n         of this :class:`_expression.Selectable`.\n         Normally the column will match if\n         it merely shares a common ancestor with one of the exported\n         columns of this :class:`_expression.Selectable`.\n\n        .. seealso::\n\n            :attr:`_expression.Selectable.exported_columns` - the\n            :class:`_expression.ColumnCollection`\n            that is used for the operation.\n\n            :meth:`_expression.ColumnCollection.corresponding_column`\n            - implementation\n            method.\n\n        '
        return self.exported_columns.corresponding_column(column, require_embedded)

class HasPrefixes:
    _prefixes: Tuple[Tuple[DQLDMLClauseElement, str], ...] = ()
    _has_prefixes_traverse_internals: _TraverseInternalsType = [('_prefixes', InternalTraversal.dp_prefix_sequence)]

    @_generative
    @_document_text_coercion('prefixes', ':meth:`_expression.HasPrefixes.prefix_with`', ':paramref:`.HasPrefixes.prefix_with.*prefixes`')
    def prefix_with(self, *prefixes: _TextCoercedExpressionArgument[Any], dialect: str='*') -> Self:
        if False:
            while True:
                i = 10
        'Add one or more expressions following the statement keyword, i.e.\n        SELECT, INSERT, UPDATE, or DELETE. Generative.\n\n        This is used to support backend-specific prefix keywords such as those\n        provided by MySQL.\n\n        E.g.::\n\n            stmt = table.insert().prefix_with("LOW_PRIORITY", dialect="mysql")\n\n            # MySQL 5.7 optimizer hints\n            stmt = select(table).prefix_with(\n                "/*+ BKA(t1) */", dialect="mysql")\n\n        Multiple prefixes can be specified by multiple calls\n        to :meth:`_expression.HasPrefixes.prefix_with`.\n\n        :param \\*prefixes: textual or :class:`_expression.ClauseElement`\n         construct which\n         will be rendered following the INSERT, UPDATE, or DELETE\n         keyword.\n        :param dialect: optional string dialect name which will\n         limit rendering of this prefix to only that dialect.\n\n        '
        self._prefixes = self._prefixes + tuple([(coercions.expect(roles.StatementOptionRole, p), dialect) for p in prefixes])
        return self

class HasSuffixes:
    _suffixes: Tuple[Tuple[DQLDMLClauseElement, str], ...] = ()
    _has_suffixes_traverse_internals: _TraverseInternalsType = [('_suffixes', InternalTraversal.dp_prefix_sequence)]

    @_generative
    @_document_text_coercion('suffixes', ':meth:`_expression.HasSuffixes.suffix_with`', ':paramref:`.HasSuffixes.suffix_with.*suffixes`')
    def suffix_with(self, *suffixes: _TextCoercedExpressionArgument[Any], dialect: str='*') -> Self:
        if False:
            i = 10
            return i + 15
        'Add one or more expressions following the statement as a whole.\n\n        This is used to support backend-specific suffix keywords on\n        certain constructs.\n\n        E.g.::\n\n            stmt = select(col1, col2).cte().suffix_with(\n                "cycle empno set y_cycle to 1 default 0", dialect="oracle")\n\n        Multiple suffixes can be specified by multiple calls\n        to :meth:`_expression.HasSuffixes.suffix_with`.\n\n        :param \\*suffixes: textual or :class:`_expression.ClauseElement`\n         construct which\n         will be rendered following the target clause.\n        :param dialect: Optional string dialect name which will\n         limit rendering of this suffix to only that dialect.\n\n        '
        self._suffixes = self._suffixes + tuple([(coercions.expect(roles.StatementOptionRole, p), dialect) for p in suffixes])
        return self

class HasHints:
    _hints: util.immutabledict[Tuple[FromClause, str], str] = util.immutabledict()
    _statement_hints: Tuple[Tuple[str, str], ...] = ()
    _has_hints_traverse_internals: _TraverseInternalsType = [('_statement_hints', InternalTraversal.dp_statement_hint_list), ('_hints', InternalTraversal.dp_table_hint_list)]

    def with_statement_hint(self, text: str, dialect_name: str='*') -> Self:
        if False:
            while True:
                i = 10
        'Add a statement hint to this :class:`_expression.Select` or\n        other selectable object.\n\n        This method is similar to :meth:`_expression.Select.with_hint`\n        except that\n        it does not require an individual table, and instead applies to the\n        statement as a whole.\n\n        Hints here are specific to the backend database and may include\n        directives such as isolation levels, file directives, fetch directives,\n        etc.\n\n        .. seealso::\n\n            :meth:`_expression.Select.with_hint`\n\n            :meth:`_expression.Select.prefix_with` - generic SELECT prefixing\n            which also can suit some database-specific HINT syntaxes such as\n            MySQL optimizer hints\n\n        '
        return self._with_hint(None, text, dialect_name)

    @_generative
    def with_hint(self, selectable: _FromClauseArgument, text: str, dialect_name: str='*') -> Self:
        if False:
            return 10
        'Add an indexing or other executional context hint for the given\n        selectable to this :class:`_expression.Select` or other selectable\n        object.\n\n        The text of the hint is rendered in the appropriate\n        location for the database backend in use, relative\n        to the given :class:`_schema.Table` or :class:`_expression.Alias`\n        passed as the\n        ``selectable`` argument. The dialect implementation\n        typically uses Python string substitution syntax\n        with the token ``%(name)s`` to render the name of\n        the table or alias. E.g. when using Oracle, the\n        following::\n\n            select(mytable).\\\n                with_hint(mytable, "index(%(name)s ix_mytable)")\n\n        Would render SQL as::\n\n            select /*+ index(mytable ix_mytable) */ ... from mytable\n\n        The ``dialect_name`` option will limit the rendering of a particular\n        hint to a particular backend. Such as, to add hints for both Oracle\n        and Sybase simultaneously::\n\n            select(mytable).\\\n                with_hint(mytable, "index(%(name)s ix_mytable)", \'oracle\').\\\n                with_hint(mytable, "WITH INDEX ix_mytable", \'mssql\')\n\n        .. seealso::\n\n            :meth:`_expression.Select.with_statement_hint`\n\n        '
        return self._with_hint(selectable, text, dialect_name)

    def _with_hint(self, selectable: Optional[_FromClauseArgument], text: str, dialect_name: str) -> Self:
        if False:
            return 10
        if selectable is None:
            self._statement_hints += ((dialect_name, text),)
        else:
            self._hints = self._hints.union({(coercions.expect(roles.FromClauseRole, selectable), dialect_name): text})
        return self

class FromClause(roles.AnonymizedFromClauseRole, Selectable):
    """Represent an element that can be used within the ``FROM``
    clause of a ``SELECT`` statement.

    The most common forms of :class:`_expression.FromClause` are the
    :class:`_schema.Table` and the :func:`_expression.select` constructs.  Key
    features common to all :class:`_expression.FromClause` objects include:

    * a :attr:`.c` collection, which provides per-name access to a collection
      of :class:`_expression.ColumnElement` objects.
    * a :attr:`.primary_key` attribute, which is a collection of all those
      :class:`_expression.ColumnElement`
      objects that indicate the ``primary_key`` flag.
    * Methods to generate various derivations of a "from" clause, including
      :meth:`_expression.FromClause.alias`,
      :meth:`_expression.FromClause.join`,
      :meth:`_expression.FromClause.select`.


    """
    __visit_name__ = 'fromclause'
    named_with_column = False

    @util.ro_non_memoized_property
    def _hide_froms(self) -> Iterable[FromClause]:
        if False:
            while True:
                i = 10
        return ()
    _is_clone_of: Optional[FromClause]
    _columns: ColumnCollection[Any, Any]
    schema: Optional[str] = None
    "Define the 'schema' attribute for this :class:`_expression.FromClause`.\n\n    This is typically ``None`` for most objects except that of\n    :class:`_schema.Table`, where it is taken as the value of the\n    :paramref:`_schema.Table.schema` argument.\n\n    "
    is_selectable = True
    _is_from_clause = True
    _is_join = False
    _use_schema_map = False

    def select(self) -> Select[Any]:
        if False:
            i = 10
            return i + 15
        'Return a SELECT of this :class:`_expression.FromClause`.\n\n\n        e.g.::\n\n            stmt = some_table.select().where(some_table.c.id == 5)\n\n        .. seealso::\n\n            :func:`_expression.select` - general purpose\n            method which allows for arbitrary column lists.\n\n        '
        return Select(self)

    def join(self, right: _FromClauseArgument, onclause: Optional[_ColumnExpressionArgument[bool]]=None, isouter: bool=False, full: bool=False) -> Join:
        if False:
            i = 10
            return i + 15
        'Return a :class:`_expression.Join` from this\n        :class:`_expression.FromClause`\n        to another :class:`FromClause`.\n\n        E.g.::\n\n            from sqlalchemy import join\n\n            j = user_table.join(address_table,\n                            user_table.c.id == address_table.c.user_id)\n            stmt = select(user_table).select_from(j)\n\n        would emit SQL along the lines of::\n\n            SELECT user.id, user.name FROM user\n            JOIN address ON user.id = address.user_id\n\n        :param right: the right side of the join; this is any\n         :class:`_expression.FromClause` object such as a\n         :class:`_schema.Table` object, and\n         may also be a selectable-compatible object such as an ORM-mapped\n         class.\n\n        :param onclause: a SQL expression representing the ON clause of the\n         join.  If left at ``None``, :meth:`_expression.FromClause.join`\n         will attempt to\n         join the two tables based on a foreign key relationship.\n\n        :param isouter: if True, render a LEFT OUTER JOIN, instead of JOIN.\n\n        :param full: if True, render a FULL OUTER JOIN, instead of LEFT OUTER\n         JOIN.  Implies :paramref:`.FromClause.join.isouter`.\n\n        .. seealso::\n\n            :func:`_expression.join` - standalone function\n\n            :class:`_expression.Join` - the type of object produced\n\n        '
        return Join(self, right, onclause, isouter, full)

    def outerjoin(self, right: _FromClauseArgument, onclause: Optional[_ColumnExpressionArgument[bool]]=None, full: bool=False) -> Join:
        if False:
            return 10
        'Return a :class:`_expression.Join` from this\n        :class:`_expression.FromClause`\n        to another :class:`FromClause`, with the "isouter" flag set to\n        True.\n\n        E.g.::\n\n            from sqlalchemy import outerjoin\n\n            j = user_table.outerjoin(address_table,\n                            user_table.c.id == address_table.c.user_id)\n\n        The above is equivalent to::\n\n            j = user_table.join(\n                address_table,\n                user_table.c.id == address_table.c.user_id,\n                isouter=True)\n\n        :param right: the right side of the join; this is any\n         :class:`_expression.FromClause` object such as a\n         :class:`_schema.Table` object, and\n         may also be a selectable-compatible object such as an ORM-mapped\n         class.\n\n        :param onclause: a SQL expression representing the ON clause of the\n         join.  If left at ``None``, :meth:`_expression.FromClause.join`\n         will attempt to\n         join the two tables based on a foreign key relationship.\n\n        :param full: if True, render a FULL OUTER JOIN, instead of\n         LEFT OUTER JOIN.\n\n        .. seealso::\n\n            :meth:`_expression.FromClause.join`\n\n            :class:`_expression.Join`\n\n        '
        return Join(self, right, onclause, True, full)

    def alias(self, name: Optional[str]=None, flat: bool=False) -> NamedFromClause:
        if False:
            for i in range(10):
                print('nop')
        "Return an alias of this :class:`_expression.FromClause`.\n\n        E.g.::\n\n            a2 = some_table.alias('a2')\n\n        The above code creates an :class:`_expression.Alias`\n        object which can be used\n        as a FROM clause in any SELECT statement.\n\n        .. seealso::\n\n            :ref:`tutorial_using_aliases`\n\n            :func:`_expression.alias`\n\n        "
        return Alias._construct(self, name=name)

    def tablesample(self, sampling: Union[float, Function[Any]], name: Optional[str]=None, seed: Optional[roles.ExpressionElementRole[Any]]=None) -> TableSample:
        if False:
            print('Hello World!')
        'Return a TABLESAMPLE alias of this :class:`_expression.FromClause`.\n\n        The return value is the :class:`_expression.TableSample`\n        construct also\n        provided by the top-level :func:`_expression.tablesample` function.\n\n        .. seealso::\n\n            :func:`_expression.tablesample` - usage guidelines and parameters\n\n        '
        return TableSample._construct(self, sampling=sampling, name=name, seed=seed)

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        if False:
            i = 10
            return i + 15
        "Return ``True`` if this :class:`_expression.FromClause` is\n        'derived' from the given ``FromClause``.\n\n        An example would be an Alias of a Table is derived from that Table.\n\n        "
        return fromclause in self._cloned_set

    def _is_lexical_equivalent(self, other: FromClause) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return ``True`` if this :class:`_expression.FromClause` and\n        the other represent the same lexical identity.\n\n        This tests if either one is a copy of the other, or\n        if they are the same via annotation identity.\n\n        '
        return bool(self._cloned_set.intersection(other._cloned_set))

    @util.ro_non_memoized_property
    def description(self) -> str:
        if False:
            print('Hello World!')
        'A brief description of this :class:`_expression.FromClause`.\n\n        Used primarily for error message formatting.\n\n        '
        return getattr(self, 'name', self.__class__.__name__ + ' object')

    def _generate_fromclause_column_proxies(self, fromclause: FromClause) -> None:
        if False:
            return 10
        fromclause._columns._populate_separate_keys((col._make_proxy(fromclause) for col in self.c))

    @util.ro_non_memoized_property
    def exported_columns(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        if False:
            for i in range(10):
                print('nop')
        'A :class:`_expression.ColumnCollection`\n        that represents the "exported"\n        columns of this :class:`_expression.Selectable`.\n\n        The "exported" columns for a :class:`_expression.FromClause`\n        object are synonymous\n        with the :attr:`_expression.FromClause.columns` collection.\n\n        .. versionadded:: 1.4\n\n        .. seealso::\n\n            :attr:`_expression.Selectable.exported_columns`\n\n            :attr:`_expression.SelectBase.exported_columns`\n\n\n        '
        return self.c

    @util.ro_non_memoized_property
    def columns(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        if False:
            for i in range(10):
                print('nop')
        'A named-based collection of :class:`_expression.ColumnElement`\n        objects maintained by this :class:`_expression.FromClause`.\n\n        The :attr:`.columns`, or :attr:`.c` collection, is the gateway\n        to the construction of SQL expressions using table-bound or\n        other selectable-bound columns::\n\n            select(mytable).where(mytable.c.somecolumn == 5)\n\n        :return: a :class:`.ColumnCollection` object.\n\n        '
        return self.c

    @util.ro_memoized_property
    def c(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        if False:
            i = 10
            return i + 15
        '\n        A synonym for :attr:`.FromClause.columns`\n\n        :return: a :class:`.ColumnCollection`\n\n        '
        if '_columns' not in self.__dict__:
            self._init_collections()
            self._populate_column_collection()
        return self._columns.as_readonly()

    @util.ro_non_memoized_property
    def entity_namespace(self) -> _EntityNamespace:
        if False:
            return 10
        'Return a namespace used for name-based access in SQL expressions.\n\n        This is the namespace that is used to resolve "filter_by()" type\n        expressions, such as::\n\n            stmt.filter_by(address=\'some address\')\n\n        It defaults to the ``.c`` collection, however internally it can\n        be overridden using the "entity_namespace" annotation to deliver\n        alternative results.\n\n        '
        return self.c

    @util.ro_memoized_property
    def primary_key(self) -> Iterable[NamedColumn[Any]]:
        if False:
            for i in range(10):
                print('nop')
        'Return the iterable collection of :class:`_schema.Column` objects\n        which comprise the primary key of this :class:`_selectable.FromClause`.\n\n        For a :class:`_schema.Table` object, this collection is represented\n        by the :class:`_schema.PrimaryKeyConstraint` which itself is an\n        iterable collection of :class:`_schema.Column` objects.\n\n        '
        self._init_collections()
        self._populate_column_collection()
        return self.primary_key

    @util.ro_memoized_property
    def foreign_keys(self) -> Iterable[ForeignKey]:
        if False:
            while True:
                i = 10
        'Return the collection of :class:`_schema.ForeignKey` marker objects\n        which this FromClause references.\n\n        Each :class:`_schema.ForeignKey` is a member of a\n        :class:`_schema.Table`-wide\n        :class:`_schema.ForeignKeyConstraint`.\n\n        .. seealso::\n\n            :attr:`_schema.Table.foreign_key_constraints`\n\n        '
        self._init_collections()
        self._populate_column_collection()
        return self.foreign_keys

    def _reset_column_collection(self) -> None:
        if False:
            return 10
        'Reset the attributes linked to the ``FromClause.c`` attribute.\n\n        This collection is separate from all the other memoized things\n        as it has shown to be sensitive to being cleared out in situations\n        where enclosing code, typically in a replacement traversal scenario,\n        has already established strong relationships\n        with the exported columns.\n\n        The collection is cleared for the case where a table is having a\n        column added to it as well as within a Join during copy internals.\n\n        '
        for key in ['_columns', 'columns', 'c', 'primary_key', 'foreign_keys']:
            self.__dict__.pop(key, None)

    @util.ro_non_memoized_property
    def _select_iterable(self) -> _SelectIterable:
        if False:
            return 10
        return (c for c in self.c if not _never_select_column(c))

    def _init_collections(self) -> None:
        if False:
            i = 10
            return i + 15
        assert '_columns' not in self.__dict__
        assert 'primary_key' not in self.__dict__
        assert 'foreign_keys' not in self.__dict__
        self._columns = ColumnCollection()
        self.primary_key = ColumnSet()
        self.foreign_keys = set()

    @property
    def _cols_populated(self) -> bool:
        if False:
            return 10
        return '_columns' in self.__dict__

    def _populate_column_collection(self) -> None:
        if False:
            i = 10
            return i + 15
        'Called on subclasses to establish the .c collection.\n\n        Each implementation has a different way of establishing\n        this collection.\n\n        '

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        if False:
            while True:
                i = 10
        'Given a column added to the .c collection of an underlying\n        selectable, produce the local version of that column, assuming this\n        selectable ultimately should proxy this column.\n\n        this is used to "ping" a derived selectable to add a new column\n        to its .c. collection when a Column has been added to one of the\n        Table objects it ultimately derives from.\n\n        If the given selectable hasn\'t populated its .c. collection yet,\n        it should at least pass on the message to the contained selectables,\n        but it will return None.\n\n        This method is currently used by Declarative to allow Table\n        columns to be added to a partially constructed inheritance\n        mapping that may have already produced joins.  The method\n        isn\'t public right now, as the full span of implications\n        and/or caveats aren\'t yet clear.\n\n        It\'s also possible that this functionality could be invoked by\n        default via an event, which would require that\n        selectables maintain a weak referencing collection of all\n        derivations.\n\n        '
        self._reset_column_collection()

    def _anonymous_fromclause(self, *, name: Optional[str]=None, flat: bool=False) -> FromClause:
        if False:
            print('Hello World!')
        return self.alias(name=name)
    if TYPE_CHECKING:

        def self_group(self, against: Optional[OperatorType]=None) -> Union[FromGrouping, Self]:
            if False:
                while True:
                    i = 10
            ...

class NamedFromClause(FromClause):
    """A :class:`.FromClause` that has a name.

    Examples include tables, subqueries, CTEs, aliased tables.

    .. versionadded:: 2.0

    """
    named_with_column = True
    name: str

    @util.preload_module('sqlalchemy.sql.sqltypes')
    def table_valued(self) -> TableValuedColumn[Any]:
        if False:
            while True:
                i = 10
        'Return a :class:`_sql.TableValuedColumn` object for this\n        :class:`_expression.FromClause`.\n\n        A :class:`_sql.TableValuedColumn` is a :class:`_sql.ColumnElement` that\n        represents a complete row in a table. Support for this construct is\n        backend dependent, and is supported in various forms by backends\n        such as PostgreSQL, Oracle and SQL Server.\n\n        E.g.:\n\n        .. sourcecode:: pycon+sql\n\n            >>> from sqlalchemy import select, column, func, table\n            >>> a = table("a", column("id"), column("x"), column("y"))\n            >>> stmt = select(func.row_to_json(a.table_valued()))\n            >>> print(stmt)\n            {printsql}SELECT row_to_json(a) AS row_to_json_1\n            FROM a\n\n        .. versionadded:: 1.4.0b2\n\n        .. seealso::\n\n            :ref:`tutorial_functions` - in the :ref:`unified_tutorial`\n\n        '
        return TableValuedColumn(self, type_api.TABLEVALUE)

class SelectLabelStyle(Enum):
    """Label style constants that may be passed to
    :meth:`_sql.Select.set_label_style`."""
    LABEL_STYLE_NONE = 0
    'Label style indicating no automatic labeling should be applied to the\n    columns clause of a SELECT statement.\n\n    Below, the columns named ``columna`` are both rendered as is, meaning that\n    the name ``columna`` can only refer to the first occurrence of this name\n    within a result set, as well as if the statement were used as a subquery:\n\n    .. sourcecode:: pycon+sql\n\n        >>> from sqlalchemy import table, column, select, true, LABEL_STYLE_NONE\n        >>> table1 = table("table1", column("columna"), column("columnb"))\n        >>> table2 = table("table2", column("columna"), column("columnc"))\n        >>> print(select(table1, table2).join(table2, true()).set_label_style(LABEL_STYLE_NONE))\n        {printsql}SELECT table1.columna, table1.columnb, table2.columna, table2.columnc\n        FROM table1 JOIN table2 ON true\n\n    Used with the :meth:`_sql.Select.set_label_style` method.\n\n    .. versionadded:: 1.4\n\n    '
    LABEL_STYLE_TABLENAME_PLUS_COL = 1
    'Label style indicating all columns should be labeled as\n    ``<tablename>_<columnname>`` when generating the columns clause of a SELECT\n    statement, to disambiguate same-named columns referenced from different\n    tables, aliases, or subqueries.\n\n    Below, all column names are given a label so that the two same-named\n    columns ``columna`` are disambiguated as ``table1_columna`` and\n    ``table2_columna``:\n\n    .. sourcecode:: pycon+sql\n\n        >>> from sqlalchemy import table, column, select, true, LABEL_STYLE_TABLENAME_PLUS_COL\n        >>> table1 = table("table1", column("columna"), column("columnb"))\n        >>> table2 = table("table2", column("columna"), column("columnc"))\n        >>> print(select(table1, table2).join(table2, true()).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL))\n        {printsql}SELECT table1.columna AS table1_columna, table1.columnb AS table1_columnb, table2.columna AS table2_columna, table2.columnc AS table2_columnc\n        FROM table1 JOIN table2 ON true\n\n    Used with the :meth:`_sql.GenerativeSelect.set_label_style` method.\n    Equivalent to the legacy method ``Select.apply_labels()``;\n    :data:`_sql.LABEL_STYLE_TABLENAME_PLUS_COL` is SQLAlchemy\'s legacy\n    auto-labeling style. :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY` provides a\n    less intrusive approach to disambiguation of same-named column expressions.\n\n\n    .. versionadded:: 1.4\n\n    '
    LABEL_STYLE_DISAMBIGUATE_ONLY = 2
    'Label style indicating that columns with a name that conflicts with\n    an existing name should be labeled with a semi-anonymizing label\n    when generating the columns clause of a SELECT statement.\n\n    Below, most column names are left unaffected, except for the second\n    occurrence of the name ``columna``, which is labeled using the\n    label ``columna_1`` to disambiguate it from that of ``tablea.columna``:\n\n    .. sourcecode:: pycon+sql\n\n        >>> from sqlalchemy import table, column, select, true, LABEL_STYLE_DISAMBIGUATE_ONLY\n        >>> table1 = table("table1", column("columna"), column("columnb"))\n        >>> table2 = table("table2", column("columna"), column("columnc"))\n        >>> print(select(table1, table2).join(table2, true()).set_label_style(LABEL_STYLE_DISAMBIGUATE_ONLY))\n        {printsql}SELECT table1.columna, table1.columnb, table2.columna AS columna_1, table2.columnc\n        FROM table1 JOIN table2 ON true\n\n    Used with the :meth:`_sql.GenerativeSelect.set_label_style` method,\n    :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY` is the default labeling style\n    for all SELECT statements outside of :term:`1.x style` ORM queries.\n\n    .. versionadded:: 1.4\n\n    '
    LABEL_STYLE_DEFAULT = LABEL_STYLE_DISAMBIGUATE_ONLY
    'The default label style, refers to\n    :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY`.\n\n    .. versionadded:: 1.4\n\n    '
    LABEL_STYLE_LEGACY_ORM = 3
(LABEL_STYLE_NONE, LABEL_STYLE_TABLENAME_PLUS_COL, LABEL_STYLE_DISAMBIGUATE_ONLY, _) = list(SelectLabelStyle)
LABEL_STYLE_DEFAULT = LABEL_STYLE_DISAMBIGUATE_ONLY

class Join(roles.DMLTableRole, FromClause):
    """Represent a ``JOIN`` construct between two
    :class:`_expression.FromClause`
    elements.

    The public constructor function for :class:`_expression.Join`
    is the module-level
    :func:`_expression.join()` function, as well as the
    :meth:`_expression.FromClause.join` method
    of any :class:`_expression.FromClause` (e.g. such as
    :class:`_schema.Table`).

    .. seealso::

        :func:`_expression.join`

        :meth:`_expression.FromClause.join`

    """
    __visit_name__ = 'join'
    _traverse_internals: _TraverseInternalsType = [('left', InternalTraversal.dp_clauseelement), ('right', InternalTraversal.dp_clauseelement), ('onclause', InternalTraversal.dp_clauseelement), ('isouter', InternalTraversal.dp_boolean), ('full', InternalTraversal.dp_boolean)]
    _is_join = True
    left: FromClause
    right: FromClause
    onclause: Optional[ColumnElement[bool]]
    isouter: bool
    full: bool

    def __init__(self, left: _FromClauseArgument, right: _FromClauseArgument, onclause: Optional[_OnClauseArgument]=None, isouter: bool=False, full: bool=False):
        if False:
            while True:
                i = 10
        'Construct a new :class:`_expression.Join`.\n\n        The usual entrypoint here is the :func:`_expression.join`\n        function or the :meth:`_expression.FromClause.join` method of any\n        :class:`_expression.FromClause` object.\n\n        '
        self.left = coercions.expect(roles.FromClauseRole, left)
        self.right = coercions.expect(roles.FromClauseRole, right).self_group()
        if onclause is None:
            self.onclause = self._match_primaries(self.left, self.right)
        else:
            self.onclause = coercions.expect(roles.OnClauseRole, onclause).self_group(against=operators._asbool)
        self.isouter = isouter
        self.full = full

    @util.ro_non_memoized_property
    def description(self) -> str:
        if False:
            print('Hello World!')
        return 'Join object on %s(%d) and %s(%d)' % (self.left.description, id(self.left), self.right.description, id(self.right))

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        if False:
            while True:
                i = 10
        return hash(fromclause) == hash(self) or self.left.is_derived_from(fromclause) or self.right.is_derived_from(fromclause)

    def self_group(self, against: Optional[OperatorType]=None) -> FromGrouping:
        if False:
            for i in range(10):
                print('nop')
        ...
        return FromGrouping(self)

    @util.preload_module('sqlalchemy.sql.util')
    def _populate_column_collection(self) -> None:
        if False:
            i = 10
            return i + 15
        sqlutil = util.preloaded.sql_util
        columns: List[KeyedColumnElement[Any]] = [c for c in self.left.c] + [c for c in self.right.c]
        self.primary_key.extend(sqlutil.reduce_columns((c for c in columns if c.primary_key), self.onclause))
        self._columns._populate_separate_keys(((col._tq_key_label, col) for col in columns))
        self.foreign_keys.update(itertools.chain(*[col.foreign_keys for col in columns]))

    def _copy_internals(self, clone: _CloneCallableType=_clone, **kw: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        all_the_froms = set(itertools.chain(_from_objects(self.left), _from_objects(self.right)))
        new_froms = {f: clone(f, **kw) for f in all_the_froms}

        def replace(obj: Union[BinaryExpression[Any], ColumnClause[Any]], **kw: Any) -> Optional[KeyedColumnElement[ColumnElement[Any]]]:
            if False:
                return 10
            if isinstance(obj, ColumnClause) and obj.table in new_froms:
                newelem = new_froms[obj.table].corresponding_column(obj)
                return newelem
            return None
        kw['replace'] = replace
        super()._copy_internals(clone=clone, **kw)
        self._reset_memoizations()

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        if False:
            return 10
        super()._refresh_for_new_column(column)
        self.left._refresh_for_new_column(column)
        self.right._refresh_for_new_column(column)

    def _match_primaries(self, left: FromClause, right: FromClause) -> ColumnElement[bool]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(left, Join):
            left_right = left.right
        else:
            left_right = None
        return self._join_condition(left, right, a_subset=left_right)

    @classmethod
    def _join_condition(cls, a: FromClause, b: FromClause, *, a_subset: Optional[FromClause]=None, consider_as_foreign_keys: Optional[AbstractSet[ColumnClause[Any]]]=None) -> ColumnElement[bool]:
        if False:
            return 10
        'Create a join condition between two tables or selectables.\n\n        See sqlalchemy.sql.util.join_condition() for full docs.\n\n        '
        constraints = cls._joincond_scan_left_right(a, a_subset, b, consider_as_foreign_keys)
        if len(constraints) > 1:
            cls._joincond_trim_constraints(a, b, constraints, consider_as_foreign_keys)
        if len(constraints) == 0:
            if isinstance(b, FromGrouping):
                hint = ' Perhaps you meant to convert the right side to a subquery using alias()?'
            else:
                hint = ''
            raise exc.NoForeignKeysError("Can't find any foreign key relationships between '%s' and '%s'.%s" % (a.description, b.description, hint))
        crit = [x == y for (x, y) in list(constraints.values())[0]]
        if len(crit) == 1:
            return crit[0]
        else:
            return and_(*crit)

    @classmethod
    def _can_join(cls, left: FromClause, right: FromClause, *, consider_as_foreign_keys: Optional[AbstractSet[ColumnClause[Any]]]=None) -> bool:
        if False:
            return 10
        if isinstance(left, Join):
            left_right = left.right
        else:
            left_right = None
        constraints = cls._joincond_scan_left_right(a=left, b=right, a_subset=left_right, consider_as_foreign_keys=consider_as_foreign_keys)
        return bool(constraints)

    @classmethod
    @util.preload_module('sqlalchemy.sql.util')
    def _joincond_scan_left_right(cls, a: FromClause, a_subset: Optional[FromClause], b: FromClause, consider_as_foreign_keys: Optional[AbstractSet[ColumnClause[Any]]]) -> collections.defaultdict[Optional[ForeignKeyConstraint], List[Tuple[ColumnClause[Any], ColumnClause[Any]]]]:
        if False:
            print('Hello World!')
        sql_util = util.preloaded.sql_util
        a = coercions.expect(roles.FromClauseRole, a)
        b = coercions.expect(roles.FromClauseRole, b)
        constraints: collections.defaultdict[Optional[ForeignKeyConstraint], List[Tuple[ColumnClause[Any], ColumnClause[Any]]]] = collections.defaultdict(list)
        for left in (a_subset, a):
            if left is None:
                continue
            for fk in sorted(b.foreign_keys, key=lambda fk: fk.parent._creation_order):
                if consider_as_foreign_keys is not None and fk.parent not in consider_as_foreign_keys:
                    continue
                try:
                    col = fk.get_referent(left)
                except exc.NoReferenceError as nrte:
                    table_names = {t.name for t in sql_util.find_tables(left)}
                    if nrte.table_name in table_names:
                        raise
                    else:
                        continue
                if col is not None:
                    constraints[fk.constraint].append((col, fk.parent))
            if left is not b:
                for fk in sorted(left.foreign_keys, key=lambda fk: fk.parent._creation_order):
                    if consider_as_foreign_keys is not None and fk.parent not in consider_as_foreign_keys:
                        continue
                    try:
                        col = fk.get_referent(b)
                    except exc.NoReferenceError as nrte:
                        table_names = {t.name for t in sql_util.find_tables(b)}
                        if nrte.table_name in table_names:
                            raise
                        else:
                            continue
                    if col is not None:
                        constraints[fk.constraint].append((col, fk.parent))
            if constraints:
                break
        return constraints

    @classmethod
    def _joincond_trim_constraints(cls, a: FromClause, b: FromClause, constraints: Dict[Any, Any], consider_as_foreign_keys: Optional[Any]) -> None:
        if False:
            print('Hello World!')
        if consider_as_foreign_keys:
            for const in list(constraints):
                if {f.parent for f in const.elements} != set(consider_as_foreign_keys):
                    del constraints[const]
        if len(constraints) > 1:
            dedupe = {tuple(crit) for crit in constraints.values()}
            if len(dedupe) == 1:
                key = list(constraints)[0]
                constraints = {key: constraints[key]}
        if len(constraints) != 1:
            raise exc.AmbiguousForeignKeysError("Can't determine join between '%s' and '%s'; tables have more than one foreign key constraint relationship between them. Please specify the 'onclause' of this join explicitly." % (a.description, b.description))

    def select(self) -> Select[Any]:
        if False:
            i = 10
            return i + 15
        'Create a :class:`_expression.Select` from this\n        :class:`_expression.Join`.\n\n        E.g.::\n\n            stmt = table_a.join(table_b, table_a.c.id == table_b.c.a_id)\n\n            stmt = stmt.select()\n\n        The above will produce a SQL string resembling::\n\n            SELECT table_a.id, table_a.col, table_b.id, table_b.a_id\n            FROM table_a JOIN table_b ON table_a.id = table_b.a_id\n\n        '
        return Select(self.left, self.right).select_from(self)

    @util.preload_module('sqlalchemy.sql.util')
    def _anonymous_fromclause(self, name: Optional[str]=None, flat: bool=False) -> TODO_Any:
        if False:
            return 10
        sqlutil = util.preloaded.sql_util
        if flat:
            if name is not None:
                raise exc.ArgumentError("Can't send name argument with flat")
            (left_a, right_a) = (self.left._anonymous_fromclause(flat=True), self.right._anonymous_fromclause(flat=True))
            adapter = sqlutil.ClauseAdapter(left_a).chain(sqlutil.ClauseAdapter(right_a))
            return left_a.join(right_a, adapter.traverse(self.onclause), isouter=self.isouter, full=self.full)
        else:
            return self.select().set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).correlate(None).alias(name)

    @util.ro_non_memoized_property
    def _hide_froms(self) -> Iterable[FromClause]:
        if False:
            print('Hello World!')
        return itertools.chain(*[_from_objects(x.left, x.right) for x in self._cloned_set])

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        if False:
            print('Hello World!')
        self_list: List[FromClause] = [self]
        return self_list + self.left._from_objects + self.right._from_objects

class NoInit:

    def __init__(self, *arg: Any, **kw: Any):
        if False:
            return 10
        raise NotImplementedError('The %s class is not intended to be constructed directly.  Please use the %s() standalone function or the %s() method available from appropriate selectable objects.' % (self.__class__.__name__, self.__class__.__name__.lower(), self.__class__.__name__.lower()))

class LateralFromClause(NamedFromClause):
    """mark a FROM clause as being able to render directly as LATERAL"""

class AliasedReturnsRows(NoInit, NamedFromClause):
    """Base class of aliases against tables, subqueries, and other
    selectables."""
    _is_from_container = True
    _supports_derived_columns = False
    element: ReturnsRows
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement), ('name', InternalTraversal.dp_anon_name)]

    @classmethod
    def _construct(cls, selectable: Any, *, name: Optional[str]=None, **kw: Any) -> Self:
        if False:
            print('Hello World!')
        obj = cls.__new__(cls)
        obj._init(selectable, name=name, **kw)
        return obj

    def _init(self, selectable: Any, *, name: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        self.element = coercions.expect(roles.ReturnsRowsRole, selectable, apply_propagate_attrs=self)
        self.element = selectable
        self._orig_name = name
        if name is None:
            if isinstance(selectable, FromClause) and selectable.named_with_column:
                name = getattr(selectable, 'name', None)
                if isinstance(name, _anonymous_label):
                    name = None
            name = _anonymous_label.safe_construct(id(self), name or 'anon')
        self.name = name

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        if False:
            while True:
                i = 10
        super()._refresh_for_new_column(column)
        self.element._refresh_for_new_column(column)

    def _populate_column_collection(self) -> None:
        if False:
            print('Hello World!')
        self.element._generate_fromclause_column_proxies(self)

    @util.ro_non_memoized_property
    def description(self) -> str:
        if False:
            print('Hello World!')
        name = self.name
        if isinstance(name, _anonymous_label):
            name = 'anon_1'
        return name

    @util.ro_non_memoized_property
    def implicit_returning(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.element.implicit_returning

    @property
    def original(self) -> ReturnsRows:
        if False:
            i = 10
            return i + 15
        'Legacy for dialects that are referring to Alias.original.'
        return self.element

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if fromclause in self._cloned_set:
            return True
        return self.element.is_derived_from(fromclause)

    def _copy_internals(self, clone: _CloneCallableType=_clone, **kw: Any) -> None:
        if False:
            print('Hello World!')
        existing_element = self.element
        super()._copy_internals(clone=clone, **kw)
        if existing_element is not self.element:
            self._reset_column_collection()

    @property
    def _from_objects(self) -> List[FromClause]:
        if False:
            print('Hello World!')
        return [self]

class FromClauseAlias(AliasedReturnsRows):
    element: FromClause

class Alias(roles.DMLTableRole, FromClauseAlias):
    """Represents an table or selectable alias (AS).

    Represents an alias, as typically applied to any table or
    sub-select within a SQL statement using the ``AS`` keyword (or
    without the keyword on certain databases such as Oracle).

    This object is constructed from the :func:`_expression.alias` module
    level function as well as the :meth:`_expression.FromClause.alias`
    method available
    on all :class:`_expression.FromClause` subclasses.

    .. seealso::

        :meth:`_expression.FromClause.alias`

    """
    __visit_name__ = 'alias'
    inherit_cache = True
    element: FromClause

    @classmethod
    def _factory(cls, selectable: FromClause, name: Optional[str]=None, flat: bool=False) -> NamedFromClause:
        if False:
            for i in range(10):
                print('nop')
        return coercions.expect(roles.FromClauseRole, selectable, allow_select=True).alias(name=name, flat=flat)

class TableValuedAlias(LateralFromClause, Alias):
    """An alias against a "table valued" SQL function.

    This construct provides for a SQL function that returns columns
    to be used in the FROM clause of a SELECT statement.   The
    object is generated using the :meth:`_functions.FunctionElement.table_valued`
    method, e.g.:

    .. sourcecode:: pycon+sql

        >>> from sqlalchemy import select, func
        >>> fn = func.json_array_elements_text('["one", "two", "three"]').table_valued("value")
        >>> print(select(fn.c.value))
        {printsql}SELECT anon_1.value
        FROM json_array_elements_text(:json_array_elements_text_1) AS anon_1

    .. versionadded:: 1.4.0b2

    .. seealso::

        :ref:`tutorial_functions_table_valued` - in the :ref:`unified_tutorial`

    """
    __visit_name__ = 'table_valued_alias'
    _supports_derived_columns = True
    _render_derived = False
    _render_derived_w_types = False
    joins_implicitly = False
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement), ('name', InternalTraversal.dp_anon_name), ('_tableval_type', InternalTraversal.dp_type), ('_render_derived', InternalTraversal.dp_boolean), ('_render_derived_w_types', InternalTraversal.dp_boolean)]

    def _init(self, selectable: Any, *, name: Optional[str]=None, table_value_type: Optional[TableValueType]=None, joins_implicitly: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        super()._init(selectable, name=name)
        self.joins_implicitly = joins_implicitly
        self._tableval_type = type_api.TABLEVALUE if table_value_type is None else table_value_type

    @HasMemoized.memoized_attribute
    def column(self) -> TableValuedColumn[Any]:
        if False:
            i = 10
            return i + 15
        'Return a column expression representing this\n        :class:`_sql.TableValuedAlias`.\n\n        This accessor is used to implement the\n        :meth:`_functions.FunctionElement.column_valued` method. See that\n        method for further details.\n\n        E.g.:\n\n        .. sourcecode:: pycon+sql\n\n            >>> print(select(func.some_func().table_valued("value").column))\n            {printsql}SELECT anon_1 FROM some_func() AS anon_1\n\n        .. seealso::\n\n            :meth:`_functions.FunctionElement.column_valued`\n\n        '
        return TableValuedColumn(self, self._tableval_type)

    def alias(self, name: Optional[str]=None, flat: bool=False) -> TableValuedAlias:
        if False:
            i = 10
            return i + 15
        'Return a new alias of this :class:`_sql.TableValuedAlias`.\n\n        This creates a distinct FROM object that will be distinguished\n        from the original one when used in a SQL statement.\n\n        '
        tva: TableValuedAlias = TableValuedAlias._construct(self, name=name, table_value_type=self._tableval_type, joins_implicitly=self.joins_implicitly)
        if self._render_derived:
            tva._render_derived = True
            tva._render_derived_w_types = self._render_derived_w_types
        return tva

    def lateral(self, name: Optional[str]=None) -> LateralFromClause:
        if False:
            while True:
                i = 10
        'Return a new :class:`_sql.TableValuedAlias` with the lateral flag\n        set, so that it renders as LATERAL.\n\n        .. seealso::\n\n            :func:`_expression.lateral`\n\n        '
        tva = self.alias(name=name)
        tva._is_lateral = True
        return tva

    def render_derived(self, name: Optional[str]=None, with_types: bool=False) -> TableValuedAlias:
        if False:
            i = 10
            return i + 15
        'Apply "render derived" to this :class:`_sql.TableValuedAlias`.\n\n        This has the effect of the individual column names listed out\n        after the alias name in the "AS" sequence, e.g.:\n\n        .. sourcecode:: pycon+sql\n\n            >>> print(\n            ...     select(\n            ...         func.unnest(array(["one", "two", "three"])).\n                        table_valued("x", with_ordinality="o").render_derived()\n            ...     )\n            ... )\n            {printsql}SELECT anon_1.x, anon_1.o\n            FROM unnest(ARRAY[%(param_1)s, %(param_2)s, %(param_3)s]) WITH ORDINALITY AS anon_1(x, o)\n\n        The ``with_types`` keyword will render column types inline within\n        the alias expression (this syntax currently applies to the\n        PostgreSQL database):\n\n        .. sourcecode:: pycon+sql\n\n            >>> print(\n            ...     select(\n            ...         func.json_to_recordset(\n            ...             \'[{"a":1,"b":"foo"},{"a":"2","c":"bar"}]\'\n            ...         )\n            ...         .table_valued(column("a", Integer), column("b", String))\n            ...         .render_derived(with_types=True)\n            ...     )\n            ... )\n            {printsql}SELECT anon_1.a, anon_1.b FROM json_to_recordset(:json_to_recordset_1)\n            AS anon_1(a INTEGER, b VARCHAR)\n\n        :param name: optional string name that will be applied to the alias\n         generated.  If left as None, a unique anonymizing name will be used.\n\n        :param with_types: if True, the derived columns will include the\n         datatype specification with each column. This is a special syntax\n         currently known to be required by PostgreSQL for some SQL functions.\n\n        '
        new_alias: TableValuedAlias = TableValuedAlias._construct(self.element, name=name, table_value_type=self._tableval_type, joins_implicitly=self.joins_implicitly)
        new_alias._render_derived = True
        new_alias._render_derived_w_types = with_types
        return new_alias

class Lateral(FromClauseAlias, LateralFromClause):
    """Represent a LATERAL subquery.

    This object is constructed from the :func:`_expression.lateral` module
    level function as well as the :meth:`_expression.FromClause.lateral`
    method available
    on all :class:`_expression.FromClause` subclasses.

    While LATERAL is part of the SQL standard, currently only more recent
    PostgreSQL versions provide support for this keyword.

    .. seealso::

        :ref:`tutorial_lateral_correlation` -  overview of usage.

    """
    __visit_name__ = 'lateral'
    _is_lateral = True
    inherit_cache = True

    @classmethod
    def _factory(cls, selectable: Union[SelectBase, _FromClauseArgument], name: Optional[str]=None) -> LateralFromClause:
        if False:
            for i in range(10):
                print('nop')
        return coercions.expect(roles.FromClauseRole, selectable, explicit_subquery=True).lateral(name=name)

class TableSample(FromClauseAlias):
    """Represent a TABLESAMPLE clause.

    This object is constructed from the :func:`_expression.tablesample` module
    level function as well as the :meth:`_expression.FromClause.tablesample`
    method
    available on all :class:`_expression.FromClause` subclasses.

    .. seealso::

        :func:`_expression.tablesample`

    """
    __visit_name__ = 'tablesample'
    _traverse_internals: _TraverseInternalsType = AliasedReturnsRows._traverse_internals + [('sampling', InternalTraversal.dp_clauseelement), ('seed', InternalTraversal.dp_clauseelement)]

    @classmethod
    def _factory(cls, selectable: _FromClauseArgument, sampling: Union[float, Function[Any]], name: Optional[str]=None, seed: Optional[roles.ExpressionElementRole[Any]]=None) -> TableSample:
        if False:
            print('Hello World!')
        return coercions.expect(roles.FromClauseRole, selectable).tablesample(sampling, name=name, seed=seed)

    @util.preload_module('sqlalchemy.sql.functions')
    def _init(self, selectable: Any, *, name: Optional[str]=None, sampling: Union[float, Function[Any]], seed: Optional[roles.ExpressionElementRole[Any]]=None) -> None:
        if False:
            return 10
        assert sampling is not None
        functions = util.preloaded.sql_functions
        if not isinstance(sampling, functions.Function):
            sampling = functions.func.system(sampling)
        self.sampling: Function[Any] = sampling
        self.seed = seed
        super()._init(selectable, name=name)

    def _get_method(self) -> Function[Any]:
        if False:
            i = 10
            return i + 15
        return self.sampling

class CTE(roles.DMLTableRole, roles.IsCTERole, Generative, HasPrefixes, HasSuffixes, AliasedReturnsRows):
    """Represent a Common Table Expression.

    The :class:`_expression.CTE` object is obtained using the
    :meth:`_sql.SelectBase.cte` method from any SELECT statement. A less often
    available syntax also allows use of the :meth:`_sql.HasCTE.cte` method
    present on :term:`DML` constructs such as :class:`_sql.Insert`,
    :class:`_sql.Update` and
    :class:`_sql.Delete`.   See the :meth:`_sql.HasCTE.cte` method for
    usage details on CTEs.

    .. seealso::

        :ref:`tutorial_subqueries_ctes` - in the 2.0 tutorial

        :meth:`_sql.HasCTE.cte` - examples of calling styles

    """
    __visit_name__ = 'cte'
    _traverse_internals: _TraverseInternalsType = AliasedReturnsRows._traverse_internals + [('_cte_alias', InternalTraversal.dp_clauseelement), ('_restates', InternalTraversal.dp_clauseelement), ('recursive', InternalTraversal.dp_boolean), ('nesting', InternalTraversal.dp_boolean)] + HasPrefixes._has_prefixes_traverse_internals + HasSuffixes._has_suffixes_traverse_internals
    element: HasCTE

    @classmethod
    def _factory(cls, selectable: HasCTE, name: Optional[str]=None, recursive: bool=False) -> CTE:
        if False:
            print('Hello World!')
        'Return a new :class:`_expression.CTE`,\n        or Common Table Expression instance.\n\n        Please see :meth:`_expression.HasCTE.cte` for detail on CTE usage.\n\n        '
        return coercions.expect(roles.HasCTERole, selectable).cte(name=name, recursive=recursive)

    def _init(self, selectable: Select[Any], *, name: Optional[str]=None, recursive: bool=False, nesting: bool=False, _cte_alias: Optional[CTE]=None, _restates: Optional[CTE]=None, _prefixes: Optional[Tuple[()]]=None, _suffixes: Optional[Tuple[()]]=None) -> None:
        if False:
            return 10
        self.recursive = recursive
        self.nesting = nesting
        self._cte_alias = _cte_alias
        self._restates = _restates
        if _prefixes:
            self._prefixes = _prefixes
        if _suffixes:
            self._suffixes = _suffixes
        super()._init(selectable, name=name)

    def _populate_column_collection(self) -> None:
        if False:
            print('Hello World!')
        if self._cte_alias is not None:
            self._cte_alias._generate_fromclause_column_proxies(self)
        else:
            self.element._generate_fromclause_column_proxies(self)

    def alias(self, name: Optional[str]=None, flat: bool=False) -> CTE:
        if False:
            while True:
                i = 10
        'Return an :class:`_expression.Alias` of this\n        :class:`_expression.CTE`.\n\n        This method is a CTE-specific specialization of the\n        :meth:`_expression.FromClause.alias` method.\n\n        .. seealso::\n\n            :ref:`tutorial_using_aliases`\n\n            :func:`_expression.alias`\n\n        '
        return CTE._construct(self.element, name=name, recursive=self.recursive, nesting=self.nesting, _cte_alias=self, _prefixes=self._prefixes, _suffixes=self._suffixes)

    def union(self, *other: _SelectStatementForCompoundArgument) -> CTE:
        if False:
            i = 10
            return i + 15
        'Return a new :class:`_expression.CTE` with a SQL ``UNION``\n        of the original CTE against the given selectables provided\n        as positional arguments.\n\n        :param \\*other: one or more elements with which to create a\n         UNION.\n\n         .. versionchanged:: 1.4.28 multiple elements are now accepted.\n\n        .. seealso::\n\n            :meth:`_sql.HasCTE.cte` - examples of calling styles\n\n        '
        assert is_select_statement(self.element), f'CTE element f{self.element} does not support union()'
        return CTE._construct(self.element.union(*other), name=self.name, recursive=self.recursive, nesting=self.nesting, _restates=self, _prefixes=self._prefixes, _suffixes=self._suffixes)

    def union_all(self, *other: _SelectStatementForCompoundArgument) -> CTE:
        if False:
            return 10
        'Return a new :class:`_expression.CTE` with a SQL ``UNION ALL``\n        of the original CTE against the given selectables provided\n        as positional arguments.\n\n        :param \\*other: one or more elements with which to create a\n         UNION.\n\n         .. versionchanged:: 1.4.28 multiple elements are now accepted.\n\n        .. seealso::\n\n            :meth:`_sql.HasCTE.cte` - examples of calling styles\n\n        '
        assert is_select_statement(self.element), f'CTE element f{self.element} does not support union_all()'
        return CTE._construct(self.element.union_all(*other), name=self.name, recursive=self.recursive, nesting=self.nesting, _restates=self, _prefixes=self._prefixes, _suffixes=self._suffixes)

    def _get_reference_cte(self) -> CTE:
        if False:
            for i in range(10):
                print('nop')
        '\n        A recursive CTE is updated to attach the recursive part.\n        Updated CTEs should still refer to the original CTE.\n        This function returns this reference identifier.\n        '
        return self._restates if self._restates is not None else self

class _CTEOpts(NamedTuple):
    nesting: bool

class _ColumnsPlusNames(NamedTuple):
    required_label_name: Optional[str]
    '\n    string label name, if non-None, must be rendered as a\n    label, i.e. "AS <name>"\n    '
    proxy_key: Optional[str]
    '\n    proxy_key that is to be part of the result map for this\n    col.  this is also the key in a fromclause.c or\n    select.selected_columns collection\n    '
    fallback_label_name: Optional[str]
    '\n    name that can be used to render an "AS <name>" when\n    we have to render a label even though\n    required_label_name was not given\n    '
    column: Union[ColumnElement[Any], TextClause]
    '\n    the ColumnElement itself\n    '
    repeated: bool
    '\n    True if this is a duplicate of a previous column\n    in the list of columns\n    '

class SelectsRows(ReturnsRows):
    """Sub-base of ReturnsRows for elements that deliver rows
    directly, namely SELECT and INSERT/UPDATE/DELETE..RETURNING"""
    _label_style: SelectLabelStyle = LABEL_STYLE_NONE

    def _generate_columns_plus_names(self, anon_for_dupe_key: bool, cols: Optional[_SelectIterable]=None) -> List[_ColumnsPlusNames]:
        if False:
            for i in range(10):
                print('nop')
        "Generate column names as rendered in a SELECT statement by\n        the compiler.\n\n        This is distinct from the _column_naming_convention generator that's\n        intended for population of .c collections and similar, which has\n        different rules.   the collection returned here calls upon the\n        _column_naming_convention as well.\n\n        "
        if cols is None:
            cols = self._all_selected_columns
        key_naming_convention = SelectState._column_naming_convention(self._label_style)
        names = {}
        result: List[_ColumnsPlusNames] = []
        result_append = result.append
        table_qualified = self._label_style is LABEL_STYLE_TABLENAME_PLUS_COL
        label_style_none = self._label_style is LABEL_STYLE_NONE
        dedupe_hash = 1
        for c in cols:
            repeated = False
            if not c._render_label_in_columns_clause:
                effective_name = required_label_name = fallback_label_name = None
            elif label_style_none:
                if TYPE_CHECKING:
                    assert is_column_element(c)
                effective_name = required_label_name = None
                fallback_label_name = c._non_anon_label or c._anon_name_label
            else:
                if TYPE_CHECKING:
                    assert is_column_element(c)
                if table_qualified:
                    required_label_name = effective_name = fallback_label_name = c._tq_label
                else:
                    effective_name = fallback_label_name = c._non_anon_label
                    required_label_name = None
                if effective_name is None:
                    expr_label = c._expression_label
                    if expr_label is None:
                        repeated = c._anon_name_label in names
                        names[c._anon_name_label] = c
                        effective_name = required_label_name = None
                        if repeated:
                            if table_qualified:
                                fallback_label_name = c._dedupe_anon_tq_label_idx(dedupe_hash)
                                dedupe_hash += 1
                            else:
                                fallback_label_name = c._dedupe_anon_label_idx(dedupe_hash)
                                dedupe_hash += 1
                        else:
                            fallback_label_name = c._anon_name_label
                    else:
                        required_label_name = effective_name = fallback_label_name = expr_label
            if effective_name is not None:
                if TYPE_CHECKING:
                    assert is_column_element(c)
                if effective_name in names:
                    if hash(names[effective_name]) != hash(c):
                        if table_qualified:
                            required_label_name = fallback_label_name = c._anon_tq_label
                        else:
                            required_label_name = fallback_label_name = c._anon_name_label
                        if anon_for_dupe_key and required_label_name in names:
                            assert hash(names[required_label_name]) == hash(c)
                            if table_qualified:
                                required_label_name = fallback_label_name = c._dedupe_anon_tq_label_idx(dedupe_hash)
                                dedupe_hash += 1
                            else:
                                required_label_name = fallback_label_name = c._dedupe_anon_label_idx(dedupe_hash)
                                dedupe_hash += 1
                            repeated = True
                        else:
                            names[required_label_name] = c
                    elif anon_for_dupe_key:
                        if table_qualified:
                            required_label_name = fallback_label_name = c._dedupe_anon_tq_label_idx(dedupe_hash)
                            dedupe_hash += 1
                        else:
                            required_label_name = fallback_label_name = c._dedupe_anon_label_idx(dedupe_hash)
                            dedupe_hash += 1
                        repeated = True
                else:
                    names[effective_name] = c
            result_append(_ColumnsPlusNames(required_label_name, key_naming_convention(c), fallback_label_name, c, repeated))
        return result

class HasCTE(roles.HasCTERole, SelectsRows):
    """Mixin that declares a class to include CTE support."""
    _has_ctes_traverse_internals: _TraverseInternalsType = [('_independent_ctes', InternalTraversal.dp_clauseelement_list), ('_independent_ctes_opts', InternalTraversal.dp_plain_obj)]
    _independent_ctes: Tuple[CTE, ...] = ()
    _independent_ctes_opts: Tuple[_CTEOpts, ...] = ()

    @_generative
    def add_cte(self, *ctes: CTE, nest_here: bool=False) -> Self:
        if False:
            while True:
                i = 10
        'Add one or more :class:`_sql.CTE` constructs to this statement.\n\n        This method will associate the given :class:`_sql.CTE` constructs with\n        the parent statement such that they will each be unconditionally\n        rendered in the WITH clause of the final statement, even if not\n        referenced elsewhere within the statement or any sub-selects.\n\n        The optional :paramref:`.HasCTE.add_cte.nest_here` parameter when set\n        to True will have the effect that each given :class:`_sql.CTE` will\n        render in a WITH clause rendered directly along with this statement,\n        rather than being moved to the top of the ultimate rendered statement,\n        even if this statement is rendered as a subquery within a larger\n        statement.\n\n        This method has two general uses. One is to embed CTE statements that\n        serve some purpose without being referenced explicitly, such as the use\n        case of embedding a DML statement such as an INSERT or UPDATE as a CTE\n        inline with a primary statement that may draw from its results\n        indirectly.  The other is to provide control over the exact placement\n        of a particular series of CTE constructs that should remain rendered\n        directly in terms of a particular statement that may be nested in a\n        larger statement.\n\n        E.g.::\n\n            from sqlalchemy import table, column, select\n            t = table(\'t\', column(\'c1\'), column(\'c2\'))\n\n            ins = t.insert().values({"c1": "x", "c2": "y"}).cte()\n\n            stmt = select(t).add_cte(ins)\n\n        Would render::\n\n            WITH anon_1 AS\n            (INSERT INTO t (c1, c2) VALUES (:param_1, :param_2))\n            SELECT t.c1, t.c2\n            FROM t\n\n        Above, the "anon_1" CTE is not referenced in the SELECT\n        statement, however still accomplishes the task of running an INSERT\n        statement.\n\n        Similarly in a DML-related context, using the PostgreSQL\n        :class:`_postgresql.Insert` construct to generate an "upsert"::\n\n            from sqlalchemy import table, column\n            from sqlalchemy.dialects.postgresql import insert\n\n            t = table("t", column("c1"), column("c2"))\n\n            delete_statement_cte = (\n                t.delete().where(t.c.c1 < 1).cte("deletions")\n            )\n\n            insert_stmt = insert(t).values({"c1": 1, "c2": 2})\n            update_statement = insert_stmt.on_conflict_do_update(\n                index_elements=[t.c.c1],\n                set_={\n                    "c1": insert_stmt.excluded.c1,\n                    "c2": insert_stmt.excluded.c2,\n                },\n            ).add_cte(delete_statement_cte)\n\n            print(update_statement)\n\n        The above statement renders as::\n\n            WITH deletions AS\n            (DELETE FROM t WHERE t.c1 < %(c1_1)s)\n            INSERT INTO t (c1, c2) VALUES (%(c1)s, %(c2)s)\n            ON CONFLICT (c1) DO UPDATE SET c1 = excluded.c1, c2 = excluded.c2\n\n        .. versionadded:: 1.4.21\n\n        :param \\*ctes: zero or more :class:`.CTE` constructs.\n\n         .. versionchanged:: 2.0  Multiple CTE instances are accepted\n\n        :param nest_here: if True, the given CTE or CTEs will be rendered\n         as though they specified the :paramref:`.HasCTE.cte.nesting` flag\n         to ``True`` when they were added to this :class:`.HasCTE`.\n         Assuming the given CTEs are not referenced in an outer-enclosing\n         statement as well, the CTEs given should render at the level of\n         this statement when this flag is given.\n\n         .. versionadded:: 2.0\n\n         .. seealso::\n\n            :paramref:`.HasCTE.cte.nesting`\n\n\n        '
        opt = _CTEOpts(nest_here)
        for cte in ctes:
            cte = coercions.expect(roles.IsCTERole, cte)
            self._independent_ctes += (cte,)
            self._independent_ctes_opts += (opt,)
        return self

    def cte(self, name: Optional[str]=None, recursive: bool=False, nesting: bool=False) -> CTE:
        if False:
            for i in range(10):
                print('nop')
        'Return a new :class:`_expression.CTE`,\n        or Common Table Expression instance.\n\n        Common table expressions are a SQL standard whereby SELECT\n        statements can draw upon secondary statements specified along\n        with the primary statement, using a clause called "WITH".\n        Special semantics regarding UNION can also be employed to\n        allow "recursive" queries, where a SELECT statement can draw\n        upon the set of rows that have previously been selected.\n\n        CTEs can also be applied to DML constructs UPDATE, INSERT\n        and DELETE on some databases, both as a source of CTE rows\n        when combined with RETURNING, as well as a consumer of\n        CTE rows.\n\n        SQLAlchemy detects :class:`_expression.CTE` objects, which are treated\n        similarly to :class:`_expression.Alias` objects, as special elements\n        to be delivered to the FROM clause of the statement as well\n        as to a WITH clause at the top of the statement.\n\n        For special prefixes such as PostgreSQL "MATERIALIZED" and\n        "NOT MATERIALIZED", the :meth:`_expression.CTE.prefix_with`\n        method may be\n        used to establish these.\n\n        .. versionchanged:: 1.3.13 Added support for prefixes.\n           In particular - MATERIALIZED and NOT MATERIALIZED.\n\n        :param name: name given to the common table expression.  Like\n         :meth:`_expression.FromClause.alias`, the name can be left as\n         ``None`` in which case an anonymous symbol will be used at query\n         compile time.\n        :param recursive: if ``True``, will render ``WITH RECURSIVE``.\n         A recursive common table expression is intended to be used in\n         conjunction with UNION ALL in order to derive rows\n         from those already selected.\n        :param nesting: if ``True``, will render the CTE locally to the\n         statement in which it is referenced.   For more complex scenarios,\n         the :meth:`.HasCTE.add_cte` method using the\n         :paramref:`.HasCTE.add_cte.nest_here`\n         parameter may also be used to more carefully\n         control the exact placement of a particular CTE.\n\n         .. versionadded:: 1.4.24\n\n         .. seealso::\n\n            :meth:`.HasCTE.add_cte`\n\n        The following examples include two from PostgreSQL\'s documentation at\n        https://www.postgresql.org/docs/current/static/queries-with.html,\n        as well as additional examples.\n\n        Example 1, non recursive::\n\n            from sqlalchemy import (Table, Column, String, Integer,\n                                    MetaData, select, func)\n\n            metadata = MetaData()\n\n            orders = Table(\'orders\', metadata,\n                Column(\'region\', String),\n                Column(\'amount\', Integer),\n                Column(\'product\', String),\n                Column(\'quantity\', Integer)\n            )\n\n            regional_sales = select(\n                                orders.c.region,\n                                func.sum(orders.c.amount).label(\'total_sales\')\n                            ).group_by(orders.c.region).cte("regional_sales")\n\n\n            top_regions = select(regional_sales.c.region).\\\n                    where(\n                        regional_sales.c.total_sales >\n                        select(\n                            func.sum(regional_sales.c.total_sales) / 10\n                        )\n                    ).cte("top_regions")\n\n            statement = select(\n                        orders.c.region,\n                        orders.c.product,\n                        func.sum(orders.c.quantity).label("product_units"),\n                        func.sum(orders.c.amount).label("product_sales")\n                ).where(orders.c.region.in_(\n                    select(top_regions.c.region)\n                )).group_by(orders.c.region, orders.c.product)\n\n            result = conn.execute(statement).fetchall()\n\n        Example 2, WITH RECURSIVE::\n\n            from sqlalchemy import (Table, Column, String, Integer,\n                                    MetaData, select, func)\n\n            metadata = MetaData()\n\n            parts = Table(\'parts\', metadata,\n                Column(\'part\', String),\n                Column(\'sub_part\', String),\n                Column(\'quantity\', Integer),\n            )\n\n            included_parts = select(\\\n                parts.c.sub_part, parts.c.part, parts.c.quantity\\\n                ).\\\n                where(parts.c.part==\'our part\').\\\n                cte(recursive=True)\n\n\n            incl_alias = included_parts.alias()\n            parts_alias = parts.alias()\n            included_parts = included_parts.union_all(\n                select(\n                    parts_alias.c.sub_part,\n                    parts_alias.c.part,\n                    parts_alias.c.quantity\n                ).\\\n                where(parts_alias.c.part==incl_alias.c.sub_part)\n            )\n\n            statement = select(\n                        included_parts.c.sub_part,\n                        func.sum(included_parts.c.quantity).\n                          label(\'total_quantity\')\n                    ).\\\n                    group_by(included_parts.c.sub_part)\n\n            result = conn.execute(statement).fetchall()\n\n        Example 3, an upsert using UPDATE and INSERT with CTEs::\n\n            from datetime import date\n            from sqlalchemy import (MetaData, Table, Column, Integer,\n                                    Date, select, literal, and_, exists)\n\n            metadata = MetaData()\n\n            visitors = Table(\'visitors\', metadata,\n                Column(\'product_id\', Integer, primary_key=True),\n                Column(\'date\', Date, primary_key=True),\n                Column(\'count\', Integer),\n            )\n\n            # add 5 visitors for the product_id == 1\n            product_id = 1\n            day = date.today()\n            count = 5\n\n            update_cte = (\n                visitors.update()\n                .where(and_(visitors.c.product_id == product_id,\n                            visitors.c.date == day))\n                .values(count=visitors.c.count + count)\n                .returning(literal(1))\n                .cte(\'update_cte\')\n            )\n\n            upsert = visitors.insert().from_select(\n                [visitors.c.product_id, visitors.c.date, visitors.c.count],\n                select(literal(product_id), literal(day), literal(count))\n                    .where(~exists(update_cte.select()))\n            )\n\n            connection.execute(upsert)\n\n        Example 4, Nesting CTE (SQLAlchemy 1.4.24 and above)::\n\n            value_a = select(\n                literal("root").label("n")\n            ).cte("value_a")\n\n            # A nested CTE with the same name as the root one\n            value_a_nested = select(\n                literal("nesting").label("n")\n            ).cte("value_a", nesting=True)\n\n            # Nesting CTEs takes ascendency locally\n            # over the CTEs at a higher level\n            value_b = select(value_a_nested.c.n).cte("value_b")\n\n            value_ab = select(value_a.c.n.label("a"), value_b.c.n.label("b"))\n\n        The above query will render the second CTE nested inside the first,\n        shown with inline parameters below as::\n\n            WITH\n                value_a AS\n                    (SELECT \'root\' AS n),\n                value_b AS\n                    (WITH value_a AS\n                        (SELECT \'nesting\' AS n)\n                    SELECT value_a.n AS n FROM value_a)\n            SELECT value_a.n AS a, value_b.n AS b\n            FROM value_a, value_b\n\n        The same CTE can be set up using the :meth:`.HasCTE.add_cte` method\n        as follows (SQLAlchemy 2.0 and above)::\n\n            value_a = select(\n                literal("root").label("n")\n            ).cte("value_a")\n\n            # A nested CTE with the same name as the root one\n            value_a_nested = select(\n                literal("nesting").label("n")\n            ).cte("value_a")\n\n            # Nesting CTEs takes ascendency locally\n            # over the CTEs at a higher level\n            value_b = (\n                select(value_a_nested.c.n).\n                add_cte(value_a_nested, nest_here=True).\n                cte("value_b")\n            )\n\n            value_ab = select(value_a.c.n.label("a"), value_b.c.n.label("b"))\n\n        Example 5, Non-Linear CTE (SQLAlchemy 1.4.28 and above)::\n\n            edge = Table(\n                "edge",\n                metadata,\n                Column("id", Integer, primary_key=True),\n                Column("left", Integer),\n                Column("right", Integer),\n            )\n\n            root_node = select(literal(1).label("node")).cte(\n                "nodes", recursive=True\n            )\n\n            left_edge = select(edge.c.left).join(\n                root_node, edge.c.right == root_node.c.node\n            )\n            right_edge = select(edge.c.right).join(\n                root_node, edge.c.left == root_node.c.node\n            )\n\n            subgraph_cte = root_node.union(left_edge, right_edge)\n\n            subgraph = select(subgraph_cte)\n\n        The above query will render 2 UNIONs inside the recursive CTE::\n\n            WITH RECURSIVE nodes(node) AS (\n                    SELECT 1 AS node\n                UNION\n                    SELECT edge."left" AS "left"\n                    FROM edge JOIN nodes ON edge."right" = nodes.node\n                UNION\n                    SELECT edge."right" AS "right"\n                    FROM edge JOIN nodes ON edge."left" = nodes.node\n            )\n            SELECT nodes.node FROM nodes\n\n        .. seealso::\n\n            :meth:`_orm.Query.cte` - ORM version of\n            :meth:`_expression.HasCTE.cte`.\n\n        '
        return CTE._construct(self, name=name, recursive=recursive, nesting=nesting)

class Subquery(AliasedReturnsRows):
    """Represent a subquery of a SELECT.

    A :class:`.Subquery` is created by invoking the
    :meth:`_expression.SelectBase.subquery` method, or for convenience the
    :meth:`_expression.SelectBase.alias` method, on any
    :class:`_expression.SelectBase` subclass
    which includes :class:`_expression.Select`,
    :class:`_expression.CompoundSelect`, and
    :class:`_expression.TextualSelect`.  As rendered in a FROM clause,
    it represents the
    body of the SELECT statement inside of parenthesis, followed by the usual
    "AS <somename>" that defines all "alias" objects.

    The :class:`.Subquery` object is very similar to the
    :class:`_expression.Alias`
    object and can be used in an equivalent way.    The difference between
    :class:`_expression.Alias` and :class:`.Subquery` is that
    :class:`_expression.Alias` always
    contains a :class:`_expression.FromClause` object whereas
    :class:`.Subquery`
    always contains a :class:`_expression.SelectBase` object.

    .. versionadded:: 1.4 The :class:`.Subquery` class was added which now
       serves the purpose of providing an aliased version of a SELECT
       statement.

    """
    __visit_name__ = 'subquery'
    _is_subquery = True
    inherit_cache = True
    element: SelectBase

    @classmethod
    def _factory(cls, selectable: SelectBase, name: Optional[str]=None) -> Subquery:
        if False:
            i = 10
            return i + 15
        'Return a :class:`.Subquery` object.'
        return coercions.expect(roles.SelectStatementRole, selectable).subquery(name=name)

    @util.deprecated('1.4', 'The :meth:`.Subquery.as_scalar` method, which was previously ``Alias.as_scalar()`` prior to version 1.4, is deprecated and will be removed in a future release; Please use the :meth:`_expression.Select.scalar_subquery` method of the :func:`_expression.select` construct before constructing a subquery object, or with the ORM use the :meth:`_query.Query.scalar_subquery` method.')
    def as_scalar(self) -> ScalarSelect[Any]:
        if False:
            for i in range(10):
                print('nop')
        return self.element.set_label_style(LABEL_STYLE_NONE).scalar_subquery()

class FromGrouping(GroupedElement, FromClause):
    """Represent a grouping of a FROM clause"""
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement)]
    element: FromClause

    def __init__(self, element: FromClause):
        if False:
            return 10
        self.element = coercions.expect(roles.FromClauseRole, element)

    def _init_collections(self) -> None:
        if False:
            print('Hello World!')
        pass

    @util.ro_non_memoized_property
    def columns(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        if False:
            return 10
        return self.element.columns

    @util.ro_non_memoized_property
    def c(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        if False:
            print('Hello World!')
        return self.element.columns

    @property
    def primary_key(self) -> Iterable[NamedColumn[Any]]:
        if False:
            while True:
                i = 10
        return self.element.primary_key

    @property
    def foreign_keys(self) -> Iterable[ForeignKey]:
        if False:
            print('Hello World!')
        return self.element.foreign_keys

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        if False:
            while True:
                i = 10
        return self.element.is_derived_from(fromclause)

    def alias(self, name: Optional[str]=None, flat: bool=False) -> NamedFromGrouping:
        if False:
            for i in range(10):
                print('nop')
        return NamedFromGrouping(self.element.alias(name=name, flat=flat))

    def _anonymous_fromclause(self, **kw: Any) -> FromGrouping:
        if False:
            print('Hello World!')
        return FromGrouping(self.element._anonymous_fromclause(**kw))

    @util.ro_non_memoized_property
    def _hide_froms(self) -> Iterable[FromClause]:
        if False:
            while True:
                i = 10
        return self.element._hide_froms

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        if False:
            for i in range(10):
                print('nop')
        return self.element._from_objects

    def __getstate__(self) -> Dict[str, FromClause]:
        if False:
            print('Hello World!')
        return {'element': self.element}

    def __setstate__(self, state: Dict[str, FromClause]) -> None:
        if False:
            print('Hello World!')
        self.element = state['element']

class NamedFromGrouping(FromGrouping, NamedFromClause):
    """represent a grouping of a named FROM clause

    .. versionadded:: 2.0

    """
    inherit_cache = True

class TableClause(roles.DMLTableRole, Immutable, NamedFromClause):
    """Represents a minimal "table" construct.

    This is a lightweight table object that has only a name, a
    collection of columns, which are typically produced
    by the :func:`_expression.column` function, and a schema::

        from sqlalchemy import table, column

        user = table("user",
                column("id"),
                column("name"),
                column("description"),
        )

    The :class:`_expression.TableClause` construct serves as the base for
    the more commonly used :class:`_schema.Table` object, providing
    the usual set of :class:`_expression.FromClause` services including
    the ``.c.`` collection and statement generation methods.

    It does **not** provide all the additional schema-level services
    of :class:`_schema.Table`, including constraints, references to other
    tables, or support for :class:`_schema.MetaData`-level services.
    It's useful
    on its own as an ad-hoc construct used to generate quick SQL
    statements when a more fully fledged :class:`_schema.Table`
    is not on hand.

    """
    __visit_name__ = 'table'
    _traverse_internals: _TraverseInternalsType = [('columns', InternalTraversal.dp_fromclause_canonical_column_collection), ('name', InternalTraversal.dp_string), ('schema', InternalTraversal.dp_string)]
    _is_table = True
    fullname: str
    implicit_returning = False
    ":class:`_expression.TableClause`\n    doesn't support having a primary key or column\n    -level defaults, so implicit returning doesn't apply."

    @util.ro_memoized_property
    def _autoincrement_column(self) -> Optional[ColumnClause[Any]]:
        if False:
            while True:
                i = 10
        'No PK or default support so no autoincrement column.'
        return None

    def __init__(self, name: str, *columns: ColumnClause[Any], **kw: Any):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.name = name
        self._columns = DedupeColumnCollection()
        self.primary_key = ColumnSet()
        self.foreign_keys = set()
        for c in columns:
            self.append_column(c)
        schema = kw.pop('schema', None)
        if schema is not None:
            self.schema = schema
        if self.schema is not None:
            self.fullname = '%s.%s' % (self.schema, self.name)
        else:
            self.fullname = self.name
        if kw:
            raise exc.ArgumentError('Unsupported argument(s): %s' % list(kw))
    if TYPE_CHECKING:

        @util.ro_non_memoized_property
        def columns(self) -> ReadOnlyColumnCollection[str, ColumnClause[Any]]:
            if False:
                i = 10
                return i + 15
            ...

        @util.ro_non_memoized_property
        def c(self) -> ReadOnlyColumnCollection[str, ColumnClause[Any]]:
            if False:
                while True:
                    i = 10
            ...

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        if self.schema is not None:
            return self.schema + '.' + self.name
        else:
            return self.name

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        if False:
            print('Hello World!')
        pass

    def _init_collections(self) -> None:
        if False:
            print('Hello World!')
        pass

    @util.ro_memoized_property
    def description(self) -> str:
        if False:
            while True:
                i = 10
        return self.name

    def append_column(self, c: ColumnClause[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        existing = c.table
        if existing is not None and existing is not self:
            raise exc.ArgumentError("column object '%s' already assigned to table '%s'" % (c.key, existing))
        self._columns.add(c)
        c.table = self

    @util.preload_module('sqlalchemy.sql.dml')
    def insert(self) -> util.preloaded.sql_dml.Insert:
        if False:
            while True:
                i = 10
        "Generate an :class:`_sql.Insert` construct against this\n        :class:`_expression.TableClause`.\n\n        E.g.::\n\n            table.insert().values(name='foo')\n\n        See :func:`_expression.insert` for argument and usage information.\n\n        "
        return util.preloaded.sql_dml.Insert(self)

    @util.preload_module('sqlalchemy.sql.dml')
    def update(self) -> Update:
        if False:
            while True:
                i = 10
        "Generate an :func:`_expression.update` construct against this\n        :class:`_expression.TableClause`.\n\n        E.g.::\n\n            table.update().where(table.c.id==7).values(name='foo')\n\n        See :func:`_expression.update` for argument and usage information.\n\n        "
        return util.preloaded.sql_dml.Update(self)

    @util.preload_module('sqlalchemy.sql.dml')
    def delete(self) -> Delete:
        if False:
            for i in range(10):
                print('nop')
        'Generate a :func:`_expression.delete` construct against this\n        :class:`_expression.TableClause`.\n\n        E.g.::\n\n            table.delete().where(table.c.id==7)\n\n        See :func:`_expression.delete` for argument and usage information.\n\n        '
        return util.preloaded.sql_dml.Delete(self)

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        if False:
            print('Hello World!')
        return [self]
ForUpdateParameter = Union['ForUpdateArg', None, bool, Dict[str, Any]]

class ForUpdateArg(ClauseElement):
    _traverse_internals: _TraverseInternalsType = [('of', InternalTraversal.dp_clauseelement_list), ('nowait', InternalTraversal.dp_boolean), ('read', InternalTraversal.dp_boolean), ('skip_locked', InternalTraversal.dp_boolean)]
    of: Optional[Sequence[ClauseElement]]
    nowait: bool
    read: bool
    skip_locked: bool

    @classmethod
    def _from_argument(cls, with_for_update: ForUpdateParameter) -> Optional[ForUpdateArg]:
        if False:
            return 10
        if isinstance(with_for_update, ForUpdateArg):
            return with_for_update
        elif with_for_update in (None, False):
            return None
        elif with_for_update is True:
            return ForUpdateArg()
        else:
            return ForUpdateArg(**cast('Dict[str, Any]', with_for_update))

    def __eq__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        return isinstance(other, ForUpdateArg) and other.nowait == self.nowait and (other.read == self.read) and (other.skip_locked == self.skip_locked) and (other.key_share == self.key_share) and (other.of is self.of)

    def __ne__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return not self.__eq__(other)

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return id(self)

    def __init__(self, *, nowait: bool=False, read: bool=False, of: Optional[_ForUpdateOfArgument]=None, skip_locked: bool=False, key_share: bool=False):
        if False:
            print('Hello World!')
        'Represents arguments specified to\n        :meth:`_expression.Select.for_update`.\n\n        '
        self.nowait = nowait
        self.read = read
        self.skip_locked = skip_locked
        self.key_share = key_share
        if of is not None:
            self.of = [coercions.expect(roles.ColumnsClauseRole, elem) for elem in util.to_list(of)]
        else:
            self.of = None

class Values(roles.InElementRole, Generative, LateralFromClause):
    """Represent a ``VALUES`` construct that can be used as a FROM element
    in a statement.

    The :class:`_expression.Values` object is created from the
    :func:`_expression.values` function.

    .. versionadded:: 1.4

    """
    __visit_name__ = 'values'
    _data: Tuple[Sequence[Tuple[Any, ...]], ...] = ()
    _unnamed: bool
    _traverse_internals: _TraverseInternalsType = [('_column_args', InternalTraversal.dp_clauseelement_list), ('_data', InternalTraversal.dp_dml_multi_values), ('name', InternalTraversal.dp_string), ('literal_binds', InternalTraversal.dp_boolean)]

    def __init__(self, *columns: ColumnClause[Any], name: Optional[str]=None, literal_binds: bool=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._column_args = columns
        if name is None:
            self._unnamed = True
            self.name = _anonymous_label.safe_construct(id(self), 'anon')
        else:
            self._unnamed = False
            self.name = name
        self.literal_binds = literal_binds
        self.named_with_column = not self._unnamed

    @property
    def _column_types(self) -> List[TypeEngine[Any]]:
        if False:
            while True:
                i = 10
        return [col.type for col in self._column_args]

    @_generative
    def alias(self, name: Optional[str]=None, flat: bool=False) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Return a new :class:`_expression.Values`\n        construct that is a copy of this\n        one with the given name.\n\n        This method is a VALUES-specific specialization of the\n        :meth:`_expression.FromClause.alias` method.\n\n        .. seealso::\n\n            :ref:`tutorial_using_aliases`\n\n            :func:`_expression.alias`\n\n        '
        non_none_name: str
        if name is None:
            non_none_name = _anonymous_label.safe_construct(id(self), 'anon')
        else:
            non_none_name = name
        self.name = non_none_name
        self.named_with_column = True
        self._unnamed = False
        return self

    @_generative
    def lateral(self, name: Optional[str]=None) -> LateralFromClause:
        if False:
            while True:
                i = 10
        'Return a new :class:`_expression.Values` with the lateral flag set,\n        so that\n        it renders as LATERAL.\n\n        .. seealso::\n\n            :func:`_expression.lateral`\n\n        '
        non_none_name: str
        if name is None:
            non_none_name = self.name
        else:
            non_none_name = name
        self._is_lateral = True
        self.name = non_none_name
        self._unnamed = False
        return self

    @_generative
    def data(self, values: Sequence[Tuple[Any, ...]]) -> Self:
        if False:
            i = 10
            return i + 15
        "Return a new :class:`_expression.Values` construct,\n        adding the given data to the data list.\n\n        E.g.::\n\n            my_values = my_values.data([(1, 'value 1'), (2, 'value2')])\n\n        :param values: a sequence (i.e. list) of tuples that map to the\n         column expressions given in the :class:`_expression.Values`\n         constructor.\n\n        "
        self._data += (values,)
        return self

    def scalar_values(self) -> ScalarValues:
        if False:
            i = 10
            return i + 15
        'Returns a scalar ``VALUES`` construct that can be used as a\n        COLUMN element in a statement.\n\n        .. versionadded:: 2.0.0b4\n\n        '
        return ScalarValues(self._column_args, self._data, self.literal_binds)

    def _populate_column_collection(self) -> None:
        if False:
            i = 10
            return i + 15
        for c in self._column_args:
            if c.table is not None and c.table is not self:
                (_, c) = c._make_proxy(self)
            else:
                c._reset_memoizations()
            self._columns.add(c)
            c.table = self

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        if False:
            for i in range(10):
                print('nop')
        return [self]

class ScalarValues(roles.InElementRole, GroupedElement, ColumnElement[Any]):
    """Represent a scalar ``VALUES`` construct that can be used as a
    COLUMN element in a statement.

    The :class:`_expression.ScalarValues` object is created from the
    :meth:`_expression.Values.scalar_values` method. It's also
    automatically generated when a :class:`_expression.Values` is used in
    an ``IN`` or ``NOT IN`` condition.

    .. versionadded:: 2.0.0b4

    """
    __visit_name__ = 'scalar_values'
    _traverse_internals: _TraverseInternalsType = [('_column_args', InternalTraversal.dp_clauseelement_list), ('_data', InternalTraversal.dp_dml_multi_values), ('literal_binds', InternalTraversal.dp_boolean)]

    def __init__(self, columns: Sequence[ColumnClause[Any]], data: Tuple[Sequence[Tuple[Any, ...]], ...], literal_binds: bool):
        if False:
            print('Hello World!')
        super().__init__()
        self._column_args = columns
        self._data = data
        self.literal_binds = literal_binds

    @property
    def _column_types(self) -> List[TypeEngine[Any]]:
        if False:
            i = 10
            return i + 15
        return [col.type for col in self._column_args]

    def __clause_element__(self) -> ScalarValues:
        if False:
            return 10
        return self

class SelectBase(roles.SelectStatementRole, roles.DMLSelectRole, roles.CompoundElementRole, roles.InElementRole, HasCTE, SupportsCloneAnnotations, Selectable):
    """Base class for SELECT statements.


    This includes :class:`_expression.Select`,
    :class:`_expression.CompoundSelect` and
    :class:`_expression.TextualSelect`.


    """
    _is_select_base = True
    is_select = True
    _label_style: SelectLabelStyle = LABEL_STYLE_NONE

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        if False:
            return 10
        self._reset_memoizations()

    @util.ro_non_memoized_property
    def selected_columns(self) -> ColumnCollection[str, ColumnElement[Any]]:
        if False:
            for i in range(10):
                print('nop')
        'A :class:`_expression.ColumnCollection`\n        representing the columns that\n        this SELECT statement or similar construct returns in its result set.\n\n        This collection differs from the :attr:`_expression.FromClause.columns`\n        collection of a :class:`_expression.FromClause` in that the columns\n        within this collection cannot be directly nested inside another SELECT\n        statement; a subquery must be applied first which provides for the\n        necessary parenthesization required by SQL.\n\n        .. note::\n\n            The :attr:`_sql.SelectBase.selected_columns` collection does not\n            include expressions established in the columns clause using the\n            :func:`_sql.text` construct; these are silently omitted from the\n            collection. To use plain textual column expressions inside of a\n            :class:`_sql.Select` construct, use the :func:`_sql.literal_column`\n            construct.\n\n        .. seealso::\n\n            :attr:`_sql.Select.selected_columns`\n\n        .. versionadded:: 1.4\n\n        '
        raise NotImplementedError()

    def _generate_fromclause_column_proxies(self, subquery: FromClause, *, proxy_compound_columns: Optional[Iterable[Sequence[ColumnElement[Any]]]]=None) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        if False:
            print('Hello World!')
        'A sequence of expressions that correspond to what is rendered\n        in the columns clause, including :class:`_sql.TextClause`\n        constructs.\n\n        .. versionadded:: 1.4.12\n\n        .. seealso::\n\n            :attr:`_sql.SelectBase.exported_columns`\n\n        '
        raise NotImplementedError()

    @property
    def exported_columns(self) -> ReadOnlyColumnCollection[str, ColumnElement[Any]]:
        if False:
            print('Hello World!')
        'A :class:`_expression.ColumnCollection`\n        that represents the "exported"\n        columns of this :class:`_expression.Selectable`, not including\n        :class:`_sql.TextClause` constructs.\n\n        The "exported" columns for a :class:`_expression.SelectBase`\n        object are synonymous\n        with the :attr:`_expression.SelectBase.selected_columns` collection.\n\n        .. versionadded:: 1.4\n\n        .. seealso::\n\n            :attr:`_expression.Select.exported_columns`\n\n            :attr:`_expression.Selectable.exported_columns`\n\n            :attr:`_expression.FromClause.exported_columns`\n\n\n        '
        return self.selected_columns.as_readonly()

    @property
    @util.deprecated('1.4', 'The :attr:`_expression.SelectBase.c` and :attr:`_expression.SelectBase.columns` attributes are deprecated and will be removed in a future release; these attributes implicitly create a subquery that should be explicit.  Please call :meth:`_expression.SelectBase.subquery` first in order to create a subquery, which then contains this attribute.  To access the columns that this SELECT object SELECTs from, use the :attr:`_expression.SelectBase.selected_columns` attribute.')
    def c(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        if False:
            return 10
        return self._implicit_subquery.columns

    @property
    def columns(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        if False:
            print('Hello World!')
        return self.c

    def get_label_style(self) -> SelectLabelStyle:
        if False:
            return 10
        '\n        Retrieve the current label style.\n\n        Implemented by subclasses.\n\n        '
        raise NotImplementedError()

    def set_label_style(self, style: SelectLabelStyle) -> Self:
        if False:
            print('Hello World!')
        'Return a new selectable with the specified label style.\n\n        Implemented by subclasses.\n\n        '
        raise NotImplementedError()

    @util.deprecated('1.4', 'The :meth:`_expression.SelectBase.select` method is deprecated and will be removed in a future release; this method implicitly creates a subquery that should be explicit.  Please call :meth:`_expression.SelectBase.subquery` first in order to create a subquery, which then can be selected.')
    def select(self, *arg: Any, **kw: Any) -> Select[Any]:
        if False:
            return 10
        return self._implicit_subquery.select(*arg, **kw)

    @HasMemoized.memoized_attribute
    def _implicit_subquery(self) -> Subquery:
        if False:
            return 10
        return self.subquery()

    def _scalar_type(self) -> TypeEngine[Any]:
        if False:
            return 10
        raise NotImplementedError()

    @util.deprecated('1.4', 'The :meth:`_expression.SelectBase.as_scalar` method is deprecated and will be removed in a future release.  Please refer to :meth:`_expression.SelectBase.scalar_subquery`.')
    def as_scalar(self) -> ScalarSelect[Any]:
        if False:
            return 10
        return self.scalar_subquery()

    def exists(self) -> Exists:
        if False:
            while True:
                i = 10
        'Return an :class:`_sql.Exists` representation of this selectable,\n        which can be used as a column expression.\n\n        The returned object is an instance of :class:`_sql.Exists`.\n\n        .. seealso::\n\n            :func:`_sql.exists`\n\n            :ref:`tutorial_exists` - in the :term:`2.0 style` tutorial.\n\n        .. versionadded:: 1.4\n\n        '
        return Exists(self)

    def scalar_subquery(self) -> ScalarSelect[Any]:
        if False:
            return 10
        "Return a 'scalar' representation of this selectable, which can be\n        used as a column expression.\n\n        The returned object is an instance of :class:`_sql.ScalarSelect`.\n\n        Typically, a select statement which has only one column in its columns\n        clause is eligible to be used as a scalar expression.  The scalar\n        subquery can then be used in the WHERE clause or columns clause of\n        an enclosing SELECT.\n\n        Note that the scalar subquery differentiates from the FROM-level\n        subquery that can be produced using the\n        :meth:`_expression.SelectBase.subquery`\n        method.\n\n        .. versionchanged: 1.4 - the ``.as_scalar()`` method was renamed to\n           :meth:`_expression.SelectBase.scalar_subquery`.\n\n        .. seealso::\n\n            :ref:`tutorial_scalar_subquery` - in the 2.0 tutorial\n\n        "
        if self._label_style is not LABEL_STYLE_NONE:
            self = self.set_label_style(LABEL_STYLE_NONE)
        return ScalarSelect(self)

    def label(self, name: Optional[str]) -> Label[Any]:
        if False:
            print('Hello World!')
        "Return a 'scalar' representation of this selectable, embedded as a\n        subquery with a label.\n\n        .. seealso::\n\n            :meth:`_expression.SelectBase.scalar_subquery`.\n\n        "
        return self.scalar_subquery().label(name)

    def lateral(self, name: Optional[str]=None) -> LateralFromClause:
        if False:
            while True:
                i = 10
        'Return a LATERAL alias of this :class:`_expression.Selectable`.\n\n        The return value is the :class:`_expression.Lateral` construct also\n        provided by the top-level :func:`_expression.lateral` function.\n\n        .. seealso::\n\n            :ref:`tutorial_lateral_correlation` -  overview of usage.\n\n        '
        return Lateral._factory(self, name)

    def subquery(self, name: Optional[str]=None) -> Subquery:
        if False:
            print('Hello World!')
        'Return a subquery of this :class:`_expression.SelectBase`.\n\n        A subquery is from a SQL perspective a parenthesized, named\n        construct that can be placed in the FROM clause of another\n        SELECT statement.\n\n        Given a SELECT statement such as::\n\n            stmt = select(table.c.id, table.c.name)\n\n        The above statement might look like::\n\n            SELECT table.id, table.name FROM table\n\n        The subquery form by itself renders the same way, however when\n        embedded into the FROM clause of another SELECT statement, it becomes\n        a named sub-element::\n\n            subq = stmt.subquery()\n            new_stmt = select(subq)\n\n        The above renders as::\n\n            SELECT anon_1.id, anon_1.name\n            FROM (SELECT table.id, table.name FROM table) AS anon_1\n\n        Historically, :meth:`_expression.SelectBase.subquery`\n        is equivalent to calling\n        the :meth:`_expression.FromClause.alias`\n        method on a FROM object; however,\n        as a :class:`_expression.SelectBase`\n        object is not directly  FROM object,\n        the :meth:`_expression.SelectBase.subquery`\n        method provides clearer semantics.\n\n        .. versionadded:: 1.4\n\n        '
        return Subquery._construct(self._ensure_disambiguated_names(), name=name)

    def _ensure_disambiguated_names(self) -> Self:
        if False:
            i = 10
            return i + 15
        'Ensure that the names generated by this selectbase will be\n        disambiguated in some way, if possible.\n\n        '
        raise NotImplementedError()

    def alias(self, name: Optional[str]=None, flat: bool=False) -> Subquery:
        if False:
            i = 10
            return i + 15
        'Return a named subquery against this\n        :class:`_expression.SelectBase`.\n\n        For a :class:`_expression.SelectBase` (as opposed to a\n        :class:`_expression.FromClause`),\n        this returns a :class:`.Subquery` object which behaves mostly the\n        same as the :class:`_expression.Alias` object that is used with a\n        :class:`_expression.FromClause`.\n\n        .. versionchanged:: 1.4 The :meth:`_expression.SelectBase.alias`\n           method is now\n           a synonym for the :meth:`_expression.SelectBase.subquery` method.\n\n        '
        return self.subquery(name=name)
_SB = TypeVar('_SB', bound=SelectBase)

class SelectStatementGrouping(GroupedElement, SelectBase, Generic[_SB]):
    """Represent a grouping of a :class:`_expression.SelectBase`.

    This differs from :class:`.Subquery` in that we are still
    an "inner" SELECT statement, this is strictly for grouping inside of
    compound selects.

    """
    __visit_name__ = 'select_statement_grouping'
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement)]
    _is_select_container = True
    element: _SB

    def __init__(self, element: _SB) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.element = cast(_SB, coercions.expect(roles.SelectStatementRole, element))

    def _ensure_disambiguated_names(self) -> SelectStatementGrouping[_SB]:
        if False:
            return 10
        new_element = self.element._ensure_disambiguated_names()
        if new_element is not self.element:
            return SelectStatementGrouping(new_element)
        else:
            return self

    def get_label_style(self) -> SelectLabelStyle:
        if False:
            while True:
                i = 10
        return self.element.get_label_style()

    def set_label_style(self, label_style: SelectLabelStyle) -> SelectStatementGrouping[_SB]:
        if False:
            for i in range(10):
                print('nop')
        return SelectStatementGrouping(self.element.set_label_style(label_style))

    @property
    def select_statement(self) -> _SB:
        if False:
            print('Hello World!')
        return self.element

    def self_group(self, against: Optional[OperatorType]=None) -> Self:
        if False:
            i = 10
            return i + 15
        ...
        return self
    if TYPE_CHECKING:

        def _ungroup(self) -> _SB:
            if False:
                i = 10
                return i + 15
            ...

    def _generate_fromclause_column_proxies(self, subquery: FromClause, *, proxy_compound_columns: Optional[Iterable[Sequence[ColumnElement[Any]]]]=None) -> None:
        if False:
            return 10
        self.element._generate_fromclause_column_proxies(subquery, proxy_compound_columns=proxy_compound_columns)

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        if False:
            print('Hello World!')
        return self.element._all_selected_columns

    @util.ro_non_memoized_property
    def selected_columns(self) -> ColumnCollection[str, ColumnElement[Any]]:
        if False:
            for i in range(10):
                print('nop')
        'A :class:`_expression.ColumnCollection`\n        representing the columns that\n        the embedded SELECT statement returns in its result set, not including\n        :class:`_sql.TextClause` constructs.\n\n        .. versionadded:: 1.4\n\n        .. seealso::\n\n            :attr:`_sql.Select.selected_columns`\n\n        '
        return self.element.selected_columns

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        if False:
            i = 10
            return i + 15
        return self.element._from_objects

class GenerativeSelect(SelectBase, Generative):
    """Base class for SELECT statements where additional elements can be
    added.

    This serves as the base for :class:`_expression.Select` and
    :class:`_expression.CompoundSelect`
    where elements such as ORDER BY, GROUP BY can be added and column
    rendering can be controlled.  Compare to
    :class:`_expression.TextualSelect`, which,
    while it subclasses :class:`_expression.SelectBase`
    and is also a SELECT construct,
    represents a fixed textual string which cannot be altered at this level,
    only wrapped as a subquery.

    """
    _order_by_clauses: Tuple[ColumnElement[Any], ...] = ()
    _group_by_clauses: Tuple[ColumnElement[Any], ...] = ()
    _limit_clause: Optional[ColumnElement[Any]] = None
    _offset_clause: Optional[ColumnElement[Any]] = None
    _fetch_clause: Optional[ColumnElement[Any]] = None
    _fetch_clause_options: Optional[Dict[str, bool]] = None
    _for_update_arg: Optional[ForUpdateArg] = None

    def __init__(self, _label_style: SelectLabelStyle=LABEL_STYLE_DEFAULT):
        if False:
            print('Hello World!')
        self._label_style = _label_style

    @_generative
    def with_for_update(self, *, nowait: bool=False, read: bool=False, of: Optional[_ForUpdateOfArgument]=None, skip_locked: bool=False, key_share: bool=False) -> Self:
        if False:
            while True:
                i = 10
        'Specify a ``FOR UPDATE`` clause for this\n        :class:`_expression.GenerativeSelect`.\n\n        E.g.::\n\n            stmt = select(table).with_for_update(nowait=True)\n\n        On a database like PostgreSQL or Oracle, the above would render a\n        statement like::\n\n            SELECT table.a, table.b FROM table FOR UPDATE NOWAIT\n\n        on other backends, the ``nowait`` option is ignored and instead\n        would produce::\n\n            SELECT table.a, table.b FROM table FOR UPDATE\n\n        When called with no arguments, the statement will render with\n        the suffix ``FOR UPDATE``.   Additional arguments can then be\n        provided which allow for common database-specific\n        variants.\n\n        :param nowait: boolean; will render ``FOR UPDATE NOWAIT`` on Oracle\n         and PostgreSQL dialects.\n\n        :param read: boolean; will render ``LOCK IN SHARE MODE`` on MySQL,\n         ``FOR SHARE`` on PostgreSQL.  On PostgreSQL, when combined with\n         ``nowait``, will render ``FOR SHARE NOWAIT``.\n\n        :param of: SQL expression or list of SQL expression elements,\n         (typically :class:`_schema.Column` objects or a compatible expression,\n         for some backends may also be a table expression) which will render\n         into a ``FOR UPDATE OF`` clause; supported by PostgreSQL, Oracle, some\n         MySQL versions and possibly others. May render as a table or as a\n         column depending on backend.\n\n        :param skip_locked: boolean, will render ``FOR UPDATE SKIP LOCKED``\n         on Oracle and PostgreSQL dialects or ``FOR SHARE SKIP LOCKED`` if\n         ``read=True`` is also specified.\n\n        :param key_share: boolean, will render ``FOR NO KEY UPDATE``,\n         or if combined with ``read=True`` will render ``FOR KEY SHARE``,\n         on the PostgreSQL dialect.\n\n        '
        self._for_update_arg = ForUpdateArg(nowait=nowait, read=read, of=of, skip_locked=skip_locked, key_share=key_share)
        return self

    def get_label_style(self) -> SelectLabelStyle:
        if False:
            i = 10
            return i + 15
        '\n        Retrieve the current label style.\n\n        .. versionadded:: 1.4\n\n        '
        return self._label_style

    def set_label_style(self, style: SelectLabelStyle) -> Self:
        if False:
            while True:
                i = 10
        'Return a new selectable with the specified label style.\n\n        There are three "label styles" available,\n        :attr:`_sql.SelectLabelStyle.LABEL_STYLE_DISAMBIGUATE_ONLY`,\n        :attr:`_sql.SelectLabelStyle.LABEL_STYLE_TABLENAME_PLUS_COL`, and\n        :attr:`_sql.SelectLabelStyle.LABEL_STYLE_NONE`.   The default style is\n        :attr:`_sql.SelectLabelStyle.LABEL_STYLE_TABLENAME_PLUS_COL`.\n\n        In modern SQLAlchemy, there is not generally a need to change the\n        labeling style, as per-expression labels are more effectively used by\n        making use of the :meth:`_sql.ColumnElement.label` method. In past\n        versions, :data:`_sql.LABEL_STYLE_TABLENAME_PLUS_COL` was used to\n        disambiguate same-named columns from different tables, aliases, or\n        subqueries; the newer :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY` now\n        applies labels only to names that conflict with an existing name so\n        that the impact of this labeling is minimal.\n\n        The rationale for disambiguation is mostly so that all column\n        expressions are available from a given :attr:`_sql.FromClause.c`\n        collection when a subquery is created.\n\n        .. versionadded:: 1.4 - the\n            :meth:`_sql.GenerativeSelect.set_label_style` method replaces the\n            previous combination of ``.apply_labels()``, ``.with_labels()`` and\n            ``use_labels=True`` methods and/or parameters.\n\n        .. seealso::\n\n            :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY`\n\n            :data:`_sql.LABEL_STYLE_TABLENAME_PLUS_COL`\n\n            :data:`_sql.LABEL_STYLE_NONE`\n\n            :data:`_sql.LABEL_STYLE_DEFAULT`\n\n        '
        if self._label_style is not style:
            self = self._generate()
            self._label_style = style
        return self

    @property
    def _group_by_clause(self) -> ClauseList:
        if False:
            return 10
        'ClauseList access to group_by_clauses for legacy dialects'
        return ClauseList._construct_raw(operators.comma_op, self._group_by_clauses)

    @property
    def _order_by_clause(self) -> ClauseList:
        if False:
            for i in range(10):
                print('nop')
        'ClauseList access to order_by_clauses for legacy dialects'
        return ClauseList._construct_raw(operators.comma_op, self._order_by_clauses)

    def _offset_or_limit_clause(self, element: _LimitOffsetType, name: Optional[str]=None, type_: Optional[_TypeEngineArgument[int]]=None) -> ColumnElement[Any]:
        if False:
            print('Hello World!')
        'Convert the given value to an "offset or limit" clause.\n\n        This handles incoming integers and converts to an expression; if\n        an expression is already given, it is passed through.\n\n        '
        return coercions.expect(roles.LimitOffsetRole, element, name=name, type_=type_)

    @overload
    def _offset_or_limit_clause_asint(self, clause: ColumnElement[Any], attrname: str) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def _offset_or_limit_clause_asint(self, clause: Optional[_OffsetLimitParam], attrname: str) -> Optional[int]:
        if False:
            while True:
                i = 10
        ...

    def _offset_or_limit_clause_asint(self, clause: Optional[ColumnElement[Any]], attrname: str) -> Union[NoReturn, Optional[int]]:
        if False:
            i = 10
            return i + 15
        'Convert the "offset or limit" clause of a select construct to an\n        integer.\n\n        This is only possible if the value is stored as a simple bound\n        parameter. Otherwise, a compilation error is raised.\n\n        '
        if clause is None:
            return None
        try:
            value = clause._limit_offset_value
        except AttributeError as err:
            raise exc.CompileError('This SELECT structure does not use a simple integer value for %s' % attrname) from err
        else:
            return util.asint(value)

    @property
    def _limit(self) -> Optional[int]:
        if False:
            print('Hello World!')
        "Get an integer value for the limit.  This should only be used\n        by code that cannot support a limit as a BindParameter or\n        other custom clause as it will throw an exception if the limit\n        isn't currently set to an integer.\n\n        "
        return self._offset_or_limit_clause_asint(self._limit_clause, 'limit')

    def _simple_int_clause(self, clause: ClauseElement) -> bool:
        if False:
            print('Hello World!')
        'True if the clause is a simple integer, False\n        if it is not present or is a SQL expression.\n        '
        return isinstance(clause, _OffsetLimitParam)

    @property
    def _offset(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        "Get an integer value for the offset.  This should only be used\n        by code that cannot support an offset as a BindParameter or\n        other custom clause as it will throw an exception if the\n        offset isn't currently set to an integer.\n\n        "
        return self._offset_or_limit_clause_asint(self._offset_clause, 'offset')

    @property
    def _has_row_limiting_clause(self) -> bool:
        if False:
            while True:
                i = 10
        return self._limit_clause is not None or self._offset_clause is not None or self._fetch_clause is not None

    @_generative
    def limit(self, limit: _LimitOffsetType) -> Self:
        if False:
            print('Hello World!')
        "Return a new selectable with the given LIMIT criterion\n        applied.\n\n        This is a numerical value which usually renders as a ``LIMIT``\n        expression in the resulting select.  Backends that don't\n        support ``LIMIT`` will attempt to provide similar\n        functionality.\n\n        .. note::\n\n           The :meth:`_sql.GenerativeSelect.limit` method will replace\n           any clause applied with :meth:`_sql.GenerativeSelect.fetch`.\n\n        :param limit: an integer LIMIT parameter, or a SQL expression\n         that provides an integer result. Pass ``None`` to reset it.\n\n        .. seealso::\n\n           :meth:`_sql.GenerativeSelect.fetch`\n\n           :meth:`_sql.GenerativeSelect.offset`\n\n        "
        self._fetch_clause = self._fetch_clause_options = None
        self._limit_clause = self._offset_or_limit_clause(limit)
        return self

    @_generative
    def fetch(self, count: _LimitOffsetType, with_ties: bool=False, percent: bool=False) -> Self:
        if False:
            print('Hello World!')
        'Return a new selectable with the given FETCH FIRST criterion\n        applied.\n\n        This is a numeric value which usually renders as\n        ``FETCH {FIRST | NEXT} [ count ] {ROW | ROWS} {ONLY | WITH TIES}``\n        expression in the resulting select. This functionality is\n        is currently implemented for Oracle, PostgreSQL, MSSQL.\n\n        Use :meth:`_sql.GenerativeSelect.offset` to specify the offset.\n\n        .. note::\n\n           The :meth:`_sql.GenerativeSelect.fetch` method will replace\n           any clause applied with :meth:`_sql.GenerativeSelect.limit`.\n\n        .. versionadded:: 1.4\n\n        :param count: an integer COUNT parameter, or a SQL expression\n         that provides an integer result. When ``percent=True`` this will\n         represent the percentage of rows to return, not the absolute value.\n         Pass ``None`` to reset it.\n\n        :param with_ties: When ``True``, the WITH TIES option is used\n         to return any additional rows that tie for the last place in the\n         result set according to the ``ORDER BY`` clause. The\n         ``ORDER BY`` may be mandatory in this case. Defaults to ``False``\n\n        :param percent: When ``True``, ``count`` represents the percentage\n         of the total number of selected rows to return. Defaults to ``False``\n\n        .. seealso::\n\n           :meth:`_sql.GenerativeSelect.limit`\n\n           :meth:`_sql.GenerativeSelect.offset`\n\n        '
        self._limit_clause = None
        if count is None:
            self._fetch_clause = self._fetch_clause_options = None
        else:
            self._fetch_clause = self._offset_or_limit_clause(count)
            self._fetch_clause_options = {'with_ties': with_ties, 'percent': percent}
        return self

    @_generative
    def offset(self, offset: _LimitOffsetType) -> Self:
        if False:
            return 10
        "Return a new selectable with the given OFFSET criterion\n        applied.\n\n\n        This is a numeric value which usually renders as an ``OFFSET``\n        expression in the resulting select.  Backends that don't\n        support ``OFFSET`` will attempt to provide similar\n        functionality.\n\n        :param offset: an integer OFFSET parameter, or a SQL expression\n         that provides an integer result. Pass ``None`` to reset it.\n\n        .. seealso::\n\n           :meth:`_sql.GenerativeSelect.limit`\n\n           :meth:`_sql.GenerativeSelect.fetch`\n\n        "
        self._offset_clause = self._offset_or_limit_clause(offset)
        return self

    @_generative
    @util.preload_module('sqlalchemy.sql.util')
    def slice(self, start: int, stop: int) -> Self:
        if False:
            for i in range(10):
                print('nop')
        "Apply LIMIT / OFFSET to this statement based on a slice.\n\n        The start and stop indices behave like the argument to Python's\n        built-in :func:`range` function. This method provides an\n        alternative to using ``LIMIT``/``OFFSET`` to get a slice of the\n        query.\n\n        For example, ::\n\n            stmt = select(User).order_by(User).id.slice(1, 3)\n\n        renders as\n\n        .. sourcecode:: sql\n\n           SELECT users.id AS users_id,\n                  users.name AS users_name\n           FROM users ORDER BY users.id\n           LIMIT ? OFFSET ?\n           (2, 1)\n\n        .. note::\n\n           The :meth:`_sql.GenerativeSelect.slice` method will replace\n           any clause applied with :meth:`_sql.GenerativeSelect.fetch`.\n\n        .. versionadded:: 1.4  Added the :meth:`_sql.GenerativeSelect.slice`\n           method generalized from the ORM.\n\n        .. seealso::\n\n           :meth:`_sql.GenerativeSelect.limit`\n\n           :meth:`_sql.GenerativeSelect.offset`\n\n           :meth:`_sql.GenerativeSelect.fetch`\n\n        "
        sql_util = util.preloaded.sql_util
        self._fetch_clause = self._fetch_clause_options = None
        (self._limit_clause, self._offset_clause) = sql_util._make_slice(self._limit_clause, self._offset_clause, start, stop)
        return self

    @_generative
    def order_by(self, __first: Union[Literal[None, _NoArg.NO_ARG], _ColumnExpressionOrStrLabelArgument[Any]]=_NoArg.NO_ARG, /, *clauses: _ColumnExpressionOrStrLabelArgument[Any]) -> Self:
        if False:
            return 10
        'Return a new selectable with the given list of ORDER BY\n        criteria applied.\n\n        e.g.::\n\n            stmt = select(table).order_by(table.c.id, table.c.name)\n\n        Calling this method multiple times is equivalent to calling it once\n        with all the clauses concatenated. All existing ORDER BY criteria may\n        be cancelled by passing ``None`` by itself.  New ORDER BY criteria may\n        then be added by invoking :meth:`_orm.Query.order_by` again, e.g.::\n\n            # will erase all ORDER BY and ORDER BY new_col alone\n            stmt = stmt.order_by(None).order_by(new_col)\n\n        :param \\*clauses: a series of :class:`_expression.ColumnElement`\n         constructs\n         which will be used to generate an ORDER BY clause.\n\n        .. seealso::\n\n            :ref:`tutorial_order_by` - in the :ref:`unified_tutorial`\n\n            :ref:`tutorial_order_by_label` - in the :ref:`unified_tutorial`\n\n        '
        if not clauses and __first is None:
            self._order_by_clauses = ()
        elif __first is not _NoArg.NO_ARG:
            self._order_by_clauses += tuple((coercions.expect(roles.OrderByRole, clause, apply_propagate_attrs=self) for clause in (__first,) + clauses))
        return self

    @_generative
    def group_by(self, __first: Union[Literal[None, _NoArg.NO_ARG], _ColumnExpressionOrStrLabelArgument[Any]]=_NoArg.NO_ARG, /, *clauses: _ColumnExpressionOrStrLabelArgument[Any]) -> Self:
        if False:
            return 10
        'Return a new selectable with the given list of GROUP BY\n        criterion applied.\n\n        All existing GROUP BY settings can be suppressed by passing ``None``.\n\n        e.g.::\n\n            stmt = select(table.c.name, func.max(table.c.stat)).\\\n            group_by(table.c.name)\n\n        :param \\*clauses: a series of :class:`_expression.ColumnElement`\n         constructs\n         which will be used to generate an GROUP BY clause.\n\n        .. seealso::\n\n            :ref:`tutorial_group_by_w_aggregates` - in the\n            :ref:`unified_tutorial`\n\n            :ref:`tutorial_order_by_label` - in the :ref:`unified_tutorial`\n\n        '
        if not clauses and __first is None:
            self._group_by_clauses = ()
        elif __first is not _NoArg.NO_ARG:
            self._group_by_clauses += tuple((coercions.expect(roles.GroupByRole, clause, apply_propagate_attrs=self) for clause in (__first,) + clauses))
        return self

@CompileState.plugin_for('default', 'compound_select')
class CompoundSelectState(CompileState):

    @util.memoized_property
    def _label_resolve_dict(self) -> Tuple[Dict[str, ColumnElement[Any]], Dict[str, ColumnElement[Any]], Dict[str, ColumnElement[Any]]]:
        if False:
            return 10
        hacky_subquery = self.statement.subquery()
        hacky_subquery.named_with_column = False
        d = {c.key: c for c in hacky_subquery.c}
        return (d, d, d)

class _CompoundSelectKeyword(Enum):
    UNION = 'UNION'
    UNION_ALL = 'UNION ALL'
    EXCEPT = 'EXCEPT'
    EXCEPT_ALL = 'EXCEPT ALL'
    INTERSECT = 'INTERSECT'
    INTERSECT_ALL = 'INTERSECT ALL'

class CompoundSelect(HasCompileState, GenerativeSelect, ExecutableReturnsRows):
    """Forms the basis of ``UNION``, ``UNION ALL``, and other
    SELECT-based set operations.


    .. seealso::

        :func:`_expression.union`

        :func:`_expression.union_all`

        :func:`_expression.intersect`

        :func:`_expression.intersect_all`

        :func:`_expression.except`

        :func:`_expression.except_all`

    """
    __visit_name__ = 'compound_select'
    _traverse_internals: _TraverseInternalsType = [('selects', InternalTraversal.dp_clauseelement_list), ('_limit_clause', InternalTraversal.dp_clauseelement), ('_offset_clause', InternalTraversal.dp_clauseelement), ('_fetch_clause', InternalTraversal.dp_clauseelement), ('_fetch_clause_options', InternalTraversal.dp_plain_dict), ('_order_by_clauses', InternalTraversal.dp_clauseelement_list), ('_group_by_clauses', InternalTraversal.dp_clauseelement_list), ('_for_update_arg', InternalTraversal.dp_clauseelement), ('keyword', InternalTraversal.dp_string)] + SupportsCloneAnnotations._clone_annotations_traverse_internals
    selects: List[SelectBase]
    _is_from_container = True
    _auto_correlate = False

    def __init__(self, keyword: _CompoundSelectKeyword, *selects: _SelectStatementForCompoundArgument):
        if False:
            i = 10
            return i + 15
        self.keyword = keyword
        self.selects = [coercions.expect(roles.CompoundElementRole, s, apply_propagate_attrs=self).self_group(against=self) for s in selects]
        GenerativeSelect.__init__(self)

    @classmethod
    def _create_union(cls, *selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
        if False:
            for i in range(10):
                print('nop')
        return CompoundSelect(_CompoundSelectKeyword.UNION, *selects)

    @classmethod
    def _create_union_all(cls, *selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
        if False:
            while True:
                i = 10
        return CompoundSelect(_CompoundSelectKeyword.UNION_ALL, *selects)

    @classmethod
    def _create_except(cls, *selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
        if False:
            for i in range(10):
                print('nop')
        return CompoundSelect(_CompoundSelectKeyword.EXCEPT, *selects)

    @classmethod
    def _create_except_all(cls, *selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
        if False:
            for i in range(10):
                print('nop')
        return CompoundSelect(_CompoundSelectKeyword.EXCEPT_ALL, *selects)

    @classmethod
    def _create_intersect(cls, *selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
        if False:
            return 10
        return CompoundSelect(_CompoundSelectKeyword.INTERSECT, *selects)

    @classmethod
    def _create_intersect_all(cls, *selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
        if False:
            print('Hello World!')
        return CompoundSelect(_CompoundSelectKeyword.INTERSECT_ALL, *selects)

    def _scalar_type(self) -> TypeEngine[Any]:
        if False:
            return 10
        return self.selects[0]._scalar_type()

    def self_group(self, against: Optional[OperatorType]=None) -> GroupedElement:
        if False:
            print('Hello World!')
        return SelectStatementGrouping(self)

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        if False:
            print('Hello World!')
        for s in self.selects:
            if s.is_derived_from(fromclause):
                return True
        return False

    def set_label_style(self, style: SelectLabelStyle) -> CompoundSelect:
        if False:
            i = 10
            return i + 15
        if self._label_style is not style:
            self = self._generate()
            select_0 = self.selects[0].set_label_style(style)
            self.selects = [select_0] + self.selects[1:]
        return self

    def _ensure_disambiguated_names(self) -> CompoundSelect:
        if False:
            for i in range(10):
                print('nop')
        new_select = self.selects[0]._ensure_disambiguated_names()
        if new_select is not self.selects[0]:
            self = self._generate()
            self.selects = [new_select] + self.selects[1:]
        return self

    def _generate_fromclause_column_proxies(self, subquery: FromClause, *, proxy_compound_columns: Optional[Iterable[Sequence[ColumnElement[Any]]]]=None) -> None:
        if False:
            while True:
                i = 10
        select_0 = self.selects[0]
        if self._label_style is not LABEL_STYLE_DEFAULT:
            select_0 = select_0.set_label_style(self._label_style)
        extra_col_iterator = zip(*[[c._annotate(dd) for c in stmt._all_selected_columns if is_column_element(c)] for (dd, stmt) in [({'weight': i + 1}, stmt) for (i, stmt) in enumerate(self.selects)]])
        select_0._generate_fromclause_column_proxies(subquery, proxy_compound_columns=extra_col_iterator)

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        super()._refresh_for_new_column(column)
        for select in self.selects:
            select._refresh_for_new_column(column)

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        if False:
            for i in range(10):
                print('nop')
        return self.selects[0]._all_selected_columns

    @util.ro_non_memoized_property
    def selected_columns(self) -> ColumnCollection[str, ColumnElement[Any]]:
        if False:
            i = 10
            return i + 15
        'A :class:`_expression.ColumnCollection`\n        representing the columns that\n        this SELECT statement or similar construct returns in its result set,\n        not including :class:`_sql.TextClause` constructs.\n\n        For a :class:`_expression.CompoundSelect`, the\n        :attr:`_expression.CompoundSelect.selected_columns`\n        attribute returns the selected\n        columns of the first SELECT statement contained within the series of\n        statements within the set operation.\n\n        .. seealso::\n\n            :attr:`_sql.Select.selected_columns`\n\n        .. versionadded:: 1.4\n\n        '
        return self.selects[0].selected_columns
for elem in _CompoundSelectKeyword:
    setattr(CompoundSelect, elem.name, elem)

@CompileState.plugin_for('default', 'select')
class SelectState(util.MemoizedSlots, CompileState):
    __slots__ = ('from_clauses', 'froms', 'columns_plus_names', '_label_resolve_dict')
    if TYPE_CHECKING:
        default_select_compile_options: CacheableOptions
    else:

        class default_select_compile_options(CacheableOptions):
            _cache_key_traversal = []
    if TYPE_CHECKING:

        @classmethod
        def get_plugin_class(cls, statement: Executable) -> Type[SelectState]:
            if False:
                return 10
            ...

    def __init__(self, statement: Select[Any], compiler: Optional[SQLCompiler], **kw: Any):
        if False:
            while True:
                i = 10
        self.statement = statement
        self.from_clauses = statement._from_obj
        for memoized_entities in statement._memoized_select_entities:
            self._setup_joins(memoized_entities._setup_joins, memoized_entities._raw_columns)
        if statement._setup_joins:
            self._setup_joins(statement._setup_joins, statement._raw_columns)
        self.froms = self._get_froms(statement)
        self.columns_plus_names = statement._generate_columns_plus_names(True)

    @classmethod
    def _plugin_not_implemented(cls) -> NoReturn:
        if False:
            return 10
        raise NotImplementedError('The default SELECT construct without plugins does not implement this method.')

    @classmethod
    def get_column_descriptions(cls, statement: Select[Any]) -> List[Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        return [{'name': name, 'type': element.type, 'expr': element} for (_, name, _, element, _) in statement._generate_columns_plus_names(False)]

    @classmethod
    def from_statement(cls, statement: Select[Any], from_statement: roles.ReturnsRowsRole) -> ExecutableReturnsRows:
        if False:
            while True:
                i = 10
        cls._plugin_not_implemented()

    @classmethod
    def get_columns_clause_froms(cls, statement: Select[Any]) -> List[FromClause]:
        if False:
            return 10
        return cls._normalize_froms(itertools.chain.from_iterable((element._from_objects for element in statement._raw_columns)))

    @classmethod
    def _column_naming_convention(cls, label_style: SelectLabelStyle) -> _LabelConventionCallable:
        if False:
            while True:
                i = 10
        table_qualified = label_style is LABEL_STYLE_TABLENAME_PLUS_COL
        dedupe = label_style is not LABEL_STYLE_NONE
        pa = prefix_anon_map()
        names = set()

        def go(c: Union[ColumnElement[Any], TextClause], col_name: Optional[str]=None) -> Optional[str]:
            if False:
                i = 10
                return i + 15
            if is_text_clause(c):
                return None
            elif TYPE_CHECKING:
                assert is_column_element(c)
            if not dedupe:
                name = c._proxy_key
                if name is None:
                    name = '_no_label'
                return name
            name = c._tq_key_label if table_qualified else c._proxy_key
            if name is None:
                name = '_no_label'
                if name in names:
                    return c._anon_label(name) % pa
                else:
                    names.add(name)
                    return name
            elif name in names:
                return c._anon_tq_key_label % pa if table_qualified else c._anon_key_label % pa
            else:
                names.add(name)
                return name
        return go

    def _get_froms(self, statement: Select[Any]) -> List[FromClause]:
        if False:
            print('Hello World!')
        ambiguous_table_name_map: _AmbiguousTableNameMap
        self._ambiguous_table_name_map = ambiguous_table_name_map = {}
        return self._normalize_froms(itertools.chain(self.from_clauses, itertools.chain.from_iterable([element._from_objects for element in statement._raw_columns]), itertools.chain.from_iterable([element._from_objects for element in statement._where_criteria])), check_statement=statement, ambiguous_table_name_map=ambiguous_table_name_map)

    @classmethod
    def _normalize_froms(cls, iterable_of_froms: Iterable[FromClause], check_statement: Optional[Select[Any]]=None, ambiguous_table_name_map: Optional[_AmbiguousTableNameMap]=None) -> List[FromClause]:
        if False:
            i = 10
            return i + 15
        'given an iterable of things to select FROM, reduce them to what\n        would actually render in the FROM clause of a SELECT.\n\n        This does the job of checking for JOINs, tables, etc. that are in fact\n        overlapping due to cloning, adaption, present in overlapping joins,\n        etc.\n\n        '
        seen: Set[FromClause] = set()
        froms: List[FromClause] = []
        for item in iterable_of_froms:
            if is_subquery(item) and item.element is check_statement:
                raise exc.InvalidRequestError('select() construct refers to itself as a FROM')
            if not seen.intersection(item._cloned_set):
                froms.append(item)
                seen.update(item._cloned_set)
        if froms:
            toremove = set(itertools.chain.from_iterable([_expand_cloned(f._hide_froms) for f in froms]))
            if toremove:
                froms = [f for f in froms if f not in toremove]
            if ambiguous_table_name_map is not None:
                ambiguous_table_name_map.update(((fr.name, _anonymous_label.safe_construct(hash(fr.name), fr.name)) for item in froms for fr in item._from_objects if is_table(fr) and fr.schema and (fr.name not in ambiguous_table_name_map)))
        return froms

    def _get_display_froms(self, explicit_correlate_froms: Optional[Sequence[FromClause]]=None, implicit_correlate_froms: Optional[Sequence[FromClause]]=None) -> List[FromClause]:
        if False:
            while True:
                i = 10
        "Return the full list of 'from' clauses to be displayed.\n\n        Takes into account a set of existing froms which may be\n        rendered in the FROM clause of enclosing selects; this Select\n        may want to leave those absent if it is automatically\n        correlating.\n\n        "
        froms = self.froms
        if self.statement._correlate:
            to_correlate = self.statement._correlate
            if to_correlate:
                froms = [f for f in froms if f not in _cloned_intersection(_cloned_intersection(froms, explicit_correlate_froms or ()), to_correlate)]
        if self.statement._correlate_except is not None:
            froms = [f for f in froms if f not in _cloned_difference(_cloned_intersection(froms, explicit_correlate_froms or ()), self.statement._correlate_except)]
        if self.statement._auto_correlate and implicit_correlate_froms and (len(froms) > 1):
            froms = [f for f in froms if f not in _cloned_intersection(froms, implicit_correlate_froms)]
            if not len(froms):
                raise exc.InvalidRequestError("Select statement '%r' returned no FROM clauses due to auto-correlation; specify correlate(<tables>) to control correlation manually." % self.statement)
        return froms

    def _memoized_attr__label_resolve_dict(self) -> Tuple[Dict[str, ColumnElement[Any]], Dict[str, ColumnElement[Any]], Dict[str, ColumnElement[Any]]]:
        if False:
            for i in range(10):
                print('nop')
        with_cols: Dict[str, ColumnElement[Any]] = {c._tq_label or c.key: c for c in self.statement._all_selected_columns if c._allow_label_resolve}
        only_froms: Dict[str, ColumnElement[Any]] = {c.key: c for c in _select_iterables(self.froms) if c._allow_label_resolve}
        only_cols: Dict[str, ColumnElement[Any]] = with_cols.copy()
        for (key, value) in only_froms.items():
            with_cols.setdefault(key, value)
        return (with_cols, only_froms, only_cols)

    @classmethod
    def determine_last_joined_entity(cls, stmt: Select[Any]) -> Optional[_JoinTargetElement]:
        if False:
            while True:
                i = 10
        if stmt._setup_joins:
            return stmt._setup_joins[-1][0]
        else:
            return None

    @classmethod
    def all_selected_columns(cls, statement: Select[Any]) -> _SelectIterable:
        if False:
            while True:
                i = 10
        return [c for c in _select_iterables(statement._raw_columns)]

    def _setup_joins(self, args: Tuple[_SetupJoinsElement, ...], raw_columns: List[_ColumnsClauseElement]) -> None:
        if False:
            while True:
                i = 10
        for (right, onclause, left, flags) in args:
            if TYPE_CHECKING:
                if onclause is not None:
                    assert isinstance(onclause, ColumnElement)
            isouter = flags['isouter']
            full = flags['full']
            if left is None:
                (left, replace_from_obj_index) = self._join_determine_implicit_left_side(raw_columns, left, right, onclause)
            else:
                replace_from_obj_index = self._join_place_explicit_left_side(left)
            if TYPE_CHECKING:
                assert isinstance(right, FromClause)
                if onclause is not None:
                    assert isinstance(onclause, ColumnElement)
            if replace_from_obj_index is not None:
                left_clause = self.from_clauses[replace_from_obj_index]
                self.from_clauses = self.from_clauses[:replace_from_obj_index] + (Join(left_clause, right, onclause, isouter=isouter, full=full),) + self.from_clauses[replace_from_obj_index + 1:]
            else:
                assert left is not None
                self.from_clauses = self.from_clauses + (Join(left, right, onclause, isouter=isouter, full=full),)

    @util.preload_module('sqlalchemy.sql.util')
    def _join_determine_implicit_left_side(self, raw_columns: List[_ColumnsClauseElement], left: Optional[FromClause], right: _JoinTargetElement, onclause: Optional[ColumnElement[Any]]) -> Tuple[Optional[FromClause], Optional[int]]:
        if False:
            return 10
        "When join conditions don't express the left side explicitly,\n        determine if an existing FROM or entity in this query\n        can serve as the left hand side.\n\n        "
        sql_util = util.preloaded.sql_util
        replace_from_obj_index: Optional[int] = None
        from_clauses = self.from_clauses
        if from_clauses:
            indexes: List[int] = sql_util.find_left_clause_to_join_from(from_clauses, right, onclause)
            if len(indexes) == 1:
                replace_from_obj_index = indexes[0]
                left = from_clauses[replace_from_obj_index]
        else:
            potential = {}
            statement = self.statement
            for from_clause in itertools.chain(itertools.chain.from_iterable([element._from_objects for element in raw_columns]), itertools.chain.from_iterable([element._from_objects for element in statement._where_criteria])):
                potential[from_clause] = ()
            all_clauses = list(potential.keys())
            indexes = sql_util.find_left_clause_to_join_from(all_clauses, right, onclause)
            if len(indexes) == 1:
                left = all_clauses[indexes[0]]
        if len(indexes) > 1:
            raise exc.InvalidRequestError("Can't determine which FROM clause to join from, there are multiple FROMS which can join to this entity. Please use the .select_from() method to establish an explicit left side, as well as providing an explicit ON clause if not present already to help resolve the ambiguity.")
        elif not indexes:
            raise exc.InvalidRequestError("Don't know how to join to %r. Please use the .select_from() method to establish an explicit left side, as well as providing an explicit ON clause if not present already to help resolve the ambiguity." % (right,))
        return (left, replace_from_obj_index)

    @util.preload_module('sqlalchemy.sql.util')
    def _join_place_explicit_left_side(self, left: FromClause) -> Optional[int]:
        if False:
            return 10
        replace_from_obj_index: Optional[int] = None
        sql_util = util.preloaded.sql_util
        from_clauses = list(self.statement._iterate_from_elements())
        if from_clauses:
            indexes: List[int] = sql_util.find_left_clause_that_matches_given(self.from_clauses, left)
        else:
            indexes = []
        if len(indexes) > 1:
            raise exc.InvalidRequestError("Can't identify which entity in which to assign the left side of this join.   Please use a more specific ON clause.")
        if indexes:
            replace_from_obj_index = indexes[0]
        return replace_from_obj_index

class _SelectFromElements:
    __slots__ = ()
    _raw_columns: List[_ColumnsClauseElement]
    _where_criteria: Tuple[ColumnElement[Any], ...]
    _from_obj: Tuple[FromClause, ...]

    def _iterate_from_elements(self) -> Iterator[FromClause]:
        if False:
            return 10
        seen = set()
        for element in self._raw_columns:
            for fr in element._from_objects:
                if fr in seen:
                    continue
                seen.add(fr)
                yield fr
        for element in self._where_criteria:
            for fr in element._from_objects:
                if fr in seen:
                    continue
                seen.add(fr)
                yield fr
        for element in self._from_obj:
            if element in seen:
                continue
            seen.add(element)
            yield element

class _MemoizedSelectEntities(cache_key.HasCacheKey, traversals.HasCopyInternals, visitors.Traversible):
    """represents partial state from a Select object, for the case
    where Select.columns() has redefined the set of columns/entities the
    statement will be SELECTing from.  This object represents
    the entities from the SELECT before that transformation was applied,
    so that transformations that were made in terms of the SELECT at that
    time, such as join() as well as options(), can access the correct context.

    In previous SQLAlchemy versions, this wasn't needed because these
    constructs calculated everything up front, like when you called join()
    or options(), it did everything to figure out how that would translate
    into specific SQL constructs that would be ready to send directly to the
    SQL compiler when needed.  But as of
    1.4, all of that stuff is done in the compilation phase, during the
    "compile state" portion of the process, so that the work can all be
    cached.  So it needs to be able to resolve joins/options2 based on what
    the list of entities was when those methods were called.


    """
    __visit_name__ = 'memoized_select_entities'
    _traverse_internals: _TraverseInternalsType = [('_raw_columns', InternalTraversal.dp_clauseelement_list), ('_setup_joins', InternalTraversal.dp_setup_join_tuple), ('_with_options', InternalTraversal.dp_executable_options)]
    _is_clone_of: Optional[ClauseElement]
    _raw_columns: List[_ColumnsClauseElement]
    _setup_joins: Tuple[_SetupJoinsElement, ...]
    _with_options: Tuple[ExecutableOption, ...]
    _annotations = util.EMPTY_DICT

    def _clone(self, **kw: Any) -> Self:
        if False:
            return 10
        c = self.__class__.__new__(self.__class__)
        c.__dict__ = {k: v for (k, v) in self.__dict__.items()}
        c._is_clone_of = self.__dict__.get('_is_clone_of', self)
        return c

    @classmethod
    def _generate_for_statement(cls, select_stmt: Select[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if select_stmt._setup_joins or select_stmt._with_options:
            self = _MemoizedSelectEntities()
            self._raw_columns = select_stmt._raw_columns
            self._setup_joins = select_stmt._setup_joins
            self._with_options = select_stmt._with_options
            select_stmt._memoized_select_entities += (self,)
            select_stmt._raw_columns = []
            select_stmt._setup_joins = select_stmt._with_options = ()

class Select(HasPrefixes, HasSuffixes, HasHints, HasCompileState, _SelectFromElements, GenerativeSelect, TypedReturnsRows[_TP]):
    """Represents a ``SELECT`` statement.

    The :class:`_sql.Select` object is normally constructed using the
    :func:`_sql.select` function.  See that function for details.

    .. seealso::

        :func:`_sql.select`

        :ref:`tutorial_selecting_data` - in the 2.0 tutorial

    """
    __visit_name__ = 'select'
    _setup_joins: Tuple[_SetupJoinsElement, ...] = ()
    _memoized_select_entities: Tuple[TODO_Any, ...] = ()
    _raw_columns: List[_ColumnsClauseElement]
    _distinct: bool = False
    _distinct_on: Tuple[ColumnElement[Any], ...] = ()
    _correlate: Tuple[FromClause, ...] = ()
    _correlate_except: Optional[Tuple[FromClause, ...]] = None
    _where_criteria: Tuple[ColumnElement[Any], ...] = ()
    _having_criteria: Tuple[ColumnElement[Any], ...] = ()
    _from_obj: Tuple[FromClause, ...] = ()
    _auto_correlate = True
    _is_select_statement = True
    _compile_options: CacheableOptions = SelectState.default_select_compile_options
    _traverse_internals: _TraverseInternalsType = [('_raw_columns', InternalTraversal.dp_clauseelement_list), ('_memoized_select_entities', InternalTraversal.dp_memoized_select_entities), ('_from_obj', InternalTraversal.dp_clauseelement_list), ('_where_criteria', InternalTraversal.dp_clauseelement_tuple), ('_having_criteria', InternalTraversal.dp_clauseelement_tuple), ('_order_by_clauses', InternalTraversal.dp_clauseelement_tuple), ('_group_by_clauses', InternalTraversal.dp_clauseelement_tuple), ('_setup_joins', InternalTraversal.dp_setup_join_tuple), ('_correlate', InternalTraversal.dp_clauseelement_tuple), ('_correlate_except', InternalTraversal.dp_clauseelement_tuple), ('_limit_clause', InternalTraversal.dp_clauseelement), ('_offset_clause', InternalTraversal.dp_clauseelement), ('_fetch_clause', InternalTraversal.dp_clauseelement), ('_fetch_clause_options', InternalTraversal.dp_plain_dict), ('_for_update_arg', InternalTraversal.dp_clauseelement), ('_distinct', InternalTraversal.dp_boolean), ('_distinct_on', InternalTraversal.dp_clauseelement_tuple), ('_label_style', InternalTraversal.dp_plain_obj)] + HasCTE._has_ctes_traverse_internals + HasPrefixes._has_prefixes_traverse_internals + HasSuffixes._has_suffixes_traverse_internals + HasHints._has_hints_traverse_internals + SupportsCloneAnnotations._clone_annotations_traverse_internals + Executable._executable_traverse_internals
    _cache_key_traversal: _CacheKeyTraversalType = _traverse_internals + [('_compile_options', InternalTraversal.dp_has_cache_key)]
    _compile_state_factory: Type[SelectState]

    @classmethod
    def _create_raw_select(cls, **kw: Any) -> Select[Any]:
        if False:
            i = 10
            return i + 15
        'Create a :class:`.Select` using raw ``__new__`` with no coercions.\n\n        Used internally to build up :class:`.Select` constructs with\n        pre-established state.\n\n        '
        stmt = Select.__new__(Select)
        stmt.__dict__.update(kw)
        return stmt

    def __init__(self, *entities: _ColumnsClauseArgument[Any]):
        if False:
            i = 10
            return i + 15
        'Construct a new :class:`_expression.Select`.\n\n        The public constructor for :class:`_expression.Select` is the\n        :func:`_sql.select` function.\n\n        '
        self._raw_columns = [coercions.expect(roles.ColumnsClauseRole, ent, apply_propagate_attrs=self) for ent in entities]
        GenerativeSelect.__init__(self)

    def _scalar_type(self) -> TypeEngine[Any]:
        if False:
            i = 10
            return i + 15
        if not self._raw_columns:
            return NULLTYPE
        elem = self._raw_columns[0]
        cols = list(elem._select_iterable)
        return cols[0].type

    def filter(self, *criteria: _ColumnExpressionArgument[bool]) -> Self:
        if False:
            print('Hello World!')
        'A synonym for the :meth:`_sql.Select.where` method.'
        return self.where(*criteria)

    def _filter_by_zero(self) -> Union[FromClause, _JoinTargetProtocol, ColumnElement[Any], TextClause]:
        if False:
            i = 10
            return i + 15
        if self._setup_joins:
            meth = SelectState.get_plugin_class(self).determine_last_joined_entity
            _last_joined_entity = meth(self)
            if _last_joined_entity is not None:
                return _last_joined_entity
        if self._from_obj:
            return self._from_obj[0]
        return self._raw_columns[0]
    if TYPE_CHECKING:

        @overload
        def scalar_subquery(self: Select[Tuple[_MAYBE_ENTITY]]) -> ScalarSelect[Any]:
            if False:
                while True:
                    i = 10
            ...

        @overload
        def scalar_subquery(self: Select[Tuple[_NOT_ENTITY]]) -> ScalarSelect[_NOT_ENTITY]:
            if False:
                while True:
                    i = 10
            ...

        @overload
        def scalar_subquery(self) -> ScalarSelect[Any]:
            if False:
                while True:
                    i = 10
            ...

        def scalar_subquery(self) -> ScalarSelect[Any]:
            if False:
                i = 10
                return i + 15
            ...

    def filter_by(self, **kwargs: Any) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'apply the given filtering criterion as a WHERE clause\n        to this select.\n\n        '
        from_entity = self._filter_by_zero()
        clauses = [_entity_namespace_key(from_entity, key) == value for (key, value) in kwargs.items()]
        return self.filter(*clauses)

    @property
    def column_descriptions(self) -> Any:
        if False:
            i = 10
            return i + 15
        "Return a :term:`plugin-enabled` 'column descriptions' structure\n        referring to the columns which are SELECTed by this statement.\n\n        This attribute is generally useful when using the ORM, as an\n        extended structure which includes information about mapped\n        entities is returned.  The section :ref:`queryguide_inspection`\n        contains more background.\n\n        For a Core-only statement, the structure returned by this accessor\n        is derived from the same objects that are returned by the\n        :attr:`.Select.selected_columns` accessor, formatted as a list of\n        dictionaries which contain the keys ``name``, ``type`` and ``expr``,\n        which indicate the column expressions to be selected::\n\n            >>> stmt = select(user_table)\n            >>> stmt.column_descriptions\n            [\n                {\n                    'name': 'id',\n                    'type': Integer(),\n                    'expr': Column('id', Integer(), ...)},\n                {\n                    'name': 'name',\n                    'type': String(length=30),\n                    'expr': Column('name', String(length=30), ...)}\n            ]\n\n        .. versionchanged:: 1.4.33 The :attr:`.Select.column_descriptions`\n           attribute returns a structure for a Core-only set of entities,\n           not just ORM-only entities.\n\n        .. seealso::\n\n            :attr:`.UpdateBase.entity_description` - entity information for\n            an :func:`.insert`, :func:`.update`, or :func:`.delete`\n\n            :ref:`queryguide_inspection` - ORM background\n\n        "
        meth = SelectState.get_plugin_class(self).get_column_descriptions
        return meth(self)

    def from_statement(self, statement: roles.ReturnsRowsRole) -> ExecutableReturnsRows:
        if False:
            print('Hello World!')
        'Apply the columns which this :class:`.Select` would select\n        onto another statement.\n\n        This operation is :term:`plugin-specific` and will raise a not\n        supported exception if this :class:`_sql.Select` does not select from\n        plugin-enabled entities.\n\n\n        The statement is typically either a :func:`_expression.text` or\n        :func:`_expression.select` construct, and should return the set of\n        columns appropriate to the entities represented by this\n        :class:`.Select`.\n\n        .. seealso::\n\n            :ref:`orm_queryguide_selecting_text` - usage examples in the\n            ORM Querying Guide\n\n        '
        meth = SelectState.get_plugin_class(self).from_statement
        return meth(self, statement)

    @_generative
    def join(self, target: _JoinTargetArgument, onclause: Optional[_OnClauseArgument]=None, *, isouter: bool=False, full: bool=False) -> Self:
        if False:
            print('Hello World!')
        "Create a SQL JOIN against this :class:`_expression.Select`\n        object's criterion\n        and apply generatively, returning the newly resulting\n        :class:`_expression.Select`.\n\n        E.g.::\n\n            stmt = select(user_table).join(address_table, user_table.c.id == address_table.c.user_id)\n\n        The above statement generates SQL similar to::\n\n            SELECT user.id, user.name FROM user JOIN address ON user.id = address.user_id\n\n        .. versionchanged:: 1.4 :meth:`_expression.Select.join` now creates\n           a :class:`_sql.Join` object between a :class:`_sql.FromClause`\n           source that is within the FROM clause of the existing SELECT,\n           and a given target :class:`_sql.FromClause`, and then adds\n           this :class:`_sql.Join` to the FROM clause of the newly generated\n           SELECT statement.    This is completely reworked from the behavior\n           in 1.3, which would instead create a subquery of the entire\n           :class:`_expression.Select` and then join that subquery to the\n           target.\n\n           This is a **backwards incompatible change** as the previous behavior\n           was mostly useless, producing an unnamed subquery rejected by\n           most databases in any case.   The new behavior is modeled after\n           that of the very successful :meth:`_orm.Query.join` method in the\n           ORM, in order to support the functionality of :class:`_orm.Query`\n           being available by using a :class:`_sql.Select` object with an\n           :class:`_orm.Session`.\n\n           See the notes for this change at :ref:`change_select_join`.\n\n\n        :param target: target table to join towards\n\n        :param onclause: ON clause of the join.  If omitted, an ON clause\n         is generated automatically based on the :class:`_schema.ForeignKey`\n         linkages between the two tables, if one can be unambiguously\n         determined, otherwise an error is raised.\n\n        :param isouter: if True, generate LEFT OUTER join.  Same as\n         :meth:`_expression.Select.outerjoin`.\n\n        :param full: if True, generate FULL OUTER join.\n\n        .. seealso::\n\n            :ref:`tutorial_select_join` - in the :doc:`/tutorial/index`\n\n            :ref:`orm_queryguide_joins` - in the :ref:`queryguide_toplevel`\n\n            :meth:`_expression.Select.join_from`\n\n            :meth:`_expression.Select.outerjoin`\n\n        "
        join_target = coercions.expect(roles.JoinTargetRole, target, apply_propagate_attrs=self)
        if onclause is not None:
            onclause_element = coercions.expect(roles.OnClauseRole, onclause)
        else:
            onclause_element = None
        self._setup_joins += ((join_target, onclause_element, None, {'isouter': isouter, 'full': full}),)
        return self

    def outerjoin_from(self, from_: _FromClauseArgument, target: _JoinTargetArgument, onclause: Optional[_OnClauseArgument]=None, *, full: bool=False) -> Self:
        if False:
            for i in range(10):
                print('nop')
        "Create a SQL LEFT OUTER JOIN against this\n        :class:`_expression.Select` object's criterion and apply generatively,\n        returning the newly resulting :class:`_expression.Select`.\n\n        Usage is the same as that of :meth:`_selectable.Select.join_from`.\n\n        "
        return self.join_from(from_, target, onclause=onclause, isouter=True, full=full)

    @_generative
    def join_from(self, from_: _FromClauseArgument, target: _JoinTargetArgument, onclause: Optional[_OnClauseArgument]=None, *, isouter: bool=False, full: bool=False) -> Self:
        if False:
            i = 10
            return i + 15
        "Create a SQL JOIN against this :class:`_expression.Select`\n        object's criterion\n        and apply generatively, returning the newly resulting\n        :class:`_expression.Select`.\n\n        E.g.::\n\n            stmt = select(user_table, address_table).join_from(\n                user_table, address_table, user_table.c.id == address_table.c.user_id\n            )\n\n        The above statement generates SQL similar to::\n\n            SELECT user.id, user.name, address.id, address.email, address.user_id\n            FROM user JOIN address ON user.id = address.user_id\n\n        .. versionadded:: 1.4\n\n        :param from\\_: the left side of the join, will be rendered in the\n         FROM clause and is roughly equivalent to using the\n         :meth:`.Select.select_from` method.\n\n        :param target: target table to join towards\n\n        :param onclause: ON clause of the join.\n\n        :param isouter: if True, generate LEFT OUTER join.  Same as\n         :meth:`_expression.Select.outerjoin`.\n\n        :param full: if True, generate FULL OUTER join.\n\n        .. seealso::\n\n            :ref:`tutorial_select_join` - in the :doc:`/tutorial/index`\n\n            :ref:`orm_queryguide_joins` - in the :ref:`queryguide_toplevel`\n\n            :meth:`_expression.Select.join`\n\n        "
        from_ = coercions.expect(roles.FromClauseRole, from_, apply_propagate_attrs=self)
        join_target = coercions.expect(roles.JoinTargetRole, target, apply_propagate_attrs=self)
        if onclause is not None:
            onclause_element = coercions.expect(roles.OnClauseRole, onclause)
        else:
            onclause_element = None
        self._setup_joins += ((join_target, onclause_element, from_, {'isouter': isouter, 'full': full}),)
        return self

    def outerjoin(self, target: _JoinTargetArgument, onclause: Optional[_OnClauseArgument]=None, *, full: bool=False) -> Self:
        if False:
            print('Hello World!')
        'Create a left outer join.\n\n        Parameters are the same as that of :meth:`_expression.Select.join`.\n\n        .. versionchanged:: 1.4 :meth:`_expression.Select.outerjoin` now\n           creates a :class:`_sql.Join` object between a\n           :class:`_sql.FromClause` source that is within the FROM clause of\n           the existing SELECT, and a given target :class:`_sql.FromClause`,\n           and then adds this :class:`_sql.Join` to the FROM clause of the\n           newly generated SELECT statement.    This is completely reworked\n           from the behavior in 1.3, which would instead create a subquery of\n           the entire\n           :class:`_expression.Select` and then join that subquery to the\n           target.\n\n           This is a **backwards incompatible change** as the previous behavior\n           was mostly useless, producing an unnamed subquery rejected by\n           most databases in any case.   The new behavior is modeled after\n           that of the very successful :meth:`_orm.Query.join` method in the\n           ORM, in order to support the functionality of :class:`_orm.Query`\n           being available by using a :class:`_sql.Select` object with an\n           :class:`_orm.Session`.\n\n           See the notes for this change at :ref:`change_select_join`.\n\n        .. seealso::\n\n            :ref:`tutorial_select_join` - in the :doc:`/tutorial/index`\n\n            :ref:`orm_queryguide_joins` - in the :ref:`queryguide_toplevel`\n\n            :meth:`_expression.Select.join`\n\n        '
        return self.join(target, onclause=onclause, isouter=True, full=full)

    def get_final_froms(self) -> Sequence[FromClause]:
        if False:
            for i in range(10):
                print('nop')
        'Compute the final displayed list of :class:`_expression.FromClause`\n        elements.\n\n        This method will run through the full computation required to\n        determine what FROM elements will be displayed in the resulting\n        SELECT statement, including shadowing individual tables with\n        JOIN objects, as well as full computation for ORM use cases including\n        eager loading clauses.\n\n        For ORM use, this accessor returns the **post compilation**\n        list of FROM objects; this collection will include elements such as\n        eagerly loaded tables and joins.  The objects will **not** be\n        ORM enabled and not work as a replacement for the\n        :meth:`_sql.Select.select_froms` collection; additionally, the\n        method is not well performing for an ORM enabled statement as it\n        will incur the full ORM construction process.\n\n        To retrieve the FROM list that\'s implied by the "columns" collection\n        passed to the :class:`_sql.Select` originally, use the\n        :attr:`_sql.Select.columns_clause_froms` accessor.\n\n        To select from an alternative set of columns while maintaining the\n        FROM list, use the :meth:`_sql.Select.with_only_columns` method and\n        pass the\n        :paramref:`_sql.Select.with_only_columns.maintain_column_froms`\n        parameter.\n\n        .. versionadded:: 1.4.23 - the :meth:`_sql.Select.get_final_froms`\n           method replaces the previous :attr:`_sql.Select.froms` accessor,\n           which is deprecated.\n\n        .. seealso::\n\n            :attr:`_sql.Select.columns_clause_froms`\n\n        '
        return self._compile_state_factory(self, None)._get_display_froms()

    @property
    @util.deprecated('1.4.23', 'The :attr:`_expression.Select.froms` attribute is moved to the :meth:`_expression.Select.get_final_froms` method.')
    def froms(self) -> Sequence[FromClause]:
        if False:
            for i in range(10):
                print('nop')
        'Return the displayed list of :class:`_expression.FromClause`\n        elements.\n\n\n        '
        return self.get_final_froms()

    @property
    def columns_clause_froms(self) -> List[FromClause]:
        if False:
            print('Hello World!')
        'Return the set of :class:`_expression.FromClause` objects implied\n        by the columns clause of this SELECT statement.\n\n        .. versionadded:: 1.4.23\n\n        .. seealso::\n\n            :attr:`_sql.Select.froms` - "final" FROM list taking the full\n            statement into account\n\n            :meth:`_sql.Select.with_only_columns` - makes use of this\n            collection to set up a new FROM list\n\n        '
        return SelectState.get_plugin_class(self).get_columns_clause_froms(self)

    @property
    def inner_columns(self) -> _SelectIterable:
        if False:
            for i in range(10):
                print('nop')
        'An iterator of all :class:`_expression.ColumnElement`\n        expressions which would\n        be rendered into the columns clause of the resulting SELECT statement.\n\n        This method is legacy as of 1.4 and is superseded by the\n        :attr:`_expression.Select.exported_columns` collection.\n\n        '
        return iter(self._all_selected_columns)

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        if False:
            return 10
        if fromclause is not None and self in fromclause._cloned_set:
            return True
        for f in self._iterate_from_elements():
            if f.is_derived_from(fromclause):
                return True
        return False

    def _copy_internals(self, clone: _CloneCallableType=_clone, **kw: Any) -> None:
        if False:
            print('Hello World!')
        all_the_froms = set(itertools.chain(_from_objects(*self._raw_columns), _from_objects(*self._where_criteria), _from_objects(*[elem[0] for elem in self._setup_joins])))
        new_froms = {f: clone(f, **kw) for f in all_the_froms}
        existing_from_obj = [clone(f, **kw) for f in self._from_obj]
        add_froms = {f for f in new_froms.values() if isinstance(f, Join)}.difference(all_the_froms).difference(existing_from_obj)
        self._from_obj = tuple(existing_from_obj) + tuple(add_froms)

        def replace(obj: Union[BinaryExpression[Any], ColumnClause[Any]], **kw: Any) -> Optional[KeyedColumnElement[ColumnElement[Any]]]:
            if False:
                return 10
            if isinstance(obj, ColumnClause) and obj.table in new_froms:
                newelem = new_froms[obj.table].corresponding_column(obj)
                return newelem
            return None
        kw['replace'] = replace
        super()._copy_internals(clone=clone, omit_attrs=('_from_obj',), **kw)
        self._reset_memoizations()

    def get_children(self, **kw: Any) -> Iterable[ClauseElement]:
        if False:
            for i in range(10):
                print('nop')
        return itertools.chain(super().get_children(omit_attrs=('_from_obj', '_correlate', '_correlate_except'), **kw), self._iterate_from_elements())

    @_generative
    def add_columns(self, *entities: _ColumnsClauseArgument[Any]) -> Select[Any]:
        if False:
            return 10
        'Return a new :func:`_expression.select` construct with\n        the given entities appended to its columns clause.\n\n        E.g.::\n\n            my_select = my_select.add_columns(table.c.new_column)\n\n        The original expressions in the columns clause remain in place.\n        To replace the original expressions with new ones, see the method\n        :meth:`_expression.Select.with_only_columns`.\n\n        :param \\*entities: column, table, or other entity expressions to be\n         added to the columns clause\n\n        .. seealso::\n\n            :meth:`_expression.Select.with_only_columns` - replaces existing\n            expressions rather than appending.\n\n            :ref:`orm_queryguide_select_multiple_entities` - ORM-centric\n            example\n\n        '
        self._reset_memoizations()
        self._raw_columns = self._raw_columns + [coercions.expect(roles.ColumnsClauseRole, column, apply_propagate_attrs=self) for column in entities]
        return self

    def _set_entities(self, entities: Iterable[_ColumnsClauseArgument[Any]]) -> None:
        if False:
            i = 10
            return i + 15
        self._raw_columns = [coercions.expect(roles.ColumnsClauseRole, ent, apply_propagate_attrs=self) for ent in util.to_list(entities)]

    @util.deprecated('1.4', 'The :meth:`_expression.Select.column` method is deprecated and will be removed in a future release.  Please use :meth:`_expression.Select.add_columns`')
    def column(self, column: _ColumnsClauseArgument[Any]) -> Select[Any]:
        if False:
            print('Hello World!')
        'Return a new :func:`_expression.select` construct with\n        the given column expression added to its columns clause.\n\n        E.g.::\n\n            my_select = my_select.column(table.c.new_column)\n\n        See the documentation for\n        :meth:`_expression.Select.with_only_columns`\n        for guidelines on adding /replacing the columns of a\n        :class:`_expression.Select` object.\n\n        '
        return self.add_columns(column)

    @util.preload_module('sqlalchemy.sql.util')
    def reduce_columns(self, only_synonyms: bool=True) -> Select[Any]:
        if False:
            i = 10
            return i + 15
        'Return a new :func:`_expression.select` construct with redundantly\n        named, equivalently-valued columns removed from the columns clause.\n\n        "Redundant" here means two columns where one refers to the\n        other either based on foreign key, or via a simple equality\n        comparison in the WHERE clause of the statement.   The primary purpose\n        of this method is to automatically construct a select statement\n        with all uniquely-named columns, without the need to use\n        table-qualified labels as\n        :meth:`_expression.Select.set_label_style`\n        does.\n\n        When columns are omitted based on foreign key, the referred-to\n        column is the one that\'s kept.  When columns are omitted based on\n        WHERE equivalence, the first column in the columns clause is the\n        one that\'s kept.\n\n        :param only_synonyms: when True, limit the removal of columns\n         to those which have the same name as the equivalent.   Otherwise,\n         all columns that are equivalent to another are removed.\n\n        '
        woc: Select[Any]
        woc = self.with_only_columns(*util.preloaded.sql_util.reduce_columns(self._all_selected_columns, *self._where_criteria + self._from_obj, only_synonyms=only_synonyms))
        return woc

    @overload
    def with_only_columns(self, __ent0: _TCCA[_T0]) -> Select[Tuple[_T0]]:
        if False:
            return 10
        ...

    @overload
    def with_only_columns(self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1]) -> Select[Tuple[_T0, _T1]]:
        if False:
            return 10
        ...

    @overload
    def with_only_columns(self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2]) -> Select[Tuple[_T0, _T1, _T2]]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def with_only_columns(self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], __ent3: _TCCA[_T3]) -> Select[Tuple[_T0, _T1, _T2, _T3]]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def with_only_columns(self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], __ent3: _TCCA[_T3], __ent4: _TCCA[_T4]) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4]]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def with_only_columns(self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], __ent5: _TCCA[_T5]) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5]]:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def with_only_columns(self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], __ent5: _TCCA[_T5], __ent6: _TCCA[_T6]) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def with_only_columns(self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], __ent5: _TCCA[_T5], __ent6: _TCCA[_T6], __ent7: _TCCA[_T7]) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def with_only_columns(self, *entities: _ColumnsClauseArgument[Any], maintain_column_froms: bool=False, **__kw: Any) -> Select[Any]:
        if False:
            while True:
                i = 10
        ...

    @_generative
    def with_only_columns(self, *entities: _ColumnsClauseArgument[Any], maintain_column_froms: bool=False, **__kw: Any) -> Select[Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return a new :func:`_expression.select` construct with its columns\n        clause replaced with the given entities.\n\n        By default, this method is exactly equivalent to as if the original\n        :func:`_expression.select` had been called with the given entities.\n        E.g. a statement::\n\n            s = select(table1.c.a, table1.c.b)\n            s = s.with_only_columns(table1.c.b)\n\n        should be exactly equivalent to::\n\n            s = select(table1.c.b)\n\n        In this mode of operation, :meth:`_sql.Select.with_only_columns`\n        will also dynamically alter the FROM clause of the\n        statement if it is not explicitly stated.\n        To maintain the existing set of FROMs including those implied by the\n        current columns clause, add the\n        :paramref:`_sql.Select.with_only_columns.maintain_column_froms`\n        parameter::\n\n            s = select(table1.c.a, table2.c.b)\n            s = s.with_only_columns(table1.c.a, maintain_column_froms=True)\n\n        The above parameter performs a transfer of the effective FROMs\n        in the columns collection to the :meth:`_sql.Select.select_from`\n        method, as though the following were invoked::\n\n            s = select(table1.c.a, table2.c.b)\n            s = s.select_from(table1, table2).with_only_columns(table1.c.a)\n\n        The :paramref:`_sql.Select.with_only_columns.maintain_column_froms`\n        parameter makes use of the :attr:`_sql.Select.columns_clause_froms`\n        collection and performs an operation equivalent to the following::\n\n            s = select(table1.c.a, table2.c.b)\n            s = s.select_from(*s.columns_clause_froms).with_only_columns(table1.c.a)\n\n        :param \\*entities: column expressions to be used.\n\n        :param maintain_column_froms: boolean parameter that will ensure the\n         FROM list implied from the current columns clause will be transferred\n         to the :meth:`_sql.Select.select_from` method first.\n\n         .. versionadded:: 1.4.23\n\n        '
        if __kw:
            raise _no_kw()
        self._assert_no_memoizations()
        if maintain_column_froms:
            self.select_from.non_generative(self, *self.columns_clause_froms)
        _MemoizedSelectEntities._generate_for_statement(self)
        self._raw_columns = [coercions.expect(roles.ColumnsClauseRole, c) for c in coercions._expression_collection_was_a_list('entities', 'Select.with_only_columns', entities)]
        return self

    @property
    def whereclause(self) -> Optional[ColumnElement[Any]]:
        if False:
            for i in range(10):
                print('nop')
        'Return the completed WHERE clause for this\n        :class:`_expression.Select` statement.\n\n        This assembles the current collection of WHERE criteria\n        into a single :class:`_expression.BooleanClauseList` construct.\n\n\n        .. versionadded:: 1.4\n\n        '
        return BooleanClauseList._construct_for_whereclause(self._where_criteria)
    _whereclause = whereclause

    @_generative
    def where(self, *whereclause: _ColumnExpressionArgument[bool]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Return a new :func:`_expression.select` construct with\n        the given expression added to\n        its WHERE clause, joined to the existing clause via AND, if any.\n\n        '
        assert isinstance(self._where_criteria, tuple)
        for criterion in whereclause:
            where_criteria: ColumnElement[Any] = coercions.expect(roles.WhereHavingRole, criterion, apply_propagate_attrs=self)
            self._where_criteria += (where_criteria,)
        return self

    @_generative
    def having(self, *having: _ColumnExpressionArgument[bool]) -> Self:
        if False:
            return 10
        'Return a new :func:`_expression.select` construct with\n        the given expression added to\n        its HAVING clause, joined to the existing clause via AND, if any.\n\n        '
        for criterion in having:
            having_criteria = coercions.expect(roles.WhereHavingRole, criterion, apply_propagate_attrs=self)
            self._having_criteria += (having_criteria,)
        return self

    @_generative
    def distinct(self, *expr: _ColumnExpressionArgument[Any]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Return a new :func:`_expression.select` construct which\n        will apply DISTINCT to its columns clause.\n\n        :param \\*expr: optional column expressions.  When present,\n         the PostgreSQL dialect will render a ``DISTINCT ON (<expressions>>)``\n         construct.\n\n         .. deprecated:: 1.4 Using \\*expr in other dialects is deprecated\n            and will raise :class:`_exc.CompileError` in a future version.\n\n        '
        if expr:
            self._distinct = True
            self._distinct_on = self._distinct_on + tuple((coercions.expect(roles.ByOfRole, e, apply_propagate_attrs=self) for e in expr))
        else:
            self._distinct = True
        return self

    @_generative
    def select_from(self, *froms: _FromClauseArgument) -> Self:
        if False:
            return 10
        'Return a new :func:`_expression.select` construct with the\n        given FROM expression(s)\n        merged into its list of FROM objects.\n\n        E.g.::\n\n            table1 = table(\'t1\', column(\'a\'))\n            table2 = table(\'t2\', column(\'b\'))\n            s = select(table1.c.a).\\\n                select_from(\n                    table1.join(table2, table1.c.a==table2.c.b)\n                )\n\n        The "from" list is a unique set on the identity of each element,\n        so adding an already present :class:`_schema.Table`\n        or other selectable\n        will have no effect.   Passing a :class:`_expression.Join` that refers\n        to an already present :class:`_schema.Table`\n        or other selectable will have\n        the effect of concealing the presence of that selectable as\n        an individual element in the rendered FROM list, instead\n        rendering it into a JOIN clause.\n\n        While the typical purpose of :meth:`_expression.Select.select_from`\n        is to\n        replace the default, derived FROM clause with a join, it can\n        also be called with individual table elements, multiple times\n        if desired, in the case that the FROM clause cannot be fully\n        derived from the columns clause::\n\n            select(func.count(\'*\')).select_from(table1)\n\n        '
        self._from_obj += tuple((coercions.expect(roles.FromClauseRole, fromclause, apply_propagate_attrs=self) for fromclause in froms))
        return self

    @_generative
    def correlate(self, *fromclauses: Union[Literal[None, False], _FromClauseArgument]) -> Self:
        if False:
            return 10
        'Return a new :class:`_expression.Select`\n        which will correlate the given FROM\n        clauses to that of an enclosing :class:`_expression.Select`.\n\n        Calling this method turns off the :class:`_expression.Select` object\'s\n        default behavior of "auto-correlation".  Normally, FROM elements\n        which appear in a :class:`_expression.Select`\n        that encloses this one via\n        its :term:`WHERE clause`, ORDER BY, HAVING or\n        :term:`columns clause` will be omitted from this\n        :class:`_expression.Select`\n        object\'s :term:`FROM clause`.\n        Setting an explicit correlation collection using the\n        :meth:`_expression.Select.correlate`\n        method provides a fixed list of FROM objects\n        that can potentially take place in this process.\n\n        When :meth:`_expression.Select.correlate`\n        is used to apply specific FROM clauses\n        for correlation, the FROM elements become candidates for\n        correlation regardless of how deeply nested this\n        :class:`_expression.Select`\n        object is, relative to an enclosing :class:`_expression.Select`\n        which refers to\n        the same FROM object.  This is in contrast to the behavior of\n        "auto-correlation" which only correlates to an immediate enclosing\n        :class:`_expression.Select`.\n        Multi-level correlation ensures that the link\n        between enclosed and enclosing :class:`_expression.Select`\n        is always via\n        at least one WHERE/ORDER BY/HAVING/columns clause in order for\n        correlation to take place.\n\n        If ``None`` is passed, the :class:`_expression.Select`\n        object will correlate\n        none of its FROM entries, and all will render unconditionally\n        in the local FROM clause.\n\n        :param \\*fromclauses: one or more :class:`.FromClause` or other\n         FROM-compatible construct such as an ORM mapped entity to become part\n         of the correlate collection; alternatively pass a single value\n         ``None`` to remove all existing correlations.\n\n        .. seealso::\n\n            :meth:`_expression.Select.correlate_except`\n\n            :ref:`tutorial_scalar_subquery`\n\n        '
        self._auto_correlate = False
        if not fromclauses or fromclauses[0] in {None, False}:
            if len(fromclauses) > 1:
                raise exc.ArgumentError('additional FROM objects not accepted when passing None/False to correlate()')
            self._correlate = ()
        else:
            self._correlate = self._correlate + tuple((coercions.expect(roles.FromClauseRole, f) for f in fromclauses))
        return self

    @_generative
    def correlate_except(self, *fromclauses: Union[Literal[None, False], _FromClauseArgument]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Return a new :class:`_expression.Select`\n        which will omit the given FROM\n        clauses from the auto-correlation process.\n\n        Calling :meth:`_expression.Select.correlate_except` turns off the\n        :class:`_expression.Select` object\'s default behavior of\n        "auto-correlation" for the given FROM elements.  An element\n        specified here will unconditionally appear in the FROM list, while\n        all other FROM elements remain subject to normal auto-correlation\n        behaviors.\n\n        If ``None`` is passed, or no arguments are passed,\n        the :class:`_expression.Select` object will correlate all of its\n        FROM entries.\n\n        :param \\*fromclauses: a list of one or more\n         :class:`_expression.FromClause`\n         constructs, or other compatible constructs (i.e. ORM-mapped\n         classes) to become part of the correlate-exception collection.\n\n        .. seealso::\n\n            :meth:`_expression.Select.correlate`\n\n            :ref:`tutorial_scalar_subquery`\n\n        '
        self._auto_correlate = False
        if not fromclauses or fromclauses[0] in {None, False}:
            if len(fromclauses) > 1:
                raise exc.ArgumentError('additional FROM objects not accepted when passing None/False to correlate_except()')
            self._correlate_except = ()
        else:
            self._correlate_except = (self._correlate_except or ()) + tuple((coercions.expect(roles.FromClauseRole, f) for f in fromclauses))
        return self

    @HasMemoized_ro_memoized_attribute
    def selected_columns(self) -> ColumnCollection[str, ColumnElement[Any]]:
        if False:
            while True:
                i = 10
        'A :class:`_expression.ColumnCollection`\n        representing the columns that\n        this SELECT statement or similar construct returns in its result set,\n        not including :class:`_sql.TextClause` constructs.\n\n        This collection differs from the :attr:`_expression.FromClause.columns`\n        collection of a :class:`_expression.FromClause` in that the columns\n        within this collection cannot be directly nested inside another SELECT\n        statement; a subquery must be applied first which provides for the\n        necessary parenthesization required by SQL.\n\n        For a :func:`_expression.select` construct, the collection here is\n        exactly what would be rendered inside the "SELECT" statement, and the\n        :class:`_expression.ColumnElement` objects are directly present as they\n        were given, e.g.::\n\n            col1 = column(\'q\', Integer)\n            col2 = column(\'p\', Integer)\n            stmt = select(col1, col2)\n\n        Above, ``stmt.selected_columns`` would be a collection that contains\n        the ``col1`` and ``col2`` objects directly. For a statement that is\n        against a :class:`_schema.Table` or other\n        :class:`_expression.FromClause`, the collection will use the\n        :class:`_expression.ColumnElement` objects that are in the\n        :attr:`_expression.FromClause.c` collection of the from element.\n\n        A use case for the :attr:`_sql.Select.selected_columns` collection is\n        to allow the existing columns to be referenced when adding additional\n        criteria, e.g.::\n\n            def filter_on_id(my_select, id):\n                return my_select.where(my_select.selected_columns[\'id\'] == id)\n\n            stmt = select(MyModel)\n\n            # adds "WHERE id=:param" to the statement\n            stmt = filter_on_id(stmt, 42)\n\n        .. note::\n\n            The :attr:`_sql.Select.selected_columns` collection does not\n            include expressions established in the columns clause using the\n            :func:`_sql.text` construct; these are silently omitted from the\n            collection. To use plain textual column expressions inside of a\n            :class:`_sql.Select` construct, use the :func:`_sql.literal_column`\n            construct.\n\n\n        .. versionadded:: 1.4\n\n        '
        conv = cast('Callable[[Any], str]', SelectState._column_naming_convention(self._label_style))
        cc: ColumnCollection[str, ColumnElement[Any]] = ColumnCollection([(conv(c), c) for c in self._all_selected_columns if is_column_element(c)])
        return cc.as_readonly()

    @HasMemoized_ro_memoized_attribute
    def _all_selected_columns(self) -> _SelectIterable:
        if False:
            for i in range(10):
                print('nop')
        meth = SelectState.get_plugin_class(self).all_selected_columns
        return list(meth(self))

    def _ensure_disambiguated_names(self) -> Select[Any]:
        if False:
            print('Hello World!')
        if self._label_style is LABEL_STYLE_NONE:
            self = self.set_label_style(LABEL_STYLE_DISAMBIGUATE_ONLY)
        return self

    def _generate_fromclause_column_proxies(self, subquery: FromClause, *, proxy_compound_columns: Optional[Iterable[Sequence[ColumnElement[Any]]]]=None) -> None:
        if False:
            while True:
                i = 10
        'Generate column proxies to place in the exported ``.c``\n        collection of a subquery.'
        if proxy_compound_columns:
            extra_col_iterator = proxy_compound_columns
            prox = [c._make_proxy(subquery, key=proxy_key, name=required_label_name, name_is_truncatable=True, compound_select_cols=extra_cols) for ((required_label_name, proxy_key, fallback_label_name, c, repeated), extra_cols) in zip(self._generate_columns_plus_names(False), extra_col_iterator) if is_column_element(c)]
        else:
            prox = [c._make_proxy(subquery, key=proxy_key, name=required_label_name, name_is_truncatable=True) for (required_label_name, proxy_key, fallback_label_name, c, repeated) in self._generate_columns_plus_names(False) if is_column_element(c)]
        subquery._columns._populate_separate_keys(prox)

    def _needs_parens_for_grouping(self) -> bool:
        if False:
            print('Hello World!')
        return self._has_row_limiting_clause or bool(self._order_by_clause.clauses)

    def self_group(self, against: Optional[OperatorType]=None) -> Union[SelectStatementGrouping[Self], Self]:
        if False:
            i = 10
            return i + 15
        ...
        "Return a 'grouping' construct as per the\n        :class:`_expression.ClauseElement` specification.\n\n        This produces an element that can be embedded in an expression. Note\n        that this method is called automatically as needed when constructing\n        expressions and should not require explicit use.\n\n        "
        if isinstance(against, CompoundSelect) and (not self._needs_parens_for_grouping()):
            return self
        else:
            return SelectStatementGrouping(self)

    def union(self, *other: _SelectStatementForCompoundArgument) -> CompoundSelect:
        if False:
            for i in range(10):
                print('nop')
        'Return a SQL ``UNION`` of this select() construct against\n        the given selectables provided as positional arguments.\n\n        :param \\*other: one or more elements with which to create a\n         UNION.\n\n         .. versionchanged:: 1.4.28\n\n            multiple elements are now accepted.\n\n        :param \\**kwargs: keyword arguments are forwarded to the constructor\n         for the newly created :class:`_sql.CompoundSelect` object.\n\n        '
        return CompoundSelect._create_union(self, *other)

    def union_all(self, *other: _SelectStatementForCompoundArgument) -> CompoundSelect:
        if False:
            return 10
        'Return a SQL ``UNION ALL`` of this select() construct against\n        the given selectables provided as positional arguments.\n\n        :param \\*other: one or more elements with which to create a\n         UNION.\n\n         .. versionchanged:: 1.4.28\n\n            multiple elements are now accepted.\n\n        :param \\**kwargs: keyword arguments are forwarded to the constructor\n         for the newly created :class:`_sql.CompoundSelect` object.\n\n        '
        return CompoundSelect._create_union_all(self, *other)

    def except_(self, *other: _SelectStatementForCompoundArgument) -> CompoundSelect:
        if False:
            i = 10
            return i + 15
        'Return a SQL ``EXCEPT`` of this select() construct against\n        the given selectable provided as positional arguments.\n\n        :param \\*other: one or more elements with which to create a\n         UNION.\n\n         .. versionchanged:: 1.4.28\n\n            multiple elements are now accepted.\n\n        '
        return CompoundSelect._create_except(self, *other)

    def except_all(self, *other: _SelectStatementForCompoundArgument) -> CompoundSelect:
        if False:
            print('Hello World!')
        'Return a SQL ``EXCEPT ALL`` of this select() construct against\n        the given selectables provided as positional arguments.\n\n        :param \\*other: one or more elements with which to create a\n         UNION.\n\n         .. versionchanged:: 1.4.28\n\n            multiple elements are now accepted.\n\n        '
        return CompoundSelect._create_except_all(self, *other)

    def intersect(self, *other: _SelectStatementForCompoundArgument) -> CompoundSelect:
        if False:
            while True:
                i = 10
        'Return a SQL ``INTERSECT`` of this select() construct against\n        the given selectables provided as positional arguments.\n\n        :param \\*other: one or more elements with which to create a\n         UNION.\n\n         .. versionchanged:: 1.4.28\n\n            multiple elements are now accepted.\n\n        :param \\**kwargs: keyword arguments are forwarded to the constructor\n         for the newly created :class:`_sql.CompoundSelect` object.\n\n        '
        return CompoundSelect._create_intersect(self, *other)

    def intersect_all(self, *other: _SelectStatementForCompoundArgument) -> CompoundSelect:
        if False:
            for i in range(10):
                print('nop')
        'Return a SQL ``INTERSECT ALL`` of this select() construct\n        against the given selectables provided as positional arguments.\n\n        :param \\*other: one or more elements with which to create a\n         UNION.\n\n         .. versionchanged:: 1.4.28\n\n            multiple elements are now accepted.\n\n        :param \\**kwargs: keyword arguments are forwarded to the constructor\n         for the newly created :class:`_sql.CompoundSelect` object.\n\n        '
        return CompoundSelect._create_intersect_all(self, *other)

class ScalarSelect(roles.InElementRole, Generative, GroupedElement, ColumnElement[_T]):
    """Represent a scalar subquery.


    A :class:`_sql.ScalarSelect` is created by invoking the
    :meth:`_sql.SelectBase.scalar_subquery` method.   The object
    then participates in other SQL expressions as a SQL column expression
    within the :class:`_sql.ColumnElement` hierarchy.

    .. seealso::

        :meth:`_sql.SelectBase.scalar_subquery`

        :ref:`tutorial_scalar_subquery` - in the 2.0 tutorial

    """
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement), ('type', InternalTraversal.dp_type)]
    _from_objects: List[FromClause] = []
    _is_from_container = True
    if not TYPE_CHECKING:
        _is_implicitly_boolean = False
    inherit_cache = True
    element: SelectBase

    def __init__(self, element: SelectBase) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.element = element
        self.type = element._scalar_type()
        self._propagate_attrs = element._propagate_attrs

    def __getattr__(self, attr: str) -> Any:
        if False:
            while True:
                i = 10
        return getattr(self.element, attr)

    def __getstate__(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return {'element': self.element, 'type': self.type}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if False:
            return 10
        self.element = state['element']
        self.type = state['type']

    @property
    def columns(self) -> NoReturn:
        if False:
            while True:
                i = 10
        raise exc.InvalidRequestError('Scalar Select expression has no columns; use this object directly within a column-level expression.')
    c = columns

    @_generative
    def where(self, crit: _ColumnExpressionArgument[bool]) -> Self:
        if False:
            i = 10
            return i + 15
        'Apply a WHERE clause to the SELECT statement referred to\n        by this :class:`_expression.ScalarSelect`.\n\n        '
        self.element = cast('Select[Any]', self.element).where(crit)
        return self

    @overload
    def self_group(self: ScalarSelect[Any], against: Optional[OperatorType]=None) -> ScalarSelect[Any]:
        if False:
            print('Hello World!')
        ...

    @overload
    def self_group(self: ColumnElement[Any], against: Optional[OperatorType]=None) -> ColumnElement[Any]:
        if False:
            print('Hello World!')
        ...

    def self_group(self, against: Optional[OperatorType]=None) -> ColumnElement[Any]:
        if False:
            return 10
        return self
    if TYPE_CHECKING:

        def _ungroup(self) -> Select[Any]:
            if False:
                print('Hello World!')
            ...

    @_generative
    def correlate(self, *fromclauses: Union[Literal[None, False], _FromClauseArgument]) -> Self:
        if False:
            print('Hello World!')
        'Return a new :class:`_expression.ScalarSelect`\n        which will correlate the given FROM\n        clauses to that of an enclosing :class:`_expression.Select`.\n\n        This method is mirrored from the :meth:`_sql.Select.correlate` method\n        of the underlying :class:`_sql.Select`.  The method applies the\n        :meth:_sql.Select.correlate` method, then returns a new\n        :class:`_sql.ScalarSelect` against that statement.\n\n        .. versionadded:: 1.4 Previously, the\n           :meth:`_sql.ScalarSelect.correlate`\n           method was only available from :class:`_sql.Select`.\n\n        :param \\*fromclauses: a list of one or more\n         :class:`_expression.FromClause`\n         constructs, or other compatible constructs (i.e. ORM-mapped\n         classes) to become part of the correlate collection.\n\n        .. seealso::\n\n            :meth:`_expression.ScalarSelect.correlate_except`\n\n            :ref:`tutorial_scalar_subquery` - in the 2.0 tutorial\n\n\n        '
        self.element = cast('Select[Any]', self.element).correlate(*fromclauses)
        return self

    @_generative
    def correlate_except(self, *fromclauses: Union[Literal[None, False], _FromClauseArgument]) -> Self:
        if False:
            i = 10
            return i + 15
        'Return a new :class:`_expression.ScalarSelect`\n        which will omit the given FROM\n        clauses from the auto-correlation process.\n\n        This method is mirrored from the\n        :meth:`_sql.Select.correlate_except` method of the underlying\n        :class:`_sql.Select`.  The method applies the\n        :meth:_sql.Select.correlate_except` method, then returns a new\n        :class:`_sql.ScalarSelect` against that statement.\n\n        .. versionadded:: 1.4 Previously, the\n           :meth:`_sql.ScalarSelect.correlate_except`\n           method was only available from :class:`_sql.Select`.\n\n        :param \\*fromclauses: a list of one or more\n         :class:`_expression.FromClause`\n         constructs, or other compatible constructs (i.e. ORM-mapped\n         classes) to become part of the correlate-exception collection.\n\n        .. seealso::\n\n            :meth:`_expression.ScalarSelect.correlate`\n\n            :ref:`tutorial_scalar_subquery` - in the 2.0 tutorial\n\n\n        '
        self.element = cast('Select[Any]', self.element).correlate_except(*fromclauses)
        return self

class Exists(UnaryExpression[bool]):
    """Represent an ``EXISTS`` clause.

    See :func:`_sql.exists` for a description of usage.

    An ``EXISTS`` clause can also be constructed from a :func:`_sql.select`
    instance by calling :meth:`_sql.SelectBase.exists`.

    """
    inherit_cache = True
    element: Union[SelectStatementGrouping[Select[Any]], ScalarSelect[Any]]

    def __init__(self, __argument: Optional[Union[_ColumnsClauseArgument[Any], SelectBase, ScalarSelect[Any]]]=None, /):
        if False:
            while True:
                i = 10
        s: ScalarSelect[Any]
        if __argument is None:
            s = Select(literal_column('*')).scalar_subquery()
        elif isinstance(__argument, SelectBase):
            s = __argument.scalar_subquery()
            s._propagate_attrs = __argument._propagate_attrs
        elif isinstance(__argument, ScalarSelect):
            s = __argument
        else:
            s = Select(__argument).scalar_subquery()
        UnaryExpression.__init__(self, s, operator=operators.exists, type_=type_api.BOOLEANTYPE, wraps_column_expression=True)

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        if False:
            print('Hello World!')
        return []

    def _regroup(self, fn: Callable[[Select[Any]], Select[Any]]) -> SelectStatementGrouping[Select[Any]]:
        if False:
            for i in range(10):
                print('nop')
        element = self.element._ungroup()
        new_element = fn(element)
        return_value = new_element.self_group(against=operators.exists)
        assert isinstance(return_value, SelectStatementGrouping)
        return return_value

    def select(self) -> Select[Any]:
        if False:
            return 10
        'Return a SELECT of this :class:`_expression.Exists`.\n\n        e.g.::\n\n            stmt = exists(some_table.c.id).where(some_table.c.id == 5).select()\n\n        This will produce a statement resembling::\n\n            SELECT EXISTS (SELECT id FROM some_table WHERE some_table = :param) AS anon_1\n\n        .. seealso::\n\n            :func:`_expression.select` - general purpose\n            method which allows for arbitrary column lists.\n\n        '
        return Select(self)

    def correlate(self, *fromclauses: Union[Literal[None, False], _FromClauseArgument]) -> Self:
        if False:
            i = 10
            return i + 15
        'Apply correlation to the subquery noted by this\n        :class:`_sql.Exists`.\n\n        .. seealso::\n\n            :meth:`_sql.ScalarSelect.correlate`\n\n        '
        e = self._clone()
        e.element = self._regroup(lambda element: element.correlate(*fromclauses))
        return e

    def correlate_except(self, *fromclauses: Union[Literal[None, False], _FromClauseArgument]) -> Self:
        if False:
            i = 10
            return i + 15
        'Apply correlation to the subquery noted by this\n        :class:`_sql.Exists`.\n\n        .. seealso::\n\n            :meth:`_sql.ScalarSelect.correlate_except`\n\n        '
        e = self._clone()
        e.element = self._regroup(lambda element: element.correlate_except(*fromclauses))
        return e

    def select_from(self, *froms: _FromClauseArgument) -> Self:
        if False:
            print('Hello World!')
        'Return a new :class:`_expression.Exists` construct,\n        applying the given\n        expression to the :meth:`_expression.Select.select_from`\n        method of the select\n        statement contained.\n\n        .. note:: it is typically preferable to build a :class:`_sql.Select`\n           statement first, including the desired WHERE clause, then use the\n           :meth:`_sql.SelectBase.exists` method to produce an\n           :class:`_sql.Exists` object at once.\n\n        '
        e = self._clone()
        e.element = self._regroup(lambda element: element.select_from(*froms))
        return e

    def where(self, *clause: _ColumnExpressionArgument[bool]) -> Self:
        if False:
            while True:
                i = 10
        'Return a new :func:`_expression.exists` construct with the\n        given expression added to\n        its WHERE clause, joined to the existing clause via AND, if any.\n\n\n        .. note:: it is typically preferable to build a :class:`_sql.Select`\n           statement first, including the desired WHERE clause, then use the\n           :meth:`_sql.SelectBase.exists` method to produce an\n           :class:`_sql.Exists` object at once.\n\n        '
        e = self._clone()
        e.element = self._regroup(lambda element: element.where(*clause))
        return e

class TextualSelect(SelectBase, ExecutableReturnsRows, Generative):
    """Wrap a :class:`_expression.TextClause` construct within a
    :class:`_expression.SelectBase`
    interface.

    This allows the :class:`_expression.TextClause` object to gain a
    ``.c`` collection
    and other FROM-like capabilities such as
    :meth:`_expression.FromClause.alias`,
    :meth:`_expression.SelectBase.cte`, etc.

    The :class:`_expression.TextualSelect` construct is produced via the
    :meth:`_expression.TextClause.columns`
    method - see that method for details.

    .. versionchanged:: 1.4 the :class:`_expression.TextualSelect`
       class was renamed
       from ``TextAsFrom``, to more correctly suit its role as a
       SELECT-oriented object and not a FROM clause.

    .. seealso::

        :func:`_expression.text`

        :meth:`_expression.TextClause.columns` - primary creation interface.

    """
    __visit_name__ = 'textual_select'
    _label_style = LABEL_STYLE_NONE
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement), ('column_args', InternalTraversal.dp_clauseelement_list)] + SupportsCloneAnnotations._clone_annotations_traverse_internals
    _is_textual = True
    is_text = True
    is_select = True

    def __init__(self, text: TextClause, columns: List[_ColumnExpressionArgument[Any]], positional: bool=False) -> None:
        if False:
            return 10
        self._init(text, [coercions.expect(roles.LabeledColumnExprRole, c) for c in columns], positional)

    def _init(self, text: TextClause, columns: List[NamedColumn[Any]], positional: bool=False) -> None:
        if False:
            return 10
        self.element = text
        self.column_args = columns
        self.positional = positional

    @HasMemoized_ro_memoized_attribute
    def selected_columns(self) -> ColumnCollection[str, KeyedColumnElement[Any]]:
        if False:
            return 10
        'A :class:`_expression.ColumnCollection`\n        representing the columns that\n        this SELECT statement or similar construct returns in its result set,\n        not including :class:`_sql.TextClause` constructs.\n\n        This collection differs from the :attr:`_expression.FromClause.columns`\n        collection of a :class:`_expression.FromClause` in that the columns\n        within this collection cannot be directly nested inside another SELECT\n        statement; a subquery must be applied first which provides for the\n        necessary parenthesization required by SQL.\n\n        For a :class:`_expression.TextualSelect` construct, the collection\n        contains the :class:`_expression.ColumnElement` objects that were\n        passed to the constructor, typically via the\n        :meth:`_expression.TextClause.columns` method.\n\n\n        .. versionadded:: 1.4\n\n        '
        return ColumnCollection(((c.key, c) for c in self.column_args)).as_readonly()

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        if False:
            i = 10
            return i + 15
        return self.column_args

    def set_label_style(self, style: SelectLabelStyle) -> TextualSelect:
        if False:
            print('Hello World!')
        return self

    def _ensure_disambiguated_names(self) -> TextualSelect:
        if False:
            print('Hello World!')
        return self

    @_generative
    def bindparams(self, *binds: BindParameter[Any], **bind_as_values: Any) -> Self:
        if False:
            for i in range(10):
                print('nop')
        self.element = self.element.bindparams(*binds, **bind_as_values)
        return self

    def _generate_fromclause_column_proxies(self, fromclause: FromClause, *, proxy_compound_columns: Optional[Iterable[Sequence[ColumnElement[Any]]]]=None) -> None:
        if False:
            print('Hello World!')
        if TYPE_CHECKING:
            assert isinstance(fromclause, Subquery)
        if proxy_compound_columns:
            fromclause._columns._populate_separate_keys((c._make_proxy(fromclause, compound_select_cols=extra_cols) for (c, extra_cols) in zip(self.column_args, proxy_compound_columns)))
        else:
            fromclause._columns._populate_separate_keys((c._make_proxy(fromclause) for c in self.column_args))

    def _scalar_type(self) -> Union[TypeEngine[Any], Any]:
        if False:
            while True:
                i = 10
        return self.column_args[0].type
TextAsFrom = TextualSelect
'Backwards compatibility with the previous name'

class AnnotatedFromClause(Annotated):

    def _copy_internals(self, **kw: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        super()._copy_internals(**kw)
        if kw.get('ind_cols_on_fromclause', False):
            ee = self._Annotated__element
            self.c = ee.__class__.c.fget(self)

    @util.ro_memoized_property
    def c(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        if False:
            return 10
        'proxy the .c collection of the underlying FromClause.\n\n        Originally implemented in 2008 as a simple load of the .c collection\n        when the annotated construct was created (see d3621ae961a), in modern\n        SQLAlchemy versions this can be expensive for statements constructed\n        with ORM aliases.   So for #8796 SQLAlchemy 2.0 we instead proxy\n        it, which works just as well.\n\n        Two different use cases seem to require the collection either copied\n        from the underlying one, or unique to this AnnotatedFromClause.\n\n        See test_selectable->test_annotated_corresponding_column\n\n        '
        ee = self._Annotated__element
        return ee.c