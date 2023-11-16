from __future__ import annotations
from typing import Any
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from .elements import ColumnClause
from .selectable import Alias
from .selectable import CompoundSelect
from .selectable import Exists
from .selectable import FromClause
from .selectable import Join
from .selectable import Lateral
from .selectable import LateralFromClause
from .selectable import NamedFromClause
from .selectable import Select
from .selectable import TableClause
from .selectable import TableSample
from .selectable import Values
if TYPE_CHECKING:
    from ._typing import _FromClauseArgument
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
    from ._typing import _T8
    from ._typing import _T9
    from ._typing import _TypedColumnClauseArgument as _TCCA
    from .functions import Function
    from .selectable import CTE
    from .selectable import HasCTE
    from .selectable import ScalarSelect
    from .selectable import SelectBase
_T = TypeVar('_T', bound=Any)

def alias(selectable: FromClause, name: Optional[str]=None, flat: bool=False) -> NamedFromClause:
    if False:
        print('Hello World!')
    'Return a named alias of the given :class:`.FromClause`.\n\n    For :class:`.Table` and :class:`.Join` objects, the return type is the\n    :class:`_expression.Alias` object. Other kinds of :class:`.NamedFromClause`\n    objects may be returned for other kinds of :class:`.FromClause` objects.\n\n    The named alias represents any :class:`_expression.FromClause` with an\n    alternate name assigned within SQL, typically using the ``AS`` clause when\n    generated, e.g. ``SELECT * FROM table AS aliasname``.\n\n    Equivalent functionality is available via the\n    :meth:`_expression.FromClause.alias`\n    method available on all :class:`_expression.FromClause` objects.\n\n    :param selectable: any :class:`_expression.FromClause` subclass,\n        such as a table, select statement, etc.\n\n    :param name: string name to be assigned as the alias.\n        If ``None``, a name will be deterministically generated at compile\n        time. Deterministic means the name is guaranteed to be unique against\n        other constructs used in the same statement, and will also be the same\n        name for each successive compilation of the same statement object.\n\n    :param flat: Will be passed through to if the given selectable\n     is an instance of :class:`_expression.Join` - see\n     :meth:`_expression.Join.alias` for details.\n\n    '
    return Alias._factory(selectable, name=name, flat=flat)

def cte(selectable: HasCTE, name: Optional[str]=None, recursive: bool=False) -> CTE:
    if False:
        return 10
    'Return a new :class:`_expression.CTE`,\n    or Common Table Expression instance.\n\n    Please see :meth:`_expression.HasCTE.cte` for detail on CTE usage.\n\n    '
    return coercions.expect(roles.HasCTERole, selectable).cte(name=name, recursive=recursive)

def except_(*selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
    if False:
        for i in range(10):
            print('nop')
    'Return an ``EXCEPT`` of multiple selectables.\n\n    The returned object is an instance of\n    :class:`_expression.CompoundSelect`.\n\n    :param \\*selects:\n      a list of :class:`_expression.Select` instances.\n\n    '
    return CompoundSelect._create_except(*selects)

def except_all(*selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
    if False:
        print('Hello World!')
    'Return an ``EXCEPT ALL`` of multiple selectables.\n\n    The returned object is an instance of\n    :class:`_expression.CompoundSelect`.\n\n    :param \\*selects:\n      a list of :class:`_expression.Select` instances.\n\n    '
    return CompoundSelect._create_except_all(*selects)

def exists(__argument: Optional[Union[_ColumnsClauseArgument[Any], SelectBase, ScalarSelect[Any]]]=None, /) -> Exists:
    if False:
        for i in range(10):
            print('nop')
    'Construct a new :class:`_expression.Exists` construct.\n\n    The :func:`_sql.exists` can be invoked by itself to produce an\n    :class:`_sql.Exists` construct, which will accept simple WHERE\n    criteria::\n\n        exists_criteria = exists().where(table1.c.col1 == table2.c.col2)\n\n    However, for greater flexibility in constructing the SELECT, an\n    existing :class:`_sql.Select` construct may be converted to an\n    :class:`_sql.Exists`, most conveniently by making use of the\n    :meth:`_sql.SelectBase.exists` method::\n\n        exists_criteria = (\n            select(table2.c.col2).\n            where(table1.c.col1 == table2.c.col2).\n            exists()\n        )\n\n    The EXISTS criteria is then used inside of an enclosing SELECT::\n\n        stmt = select(table1.c.col1).where(exists_criteria)\n\n    The above statement will then be of the form::\n\n        SELECT col1 FROM table1 WHERE EXISTS\n        (SELECT table2.col2 FROM table2 WHERE table2.col2 = table1.col1)\n\n    .. seealso::\n\n        :ref:`tutorial_exists` - in the :term:`2.0 style` tutorial.\n\n        :meth:`_sql.SelectBase.exists` - method to transform a ``SELECT`` to an\n        ``EXISTS`` clause.\n\n    '
    return Exists(__argument)

def intersect(*selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
    if False:
        for i in range(10):
            print('nop')
    'Return an ``INTERSECT`` of multiple selectables.\n\n    The returned object is an instance of\n    :class:`_expression.CompoundSelect`.\n\n    :param \\*selects:\n      a list of :class:`_expression.Select` instances.\n\n    '
    return CompoundSelect._create_intersect(*selects)

def intersect_all(*selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
    if False:
        for i in range(10):
            print('nop')
    'Return an ``INTERSECT ALL`` of multiple selectables.\n\n    The returned object is an instance of\n    :class:`_expression.CompoundSelect`.\n\n    :param \\*selects:\n      a list of :class:`_expression.Select` instances.\n\n\n    '
    return CompoundSelect._create_intersect_all(*selects)

def join(left: _FromClauseArgument, right: _FromClauseArgument, onclause: Optional[_OnClauseArgument]=None, isouter: bool=False, full: bool=False) -> Join:
    if False:
        for i in range(10):
            print('nop')
    'Produce a :class:`_expression.Join` object, given two\n    :class:`_expression.FromClause`\n    expressions.\n\n    E.g.::\n\n        j = join(user_table, address_table,\n                 user_table.c.id == address_table.c.user_id)\n        stmt = select(user_table).select_from(j)\n\n    would emit SQL along the lines of::\n\n        SELECT user.id, user.name FROM user\n        JOIN address ON user.id = address.user_id\n\n    Similar functionality is available given any\n    :class:`_expression.FromClause` object (e.g. such as a\n    :class:`_schema.Table`) using\n    the :meth:`_expression.FromClause.join` method.\n\n    :param left: The left side of the join.\n\n    :param right: the right side of the join; this is any\n     :class:`_expression.FromClause` object such as a\n     :class:`_schema.Table` object, and\n     may also be a selectable-compatible object such as an ORM-mapped\n     class.\n\n    :param onclause: a SQL expression representing the ON clause of the\n     join.  If left at ``None``, :meth:`_expression.FromClause.join`\n     will attempt to\n     join the two tables based on a foreign key relationship.\n\n    :param isouter: if True, render a LEFT OUTER JOIN, instead of JOIN.\n\n    :param full: if True, render a FULL OUTER JOIN, instead of JOIN.\n\n    .. seealso::\n\n        :meth:`_expression.FromClause.join` - method form,\n        based on a given left side.\n\n        :class:`_expression.Join` - the type of object produced.\n\n    '
    return Join(left, right, onclause, isouter, full)

def lateral(selectable: Union[SelectBase, _FromClauseArgument], name: Optional[str]=None) -> LateralFromClause:
    if False:
        for i in range(10):
            print('nop')
    'Return a :class:`_expression.Lateral` object.\n\n    :class:`_expression.Lateral` is an :class:`_expression.Alias`\n    subclass that represents\n    a subquery with the LATERAL keyword applied to it.\n\n    The special behavior of a LATERAL subquery is that it appears in the\n    FROM clause of an enclosing SELECT, but may correlate to other\n    FROM clauses of that SELECT.   It is a special case of subquery\n    only supported by a small number of backends, currently more recent\n    PostgreSQL versions.\n\n    .. seealso::\n\n        :ref:`tutorial_lateral_correlation` -  overview of usage.\n\n    '
    return Lateral._factory(selectable, name=name)

def outerjoin(left: _FromClauseArgument, right: _FromClauseArgument, onclause: Optional[_OnClauseArgument]=None, full: bool=False) -> Join:
    if False:
        while True:
            i = 10
    'Return an ``OUTER JOIN`` clause element.\n\n    The returned object is an instance of :class:`_expression.Join`.\n\n    Similar functionality is also available via the\n    :meth:`_expression.FromClause.outerjoin` method on any\n    :class:`_expression.FromClause`.\n\n    :param left: The left side of the join.\n\n    :param right: The right side of the join.\n\n    :param onclause:  Optional criterion for the ``ON`` clause, is\n      derived from foreign key relationships established between\n      left and right otherwise.\n\n    To chain joins together, use the :meth:`_expression.FromClause.join`\n    or\n    :meth:`_expression.FromClause.outerjoin` methods on the resulting\n    :class:`_expression.Join` object.\n\n    '
    return Join(left, right, onclause, isouter=True, full=full)

@overload
def select(__ent0: _TCCA[_T0], /) -> Select[Tuple[_T0]]:
    if False:
        print('Hello World!')
    ...

@overload
def select(__ent0: _TCCA[_T0], __ent1: _TCCA[_T1], /) -> Select[Tuple[_T0, _T1]]:
    if False:
        print('Hello World!')
    ...

@overload
def select(__ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], /) -> Select[Tuple[_T0, _T1, _T2]]:
    if False:
        return 10
    ...

@overload
def select(__ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], __ent3: _TCCA[_T3], /) -> Select[Tuple[_T0, _T1, _T2, _T3]]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def select(__ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], /) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4]]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def select(__ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], __ent5: _TCCA[_T5], /) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5]]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def select(__ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], __ent5: _TCCA[_T5], __ent6: _TCCA[_T6], /) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]:
    if False:
        while True:
            i = 10
    ...

@overload
def select(__ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], __ent5: _TCCA[_T5], __ent6: _TCCA[_T6], __ent7: _TCCA[_T7], /) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]]:
    if False:
        print('Hello World!')
    ...

@overload
def select(__ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], __ent5: _TCCA[_T5], __ent6: _TCCA[_T6], __ent7: _TCCA[_T7], __ent8: _TCCA[_T8], /) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8]]:
    if False:
        print('Hello World!')
    ...

@overload
def select(__ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2], __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], __ent5: _TCCA[_T5], __ent6: _TCCA[_T6], __ent7: _TCCA[_T7], __ent8: _TCCA[_T8], __ent9: _TCCA[_T9], /) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8, _T9]]:
    if False:
        while True:
            i = 10
    ...

@overload
def select(*entities: _ColumnsClauseArgument[Any], **__kw: Any) -> Select[Any]:
    if False:
        return 10
    ...

def select(*entities: _ColumnsClauseArgument[Any], **__kw: Any) -> Select[Any]:
    if False:
        for i in range(10):
            print('nop')
    'Construct a new :class:`_expression.Select`.\n\n\n    .. versionadded:: 1.4 - The :func:`_sql.select` function now accepts\n       column arguments positionally.   The top-level :func:`_sql.select`\n       function will automatically use the 1.x or 2.x style API based on\n       the incoming arguments; using :func:`_sql.select` from the\n       ``sqlalchemy.future`` module will enforce that only the 2.x style\n       constructor is used.\n\n    Similar functionality is also available via the\n    :meth:`_expression.FromClause.select` method on any\n    :class:`_expression.FromClause`.\n\n    .. seealso::\n\n        :ref:`tutorial_selecting_data` - in the :ref:`unified_tutorial`\n\n    :param \\*entities:\n      Entities to SELECT from.  For Core usage, this is typically a series\n      of :class:`_expression.ColumnElement` and / or\n      :class:`_expression.FromClause`\n      objects which will form the columns clause of the resulting\n      statement.   For those objects that are instances of\n      :class:`_expression.FromClause` (typically :class:`_schema.Table`\n      or :class:`_expression.Alias`\n      objects), the :attr:`_expression.FromClause.c`\n      collection is extracted\n      to form a collection of :class:`_expression.ColumnElement` objects.\n\n      This parameter will also accept :class:`_expression.TextClause`\n      constructs as\n      given, as well as ORM-mapped classes.\n\n    '
    if __kw:
        raise _no_kw()
    return Select(*entities)

def table(name: str, *columns: ColumnClause[Any], **kw: Any) -> TableClause:
    if False:
        i = 10
        return i + 15
    'Produce a new :class:`_expression.TableClause`.\n\n    The object returned is an instance of\n    :class:`_expression.TableClause`, which\n    represents the "syntactical" portion of the schema-level\n    :class:`_schema.Table` object.\n    It may be used to construct lightweight table constructs.\n\n    :param name: Name of the table.\n\n    :param columns: A collection of :func:`_expression.column` constructs.\n\n    :param schema: The schema name for this table.\n\n        .. versionadded:: 1.3.18 :func:`_expression.table` can now\n           accept a ``schema`` argument.\n    '
    return TableClause(name, *columns, **kw)

def tablesample(selectable: _FromClauseArgument, sampling: Union[float, Function[Any]], name: Optional[str]=None, seed: Optional[roles.ExpressionElementRole[Any]]=None) -> TableSample:
    if False:
        i = 10
        return i + 15
    "Return a :class:`_expression.TableSample` object.\n\n    :class:`_expression.TableSample` is an :class:`_expression.Alias`\n    subclass that represents\n    a table with the TABLESAMPLE clause applied to it.\n    :func:`_expression.tablesample`\n    is also available from the :class:`_expression.FromClause`\n    class via the\n    :meth:`_expression.FromClause.tablesample` method.\n\n    The TABLESAMPLE clause allows selecting a randomly selected approximate\n    percentage of rows from a table. It supports multiple sampling methods,\n    most commonly BERNOULLI and SYSTEM.\n\n    e.g.::\n\n        from sqlalchemy import func\n\n        selectable = people.tablesample(\n                    func.bernoulli(1),\n                    name='alias',\n                    seed=func.random())\n        stmt = select(selectable.c.people_id)\n\n    Assuming ``people`` with a column ``people_id``, the above\n    statement would render as::\n\n        SELECT alias.people_id FROM\n        people AS alias TABLESAMPLE bernoulli(:bernoulli_1)\n        REPEATABLE (random())\n\n    :param sampling: a ``float`` percentage between 0 and 100 or\n        :class:`_functions.Function`.\n\n    :param name: optional alias name\n\n    :param seed: any real-valued SQL expression.  When specified, the\n     REPEATABLE sub-clause is also rendered.\n\n    "
    return TableSample._factory(selectable, sampling, name=name, seed=seed)

def union(*selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
    if False:
        return 10
    'Return a ``UNION`` of multiple selectables.\n\n    The returned object is an instance of\n    :class:`_expression.CompoundSelect`.\n\n    A similar :func:`union()` method is available on all\n    :class:`_expression.FromClause` subclasses.\n\n    :param \\*selects:\n      a list of :class:`_expression.Select` instances.\n\n    :param \\**kwargs:\n      available keyword arguments are the same as those of\n      :func:`select`.\n\n    '
    return CompoundSelect._create_union(*selects)

def union_all(*selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
    if False:
        print('Hello World!')
    'Return a ``UNION ALL`` of multiple selectables.\n\n    The returned object is an instance of\n    :class:`_expression.CompoundSelect`.\n\n    A similar :func:`union_all()` method is available on all\n    :class:`_expression.FromClause` subclasses.\n\n    :param \\*selects:\n      a list of :class:`_expression.Select` instances.\n\n    '
    return CompoundSelect._create_union_all(*selects)

def values(*columns: ColumnClause[Any], name: Optional[str]=None, literal_binds: bool=False) -> Values:
    if False:
        return 10
    'Construct a :class:`_expression.Values` construct.\n\n    The column expressions and the actual data for\n    :class:`_expression.Values` are given in two separate steps.  The\n    constructor receives the column expressions typically as\n    :func:`_expression.column` constructs,\n    and the data is then passed via the\n    :meth:`_expression.Values.data` method as a list,\n    which can be called multiple\n    times to add more data, e.g.::\n\n        from sqlalchemy import column\n        from sqlalchemy import values\n\n        value_expr = values(\n            column(\'id\', Integer),\n            column(\'name\', String),\n            name="my_values"\n        ).data(\n            [(1, \'name1\'), (2, \'name2\'), (3, \'name3\')]\n        )\n\n    :param \\*columns: column expressions, typically composed using\n     :func:`_expression.column` objects.\n\n    :param name: the name for this VALUES construct.  If omitted, the\n     VALUES construct will be unnamed in a SQL expression.   Different\n     backends may have different requirements here.\n\n    :param literal_binds: Defaults to False.  Whether or not to render\n     the data values inline in the SQL output, rather than using bound\n     parameters.\n\n    '
    return Values(*columns, literal_binds=literal_binds, name=name)