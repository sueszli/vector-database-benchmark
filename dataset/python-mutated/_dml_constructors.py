from __future__ import annotations
from typing import TYPE_CHECKING
from .dml import Delete
from .dml import Insert
from .dml import Update
if TYPE_CHECKING:
    from ._typing import _DMLTableArgument

def insert(table: _DMLTableArgument) -> Insert:
    if False:
        return 10
    'Construct an :class:`_expression.Insert` object.\n\n    E.g.::\n\n        from sqlalchemy import insert\n\n        stmt = (\n            insert(user_table).\n            values(name=\'username\', fullname=\'Full Username\')\n        )\n\n    Similar functionality is available via the\n    :meth:`_expression.TableClause.insert` method on\n    :class:`_schema.Table`.\n\n    .. seealso::\n\n        :ref:`tutorial_core_insert` - in the :ref:`unified_tutorial`\n\n\n    :param table: :class:`_expression.TableClause`\n     which is the subject of the\n     insert.\n\n    :param values: collection of values to be inserted; see\n     :meth:`_expression.Insert.values`\n     for a description of allowed formats here.\n     Can be omitted entirely; a :class:`_expression.Insert` construct\n     will also dynamically render the VALUES clause at execution time\n     based on the parameters passed to :meth:`_engine.Connection.execute`.\n\n    :param inline: if True, no attempt will be made to retrieve the\n     SQL-generated default values to be provided within the statement;\n     in particular,\n     this allows SQL expressions to be rendered \'inline\' within the\n     statement without the need to pre-execute them beforehand; for\n     backends that support "returning", this turns off the "implicit\n     returning" feature for the statement.\n\n    If both :paramref:`_expression.insert.values` and compile-time bind\n    parameters are present, the compile-time bind parameters override the\n    information specified within :paramref:`_expression.insert.values` on a\n    per-key basis.\n\n    The keys within :paramref:`_expression.Insert.values` can be either\n    :class:`~sqlalchemy.schema.Column` objects or their string\n    identifiers. Each key may reference one of:\n\n    * a literal data value (i.e. string, number, etc.);\n    * a Column object;\n    * a SELECT statement.\n\n    If a ``SELECT`` statement is specified which references this\n    ``INSERT`` statement\'s table, the statement will be correlated\n    against the ``INSERT`` statement.\n\n    .. seealso::\n\n        :ref:`tutorial_core_insert` - in the :ref:`unified_tutorial`\n\n    '
    return Insert(table)

def update(table: _DMLTableArgument) -> Update:
    if False:
        return 10
    "Construct an :class:`_expression.Update` object.\n\n    E.g.::\n\n        from sqlalchemy import update\n\n        stmt = (\n            update(user_table).\n            where(user_table.c.id == 5).\n            values(name='user #5')\n        )\n\n    Similar functionality is available via the\n    :meth:`_expression.TableClause.update` method on\n    :class:`_schema.Table`.\n\n    :param table: A :class:`_schema.Table`\n     object representing the database\n     table to be updated.\n\n\n    .. seealso::\n\n        :ref:`tutorial_core_update_delete` - in the :ref:`unified_tutorial`\n\n\n    "
    return Update(table)

def delete(table: _DMLTableArgument) -> Delete:
    if False:
        return 10
    'Construct :class:`_expression.Delete` object.\n\n    E.g.::\n\n        from sqlalchemy import delete\n\n        stmt = (\n            delete(user_table).\n            where(user_table.c.id == 5)\n        )\n\n    Similar functionality is available via the\n    :meth:`_expression.TableClause.delete` method on\n    :class:`_schema.Table`.\n\n    :param table: The table to delete rows from.\n\n    .. seealso::\n\n        :ref:`tutorial_core_update_delete` - in the :ref:`unified_tutorial`\n\n\n    '
    return Delete(table)