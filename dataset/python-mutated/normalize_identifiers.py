from __future__ import annotations
import typing as t
from sqlglot import exp
from sqlglot._typing import E
from sqlglot.dialects.dialect import Dialect, DialectType

@t.overload
def normalize_identifiers(expression: E, dialect: DialectType=None) -> E:
    if False:
        i = 10
        return i + 15
    ...

@t.overload
def normalize_identifiers(expression: str, dialect: DialectType=None) -> exp.Identifier:
    if False:
        for i in range(10):
            print('nop')
    ...

def normalize_identifiers(expression, dialect=None):
    if False:
        while True:
            i = 10
    '\n    Normalize all unquoted identifiers to either lower or upper case, depending\n    on the dialect. This essentially makes those identifiers case-insensitive.\n\n    It\'s possible to make this a no-op by adding a special comment next to the\n    identifier of interest:\n\n        SELECT a /* sqlglot.meta case_sensitive */ FROM table\n\n    In this example, the identifier `a` will not be normalized.\n\n    Note:\n        Some dialects (e.g. BigQuery) treat identifiers as case-insensitive even\n        when they\'re quoted, so in these cases all identifiers are normalized.\n\n    Example:\n        >>> import sqlglot\n        >>> expression = sqlglot.parse_one(\'SELECT Bar.A AS A FROM "Foo".Bar\')\n        >>> normalize_identifiers(expression).sql()\n        \'SELECT bar.a AS a FROM "Foo".bar\'\n        >>> normalize_identifiers("foo", dialect="snowflake").sql(dialect="snowflake")\n        \'FOO\'\n\n    Args:\n        expression: The expression to transform.\n        dialect: The dialect to use in order to decide how to normalize identifiers.\n\n    Returns:\n        The transformed expression.\n    '
    if isinstance(expression, str):
        expression = exp.parse_identifier(expression, dialect=dialect)
    dialect = Dialect.get_or_raise(dialect)

    def _normalize(node: E) -> E:
        if False:
            print('Hello World!')
        if not node.meta.get('case_sensitive'):
            exp.replace_children(node, _normalize)
            node = dialect.normalize_identifier(node)
        return node
    return _normalize(expression)