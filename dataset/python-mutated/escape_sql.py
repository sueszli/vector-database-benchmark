import re
from datetime import datetime, date
from typing import Optional, Any, Literal, List, Tuple
from uuid import UUID
from zoneinfo import ZoneInfo
import math
from posthog.hogql.errors import HogQLException
from posthog.models.utils import UUIDT
escape_chars_map = {'\x08': '\\b', '\x0c': '\\f', '\r': '\\r', '\n': '\\n', '\t': '\\t', '\x00': '\\0', '\x07': '\\a', '\x0b': '\\v', '\\': '\\\\'}
singlequote_escape_chars_map = {**escape_chars_map, "'": "\\'"}
backquote_escape_chars_map = {**escape_chars_map, '`': '\\`'}

def escape_param_clickhouse(value: str) -> str:
    if False:
        while True:
            i = 10
    return "'%s'" % ''.join((singlequote_escape_chars_map.get(c, c) for c in str(value)))

def escape_hogql_identifier(identifier: str | int) -> str:
    if False:
        return 10
    if isinstance(identifier, int):
        return str(identifier)
    if '%' in identifier:
        raise HogQLException(f'The HogQL identifier "{identifier}" is not permitted as it contains the "%" character')
    if re.match('^[A-Za-z_$][A-Za-z0-9_$]*$', identifier):
        return identifier
    return '`%s`' % ''.join((backquote_escape_chars_map.get(c, c) for c in identifier))

def escape_clickhouse_identifier(identifier: str) -> str:
    if False:
        print('Hello World!')
    if '%' in identifier:
        raise HogQLException(f'The HogQL identifier "{identifier}" is not permitted as it contains the "%" character')
    if re.match('^[A-Za-z_][A-Za-z0-9_]*$', identifier):
        return identifier
    return '`%s`' % ''.join((backquote_escape_chars_map.get(c, c) for c in identifier))

def escape_hogql_string(name: float | int | str | list | tuple | date | datetime | UUID | UUIDT, timezone: Optional[str]=None) -> str:
    if False:
        while True:
            i = 10
    return SQLValueEscaper(timezone=timezone, dialect='hogql').visit(name)

def escape_clickhouse_string(name: float | int | str | list | tuple | date | datetime | UUID | UUIDT, timezone: Optional[str]=None) -> str:
    if False:
        i = 10
        return i + 15
    return SQLValueEscaper(timezone=timezone, dialect='clickhouse').visit(name)

class SQLValueEscaper:

    def __init__(self, timezone: Optional[str]=None, dialect: Literal['hogql', 'clickhouse']='clickhouse'):
        if False:
            return 10
        self._timezone = timezone or 'UTC'
        self._dialect = dialect

    def visit(self, node: Any) -> str:
        if False:
            for i in range(10):
                print('nop')
        method_name = f'visit_{node.__class__.__name__.lower()}'
        if hasattr(self, method_name):
            return getattr(self, method_name)(node)
        raise HogQLException(f'SQLValueEscaper has no method {method_name}')

    def visit_nonetype(self, value: None):
        if False:
            return 10
        return 'NULL'

    def visit_str(self, value: str):
        if False:
            for i in range(10):
                print('nop')
        return escape_param_clickhouse(value)

    def visit_bool(self, value: bool):
        if False:
            print('Hello World!')
        return 'true' if value is True else 'false'

    def visit_int(self, value: int):
        if False:
            while True:
                i = 10
        return str(value)

    def visit_float(self, value: float):
        if False:
            return 10
        if math.isnan(value):
            return 'NaN'
        if math.isinf(value):
            if value == float('-inf'):
                return '-Inf'
            return 'Inf'
        return str(value)

    def visit_uuid(self, value: UUID):
        if False:
            print('Hello World!')
        if self._dialect == 'hogql':
            return f'toUUID({self.visit(str(value))})'
        return f'toUUIDOrNull({self.visit(str(value))})'

    def visit_uuidt(self, value: UUIDT):
        if False:
            while True:
                i = 10
        if self._dialect == 'hogql':
            return f'toUUID({self.visit(str(value))})'
        return f'toUUIDOrNull({self.visit(str(value))})'

    def visit_fakedatetime(self, value: datetime):
        if False:
            for i in range(10):
                print('nop')
        return self.visit_datetime(value)

    def visit_datetime(self, value: datetime):
        if False:
            while True:
                i = 10
        datetime_string = value.astimezone(ZoneInfo(self._timezone)).strftime('%Y-%m-%d %H:%M:%S.%f')
        if self._dialect == 'hogql':
            return f'toDateTime({self.visit(datetime_string)})'
        return f'toDateTime64({self.visit(datetime_string)}, 6, {self.visit(self._timezone)})'

    def visit_date(self, value: date):
        if False:
            print('Hello World!')
        return f"toDate({self.visit(value.strftime('%Y-%m-%d'))})"

    def visit_list(self, value: List):
        if False:
            i = 10
            return i + 15
        return f"[{', '.join((str(self.visit(x)) for x in value))}]"

    def visit_tuple(self, value: Tuple):
        if False:
            for i in range(10):
                print('nop')
        return f"({', '.join((str(self.visit(x)) for x in value))})"