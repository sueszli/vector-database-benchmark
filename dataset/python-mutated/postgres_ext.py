"""
Collection of postgres-specific extensions, currently including:

* Support for hstore, a key/value type storage
"""
import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
from peewee import __exception_wrapper__
try:
    from psycopg2cffi import compat
    compat.register()
except ImportError:
    pass
try:
    from psycopg2.extras import register_hstore
except ImportError:

    def register_hstore(c, globally):
        if False:
            print('Hello World!')
        pass
try:
    from psycopg2.extras import Json
except:
    Json = None
logger = logging.getLogger('peewee')
HCONTAINS_DICT = '@>'
HCONTAINS_KEYS = '?&'
HCONTAINS_KEY = '?'
HCONTAINS_ANY_KEY = '?|'
HKEY = '->'
HUPDATE = '||'
ACONTAINS = '@>'
ACONTAINED_BY = '<@'
ACONTAINS_ANY = '&&'
TS_MATCH = '@@'
JSONB_CONTAINS = '@>'
JSONB_CONTAINED_BY = '<@'
JSONB_CONTAINS_KEY = '?'
JSONB_CONTAINS_ANY_KEY = '?|'
JSONB_CONTAINS_ALL_KEYS = '?&'
JSONB_EXISTS = '?'
JSONB_REMOVE = '-'

class _LookupNode(ColumnBase):

    def __init__(self, node, parts):
        if False:
            print('Hello World!')
        self.node = node
        self.parts = parts
        super(_LookupNode, self).__init__()

    def clone(self):
        if False:
            return 10
        return type(self)(self.node, list(self.parts))

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self.__class__.__name__, id(self)))

class _JsonLookupBase(_LookupNode):

    def __init__(self, node, parts, as_json=False):
        if False:
            i = 10
            return i + 15
        super(_JsonLookupBase, self).__init__(node, parts)
        self._as_json = as_json

    def clone(self):
        if False:
            i = 10
            return i + 15
        return type(self)(self.node, list(self.parts), self._as_json)

    @Node.copy
    def as_json(self, as_json=True):
        if False:
            while True:
                i = 10
        self._as_json = as_json

    def concat(self, rhs):
        if False:
            i = 10
            return i + 15
        if not isinstance(rhs, Node):
            rhs = Json(rhs)
        return Expression(self.as_json(True), OP.CONCAT, rhs)

    def contains(self, other):
        if False:
            return 10
        clone = self.as_json(True)
        if isinstance(other, (list, dict)):
            return Expression(clone, JSONB_CONTAINS, Json(other))
        return Expression(clone, JSONB_EXISTS, other)

    def contains_any(self, *keys):
        if False:
            while True:
                i = 10
        return Expression(self.as_json(True), JSONB_CONTAINS_ANY_KEY, Value(list(keys), unpack=False))

    def contains_all(self, *keys):
        if False:
            for i in range(10):
                print('nop')
        return Expression(self.as_json(True), JSONB_CONTAINS_ALL_KEYS, Value(list(keys), unpack=False))

    def has_key(self, key):
        if False:
            print('Hello World!')
        return Expression(self.as_json(True), JSONB_CONTAINS_KEY, key)

class JsonLookup(_JsonLookupBase):

    def __getitem__(self, value):
        if False:
            while True:
                i = 10
        return JsonLookup(self.node, self.parts + [value], self._as_json)

    def __sql__(self, ctx):
        if False:
            i = 10
            return i + 15
        ctx.sql(self.node)
        for part in self.parts[:-1]:
            ctx.literal('->').sql(part)
        if self.parts:
            ctx.literal('->' if self._as_json else '->>').sql(self.parts[-1])
        return ctx

class JsonPath(_JsonLookupBase):

    def __sql__(self, ctx):
        if False:
            while True:
                i = 10
        return ctx.sql(self.node).literal('#>' if self._as_json else '#>>').sql(Value('{%s}' % ','.join(map(str, self.parts))))

class ObjectSlice(_LookupNode):

    @classmethod
    def create(cls, node, value):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, slice):
            parts = [value.start or 0, value.stop or 0]
        elif isinstance(value, int):
            parts = [value]
        elif isinstance(value, Node):
            parts = value
        else:
            parts = [int(i) for i in value.split(':')]
        return cls(node, parts)

    def __sql__(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        ctx.sql(self.node)
        if isinstance(self.parts, Node):
            ctx.literal('[').sql(self.parts).literal(']')
        else:
            ctx.literal('[%s]' % ':'.join((str(p + 1) for p in self.parts)))
        return ctx

    def __getitem__(self, value):
        if False:
            i = 10
            return i + 15
        return ObjectSlice.create(self, value)

class IndexedFieldMixin(object):
    default_index_type = 'GIN'

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs.setdefault('index', True)
        super(IndexedFieldMixin, self).__init__(*args, **kwargs)

class ArrayField(IndexedFieldMixin, Field):
    passthrough = True

    def __init__(self, field_class=IntegerField, field_kwargs=None, dimensions=1, convert_values=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.__field = field_class(**field_kwargs or {})
        self.dimensions = dimensions
        self.convert_values = convert_values
        self.field_type = self.__field.field_type
        super(ArrayField, self).__init__(*args, **kwargs)

    def bind(self, model, name, set_attribute=True):
        if False:
            print('Hello World!')
        ret = super(ArrayField, self).bind(model, name, set_attribute)
        self.__field.bind(model, '__array_%s' % name, False)
        return ret

    def ddl_datatype(self, ctx):
        if False:
            i = 10
            return i + 15
        data_type = self.__field.ddl_datatype(ctx)
        return NodeList((data_type, SQL('[]' * self.dimensions)), glue='')

    def db_value(self, value):
        if False:
            while True:
                i = 10
        if value is None or isinstance(value, Node):
            return value
        elif self.convert_values:
            return self._process(self.__field.db_value, value, self.dimensions)
        else:
            return value if isinstance(value, list) else list(value)

    def python_value(self, value):
        if False:
            return 10
        if self.convert_values and value is not None:
            conv = self.__field.python_value
            if isinstance(value, list):
                return self._process(conv, value, self.dimensions)
            else:
                return conv(value)
        else:
            return value

    def _process(self, conv, value, dimensions):
        if False:
            i = 10
            return i + 15
        dimensions -= 1
        if dimensions == 0:
            return [conv(v) for v in value]
        else:
            return [self._process(conv, v, dimensions) for v in value]

    def __getitem__(self, value):
        if False:
            return 10
        return ObjectSlice.create(self, value)

    def _e(op):
        if False:
            return 10

        def inner(self, rhs):
            if False:
                return 10
            return Expression(self, op, ArrayValue(self, rhs))
        return inner
    __eq__ = _e(OP.EQ)
    __ne__ = _e(OP.NE)
    __gt__ = _e(OP.GT)
    __ge__ = _e(OP.GTE)
    __lt__ = _e(OP.LT)
    __le__ = _e(OP.LTE)
    __hash__ = Field.__hash__

    def contains(self, *items):
        if False:
            return 10
        return Expression(self, ACONTAINS, ArrayValue(self, items))

    def contains_any(self, *items):
        if False:
            while True:
                i = 10
        return Expression(self, ACONTAINS_ANY, ArrayValue(self, items))

    def contained_by(self, *items):
        if False:
            for i in range(10):
                print('nop')
        return Expression(self, ACONTAINED_BY, ArrayValue(self, items))

class ArrayValue(Node):

    def __init__(self, field, value):
        if False:
            for i in range(10):
                print('nop')
        self.field = field
        self.value = value

    def __sql__(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        return ctx.sql(Value(self.value, unpack=False)).literal('::').sql(self.field.ddl_datatype(ctx))

class DateTimeTZField(DateTimeField):
    field_type = 'TIMESTAMPTZ'

class HStoreField(IndexedFieldMixin, Field):
    field_type = 'HSTORE'
    __hash__ = Field.__hash__

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return Expression(self, HKEY, Value(key))

    def keys(self):
        if False:
            while True:
                i = 10
        return fn.akeys(self)

    def values(self):
        if False:
            for i in range(10):
                print('nop')
        return fn.avals(self)

    def items(self):
        if False:
            return 10
        return fn.hstore_to_matrix(self)

    def slice(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return fn.slice(self, Value(list(args), unpack=False))

    def exists(self, key):
        if False:
            while True:
                i = 10
        return fn.exist(self, key)

    def defined(self, key):
        if False:
            return 10
        return fn.defined(self, key)

    def update(self, **data):
        if False:
            print('Hello World!')
        return Expression(self, HUPDATE, data)

    def delete(self, *keys):
        if False:
            while True:
                i = 10
        return fn.delete(self, Value(list(keys), unpack=False))

    def contains(self, value):
        if False:
            i = 10
            return i + 15
        if isinstance(value, dict):
            rhs = Value(value, unpack=False)
            return Expression(self, HCONTAINS_DICT, rhs)
        elif isinstance(value, (list, tuple)):
            rhs = Value(value, unpack=False)
            return Expression(self, HCONTAINS_KEYS, rhs)
        return Expression(self, HCONTAINS_KEY, value)

    def contains_any(self, *keys):
        if False:
            return 10
        return Expression(self, HCONTAINS_ANY_KEY, Value(list(keys), unpack=False))

class JSONField(Field):
    field_type = 'JSON'
    _json_datatype = 'json'

    def __init__(self, dumps=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if Json is None:
            raise Exception('Your version of psycopg2 does not support JSON.')
        self.dumps = dumps or json.dumps
        super(JSONField, self).__init__(*args, **kwargs)

    def db_value(self, value):
        if False:
            print('Hello World!')
        if value is None:
            return value
        if not isinstance(value, Json):
            return Cast(self.dumps(value), self._json_datatype)
        return value

    def __getitem__(self, value):
        if False:
            return 10
        return JsonLookup(self, [value])

    def path(self, *keys):
        if False:
            print('Hello World!')
        return JsonPath(self, keys)

    def concat(self, value):
        if False:
            while True:
                i = 10
        if not isinstance(value, Node):
            value = Json(value)
        return super(JSONField, self).concat(value)

def cast_jsonb(node):
    if False:
        i = 10
        return i + 15
    return NodeList((node, SQL('::jsonb')), glue='')

class BinaryJSONField(IndexedFieldMixin, JSONField):
    field_type = 'JSONB'
    _json_datatype = 'jsonb'
    __hash__ = Field.__hash__

    def contains(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, (list, dict)):
            return Expression(self, JSONB_CONTAINS, Json(other))
        elif isinstance(other, JSONField):
            return Expression(self, JSONB_CONTAINS, other)
        return Expression(cast_jsonb(self), JSONB_EXISTS, other)

    def contained_by(self, other):
        if False:
            for i in range(10):
                print('nop')
        return Expression(cast_jsonb(self), JSONB_CONTAINED_BY, Json(other))

    def contains_any(self, *items):
        if False:
            i = 10
            return i + 15
        return Expression(cast_jsonb(self), JSONB_CONTAINS_ANY_KEY, Value(list(items), unpack=False))

    def contains_all(self, *items):
        if False:
            for i in range(10):
                print('nop')
        return Expression(cast_jsonb(self), JSONB_CONTAINS_ALL_KEYS, Value(list(items), unpack=False))

    def has_key(self, key):
        if False:
            print('Hello World!')
        return Expression(cast_jsonb(self), JSONB_CONTAINS_KEY, key)

    def remove(self, *items):
        if False:
            return 10
        return Expression(cast_jsonb(self), JSONB_REMOVE, Value(list(items), unpack=False))

class TSVectorField(IndexedFieldMixin, TextField):
    field_type = 'TSVECTOR'
    __hash__ = Field.__hash__

    def match(self, query, language=None, plain=False):
        if False:
            print('Hello World!')
        params = (language, query) if language is not None else (query,)
        func = fn.plainto_tsquery if plain else fn.to_tsquery
        return Expression(self, TS_MATCH, func(*params))

def Match(field, query, language=None):
    if False:
        print('Hello World!')
    params = (language, query) if language is not None else (query,)
    field_params = (language, field) if language is not None else (field,)
    return Expression(fn.to_tsvector(*field_params), TS_MATCH, fn.to_tsquery(*params))

class IntervalField(Field):
    field_type = 'INTERVAL'

class FetchManyCursor(object):
    __slots__ = ('cursor', 'array_size', 'exhausted', 'iterable')

    def __init__(self, cursor, array_size=None):
        if False:
            return 10
        self.cursor = cursor
        self.array_size = array_size or cursor.itersize
        self.exhausted = False
        self.iterable = self.row_gen()

    @property
    def description(self):
        if False:
            print('Hello World!')
        return self.cursor.description

    def close(self):
        if False:
            return 10
        self.cursor.close()

    def row_gen(self):
        if False:
            while True:
                i = 10
        while True:
            rows = self.cursor.fetchmany(self.array_size)
            if not rows:
                return
            for row in rows:
                yield row

    def fetchone(self):
        if False:
            print('Hello World!')
        if self.exhausted:
            return
        try:
            return next(self.iterable)
        except StopIteration:
            self.exhausted = True

class ServerSideQuery(Node):

    def __init__(self, query, array_size=None):
        if False:
            i = 10
            return i + 15
        self.query = query
        self.array_size = array_size
        self._cursor_wrapper = None

    def __sql__(self, ctx):
        if False:
            print('Hello World!')
        return self.query.__sql__(ctx)

    def __iter__(self):
        if False:
            print('Hello World!')
        if self._cursor_wrapper is None:
            self._execute(self.query._database)
        return iter(self._cursor_wrapper.iterator())

    def _execute(self, database):
        if False:
            return 10
        if self._cursor_wrapper is None:
            cursor = database.execute(self.query, named_cursor=True, array_size=self.array_size)
            self._cursor_wrapper = self.query._get_cursor_wrapper(cursor)
        return self._cursor_wrapper

def ServerSide(query, database=None, array_size=None):
    if False:
        print('Hello World!')
    if database is None:
        database = query._database
    with database.transaction():
        server_side_query = ServerSideQuery(query, array_size=array_size)
        for row in server_side_query:
            yield row

class _empty_object(object):
    __slots__ = ()

    def __nonzero__(self):
        if False:
            return 10
        return False
    __bool__ = __nonzero__

class PostgresqlExtDatabase(PostgresqlDatabase):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._register_hstore = kwargs.pop('register_hstore', False)
        self._server_side_cursors = kwargs.pop('server_side_cursors', False)
        super(PostgresqlExtDatabase, self).__init__(*args, **kwargs)

    def _connect(self):
        if False:
            for i in range(10):
                print('nop')
        conn = super(PostgresqlExtDatabase, self)._connect()
        if self._register_hstore:
            register_hstore(conn, globally=True)
        return conn

    def cursor(self, commit=None, named_cursor=None):
        if False:
            while True:
                i = 10
        if commit is not None:
            __deprecated__('"commit" has been deprecated and is a no-op.')
        if self.is_closed():
            if self.autoconnect:
                self.connect()
            else:
                raise InterfaceError('Error, database connection not opened.')
        if named_cursor:
            curs = self._state.conn.cursor(name=str(uuid.uuid1()))
            return curs
        return self._state.conn.cursor()

    def execute(self, query, commit=None, named_cursor=False, array_size=None, **context_options):
        if False:
            i = 10
            return i + 15
        if commit is not None:
            __deprecated__('"commit" has been deprecated and is a no-op.')
        ctx = self.get_sql_context(**context_options)
        (sql, params) = ctx.sql(query).query()
        named_cursor = named_cursor or (self._server_side_cursors and sql[:6].lower() == 'select')
        cursor = self.execute_sql(sql, params)
        if named_cursor:
            cursor = FetchManyCursor(cursor, array_size)
        return cursor