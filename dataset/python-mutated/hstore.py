import re
from .array import ARRAY
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import GETITEM
from .operators import HAS_ALL
from .operators import HAS_ANY
from .operators import HAS_KEY
from ... import types as sqltypes
from ...sql import functions as sqlfunc
__all__ = ('HSTORE', 'hstore')

class HSTORE(sqltypes.Indexable, sqltypes.Concatenable, sqltypes.TypeEngine):
    """Represent the PostgreSQL HSTORE type.

    The :class:`.HSTORE` type stores dictionaries containing strings, e.g.::

        data_table = Table('data_table', metadata,
            Column('id', Integer, primary_key=True),
            Column('data', HSTORE)
        )

        with engine.connect() as conn:
            conn.execute(
                data_table.insert(),
                data = {"key1": "value1", "key2": "value2"}
            )

    :class:`.HSTORE` provides for a wide range of operations, including:

    * Index operations::

        data_table.c.data['some key'] == 'some value'

    * Containment operations::

        data_table.c.data.has_key('some key')

        data_table.c.data.has_all(['one', 'two', 'three'])

    * Concatenation::

        data_table.c.data + {"k1": "v1"}

    For a full list of special methods see
    :class:`.HSTORE.comparator_factory`.

    .. container:: topic

        **Detecting Changes in HSTORE columns when using the ORM**

        For usage with the SQLAlchemy ORM, it may be desirable to combine the
        usage of :class:`.HSTORE` with :class:`.MutableDict` dictionary now
        part of the :mod:`sqlalchemy.ext.mutable` extension. This extension
        will allow "in-place" changes to the dictionary, e.g. addition of new
        keys or replacement/removal of existing keys to/from the current
        dictionary, to produce events which will be detected by the unit of
        work::

            from sqlalchemy.ext.mutable import MutableDict

            class MyClass(Base):
                __tablename__ = 'data_table'

                id = Column(Integer, primary_key=True)
                data = Column(MutableDict.as_mutable(HSTORE))

            my_object = session.query(MyClass).one()

            # in-place mutation, requires Mutable extension
            # in order for the ORM to detect
            my_object.data['some_key'] = 'some value'

            session.commit()

        When the :mod:`sqlalchemy.ext.mutable` extension is not used, the ORM
        will not be alerted to any changes to the contents of an existing
        dictionary, unless that dictionary value is re-assigned to the
        HSTORE-attribute itself, thus generating a change event.

    .. seealso::

        :class:`.hstore` - render the PostgreSQL ``hstore()`` function.


    """
    __visit_name__ = 'HSTORE'
    hashable = False
    text_type = sqltypes.Text()

    def __init__(self, text_type=None):
        if False:
            print('Hello World!')
        'Construct a new :class:`.HSTORE`.\n\n        :param text_type: the type that should be used for indexed values.\n         Defaults to :class:`_types.Text`.\n\n        '
        if text_type is not None:
            self.text_type = text_type

    class Comparator(sqltypes.Indexable.Comparator, sqltypes.Concatenable.Comparator):
        """Define comparison operations for :class:`.HSTORE`."""

        def has_key(self, other):
            if False:
                while True:
                    i = 10
            'Boolean expression.  Test for presence of a key.  Note that the\n            key may be a SQLA expression.\n            '
            return self.operate(HAS_KEY, other, result_type=sqltypes.Boolean)

        def has_all(self, other):
            if False:
                return 10
            'Boolean expression.  Test for presence of all keys in jsonb'
            return self.operate(HAS_ALL, other, result_type=sqltypes.Boolean)

        def has_any(self, other):
            if False:
                return 10
            'Boolean expression.  Test for presence of any key in jsonb'
            return self.operate(HAS_ANY, other, result_type=sqltypes.Boolean)

        def contains(self, other, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            'Boolean expression.  Test if keys (or array) are a superset\n            of/contained the keys of the argument jsonb expression.\n\n            kwargs may be ignored by this operator but are required for API\n            conformance.\n            '
            return self.operate(CONTAINS, other, result_type=sqltypes.Boolean)

        def contained_by(self, other):
            if False:
                while True:
                    i = 10
            'Boolean expression.  Test if keys are a proper subset of the\n            keys of the argument jsonb expression.\n            '
            return self.operate(CONTAINED_BY, other, result_type=sqltypes.Boolean)

        def _setup_getitem(self, index):
            if False:
                return 10
            return (GETITEM, index, self.type.text_type)

        def defined(self, key):
            if False:
                i = 10
                return i + 15
            'Boolean expression.  Test for presence of a non-NULL value for\n            the key.  Note that the key may be a SQLA expression.\n            '
            return _HStoreDefinedFunction(self.expr, key)

        def delete(self, key):
            if False:
                while True:
                    i = 10
            'HStore expression.  Returns the contents of this hstore with the\n            given key deleted.  Note that the key may be a SQLA expression.\n            '
            if isinstance(key, dict):
                key = _serialize_hstore(key)
            return _HStoreDeleteFunction(self.expr, key)

        def slice(self, array):
            if False:
                i = 10
                return i + 15
            'HStore expression.  Returns a subset of an hstore defined by\n            array of keys.\n            '
            return _HStoreSliceFunction(self.expr, array)

        def keys(self):
            if False:
                for i in range(10):
                    print('nop')
            'Text array expression.  Returns array of keys.'
            return _HStoreKeysFunction(self.expr)

        def vals(self):
            if False:
                return 10
            'Text array expression.  Returns array of values.'
            return _HStoreValsFunction(self.expr)

        def array(self):
            if False:
                print('Hello World!')
            'Text array expression.  Returns array of alternating keys and\n            values.\n            '
            return _HStoreArrayFunction(self.expr)

        def matrix(self):
            if False:
                return 10
            'Text array expression.  Returns array of [key, value] pairs.'
            return _HStoreMatrixFunction(self.expr)
    comparator_factory = Comparator

    def bind_processor(self, dialect):
        if False:
            return 10

        def process(value):
            if False:
                return 10
            if isinstance(value, dict):
                return _serialize_hstore(value)
            else:
                return value
        return process

    def result_processor(self, dialect, coltype):
        if False:
            print('Hello World!')

        def process(value):
            if False:
                i = 10
                return i + 15
            if value is not None:
                return _parse_hstore(value)
            else:
                return value
        return process

class hstore(sqlfunc.GenericFunction):
    """Construct an hstore value within a SQL expression using the
    PostgreSQL ``hstore()`` function.

    The :class:`.hstore` function accepts one or two arguments as described
    in the PostgreSQL documentation.

    E.g.::

        from sqlalchemy.dialects.postgresql import array, hstore

        select(hstore('key1', 'value1'))

        select(
            hstore(
                array(['key1', 'key2', 'key3']),
                array(['value1', 'value2', 'value3'])
            )
        )

    .. seealso::

        :class:`.HSTORE` - the PostgreSQL ``HSTORE`` datatype.

    """
    type = HSTORE
    name = 'hstore'
    inherit_cache = True

class _HStoreDefinedFunction(sqlfunc.GenericFunction):
    type = sqltypes.Boolean
    name = 'defined'
    inherit_cache = True

class _HStoreDeleteFunction(sqlfunc.GenericFunction):
    type = HSTORE
    name = 'delete'
    inherit_cache = True

class _HStoreSliceFunction(sqlfunc.GenericFunction):
    type = HSTORE
    name = 'slice'
    inherit_cache = True

class _HStoreKeysFunction(sqlfunc.GenericFunction):
    type = ARRAY(sqltypes.Text)
    name = 'akeys'
    inherit_cache = True

class _HStoreValsFunction(sqlfunc.GenericFunction):
    type = ARRAY(sqltypes.Text)
    name = 'avals'
    inherit_cache = True

class _HStoreArrayFunction(sqlfunc.GenericFunction):
    type = ARRAY(sqltypes.Text)
    name = 'hstore_to_array'
    inherit_cache = True

class _HStoreMatrixFunction(sqlfunc.GenericFunction):
    type = ARRAY(sqltypes.Text)
    name = 'hstore_to_matrix'
    inherit_cache = True
HSTORE_PAIR_RE = re.compile('\n(\n  "(?P<key> (\\\\ . | [^"])* )"       # Quoted key\n)\n[ ]* => [ ]*    # Pair operator, optional adjoining whitespace\n(\n    (?P<value_null> NULL )          # NULL value\n  | "(?P<value> (\\\\ . | [^"])* )"   # Quoted value\n)\n', re.VERBOSE)
HSTORE_DELIMITER_RE = re.compile('\n[ ]* , [ ]*\n', re.VERBOSE)

def _parse_error(hstore_str, pos):
    if False:
        return 10
    'format an unmarshalling error.'
    ctx = 20
    hslen = len(hstore_str)
    parsed_tail = hstore_str[max(pos - ctx - 1, 0):min(pos, hslen)]
    residual = hstore_str[min(pos, hslen):min(pos + ctx + 1, hslen)]
    if len(parsed_tail) > ctx:
        parsed_tail = '[...]' + parsed_tail[1:]
    if len(residual) > ctx:
        residual = residual[:-1] + '[...]'
    return 'After %r, could not parse residual at position %d: %r' % (parsed_tail, pos, residual)

def _parse_hstore(hstore_str):
    if False:
        i = 10
        return i + 15
    "Parse an hstore from its literal string representation.\n\n    Attempts to approximate PG's hstore input parsing rules as closely as\n    possible. Although currently this is not strictly necessary, since the\n    current implementation of hstore's output syntax is stricter than what it\n    accepts as input, the documentation makes no guarantees that will always\n    be the case.\n\n\n\n    "
    result = {}
    pos = 0
    pair_match = HSTORE_PAIR_RE.match(hstore_str)
    while pair_match is not None:
        key = pair_match.group('key').replace('\\"', '"').replace('\\\\', '\\')
        if pair_match.group('value_null'):
            value = None
        else:
            value = pair_match.group('value').replace('\\"', '"').replace('\\\\', '\\')
        result[key] = value
        pos += pair_match.end()
        delim_match = HSTORE_DELIMITER_RE.match(hstore_str[pos:])
        if delim_match is not None:
            pos += delim_match.end()
        pair_match = HSTORE_PAIR_RE.match(hstore_str[pos:])
    if pos != len(hstore_str):
        raise ValueError(_parse_error(hstore_str, pos))
    return result

def _serialize_hstore(val):
    if False:
        i = 10
        return i + 15
    'Serialize a dictionary into an hstore literal.  Keys and values must\n    both be strings (except None for values).\n\n    '

    def esc(s, position):
        if False:
            for i in range(10):
                print('nop')
        if position == 'value' and s is None:
            return 'NULL'
        elif isinstance(s, str):
            return '"%s"' % s.replace('\\', '\\\\').replace('"', '\\"')
        else:
            raise ValueError('%r in %s position is not a string.' % (s, position))
    return ', '.join(('%s=>%s' % (esc(k, 'key'), esc(v, 'value')) for (k, v) in val.items()))