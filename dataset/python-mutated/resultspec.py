import sqlalchemy as sa
from twisted.python import log
from buildbot.data import base

class FieldBase:
    """
    This class implements a basic behavior
    to wrap value into a `Field` instance

    """
    __slots__ = ['field', 'op', 'values']
    singular_operators = {'eq': lambda d, v: d == v[0], 'ne': lambda d, v: d != v[0], 'lt': lambda d, v: d < v[0], 'le': lambda d, v: d <= v[0], 'gt': lambda d, v: d > v[0], 'ge': lambda d, v: d >= v[0], 'contains': lambda d, v: v[0] in d, 'in': lambda d, v: d in v, 'notin': lambda d, v: d not in v}
    singular_operators_sql = {'eq': lambda d, v: d == v[0], 'ne': lambda d, v: d != v[0], 'lt': lambda d, v: d < v[0], 'le': lambda d, v: d <= v[0], 'gt': lambda d, v: d > v[0], 'ge': lambda d, v: d >= v[0], 'contains': lambda d, v: d.contains(v[0]), 'in': lambda d, v: d.in_(v), 'notin': lambda d, v: d.notin_(v)}
    plural_operators = {'eq': lambda d, v: d in v, 'ne': lambda d, v: d not in v, 'contains': lambda d, v: len(set(v).intersection(set(d))) > 0, 'in': lambda d, v: d in v, 'notin': lambda d, v: d not in v}
    plural_operators_sql = {'eq': lambda d, v: d.in_(v), 'ne': lambda d, v: d.notin_(v), 'contains': lambda d, vs: sa.or_(*[d.contains(v) for v in vs]), 'in': lambda d, v: d.in_(v), 'notin': lambda d, v: d.notin_(v)}

    def __init__(self, field, op, values):
        if False:
            print('Hello World!')
        self.field = field
        self.op = op
        self.values = values

    def getOperator(self, sqlMode=False):
        if False:
            i = 10
            return i + 15
        v = self.values
        if len(v) == 1:
            if sqlMode:
                ops = self.singular_operators_sql
            else:
                ops = self.singular_operators
        else:
            if sqlMode:
                ops = self.plural_operators_sql
            else:
                ops = self.plural_operators
            v = set(v)
        return ops[self.op]

    def apply(self, data):
        if False:
            print('Hello World!')
        fld = self.field
        v = self.values
        f = self.getOperator()
        return (d for d in data if f(d[fld], v))

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f"resultspec.{self.__class__.__name__}('{self.field}','{self.op}',{self.values})"

    def __eq__(self, b):
        if False:
            print('Hello World!')
        for i in self.__slots__:
            if getattr(self, i) != getattr(b, i):
                return False
        return True

    def __ne__(self, b):
        if False:
            for i in range(10):
                print('nop')
        return not self == b

class Property(FieldBase):
    """
    Wraps ``property`` type value(s)

    """

class Filter(FieldBase):
    """
    Wraps ``filter`` type value(s)

    """

class NoneComparator:
    """
    Object which wraps 'None' when doing comparisons in sorted().
    '> None' and '< None' are not supported
    in Python 3.
    """

    def __init__(self, value):
        if False:
            return 10
        self.value = value

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if self.value is None and other.value is None:
            return False
        elif self.value is None:
            return True
        elif other.value is None:
            return False
        return self.value < other.value

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.value == other.value

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return self.value != other.value

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if self.value is None and other.value is None:
            return False
        elif self.value is None:
            return False
        elif other.value is None:
            return True
        return self.value > other.value

class ReverseComparator:
    """
    Object which swaps '<' and '>' so
    instead of a < b, it does b < a,
    and instead of a > b, it does b > a.
    This can be used in reverse comparisons.
    """

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.value = value

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return other.value < self.value

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return other.value == self.value

    def __ne__(self, other):
        if False:
            return 10
        return other.value != self.value

    def __gt__(self, other):
        if False:
            print('Hello World!')
        return other.value > self.value

class ResultSpec:
    __slots__ = ['filters', 'fields', 'properties', 'order', 'limit', 'offset', 'fieldMapping']

    def __init__(self, filters=None, fields=None, properties=None, order=None, limit=None, offset=None):
        if False:
            return 10
        self.filters = filters or []
        self.properties = properties or []
        self.fields = fields
        self.order = order
        self.limit = limit
        self.offset = offset
        self.fieldMapping = {}

    def __repr__(self):
        if False:
            return 10
        return f"ResultSpec(**{{'filters': {self.filters}, 'fields': {self.fields}, 'properties': {self.properties}, 'order': {self.order}, 'limit': {self.limit}, 'offset': {self.offset}" + '})'

    def __eq__(self, b):
        if False:
            while True:
                i = 10
        for i in ['filters', 'fields', 'properties', 'order', 'limit', 'offset']:
            if getattr(self, i) != getattr(b, i):
                return False
        return True

    def __ne__(self, b):
        if False:
            print('Hello World!')
        return not self == b

    def popProperties(self):
        if False:
            return 10
        values = []
        for p in self.properties:
            if p.field == b'property' and p.op == 'eq':
                self.properties.remove(p)
                values = p.values
                break
        return values

    def popFilter(self, field, op):
        if False:
            while True:
                i = 10
        for f in self.filters:
            if f.field == field and f.op == op:
                self.filters.remove(f)
                return f.values
        return None

    def popOneFilter(self, field, op):
        if False:
            while True:
                i = 10
        v = self.popFilter(field, op)
        return v[0] if v is not None else None

    def popBooleanFilter(self, field):
        if False:
            while True:
                i = 10
        eqVals = self.popFilter(field, 'eq')
        if eqVals and len(eqVals) == 1:
            return eqVals[0]
        neVals = self.popFilter(field, 'ne')
        if neVals and len(neVals) == 1:
            return not neVals[0]
        return None

    def popStringFilter(self, field):
        if False:
            print('Hello World!')
        eqVals = self.popFilter(field, 'eq')
        if eqVals and len(eqVals) == 1:
            return eqVals[0]
        return None

    def popIntegerFilter(self, field):
        if False:
            while True:
                i = 10
        eqVals = self.popFilter(field, 'eq')
        if eqVals and len(eqVals) == 1:
            try:
                return int(eqVals[0])
            except ValueError as e:
                raise ValueError(f'Filter value for {field} should be integer, but got: {eqVals[0]}') from e
        return None

    def removePagination(self):
        if False:
            i = 10
            return i + 15
        self.limit = self.offset = None

    def removeOrder(self):
        if False:
            for i in range(10):
                print('nop')
        self.order = None

    def popField(self, field):
        if False:
            i = 10
            return i + 15
        try:
            i = self.fields.index(field)
        except ValueError:
            return False
        del self.fields[i]
        return True

    def findColumn(self, query, field):
        if False:
            for i in range(10):
                print('nop')
        mapped = self.fieldMapping[field]
        for col in query.inner_columns:
            if str(col) == mapped:
                return col
        raise KeyError(f'unable to find field {field} in query')

    def applyFilterToSQLQuery(self, query, f):
        if False:
            for i in range(10):
                print('nop')
        field = f.field
        col = self.findColumn(query, field)
        return query.where(f.getOperator(sqlMode=True)(col, f.values))

    def applyOrderToSQLQuery(self, query, o):
        if False:
            i = 10
            return i + 15
        reverse = False
        if o.startswith('-'):
            reverse = True
            o = o[1:]
        col = self.findColumn(query, o)
        if reverse:
            col = col.desc()
        return query.order_by(col)

    def applyToSQLQuery(self, query):
        if False:
            print('Hello World!')
        filters = self.filters
        order = self.order
        unmatched_filters = []
        unmatched_order = []
        for f in filters:
            try:
                query = self.applyFilterToSQLQuery(query, f)
            except KeyError:
                unmatched_filters.append(f)
        if order:
            for o in order:
                try:
                    query = self.applyOrderToSQLQuery(query, o)
                except KeyError:
                    unmatched_order.append(o)
        if unmatched_filters or unmatched_order:
            if self.offset is not None or self.limit is not None:
                log.msg('Warning: limited data api query is not backed by db because of following filters', unmatched_filters, unmatched_order)
            self.filters = unmatched_filters
            self.order = tuple(unmatched_order)
            return (query, None)
        count_query = sa.select([sa.func.count()]).select_from(query.alias('query'))
        self.order = None
        self.filters = []
        if self.offset is not None:
            query = query.offset(self.offset)
            self.offset = None
        if self.limit is not None:
            query = query.limit(self.limit)
            self.limit = None
        return (query, count_query)

    def thd_execute(self, conn, q, dictFromRow):
        if False:
            i = 10
            return i + 15
        (offset, limit) = (self.offset, self.limit)
        (q, qc) = self.applyToSQLQuery(q)
        res = conn.execute(q)
        rv = [dictFromRow(row) for row in res.fetchall()]
        if qc is not None and (offset or limit):
            total = conn.execute(qc).scalar()
            rv = base.ListResult(rv)
            (rv.offset, rv.total, rv.limit) = (offset, total, limit)
        return rv

    def apply(self, data):
        if False:
            for i in range(10):
                print('nop')
        if data is None:
            return data
        if self.fields:
            fields = set(self.fields)

            def includeFields(d):
                if False:
                    for i in range(10):
                        print('nop')
                return dict(((k, v) for (k, v) in d.items() if k in fields))
            applyFields = includeFields
        else:
            fields = None
        if isinstance(data, dict):
            if fields:
                data = applyFields(data)
            return data
        else:
            filters = self.filters
            order = self.order
            if isinstance(data, base.ListResult):
                assert not fields and (not order) and (not filters), 'endpoint must apply fields, order, and filters if it performs pagination'
                (offset, total) = (data.offset, data.total)
                limit = data.limit
            else:
                (offset, total) = (None, None)
                limit = None
            if fields:
                data = (applyFields(d) for d in data)
            for f in self.filters:
                data = f.apply(data)
            data = list(data)
            if total is None:
                total = len(data)
            if self.order:

                def keyFunc(elem, order=self.order):
                    if False:
                        print('Hello World!')
                    "\n                    Do a multi-level sort by passing in the keys\n                    to sort by.\n\n                    @param elem: each item in the list to sort.  It must be\n                              a C{dict}\n                    @param order: a list of keys to sort by, such as:\n                                ('lastName', 'firstName', 'age')\n                    @return: a key used by sorted(). This will be a\n                             list such as:\n                             [a['lastName', a['firstName'], a['age']]\n                    @rtype: a C{list}\n                    "
                    compareKey = []
                    for k in order:
                        doReverse = False
                        if k[0] == '-':
                            k = k[1:]
                            doReverse = True
                        val = NoneComparator(elem[k])
                        if doReverse:
                            val = ReverseComparator(val)
                        compareKey.append(val)
                    return compareKey
                data.sort(key=keyFunc)
            if self.offset is not None or self.limit is not None:
                if offset is not None or limit is not None:
                    raise AssertionError('endpoint must clear offset/limit')
                end = (self.offset or 0) + self.limit if self.limit is not None else None
                data = data[self.offset:end]
                offset = self.offset
                limit = self.limit
            rv = base.ListResult(data)
            (rv.offset, rv.total) = (offset, total)
            rv.limit = limit
            return rv

class OptimisedResultSpec(ResultSpec):

    def apply(self, data):
        if False:
            return 10
        return data