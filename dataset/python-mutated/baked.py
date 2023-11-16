"""Baked query extension.

Provides a creational pattern for the :class:`.query.Query` object which
allows the fully constructed object, Core select statement, and string
compiled result to be fully cached.


"""
import collections.abc as collections_abc
import logging
from .. import exc as sa_exc
from .. import util
from ..orm import exc as orm_exc
from ..orm.query import Query
from ..orm.session import Session
from ..sql import func
from ..sql import literal_column
from ..sql import util as sql_util
log = logging.getLogger(__name__)

class Bakery:
    """Callable which returns a :class:`.BakedQuery`.

    This object is returned by the class method
    :meth:`.BakedQuery.bakery`.  It exists as an object
    so that the "cache" can be easily inspected.

    .. versionadded:: 1.2


    """
    __slots__ = ('cls', 'cache')

    def __init__(self, cls_, cache):
        if False:
            print('Hello World!')
        self.cls = cls_
        self.cache = cache

    def __call__(self, initial_fn, *args):
        if False:
            i = 10
            return i + 15
        return self.cls(self.cache, initial_fn, args)

class BakedQuery:
    """A builder object for :class:`.query.Query` objects."""
    __slots__ = ('steps', '_bakery', '_cache_key', '_spoiled')

    def __init__(self, bakery, initial_fn, args=()):
        if False:
            return 10
        self._cache_key = ()
        self._update_cache_key(initial_fn, args)
        self.steps = [initial_fn]
        self._spoiled = False
        self._bakery = bakery

    @classmethod
    def bakery(cls, size=200, _size_alert=None):
        if False:
            for i in range(10):
                print('nop')
        'Construct a new bakery.\n\n        :return: an instance of :class:`.Bakery`\n\n        '
        return Bakery(cls, util.LRUCache(size, size_alert=_size_alert))

    def _clone(self):
        if False:
            i = 10
            return i + 15
        b1 = BakedQuery.__new__(BakedQuery)
        b1._cache_key = self._cache_key
        b1.steps = list(self.steps)
        b1._bakery = self._bakery
        b1._spoiled = self._spoiled
        return b1

    def _update_cache_key(self, fn, args=()):
        if False:
            i = 10
            return i + 15
        self._cache_key += (fn.__code__,) + args

    def __iadd__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, tuple):
            self.add_criteria(*other)
        else:
            self.add_criteria(other)
        return self

    def __add__(self, other):
        if False:
            return 10
        if isinstance(other, tuple):
            return self.with_criteria(*other)
        else:
            return self.with_criteria(other)

    def add_criteria(self, fn, *args):
        if False:
            print('Hello World!')
        'Add a criteria function to this :class:`.BakedQuery`.\n\n        This is equivalent to using the ``+=`` operator to\n        modify a :class:`.BakedQuery` in-place.\n\n        '
        self._update_cache_key(fn, args)
        self.steps.append(fn)
        return self

    def with_criteria(self, fn, *args):
        if False:
            return 10
        'Add a criteria function to a :class:`.BakedQuery` cloned from this\n        one.\n\n        This is equivalent to using the ``+`` operator to\n        produce a new :class:`.BakedQuery` with modifications.\n\n        '
        return self._clone().add_criteria(fn, *args)

    def for_session(self, session):
        if False:
            for i in range(10):
                print('nop')
        'Return a :class:`_baked.Result` object for this\n        :class:`.BakedQuery`.\n\n        This is equivalent to calling the :class:`.BakedQuery` as a\n        Python callable, e.g. ``result = my_baked_query(session)``.\n\n        '
        return Result(self, session)

    def __call__(self, session):
        if False:
            print('Hello World!')
        return self.for_session(session)

    def spoil(self, full=False):
        if False:
            print('Hello World!')
        'Cancel any query caching that will occur on this BakedQuery object.\n\n        The BakedQuery can continue to be used normally, however additional\n        creational functions will not be cached; they will be called\n        on every invocation.\n\n        This is to support the case where a particular step in constructing\n        a baked query disqualifies the query from being cacheable, such\n        as a variant that relies upon some uncacheable value.\n\n        :param full: if False, only functions added to this\n         :class:`.BakedQuery` object subsequent to the spoil step will be\n         non-cached; the state of the :class:`.BakedQuery` up until\n         this point will be pulled from the cache.   If True, then the\n         entire :class:`_query.Query` object is built from scratch each\n         time, with all creational functions being called on each\n         invocation.\n\n        '
        if not full and (not self._spoiled):
            _spoil_point = self._clone()
            _spoil_point._cache_key += ('_query_only',)
            self.steps = [_spoil_point._retrieve_baked_query]
        self._spoiled = True
        return self

    def _effective_key(self, session):
        if False:
            return 10
        "Return the key that actually goes into the cache dictionary for\n        this :class:`.BakedQuery`, taking into account the given\n        :class:`.Session`.\n\n        This basically means we also will include the session's query_class,\n        as the actual :class:`_query.Query` object is part of what's cached\n        and needs to match the type of :class:`_query.Query` that a later\n        session will want to use.\n\n        "
        return self._cache_key + (session._query_cls,)

    def _with_lazyload_options(self, options, effective_path, cache_path=None):
        if False:
            for i in range(10):
                print('nop')
        'Cloning version of _add_lazyload_options.'
        q = self._clone()
        q._add_lazyload_options(options, effective_path, cache_path=cache_path)
        return q

    def _add_lazyload_options(self, options, effective_path, cache_path=None):
        if False:
            while True:
                i = 10
        'Used by per-state lazy loaders to add options to the\n        "lazy load" query from a parent query.\n\n        Creates a cache key based on given load path and query options;\n        if a repeatable cache key cannot be generated, the query is\n        "spoiled" so that it won\'t use caching.\n\n        '
        key = ()
        if not cache_path:
            cache_path = effective_path
        for opt in options:
            if opt._is_legacy_option or opt._is_compile_state:
                ck = opt._generate_cache_key()
                if ck is None:
                    self.spoil(full=True)
                else:
                    assert not ck[1], 'loader options with variable bound parameters not supported with baked queries.  Please use new-style select() statements for cached ORM queries.'
                    key += ck[0]
        self.add_criteria(lambda q: q._with_current_path(effective_path).options(*options), cache_path.path, key)

    def _retrieve_baked_query(self, session):
        if False:
            return 10
        query = self._bakery.get(self._effective_key(session), None)
        if query is None:
            query = self._as_query(session)
            self._bakery[self._effective_key(session)] = query.with_session(None)
        return query.with_session(session)

    def _bake(self, session):
        if False:
            print('Hello World!')
        query = self._as_query(session)
        query.session = None
        statement = query._statement_20()
        if statement._compile_options._bake_ok:
            self._bakery[self._effective_key(session)] = (query, statement)
        return (query, statement)

    def to_query(self, query_or_session):
        if False:
            i = 10
            return i + 15
        'Return the :class:`_query.Query` object for use as a subquery.\n\n        This method should be used within the lambda callable being used\n        to generate a step of an enclosing :class:`.BakedQuery`.   The\n        parameter should normally be the :class:`_query.Query` object that\n        is passed to the lambda::\n\n            sub_bq = self.bakery(lambda s: s.query(User.name))\n            sub_bq += lambda q: q.filter(\n                User.id == Address.user_id).correlate(Address)\n\n            main_bq = self.bakery(lambda s: s.query(Address))\n            main_bq += lambda q: q.filter(\n                sub_bq.to_query(q).exists())\n\n        In the case where the subquery is used in the first callable against\n        a :class:`.Session`, the :class:`.Session` is also accepted::\n\n            sub_bq = self.bakery(lambda s: s.query(User.name))\n            sub_bq += lambda q: q.filter(\n                User.id == Address.user_id).correlate(Address)\n\n            main_bq = self.bakery(\n                lambda s: s.query(\n                Address.id, sub_bq.to_query(q).scalar_subquery())\n            )\n\n        :param query_or_session: a :class:`_query.Query` object or a class\n         :class:`.Session` object, that is assumed to be within the context\n         of an enclosing :class:`.BakedQuery` callable.\n\n\n         .. versionadded:: 1.3\n\n\n        '
        if isinstance(query_or_session, Session):
            session = query_or_session
        elif isinstance(query_or_session, Query):
            session = query_or_session.session
            if session is None:
                raise sa_exc.ArgumentError('Given Query needs to be associated with a Session')
        else:
            raise TypeError('Query or Session object expected, got %r.' % type(query_or_session))
        return self._as_query(session)

    def _as_query(self, session):
        if False:
            i = 10
            return i + 15
        query = self.steps[0](session)
        for step in self.steps[1:]:
            query = step(query)
        return query

class Result:
    """Invokes a :class:`.BakedQuery` against a :class:`.Session`.

    The :class:`_baked.Result` object is where the actual :class:`.query.Query`
    object gets created, or retrieved from the cache,
    against a target :class:`.Session`, and is then invoked for results.

    """
    __slots__ = ('bq', 'session', '_params', '_post_criteria')

    def __init__(self, bq, session):
        if False:
            print('Hello World!')
        self.bq = bq
        self.session = session
        self._params = {}
        self._post_criteria = []

    def params(self, *args, **kw):
        if False:
            return 10
        'Specify parameters to be replaced into the string SQL statement.'
        if len(args) == 1:
            kw.update(args[0])
        elif len(args) > 0:
            raise sa_exc.ArgumentError('params() takes zero or one positional argument, which is a dictionary.')
        self._params.update(kw)
        return self

    def _using_post_criteria(self, fns):
        if False:
            i = 10
            return i + 15
        if fns:
            self._post_criteria.extend(fns)
        return self

    def with_post_criteria(self, fn):
        if False:
            while True:
                i = 10
        "Add a criteria function that will be applied post-cache.\n\n        This adds a function that will be run against the\n        :class:`_query.Query` object after it is retrieved from the\n        cache.    This currently includes **only** the\n        :meth:`_query.Query.params` and :meth:`_query.Query.execution_options`\n        methods.\n\n        .. warning::  :meth:`_baked.Result.with_post_criteria`\n           functions are applied\n           to the :class:`_query.Query`\n           object **after** the query's SQL statement\n           object has been retrieved from the cache.   Only\n           :meth:`_query.Query.params` and\n           :meth:`_query.Query.execution_options`\n           methods should be used.\n\n\n        .. versionadded:: 1.2\n\n\n        "
        return self._using_post_criteria([fn])

    def _as_query(self):
        if False:
            for i in range(10):
                print('nop')
        q = self.bq._as_query(self.session).params(self._params)
        for fn in self._post_criteria:
            q = fn(q)
        return q

    def __str__(self):
        if False:
            while True:
                i = 10
        return str(self._as_query())

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self._iter().__iter__()

    def _iter(self):
        if False:
            i = 10
            return i + 15
        bq = self.bq
        if not self.session.enable_baked_queries or bq._spoiled:
            return self._as_query()._iter()
        (query, statement) = bq._bakery.get(bq._effective_key(self.session), (None, None))
        if query is None:
            (query, statement) = bq._bake(self.session)
        if self._params:
            q = query.params(self._params)
        else:
            q = query
        for fn in self._post_criteria:
            q = fn(q)
        params = q._params
        execution_options = dict(q._execution_options)
        execution_options.update({'_sa_orm_load_options': q.load_options, 'compiled_cache': bq._bakery})
        result = self.session.execute(statement, params, execution_options=execution_options)
        if result._attributes.get('is_single_entity', False):
            result = result.scalars()
        if result._attributes.get('filtered', False):
            result = result.unique()
        return result

    def count(self):
        if False:
            while True:
                i = 10
        "return the 'count'.\n\n        Equivalent to :meth:`_query.Query.count`.\n\n        Note this uses a subquery to ensure an accurate count regardless\n        of the structure of the original statement.\n\n        "
        col = func.count(literal_column('*'))
        bq = self.bq.with_criteria(lambda q: q._legacy_from_self(col))
        return bq.for_session(self.session).params(self._params).scalar()

    def scalar(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the first element of the first result or None\n        if no rows present.  If multiple rows are returned,\n        raises MultipleResultsFound.\n\n        Equivalent to :meth:`_query.Query.scalar`.\n\n        '
        try:
            ret = self.one()
            if not isinstance(ret, collections_abc.Sequence):
                return ret
            return ret[0]
        except orm_exc.NoResultFound:
            return None

    def first(self):
        if False:
            while True:
                i = 10
        'Return the first row.\n\n        Equivalent to :meth:`_query.Query.first`.\n\n        '
        bq = self.bq.with_criteria(lambda q: q.slice(0, 1))
        return bq.for_session(self.session).params(self._params)._using_post_criteria(self._post_criteria)._iter().first()

    def one(self):
        if False:
            while True:
                i = 10
        'Return exactly one result or raise an exception.\n\n        Equivalent to :meth:`_query.Query.one`.\n\n        '
        return self._iter().one()

    def one_or_none(self):
        if False:
            i = 10
            return i + 15
        'Return one or zero results, or raise an exception for multiple\n        rows.\n\n        Equivalent to :meth:`_query.Query.one_or_none`.\n\n        '
        return self._iter().one_or_none()

    def all(self):
        if False:
            print('Hello World!')
        'Return all rows.\n\n        Equivalent to :meth:`_query.Query.all`.\n\n        '
        return self._iter().all()

    def get(self, ident):
        if False:
            while True:
                i = 10
        'Retrieve an object based on identity.\n\n        Equivalent to :meth:`_query.Query.get`.\n\n        '
        query = self.bq.steps[0](self.session)
        return query._get_impl(ident, self._load_on_pk_identity)

    def _load_on_pk_identity(self, session, query, primary_key_identity, **kw):
        if False:
            i = 10
            return i + 15
        'Load the given primary key identity from the database.'
        mapper = query._raw_columns[0]._annotations['parententity']
        (_get_clause, _get_params) = mapper._get_clause

        def setup(query):
            if False:
                i = 10
                return i + 15
            _lcl_get_clause = _get_clause
            q = query._clone()
            q._get_condition()
            q._order_by = None
            if None in primary_key_identity:
                nones = {_get_params[col].key for (col, value) in zip(mapper.primary_key, primary_key_identity) if value is None}
                _lcl_get_clause = sql_util.adapt_criterion_to_null(_lcl_get_clause, nones)
            q._where_criteria = (sql_util._deep_annotate(_lcl_get_clause, {'_orm_adapt': True}),)
            for fn in self._post_criteria:
                q = fn(q)
            return q
        bq = self.bq
        bq = bq._clone()
        bq._cache_key += (_get_clause,)
        bq = bq.with_criteria(setup, tuple((elem is None for elem in primary_key_identity)))
        params = {_get_params[primary_key].key: id_val for (id_val, primary_key) in zip(primary_key_identity, mapper.primary_key)}
        result = list(bq.for_session(self.session).params(**params))
        l = len(result)
        if l > 1:
            raise orm_exc.MultipleResultsFound()
        elif l:
            return result[0]
        else:
            return None
bakery = BakedQuery.bakery