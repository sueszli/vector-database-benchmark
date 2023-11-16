"""Represent functions and classes
which allow the usage of Dogpile caching with SQLAlchemy.
Introduces a query option called FromCache.

.. versionchanged:: 1.4  the caching approach has been altered to work
   based on a session event.


The three new concepts introduced here are:

 * ORMCache - an extension for an ORM :class:`.Session`
   retrieves results in/from dogpile.cache.
 * FromCache - a query option that establishes caching
   parameters on a Query
 * RelationshipCache - a variant of FromCache which is specific
   to a query invoked during a lazy load.

The rest of what's here are standard SQLAlchemy and
dogpile.cache constructs.

"""
from dogpile.cache.api import NO_VALUE
from sqlalchemy import event
from sqlalchemy.orm import loading
from sqlalchemy.orm import Query
from sqlalchemy.orm.interfaces import UserDefinedOption

class ORMCache:
    """An add-on for an ORM :class:`.Session` optionally loads full results
    from a dogpile cache region.


    """

    def __init__(self, regions):
        if False:
            i = 10
            return i + 15
        self.cache_regions = regions
        self._statement_cache = {}

    def listen_on_session(self, session_factory):
        if False:
            while True:
                i = 10
        event.listen(session_factory, 'do_orm_execute', self._do_orm_execute)

    def _do_orm_execute(self, orm_context):
        if False:
            for i in range(10):
                print('nop')
        for opt in orm_context.user_defined_options:
            if isinstance(opt, RelationshipCache):
                opt = opt._process_orm_context(orm_context)
                if opt is None:
                    continue
            if isinstance(opt, FromCache):
                dogpile_region = self.cache_regions[opt.region]
                our_cache_key = opt._generate_cache_key(orm_context.statement, orm_context.parameters or {}, self)
                if opt.ignore_expiration:
                    cached_value = dogpile_region.get(our_cache_key, expiration_time=opt.expiration_time, ignore_expiration=opt.ignore_expiration)
                else:

                    def createfunc():
                        if False:
                            i = 10
                            return i + 15
                        return orm_context.invoke_statement().freeze()
                    cached_value = dogpile_region.get_or_create(our_cache_key, createfunc, expiration_time=opt.expiration_time)
                if cached_value is NO_VALUE:
                    raise KeyError()
                orm_result = loading.merge_frozen_result(orm_context.session, orm_context.statement, cached_value, load=False)
                return orm_result()
        else:
            return None

    def invalidate(self, statement, parameters, opt):
        if False:
            while True:
                i = 10
        'Invalidate the cache value represented by a statement.'
        if isinstance(statement, Query):
            statement = statement.__clause_element__()
        dogpile_region = self.cache_regions[opt.region]
        cache_key = opt._generate_cache_key(statement, parameters, self)
        dogpile_region.delete(cache_key)

class FromCache(UserDefinedOption):
    """Specifies that a Query should load results from a cache."""
    propagate_to_loaders = False

    def __init__(self, region='default', cache_key=None, expiration_time=None, ignore_expiration=False):
        if False:
            print('Hello World!')
        'Construct a new FromCache.\n\n        :param region: the cache region.  Should be a\n         region configured in the dictionary of dogpile\n         regions.\n\n        :param cache_key: optional.  A string cache key\n         that will serve as the key to the query.   Use this\n         if your query has a huge amount of parameters (such\n         as when using in_()) which correspond more simply to\n         some other identifier.\n\n        '
        self.region = region
        self.cache_key = cache_key
        self.expiration_time = expiration_time
        self.ignore_expiration = ignore_expiration

    def _gen_cache_key(self, anon_map, bindparams):
        if False:
            i = 10
            return i + 15
        return None

    def _generate_cache_key(self, statement, parameters, orm_cache):
        if False:
            return 10
        'generate a cache key with which to key the results of a statement.\n\n        This leverages the use of the SQL compilation cache key which is\n        repurposed as a SQL results key.\n\n        '
        statement_cache_key = statement._generate_cache_key()
        key = statement_cache_key.to_offline_string(orm_cache._statement_cache, statement, parameters) + repr(self.cache_key)
        return key

class RelationshipCache(FromCache):
    """Specifies that a Query as called within a "lazy load"
    should load results from a cache."""
    propagate_to_loaders = True

    def __init__(self, attribute, region='default', cache_key=None, expiration_time=None, ignore_expiration=False):
        if False:
            return 10
        'Construct a new RelationshipCache.\n\n        :param attribute: A Class.attribute which\n         indicates a particular class relationship() whose\n         lazy loader should be pulled from the cache.\n\n        :param region: name of the cache region.\n\n        :param cache_key: optional.  A string cache key\n         that will serve as the key to the query, bypassing\n         the usual means of forming a key from the Query itself.\n\n        '
        self.region = region
        self.cache_key = cache_key
        self.expiration_time = expiration_time
        self.ignore_expiration = ignore_expiration
        self._relationship_options = {(attribute.property.parent.class_, attribute.property.key): self}

    def _process_orm_context(self, orm_context):
        if False:
            for i in range(10):
                print('nop')
        current_path = orm_context.loader_strategy_path
        if current_path:
            (mapper, prop) = current_path[-2:]
            key = prop.key
            for cls in mapper.class_.__mro__:
                if (cls, key) in self._relationship_options:
                    relationship_option = self._relationship_options[cls, key]
                    return relationship_option

    def and_(self, option):
        if False:
            while True:
                i = 10
        'Chain another RelationshipCache option to this one.\n\n        While many RelationshipCache objects can be specified on a single\n        Query separately, chaining them together allows for a more efficient\n        lookup during load.\n\n        '
        self._relationship_options.update(option._relationship_options)
        return self