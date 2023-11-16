"""This example creates a new dogpile.cache backend that will persist data in a
dictionary which is local to the current session.   remove() the session and
the cache is gone.

Create a new Dogpile cache backend that will store
cached data local to the current Session.

This is an advanced example which assumes familiarity
with the basic operation of CachingQuery.

"""
from dogpile.cache import make_region
from dogpile.cache.api import CacheBackend
from dogpile.cache.api import NO_VALUE
from dogpile.cache.region import register_backend
from sqlalchemy import select
from . import environment
from .caching_query import FromCache
from .environment import regions
from .environment import Session

class ScopedSessionBackend(CacheBackend):
    """A dogpile backend which will cache objects locally on
    the current session.

    When used with the query_cache system, the effect is that the objects
    in the cache are the same as that within the session - the merge()
    is a formality that doesn't actually create a second instance.
    This makes it safe to use for updates of data from an identity
    perspective (still not ideal for deletes though).

    When the session is removed, the cache is gone too, so the cache
    is automatically disposed upon session.remove().

    """

    def __init__(self, arguments):
        if False:
            print('Hello World!')
        self.scoped_session = arguments['scoped_session']

    def get(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self._cache_dictionary.get(key, NO_VALUE)

    def set(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self._cache_dictionary[key] = value

    def delete(self, key):
        if False:
            return 10
        self._cache_dictionary.pop(key, None)

    @property
    def _cache_dictionary(self):
        if False:
            return 10
        'Return the cache dictionary linked to the current Session.'
        sess = self.scoped_session()
        try:
            cache_dict = sess._cache_dictionary
        except AttributeError:
            sess._cache_dictionary = cache_dict = {}
        return cache_dict
register_backend('sqlalchemy.session', __name__, 'ScopedSessionBackend')
if __name__ == '__main__':
    regions['local_session'] = make_region().configure('sqlalchemy.session', arguments={'scoped_session': Session})
    from .model import Person
    q = select(Person).filter(Person.name == 'person 10').options(FromCache('local_session'))
    person10 = Session.scalars(q).one()
    person10 = Session.scalars(q).one()
    Session.remove()
    person10 = Session.scalars(q).one()
    cache_key = FromCache('local_session')._generate_cache_key(q, {}, environment.cache)
    assert person10 is regions['local_session'].get(cache_key)().scalar()