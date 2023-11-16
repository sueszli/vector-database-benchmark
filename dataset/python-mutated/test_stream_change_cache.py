from synapse.util.caches.stream_change_cache import StreamChangeCache
from tests import unittest

class StreamChangeCacheTests(unittest.HomeserverTestCase):
    """
    Tests for StreamChangeCache.
    """

    def test_prefilled_cache(self) -> None:
        if False:
            return 10
        '\n        Providing a prefilled cache to StreamChangeCache will result in a cache\n        with the prefilled-cache entered in.\n        '
        cache = StreamChangeCache('#test', 1, prefilled_cache={'user@foo.com': 2})
        self.assertTrue(cache.has_entity_changed('user@foo.com', 1))

    def test_has_entity_changed(self) -> None:
        if False:
            print('Hello World!')
        '\n        StreamChangeCache.entity_has_changed will mark entities as changed, and\n        has_entity_changed will observe the changed entities.\n        '
        cache = StreamChangeCache('#test', 3)
        cache.entity_has_changed('user@foo.com', 6)
        cache.entity_has_changed('bar@baz.net', 7)
        cache.entity_has_changed('user2@foo.com', 8)
        cache.entity_has_changed('bar2@baz.net', 8)
        self.assertTrue(cache.has_entity_changed('user@foo.com', 4))
        self.assertTrue(cache.has_entity_changed('bar@baz.net', 4))
        self.assertTrue(cache.has_entity_changed('bar2@baz.net', 4))
        self.assertTrue(cache.has_entity_changed('user2@foo.com', 4))
        self.assertFalse(cache.has_entity_changed('user@foo.com', 6))
        self.assertFalse(cache.has_entity_changed('user2@foo.com', 8))
        self.assertFalse(cache.has_entity_changed('user@foo.com', 7))
        self.assertFalse(cache.has_entity_changed('user2@foo.com', 9))
        self.assertFalse(cache.has_entity_changed('not@here.website', 9))
        self.assertTrue(cache.has_entity_changed('user@foo.com', 0))
        self.assertTrue(cache.has_entity_changed('not@here.website', 0))
        self.assertTrue(cache.has_entity_changed('user@foo.com', 3))
        self.assertTrue(cache.has_entity_changed('not@here.website', 3))

    def test_entity_has_changed_pops_off_start(self) -> None:
        if False:
            print('Hello World!')
        '\n        StreamChangeCache.entity_has_changed will respect the max size and\n        purge the oldest items upon reaching that max size.\n        '
        cache = StreamChangeCache('#test', 1, max_size=2)
        cache.entity_has_changed('user@foo.com', 2)
        cache.entity_has_changed('bar@baz.net', 3)
        cache.entity_has_changed('user@elsewhere.org', 4)
        self.assertEqual(len(cache._cache), 2)
        self.assertEqual(cache._earliest_known_stream_pos, 2)
        self.assertTrue('user@foo.com' not in cache._entity_to_key)
        self.assertEqual(cache.get_all_entities_changed(3).entities, ['user@elsewhere.org'])
        self.assertFalse(cache.get_all_entities_changed(2).hit)
        cache.entity_has_changed('bar@baz.net', 5)
        self.assertEqual({'bar@baz.net', 'user@elsewhere.org'}, set(cache._entity_to_key))
        self.assertEqual(cache.get_all_entities_changed(3).entities, ['user@elsewhere.org', 'bar@baz.net'])
        self.assertFalse(cache.get_all_entities_changed(2).hit)

    def test_get_all_entities_changed(self) -> None:
        if False:
            return 10
        '\n        StreamChangeCache.get_all_entities_changed will return all changed\n        entities since the given position.  If the position is before the start\n        of the known stream, it returns None instead.\n        '
        cache = StreamChangeCache('#test', 1)
        cache.entity_has_changed('user@foo.com', 2)
        cache.entity_has_changed('bar@baz.net', 3)
        cache.entity_has_changed('anotheruser@foo.com', 3)
        cache.entity_has_changed('user@elsewhere.org', 4)
        r = cache.get_all_entities_changed(2)
        ok1 = ['bar@baz.net', 'anotheruser@foo.com', 'user@elsewhere.org']
        ok2 = ['anotheruser@foo.com', 'bar@baz.net', 'user@elsewhere.org']
        self.assertTrue(r.entities == ok1 or r.entities == ok2)
        self.assertEqual(cache.get_all_entities_changed(3).entities, ['user@elsewhere.org'])
        self.assertFalse(cache.get_all_entities_changed(1).hit)
        cache.entity_has_changed('user@foo.com', 5)
        cache.entity_has_changed('bar@baz.net', 5)
        cache.entity_has_changed('anotheruser@foo.com', 6)
        ok1 = ['user@elsewhere.org', 'user@foo.com', 'bar@baz.net', 'anotheruser@foo.com']
        ok2 = ['user@elsewhere.org', 'bar@baz.net', 'user@foo.com', 'anotheruser@foo.com']
        r = cache.get_all_entities_changed(3)
        self.assertTrue(r.entities == ok1 or r.entities == ok2)

    def test_has_any_entity_changed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        StreamChangeCache.has_any_entity_changed will return True if any\n        entities have been changed since the provided stream position, and\n        False if they have not.  If the cache has entries and the provided\n        stream position is before it, it will return True, otherwise False if\n        the cache has no entries.\n        '
        cache = StreamChangeCache('#test', 1)
        self.assertTrue(cache.has_any_entity_changed(0))
        self.assertTrue(cache.has_any_entity_changed(1))
        self.assertFalse(cache.has_any_entity_changed(2))
        cache.entity_has_changed('user@foo.com', 2)
        self.assertTrue(cache.has_any_entity_changed(0))
        self.assertTrue(cache.has_any_entity_changed(1))
        self.assertFalse(cache.has_any_entity_changed(2))
        self.assertFalse(cache.has_any_entity_changed(3))

    def test_get_entities_changed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        StreamChangeCache.get_entities_changed will return the entities in the\n        given list that have changed since the provided stream ID.  If the\n        stream position is earlier than the earliest known position, it will\n        return all of the entities queried for.\n        '
        cache = StreamChangeCache('#test', 1)
        cache.entity_has_changed('user@foo.com', 2)
        cache.entity_has_changed('bar@baz.net', 3)
        cache.entity_has_changed('user@elsewhere.org', 4)
        self.assertEqual(cache.get_entities_changed(['user@foo.com', 'bar@baz.net', 'user@elsewhere.org'], stream_pos=2), {'bar@baz.net', 'user@elsewhere.org'})
        self.assertEqual(cache.get_entities_changed(['user@foo.com', 'bar@baz.net', 'user@elsewhere.org', 'not@here.website'], stream_pos=2), {'bar@baz.net', 'user@elsewhere.org'})
        self.assertEqual(cache.get_entities_changed(['user@foo.com', 'bar@baz.net', 'user@elsewhere.org', 'not@here.website'], stream_pos=0), {'user@foo.com', 'bar@baz.net', 'user@elsewhere.org', 'not@here.website'})
        self.assertEqual(cache.get_entities_changed(['bar@baz.net'], stream_pos=2), {'bar@baz.net'})

    def test_max_pos(self) -> None:
        if False:
            return 10
        '\n        StreamChangeCache.get_max_pos_of_last_change will return the most\n        recent point where the entity could have changed.  If the entity is not\n        known, the stream start is provided instead.\n        '
        cache = StreamChangeCache('#test', 1)
        cache.entity_has_changed('user@foo.com', 2)
        cache.entity_has_changed('bar@baz.net', 3)
        cache.entity_has_changed('user@elsewhere.org', 4)
        self.assertEqual(cache.get_max_pos_of_last_change('user@foo.com'), 2)
        self.assertEqual(cache.get_max_pos_of_last_change('bar@baz.net'), 3)
        self.assertEqual(cache.get_max_pos_of_last_change('user@elsewhere.org'), 4)
        self.assertEqual(cache.get_max_pos_of_last_change('not@here.website'), 1)