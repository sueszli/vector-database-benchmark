from mock import MagicMock, patch
from r2.lib.db.thing import CreationError, hooks, NotFound, tdb, Thing
from r2.lib.lock import TimeoutExpired
from r2.tests import RedditTestCase

class SimpleThing(Thing):
    _nodb = True
    _type_name = 'simplething'
    _type_id = 100
    _cache = MagicMock()
    _data_int_props = ('prop_for_data',)
    _defaults = {'prop_for_data': 0}

class TestThingReadCaching(RedditTestCase):

    def setUp(self):
        if False:
            return 10
        self.get_things_from_cache = self.autopatch(Thing, 'get_things_from_cache')
        self.get_things_from_db = self.autopatch(Thing, 'get_things_from_db')
        self.write_things_to_cache = self.autopatch(Thing, 'write_things_to_cache')

    def test_not_found(self):
        if False:
            print('Hello World!')
        things_by_id = {}
        self.get_things_from_cache.return_value = {}
        self.get_things_from_db.return_value = things_by_id
        with self.assertRaises(NotFound):
            SimpleThing._byID([1, 2, 3], stale=False)
        self.get_things_from_cache.assert_called_once_with([1, 2, 3], stale=False)
        self.get_things_from_db.assert_called_once_with([1, 2, 3])
        self.write_things_to_cache.assert_not_called()

    def test_partial_not_found(self):
        if False:
            return 10
        things_by_id = {1: 'one'}
        self.get_things_from_cache.return_value = {}
        self.get_things_from_db.return_value = things_by_id
        with self.assertRaises(NotFound):
            SimpleThing._byID([1, 2, 3], stale=False)
        self.get_things_from_cache.assert_called_once_with([1, 2, 3], stale=False)
        self.get_things_from_db.assert_called_once_with([1, 2, 3])
        self.write_things_to_cache.assert_called_once_with(things_by_id)

    def test_partial_not_found_ignore(self):
        if False:
            print('Hello World!')
        things_by_id = {1: 'one'}
        self.get_things_from_cache.return_value = {}
        self.get_things_from_db.return_value = things_by_id
        ret = SimpleThing._byID([1, 2, 3], stale=False, ignore_missing=True)
        self.get_things_from_cache.assert_called_once_with([1, 2, 3], stale=False)
        self.get_things_from_db.assert_called_once_with([1, 2, 3])
        self.write_things_to_cache.assert_called_once_with(things_by_id)
        self.assertEqual(ret, things_by_id)

    def test_cache_miss(self):
        if False:
            for i in range(10):
                print('nop')
        things_by_id = {1: 'one', 2: 'two', 3: 'three'}
        self.get_things_from_cache.return_value = {}
        self.get_things_from_db.return_value = things_by_id
        ret = SimpleThing._byID([1, 2, 3], stale=False)
        self.get_things_from_cache.assert_called_once_with([1, 2, 3], stale=False)
        self.get_things_from_db.assert_called_once_with([1, 2, 3])
        self.write_things_to_cache.assert_called_once_with(things_by_id)
        self.assertEqual(ret, things_by_id)

    def test_cache_hit(self):
        if False:
            while True:
                i = 10
        things_by_id = {1: 'one', 2: 'two', 3: 'three'}
        self.get_things_from_cache.return_value = things_by_id
        ret = SimpleThing._byID([1, 2, 3], stale=False)
        self.get_things_from_cache.assert_called_once_with([1, 2, 3], stale=False)
        self.get_things_from_db.assert_not_called()
        self.write_things_to_cache.assert_not_called()
        self.assertEqual(ret, things_by_id)

    def test_partial_hit(self):
        if False:
            for i in range(10):
                print('nop')
        things_by_id = {1: 'one', 2: 'two', 3: 'three'}
        self.get_things_from_cache.return_value = {1: 'one'}
        self.get_things_from_db.return_value = {2: 'two', 3: 'three'}
        ret = SimpleThing._byID([1, 2, 3], stale=False)
        self.get_things_from_cache.assert_called_once_with([1, 2, 3], stale=False)
        self.get_things_from_db.assert_called_once_with([2, 3])
        self.write_things_to_cache.assert_called_once_with({2: 'two', 3: 'three'})
        self.assertEqual(ret, things_by_id)

    def test_return_list(self):
        if False:
            print('Hello World!')
        things_by_id = {1: 'one', 2: 'two', 3: 'three'}
        self.get_things_from_cache.return_value = things_by_id
        self.get_things_from_db.return_value = things_by_id
        ret = SimpleThing._byID([1, 2, 3], stale=False, return_dict=False)
        self.get_things_from_cache.assert_called_once_with([1, 2, 3], stale=False)
        self.get_things_from_db.assert_not_called()
        self.write_things_to_cache.assert_not_called()
        self.assertEqual(ret, ['one', 'two', 'three'])

    def test_single_not_found(self):
        if False:
            for i in range(10):
                print('nop')
        things_by_id = {}
        self.get_things_from_cache.return_value = {}
        self.get_things_from_db.return_value = {}
        with self.assertRaises(NotFound):
            SimpleThing._byID(1, stale=False)
        self.get_things_from_cache.assert_called_once_with((1,), stale=False)
        self.get_things_from_db.assert_called_once_with([1])
        self.write_things_to_cache.assert_not_called()

    def test_single_miss(self):
        if False:
            print('Hello World!')
        things_by_id = {1: 'one'}
        self.get_things_from_cache.return_value = {}
        self.get_things_from_db.return_value = things_by_id
        ret = SimpleThing._byID(1, stale=False)
        self.get_things_from_cache.assert_called_once_with((1,), stale=False)
        self.get_things_from_db.assert_called_once_with([1])
        self.write_things_to_cache.assert_called_once_with(things_by_id)
        self.assertEqual(ret, 'one')

    def test_single_hit(self):
        if False:
            for i in range(10):
                print('nop')
        things_by_id = {1: 'one'}
        self.get_things_from_cache.return_value = things_by_id
        self.get_things_from_db.return_value = things_by_id
        ret = SimpleThing._byID(1, stale=False)
        self.get_things_from_cache.assert_called_once_with((1,), stale=False)
        self.get_things_from_db.assert_not_called()
        self.write_things_to_cache.assert_not_called()
        self.assertEqual(ret, 'one')

class FakeLock(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.have_lock = True

    def acquire(self):
        if False:
            while True:
                i = 10
        return

    def release(self):
        if False:
            for i in range(10):
                print('nop')
        return

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, *args):
        if False:
            return 10
        return

class TestThingWrite(RedditTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.lock = FakeLock()
        self.thing_id = 333
        self.autopatch(tdb, 'transactions')
        self.autopatch(hooks, 'get_hook')
        self.autopatch(Thing, 'write_new_thing_to_db', return_value=self.thing_id)
        self.autopatch(Thing, 'get_read_modify_write_lock', return_value=self.lock)
        self.autopatch(Thing, 'write_props_to_db')
        self.autopatch(Thing, 'write_thing_to_cache')
        self.autopatch(Thing, 'update_from_cache')

    def reset_mocks(self):
        if False:
            print('Hello World!')
        SimpleThing.write_new_thing_to_db.reset_mock()
        SimpleThing.get_read_modify_write_lock.reset_mock()
        SimpleThing.update_from_cache.reset_mock()
        SimpleThing.write_props_to_db.reset_mock()
        SimpleThing.write_thing_to_cache.reset_mock()

    def test_create(self):
        if False:
            while True:
                i = 10
        thing = SimpleThing(ups=1, downs=0, spam=False, deleted=False)
        thing.other_prop = 100
        thing._commit()
        SimpleThing.write_new_thing_to_db.assert_called_once_with()
        SimpleThing.update_from_cache.assert_not_called()
        SimpleThing.write_props_to_db.assert_called_once_with({}, {'other_prop': 100}, True)
        SimpleThing.write_thing_to_cache.assert_called_once_with(lock=None, brand_new_thing=True)

    def test_modify(self):
        if False:
            print('Hello World!')
        thing = SimpleThing(ups=1, downs=0, spam=False, deleted=False)
        thing.other_prop = 100
        thing._commit()
        self.reset_mocks()
        thing.other_prop = 101
        thing._ups = 12
        thing._commit()
        SimpleThing.write_new_thing_to_db.assert_not_called()
        SimpleThing.get_read_modify_write_lock.assert_called_once_with()
        SimpleThing.update_from_cache.assert_called_once_with(self.lock)
        SimpleThing.write_props_to_db.assert_called_once_with({'ups': 12}, {'other_prop': 101}, False)
        SimpleThing.write_thing_to_cache.assert_called_once_with(self.lock)

class TestThingIncr(RedditTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.lock = FakeLock()
        self.thing_id = 333
        self.autopatch(tdb, 'transactions')
        self.autopatch(hooks, 'get_hook')
        self.autopatch(Thing, 'write_new_thing_to_db', return_value=self.thing_id)
        self.autopatch(Thing, 'get_read_modify_write_lock', return_value=self.lock)
        self.autopatch(Thing, 'write_props_to_db')
        self.autopatch(Thing, 'write_thing_to_cache')
        self.autopatch(Thing, 'update_from_cache')

    def reset_mocks(self):
        if False:
            while True:
                i = 10
        SimpleThing.write_new_thing_to_db.reset_mock()
        SimpleThing.get_read_modify_write_lock.reset_mock()
        SimpleThing.update_from_cache.reset_mock()
        SimpleThing.write_props_to_db.reset_mock()
        SimpleThing.write_thing_to_cache.reset_mock()

    @patch('r2.lib.db.tdb_sql.incr_thing_prop')
    def test_incr_base_prop(self, incr_thing_prop):
        if False:
            while True:
                i = 10
        thing = SimpleThing(ups=1, downs=0, spam=False, deleted=False)
        thing._commit()
        self.reset_mocks()
        thing._incr('_ups')
        incr_thing_prop.assert_called_once_with(type_id=SimpleThing._type_id, thing_id=thing._id, prop='ups', amount=1)

    @patch('r2.lib.db.tdb_sql.incr_thing_data')
    def test_incr_data_prop(self, incr_thing_data):
        if False:
            print('Hello World!')
        thing = SimpleThing(ups=1, downs=0, spam=False, deleted=False)
        thing.prop_for_data = 100
        thing._commit()
        self.reset_mocks()
        thing._incr('prop_for_data')
        incr_thing_data.assert_called_once_with(type_id=SimpleThing._type_id, thing_id=thing._id, prop='prop_for_data', amount=1)

    @patch('r2.lib.db.tdb_sql.set_thing_data')
    def test_incr_unset_data_prop(self, set_thing_data):
        if False:
            while True:
                i = 10
        thing = SimpleThing(ups=1, downs=0, spam=False, deleted=False)
        thing._commit()
        self.reset_mocks()
        thing._incr('prop_for_data')
        set_thing_data.assert_called_once_with(type_id=SimpleThing._type_id, thing_id=thing._id, brand_new_thing=False, prop_for_data=1)

    def test_incr_dirty(self):
        if False:
            print('Hello World!')
        thing = SimpleThing(ups=1, downs=0, spam=False, deleted=False)
        thing._commit()
        self.reset_mocks()
        thing.other_prop = 100
        with self.assertRaises(AssertionError):
            thing._incr('_ups')

class TestThingWriteConflict(RedditTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.lock = FakeLock()
        self.thing_id = 333
        self.autopatch(tdb, 'transactions')
        self.autopatch(hooks, 'get_hook')
        self.autopatch(Thing, 'write_new_thing_to_db', return_value=self.thing_id)
        self.autopatch(Thing, 'get_read_modify_write_lock', return_value=self.lock)
        self.autopatch(Thing, 'write_props_to_db')
        self.autopatch(Thing, 'write_thing_to_cache')

    def reset_mocks(self):
        if False:
            for i in range(10):
                print('nop')
        SimpleThing.write_new_thing_to_db.reset_mock()
        SimpleThing.get_read_modify_write_lock.reset_mock()
        SimpleThing.write_props_to_db.reset_mock()
        SimpleThing.write_thing_to_cache.reset_mock()

    @patch('r2.lib.db.thing.Thing.get_things_from_cache')
    def test_dont_overwrite(self, get_things_from_cache):
        if False:
            for i in range(10):
                print('nop')
        other_thing = SimpleThing(ups=2, downs=0, spam=False, deleted=False, id=self.thing_id)
        other_thing.__setattr__('another_prop', 3, make_dirty=False)
        get_things_from_cache.return_value = {self.thing_id: other_thing}
        thing = SimpleThing(ups=1, downs=0, spam=False, deleted=False)
        thing.other_prop = 100
        thing._commit()
        self.reset_mocks()
        thing._downs = 1
        thing.other_prop = 102
        thing._commit()
        SimpleThing.write_new_thing_to_db.assert_not_called()
        SimpleThing.get_read_modify_write_lock.assert_called_once_with()
        get_things_from_cache.assert_called_once_with([self.thing_id], allow_local=False)
        SimpleThing.write_props_to_db.assert_called_once_with({'downs': 1}, {'other_prop': 102}, False)
        SimpleThing.write_thing_to_cache.assert_called_once_with(self.lock)
        self.assertEqual(thing.another_prop, 3)

    @patch('r2.lib.db.thing.Thing.get_read_modify_write_lock')
    def test_lock_fail(self, get_read_modify_write_lock):
        if False:
            return 10
        get_read_modify_write_lock.side_effect = TimeoutExpired()
        thing = SimpleThing(ups=2, downs=0, spam=False, deleted=False, id=self.thing_id)
        self.reset_mocks()
        thing._ups = 3
        with self.assertRaises(TimeoutExpired):
            thing._commit()
        tdb.transactions.rollback.assert_not_called()

    @patch('r2.lib.db.thing.Thing.write_new_thing_to_db')
    def test_create_fail(self, write_new_thing_to_db):
        if False:
            for i in range(10):
                print('nop')
        write_new_thing_to_db.side_effect = CreationError()
        thing = SimpleThing(ups=2, downs=0, spam=False, deleted=False)
        thing.other_prop = 13
        with self.assertRaises(CreationError):
            thing._commit()
        tdb.transactions.rollback.assert_called_once_with()

    @patch('r2.lib.db.thing.Thing.write_changes_to_db')
    def test_modify_fail(self, write_changes_to_db):
        if False:
            return 10
        write_changes_to_db.side_effect = CreationError()
        thing = SimpleThing(ups=2, downs=0, spam=False, deleted=False, id=self.thing_id)
        self.reset_mocks()
        thing._ups = 3
        with self.assertRaises(CreationError):
            thing._commit()
        tdb.transactions.rollback.assert_called_once_with()