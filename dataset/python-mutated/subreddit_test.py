import unittest
from mock import MagicMock
from pylons import app_globals as g
from r2.lib.permissions import PermissionSet
from r2.models import NotFound
from r2.models.account import Account
from r2.models.subreddit import SRMember, Subreddit

class TestPermissionSet(PermissionSet):
    info = dict(x={}, y={})

class SRMemberTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        a = Account()
        a._id = 1
        sr = Subreddit()
        sr._id = 2
        self.rel = SRMember(sr, a, 'test')

    def test_get_permissions(self):
        if False:
            while True:
                i = 10
        self.assertRaises(NotImplementedError, self.rel.get_permissions)
        self.rel._permission_class = TestPermissionSet
        self.assertEquals('', self.rel.get_permissions().dumps())
        self.rel.encoded_permissions = '+x,-y'
        self.assertEquals('+x,-y', self.rel.get_permissions().dumps())

    def test_has_permission(self):
        if False:
            return 10
        self.assertRaises(NotImplementedError, self.rel.has_permission, 'x')
        self.rel._permission_class = TestPermissionSet
        self.assertFalse(self.rel.has_permission('x'))
        self.rel.encoded_permissions = '+x,-y'
        self.assertTrue(self.rel.has_permission('x'))
        self.assertFalse(self.rel.has_permission('y'))
        self.rel.encoded_permissions = '+all'
        self.assertTrue(self.rel.has_permission('x'))
        self.assertTrue(self.rel.has_permission('y'))
        self.assertFalse(self.rel.has_permission('z'))

    def test_update_permissions(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(NotImplementedError, self.rel.update_permissions, x=True)
        self.rel._permission_class = TestPermissionSet
        self.rel.update_permissions(x=True, y=False)
        self.assertEquals('+x,-y', self.rel.encoded_permissions)
        self.rel.update_permissions(x=None)
        self.assertEquals('-y', self.rel.encoded_permissions)
        self.rel.update_permissions(y=None, z=None)
        self.assertEquals('', self.rel.encoded_permissions)
        self.rel.update_permissions(x=True, y=False, all=True)
        self.assertEquals('+all', self.rel.encoded_permissions)

    def test_set_permissions(self):
        if False:
            return 10
        self.rel.set_permissions(PermissionSet(x=True, y=False))
        self.assertEquals('+x,-y', self.rel.encoded_permissions)

    def test_is_superuser(self):
        if False:
            while True:
                i = 10
        self.assertRaises(NotImplementedError, self.rel.is_superuser)
        self.rel._permission_class = TestPermissionSet
        self.assertFalse(self.rel.is_superuser())
        self.rel.encoded_permissions = '+all'
        self.assertTrue(self.rel.is_superuser())

class IsValidNameTest(unittest.TestCase):

    def test_empty(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(Subreddit.is_valid_name(None))

    def test_short(self):
        if False:
            print('Hello World!')
        self.assertTrue(Subreddit.is_valid_name('aaa'))

    def test_too_short(self):
        if False:
            while True:
                i = 10
        self.assertFalse(Subreddit.is_valid_name('aa'))

    def test_long(self):
        if False:
            while True:
                i = 10
        self.assertTrue(Subreddit.is_valid_name('aaaaaaaaaaaaaaaaaaaaa'))

    def test_too_long(self):
        if False:
            print('Hello World!')
        self.assertFalse(Subreddit.is_valid_name('aaaaaaaaaaaaaaaaaaaaaa'))

    def test_underscore(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(Subreddit.is_valid_name('a_a'))

    def test_leading_underscore(self):
        if False:
            return 10
        self.assertFalse(Subreddit.is_valid_name('_aa'))

    def test_capitals(self):
        if False:
            return 10
        self.assertTrue(Subreddit.is_valid_name('AZA'))

    def test_numerics(self):
        if False:
            return 10
        self.assertTrue(Subreddit.is_valid_name('090'))

class ByNameTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.cache = MagicMock()
        g.gencache = self.cache
        self.subreddit_byID = MagicMock()
        Subreddit._byID = self.subreddit_byID
        self.subreddit_query = MagicMock()
        Subreddit._query = self.subreddit_query

    def testSingleCached(self):
        if False:
            for i in range(10):
                print('nop')
        subreddit = Subreddit(id=1, name='exists')
        self.cache.get_multi.return_value = {'exists': subreddit._id}
        self.subreddit_byID.return_value = [subreddit]
        ret = Subreddit._by_name('exists')
        self.assertEqual(ret, subreddit)
        self.assertEqual(self.subreddit_query.call_count, 0)

    def testSingleFromDB(self):
        if False:
            i = 10
            return i + 15
        subreddit = Subreddit(id=1, name='exists')
        self.cache.get_multi.return_value = {}
        self.subreddit_query.return_value = [subreddit]
        self.subreddit_byID.return_value = [subreddit]
        ret = Subreddit._by_name('exists')
        self.assertEqual(ret, subreddit)
        self.assertEqual(self.cache.set_multi.call_count, 1)

    def testSingleNotFound(self):
        if False:
            return 10
        self.cache.get_multi.return_value = {}
        self.subreddit_query.return_value = []
        with self.assertRaises(NotFound):
            Subreddit._by_name('doesnotexist')

    def testSingleInvalid(self):
        if False:
            return 10
        with self.assertRaises(NotFound):
            Subreddit._by_name('_illegalunderscore')
        self.assertEqual(self.cache.get_multi.call_count, 0)
        self.assertEqual(self.subreddit_query.call_count, 0)

    def testMultiCached(self):
        if False:
            while True:
                i = 10
        srs = [Subreddit(id=1, name='exists'), Subreddit(id=2, name='also')]
        self.cache.get_multi.return_value = {sr.name: sr._id for sr in srs}
        self.subreddit_byID.return_value = srs
        ret = Subreddit._by_name(['exists', 'also'])
        self.assertEqual(ret, {sr.name: sr for sr in srs})
        self.assertEqual(self.subreddit_query.call_count, 0)

    def testMultiCacheMissesAllExist(self):
        if False:
            i = 10
            return i + 15
        srs = [Subreddit(id=1, name='exists'), Subreddit(id=2, name='also')]
        self.cache.get_multi.return_value = {}
        self.subreddit_query.return_value = srs
        self.subreddit_byID.return_value = srs
        ret = Subreddit._by_name(['exists', 'also'])
        self.assertEqual(ret, {sr.name: sr for sr in srs})
        self.assertEqual(self.cache.get_multi.call_count, 1)
        self.assertEqual(self.subreddit_query.call_count, 1)

    def testMultiSomeDontExist(self):
        if False:
            return 10
        sr = Subreddit(id=1, name='exists')
        self.cache.get_multi.return_value = {sr.name: sr._id}
        self.subreddit_query.return_value = []
        self.subreddit_byID.return_value = [sr]
        ret = Subreddit._by_name(['exists', 'doesnt'])
        self.assertEqual(ret, {sr.name: sr})
        self.assertEqual(self.cache.get_multi.call_count, 1)
        self.assertEqual(self.subreddit_query.call_count, 1)

    def testMultiSomeInvalid(self):
        if False:
            i = 10
            return i + 15
        sr = Subreddit(id=1, name='exists')
        self.cache.get_multi.return_value = {sr.name: sr._id}
        self.subreddit_query.return_value = []
        self.subreddit_byID.return_value = [sr]
        ret = Subreddit._by_name(['exists', '_illegalunderscore'])
        self.assertEqual(ret, {sr.name: sr})
        self.assertEqual(self.cache.get_multi.call_count, 1)
        self.assertEqual(self.subreddit_query.call_count, 0)

    def testForceUpdate(self):
        if False:
            for i in range(10):
                print('nop')
        sr = Subreddit(id=1, name='exists')
        self.cache.get_multi.return_value = {sr.name: sr._id}
        self.subreddit_query.return_value = [sr]
        self.subreddit_byID.return_value = [sr]
        ret = Subreddit._by_name('exists', _update=True)
        self.assertEqual(ret, sr)
        self.cache.set_multi.assert_called_once_with(keys={sr.name: sr._id}, prefix='srid:', time=43200)

    def testCacheNegativeResults(self):
        if False:
            print('Hello World!')
        self.cache.get_multi.return_value = {}
        self.subreddit_query.return_value = []
        self.subreddit_byID.return_value = []
        with self.assertRaises(NotFound):
            Subreddit._by_name('doesnotexist')
        self.cache.set_multi.assert_called_once_with(keys={'doesnotexist': Subreddit.SRNAME_NOTFOUND}, prefix='srid:', time=43200)

    def testExcludeNegativeLookups(self):
        if False:
            return 10
        self.cache.get_multi.return_value = {'doesnotexist': Subreddit.SRNAME_NOTFOUND}
        with self.assertRaises(NotFound):
            Subreddit._by_name('doesnotexist')
        self.assertEqual(self.subreddit_query.call_count, 0)
        self.assertEqual(self.subreddit_byID.call_count, 0)
        self.assertEqual(self.cache.set_multi.call_count, 0)
if __name__ == '__main__':
    unittest.main()