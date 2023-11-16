"""Tests for the IdentityMap class."""
import bzrlib.errors as errors
from bzrlib.tests import TestCase
import bzrlib.identitymap as identitymap

class TestIdentityMap(TestCase):

    def test_symbols(self):
        if False:
            return 10
        from bzrlib.identitymap import IdentityMap

    def test_construct(self):
        if False:
            return 10
        identitymap.IdentityMap()

    def test_add_weave(self):
        if False:
            while True:
                i = 10
        map = identitymap.IdentityMap()
        weave = 'foo'
        map.add_weave('id', weave)
        self.assertEqual(weave, map.find_weave('id'))

    def test_double_add_weave(self):
        if False:
            while True:
                i = 10
        map = identitymap.IdentityMap()
        weave = 'foo'
        map.add_weave('id', weave)
        self.assertRaises(errors.BzrError, map.add_weave, 'id', weave)
        self.assertEqual(weave, map.find_weave('id'))

    def test_remove_object(self):
        if False:
            while True:
                i = 10
        map = identitymap.IdentityMap()
        weave = 'foo'
        map.add_weave('id', weave)
        map.remove_object(weave)
        map.add_weave('id', weave)

class TestNullIdentityMap(TestCase):

    def test_symbols(self):
        if False:
            for i in range(10):
                print('nop')
        from bzrlib.identitymap import NullIdentityMap

    def test_construct(self):
        if False:
            print('Hello World!')
        identitymap.NullIdentityMap()

    def test_add_weave(self):
        if False:
            i = 10
            return i + 15
        map = identitymap.NullIdentityMap()
        weave = 'foo'
        map.add_weave('id', weave)
        self.assertEqual(None, map.find_weave('id'))

    def test_double_add_weave(self):
        if False:
            for i in range(10):
                print('nop')
        map = identitymap.NullIdentityMap()
        weave = 'foo'
        map.add_weave('id', weave)
        map.add_weave('id', weave)
        self.assertEqual(None, map.find_weave('id'))

    def test_null_identity_map_has_no_remove(self):
        if False:
            i = 10
            return i + 15
        map = identitymap.NullIdentityMap()
        self.assertEqual(None, getattr(map, 'remove_object', None))