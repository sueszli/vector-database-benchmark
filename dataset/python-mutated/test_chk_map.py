"""Tests for maps built on a CHK versionedfiles facility."""
from bzrlib import chk_map, errors, groupcompress, osutils, tests
from bzrlib.chk_map import CHKMap, InternalNode, LeafNode, Node
from bzrlib.static_tuple import StaticTuple

class TestNode(tests.TestCase):

    def assertCommonPrefix(self, expected_common, prefix, key):
        if False:
            i = 10
            return i + 15
        common = Node.common_prefix(prefix, key)
        self.assertTrue(len(common) <= len(prefix))
        self.assertTrue(len(common) <= len(key))
        self.assertStartsWith(prefix, common)
        self.assertStartsWith(key, common)
        self.assertEqual(expected_common, common)

    def test_common_prefix(self):
        if False:
            while True:
                i = 10
        self.assertCommonPrefix('beg', 'beg', 'begin')

    def test_no_common_prefix(self):
        if False:
            return 10
        self.assertCommonPrefix('', 'begin', 'end')

    def test_equal(self):
        if False:
            return 10
        self.assertCommonPrefix('begin', 'begin', 'begin')

    def test_not_a_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertCommonPrefix('b', 'begin', 'b')

    def test_empty(self):
        if False:
            return 10
        self.assertCommonPrefix('', '', 'end')
        self.assertCommonPrefix('', 'begin', '')
        self.assertCommonPrefix('', '', '')

class TestCaseWithStore(tests.TestCaseWithMemoryTransport):

    def get_chk_bytes(self):
        if False:
            i = 10
            return i + 15
        factory = groupcompress.make_pack_factory(False, False, 1)
        self.chk_bytes = factory(self.get_transport())
        return self.chk_bytes

    def _get_map(self, a_dict, maximum_size=0, chk_bytes=None, key_width=1, search_key_func=None):
        if False:
            while True:
                i = 10
        if chk_bytes is None:
            chk_bytes = self.get_chk_bytes()
        root_key = CHKMap.from_dict(chk_bytes, a_dict, maximum_size=maximum_size, key_width=key_width, search_key_func=search_key_func)
        root_key2 = CHKMap._create_via_map(chk_bytes, a_dict, maximum_size=maximum_size, key_width=key_width, search_key_func=search_key_func)
        self.assertEqual(root_key, root_key2, 'CHKMap.from_dict() did not match CHKMap._create_via_map')
        chkmap = CHKMap(chk_bytes, root_key, search_key_func=search_key_func)
        return chkmap

    def read_bytes(self, chk_bytes, key):
        if False:
            print('Hello World!')
        stream = chk_bytes.get_record_stream([key], 'unordered', True)
        record = stream.next()
        if record.storage_kind == 'absent':
            self.fail('Store does not contain the key %s' % (key,))
        return record.get_bytes_as('fulltext')

    def to_dict(self, node, *args):
        if False:
            print('Hello World!')
        return dict(node.iteritems(*args))

class TestCaseWithExampleMaps(TestCaseWithStore):

    def get_chk_bytes(self):
        if False:
            i = 10
            return i + 15
        if getattr(self, '_chk_bytes', None) is None:
            self._chk_bytes = super(TestCaseWithExampleMaps, self).get_chk_bytes()
        return self._chk_bytes

    def get_map(self, a_dict, maximum_size=100, search_key_func=None):
        if False:
            while True:
                i = 10
        c_map = self._get_map(a_dict, maximum_size=maximum_size, chk_bytes=self.get_chk_bytes(), search_key_func=search_key_func)
        return c_map

    def make_root_only_map(self, search_key_func=None):
        if False:
            i = 10
            return i + 15
        return self.get_map({('aaa',): 'initial aaa content', ('abb',): 'initial abb content'}, search_key_func=search_key_func)

    def make_root_only_aaa_ddd_map(self, search_key_func=None):
        if False:
            i = 10
            return i + 15
        return self.get_map({('aaa',): 'initial aaa content', ('ddd',): 'initial ddd content'}, search_key_func=search_key_func)

    def make_one_deep_map(self, search_key_func=None):
        if False:
            for i in range(10):
                print('nop')
        return self.get_map({('aaa',): 'initial aaa content', ('abb',): 'initial abb content', ('ccc',): 'initial ccc content', ('ddd',): 'initial ddd content'}, search_key_func=search_key_func)

    def make_two_deep_map(self, search_key_func=None):
        if False:
            for i in range(10):
                print('nop')
        return self.get_map({('aaa',): 'initial aaa content', ('abb',): 'initial abb content', ('acc',): 'initial acc content', ('ace',): 'initial ace content', ('add',): 'initial add content', ('adh',): 'initial adh content', ('adl',): 'initial adl content', ('ccc',): 'initial ccc content', ('ddd',): 'initial ddd content'}, search_key_func=search_key_func)

    def make_one_deep_two_prefix_map(self, search_key_func=None):
        if False:
            return 10
        'Create a map with one internal node, but references are extra long.\n\n        Otherwise has similar content to make_two_deep_map.\n        '
        return self.get_map({('aaa',): 'initial aaa content', ('add',): 'initial add content', ('adh',): 'initial adh content', ('adl',): 'initial adl content'}, search_key_func=search_key_func)

    def make_one_deep_one_prefix_map(self, search_key_func=None):
        if False:
            for i in range(10):
                print('nop')
        'Create a map with one internal node, but references are extra long.\n\n        Similar to make_one_deep_two_prefix_map, except the split is at the\n        first char, rather than the second.\n        '
        return self.get_map({('add',): 'initial add content', ('adh',): 'initial adh content', ('adl',): 'initial adl content', ('bbb',): 'initial bbb content'}, search_key_func=search_key_func)

class TestTestCaseWithExampleMaps(TestCaseWithExampleMaps):
    """Actual tests for the provided examples."""

    def test_root_only_map_plain(self):
        if False:
            for i in range(10):
                print('nop')
        c_map = self.make_root_only_map()
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('abb',) 'initial abb content'\n", c_map._dump_tree())

    def test_root_only_map_16(self):
        if False:
            while True:
                i = 10
        c_map = self.make_root_only_map(search_key_func=chk_map._search_key_16)
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('abb',) 'initial abb content'\n", c_map._dump_tree())

    def test_one_deep_map_plain(self):
        if False:
            return 10
        c_map = self.make_one_deep_map()
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('abb',) 'initial abb content'\n  'c' LeafNode\n      ('ccc',) 'initial ccc content'\n  'd' LeafNode\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())

    def test_one_deep_map_16(self):
        if False:
            while True:
                i = 10
        c_map = self.make_one_deep_map(search_key_func=chk_map._search_key_16)
        self.assertEqualDiff("'' InternalNode\n  '2' LeafNode\n      ('ccc',) 'initial ccc content'\n  '4' LeafNode\n      ('abb',) 'initial abb content'\n  'F' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())

    def test_root_only_aaa_ddd_plain(self):
        if False:
            for i in range(10):
                print('nop')
        c_map = self.make_root_only_aaa_ddd_map()
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())

    def test_root_only_aaa_ddd_16(self):
        if False:
            i = 10
            return i + 15
        c_map = self.make_root_only_aaa_ddd_map(search_key_func=chk_map._search_key_16)
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())

    def test_two_deep_map_plain(self):
        if False:
            i = 10
            return i + 15
        c_map = self.make_two_deep_map()
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aa' LeafNode\n      ('aaa',) 'initial aaa content'\n    'ab' LeafNode\n      ('abb',) 'initial abb content'\n    'ac' LeafNode\n      ('acc',) 'initial acc content'\n      ('ace',) 'initial ace content'\n    'ad' LeafNode\n      ('add',) 'initial add content'\n      ('adh',) 'initial adh content'\n      ('adl',) 'initial adl content'\n  'c' LeafNode\n      ('ccc',) 'initial ccc content'\n  'd' LeafNode\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())

    def test_two_deep_map_16(self):
        if False:
            while True:
                i = 10
        c_map = self.make_two_deep_map(search_key_func=chk_map._search_key_16)
        self.assertEqualDiff("'' InternalNode\n  '2' LeafNode\n      ('acc',) 'initial acc content'\n      ('ccc',) 'initial ccc content'\n  '4' LeafNode\n      ('abb',) 'initial abb content'\n  'C' LeafNode\n      ('ace',) 'initial ace content'\n  'F' InternalNode\n    'F0' LeafNode\n      ('aaa',) 'initial aaa content'\n    'F3' LeafNode\n      ('adl',) 'initial adl content'\n    'F4' LeafNode\n      ('adh',) 'initial adh content'\n    'FB' LeafNode\n      ('ddd',) 'initial ddd content'\n    'FD' LeafNode\n      ('add',) 'initial add content'\n", c_map._dump_tree())

    def test_one_deep_two_prefix_map_plain(self):
        if False:
            print('Hello World!')
        c_map = self.make_one_deep_two_prefix_map()
        self.assertEqualDiff("'' InternalNode\n  'aa' LeafNode\n      ('aaa',) 'initial aaa content'\n  'ad' LeafNode\n      ('add',) 'initial add content'\n      ('adh',) 'initial adh content'\n      ('adl',) 'initial adl content'\n", c_map._dump_tree())

    def test_one_deep_two_prefix_map_16(self):
        if False:
            print('Hello World!')
        c_map = self.make_one_deep_two_prefix_map(search_key_func=chk_map._search_key_16)
        self.assertEqualDiff("'' InternalNode\n  'F0' LeafNode\n      ('aaa',) 'initial aaa content'\n  'F3' LeafNode\n      ('adl',) 'initial adl content'\n  'F4' LeafNode\n      ('adh',) 'initial adh content'\n  'FD' LeafNode\n      ('add',) 'initial add content'\n", c_map._dump_tree())

    def test_one_deep_one_prefix_map_plain(self):
        if False:
            while True:
                i = 10
        c_map = self.make_one_deep_one_prefix_map()
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('add',) 'initial add content'\n      ('adh',) 'initial adh content'\n      ('adl',) 'initial adl content'\n  'b' LeafNode\n      ('bbb',) 'initial bbb content'\n", c_map._dump_tree())

    def test_one_deep_one_prefix_map_16(self):
        if False:
            for i in range(10):
                print('nop')
        c_map = self.make_one_deep_one_prefix_map(search_key_func=chk_map._search_key_16)
        self.assertEqualDiff("'' InternalNode\n  '4' LeafNode\n      ('bbb',) 'initial bbb content'\n  'F' LeafNode\n      ('add',) 'initial add content'\n      ('adh',) 'initial adh content'\n      ('adl',) 'initial adl content'\n", c_map._dump_tree())

class TestMap(TestCaseWithStore):

    def assertHasABMap(self, chk_bytes):
        if False:
            while True:
                i = 10
        ab_leaf_bytes = 'chkleaf:\n0\n1\n1\na\n\x001\nb\n'
        ab_sha1 = osutils.sha_string(ab_leaf_bytes)
        self.assertEqual('90986195696b177c8895d48fdb4b7f2366f798a0', ab_sha1)
        root_key = ('sha1:' + ab_sha1,)
        self.assertEqual(ab_leaf_bytes, self.read_bytes(chk_bytes, root_key))
        return root_key

    def assertHasEmptyMap(self, chk_bytes):
        if False:
            return 10
        empty_leaf_bytes = 'chkleaf:\n0\n1\n0\n\n'
        empty_sha1 = osutils.sha_string(empty_leaf_bytes)
        self.assertEqual('8571e09bf1bcc5b9621ce31b3d4c93d6e9a1ed26', empty_sha1)
        root_key = ('sha1:' + empty_sha1,)
        self.assertEqual(empty_leaf_bytes, self.read_bytes(chk_bytes, root_key))
        return root_key

    def assertMapLayoutEqual(self, map_one, map_two):
        if False:
            while True:
                i = 10
        'Assert that the internal structure is identical between the maps.'
        map_one._ensure_root()
        node_one_stack = [map_one._root_node]
        map_two._ensure_root()
        node_two_stack = [map_two._root_node]
        while node_one_stack:
            node_one = node_one_stack.pop()
            node_two = node_two_stack.pop()
            if node_one.__class__ != node_two.__class__:
                self.assertEqualDiff(map_one._dump_tree(include_keys=True), map_two._dump_tree(include_keys=True))
            self.assertEqual(node_one._search_prefix, node_two._search_prefix)
            if isinstance(node_one, InternalNode):
                self.assertEqual(sorted(node_one._items.keys()), sorted(node_two._items.keys()))
                node_one_stack.extend([n for (n, _) in node_one._iter_nodes(map_one._store)])
                node_two_stack.extend([n for (n, _) in node_two._iter_nodes(map_two._store)])
            else:
                self.assertEqual(node_one._items, node_two._items)
        self.assertEqual([], node_two_stack)

    def assertCanonicalForm(self, chkmap):
        if False:
            for i in range(10):
                print('nop')
        "Assert that the chkmap is in 'canonical' form.\n\n        We do this by adding all of the key value pairs from scratch, both in\n        forward order and reverse order, and assert that the final tree layout\n        is identical.\n        "
        items = list(chkmap.iteritems())
        map_forward = chk_map.CHKMap(None, None)
        map_forward._root_node.set_maximum_size(chkmap._root_node.maximum_size)
        for (key, value) in items:
            map_forward.map(key, value)
        self.assertMapLayoutEqual(map_forward, chkmap)
        map_reverse = chk_map.CHKMap(None, None)
        map_reverse._root_node.set_maximum_size(chkmap._root_node.maximum_size)
        for (key, value) in reversed(items):
            map_reverse.map(key, value)
        self.assertMapLayoutEqual(map_reverse, chkmap)

    def test_assert_map_layout_equal(self):
        if False:
            while True:
                i = 10
        store = self.get_chk_bytes()
        map_one = CHKMap(store, None)
        map_one._root_node.set_maximum_size(20)
        map_two = CHKMap(store, None)
        map_two._root_node.set_maximum_size(20)
        self.assertMapLayoutEqual(map_one, map_two)
        map_one.map('aaa', 'value')
        self.assertRaises(AssertionError, self.assertMapLayoutEqual, map_one, map_two)
        map_two.map('aaa', 'value')
        self.assertMapLayoutEqual(map_one, map_two)
        map_one.map('aab', 'value')
        self.assertIsInstance(map_one._root_node, InternalNode)
        self.assertRaises(AssertionError, self.assertMapLayoutEqual, map_one, map_two)
        map_two.map('aab', 'value')
        self.assertMapLayoutEqual(map_one, map_two)
        map_one.map('aac', 'value')
        self.assertRaises(AssertionError, self.assertMapLayoutEqual, map_one, map_two)
        self.assertCanonicalForm(map_one)

    def test_from_dict_empty(self):
        if False:
            while True:
                i = 10
        chk_bytes = self.get_chk_bytes()
        root_key = CHKMap.from_dict(chk_bytes, {})
        expected_root_key = self.assertHasEmptyMap(chk_bytes)
        self.assertEqual(expected_root_key, root_key)

    def test_from_dict_ab(self):
        if False:
            return 10
        chk_bytes = self.get_chk_bytes()
        root_key = CHKMap.from_dict(chk_bytes, {'a': 'b'})
        expected_root_key = self.assertHasABMap(chk_bytes)
        self.assertEqual(expected_root_key, root_key)

    def test_apply_empty_ab(self):
        if False:
            for i in range(10):
                print('nop')
        chk_bytes = self.get_chk_bytes()
        root_key = CHKMap.from_dict(chk_bytes, {})
        chkmap = CHKMap(chk_bytes, root_key)
        new_root = chkmap.apply_delta([(None, 'a', 'b')])
        expected_root_key = self.assertHasABMap(chk_bytes)
        self.assertEqual(expected_root_key, new_root)
        self.assertEqual(new_root, chkmap._root_node._key)

    def test_apply_ab_empty(self):
        if False:
            i = 10
            return i + 15
        chk_bytes = self.get_chk_bytes()
        root_key = CHKMap.from_dict(chk_bytes, {('a',): 'b'})
        chkmap = CHKMap(chk_bytes, root_key)
        new_root = chkmap.apply_delta([(('a',), None, None)])
        expected_root_key = self.assertHasEmptyMap(chk_bytes)
        self.assertEqual(expected_root_key, new_root)
        self.assertEqual(new_root, chkmap._root_node._key)

    def test_apply_delete_to_internal_node(self):
        if False:
            print('Hello World!')
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(100)
        chkmap.map(('small',), 'value')
        chkmap.map(('little',), 'value')
        chkmap.map(('very-big',), 'x' * 100)
        self.assertIsInstance(chkmap._root_node, InternalNode)
        delta = [(('very-big',), None, None)]
        chkmap.apply_delta(delta)
        self.assertCanonicalForm(chkmap)
        self.assertIsInstance(chkmap._root_node, LeafNode)

    def test_apply_new_keys_must_be_new(self):
        if False:
            return 10
        chk_bytes = self.get_chk_bytes()
        root_key = CHKMap.from_dict(chk_bytes, {('a',): 'b'})
        chkmap = CHKMap(chk_bytes, root_key)
        self.assertRaises(errors.InconsistentDelta, chkmap.apply_delta, [(None, ('a',), 'b')])
        self.assertEqual(root_key, chkmap._root_node._key)

    def test_apply_delta_is_deterministic(self):
        if False:
            i = 10
            return i + 15
        chk_bytes = self.get_chk_bytes()
        chkmap1 = CHKMap(chk_bytes, None)
        chkmap1._root_node.set_maximum_size(10)
        chkmap1.apply_delta([(None, ('aaa',), 'common'), (None, ('bba',), 'target2'), (None, ('bbb',), 'common')])
        root_key1 = chkmap1._save()
        self.assertCanonicalForm(chkmap1)
        chkmap2 = CHKMap(chk_bytes, None)
        chkmap2._root_node.set_maximum_size(10)
        chkmap2.apply_delta([(None, ('bbb',), 'common'), (None, ('bba',), 'target2'), (None, ('aaa',), 'common')])
        root_key2 = chkmap2._save()
        self.assertEqualDiff(chkmap1._dump_tree(include_keys=True), chkmap2._dump_tree(include_keys=True))
        self.assertEqual(root_key1, root_key2)
        self.assertCanonicalForm(chkmap2)

    def test_stable_splitting(self):
        if False:
            while True:
                i = 10
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(35)
        chkmap.map(('aaa',), 'v')
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n", chkmap._dump_tree())
        chkmap.map(('aab',), 'v')
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n      ('aab',) 'v'\n", chkmap._dump_tree())
        self.assertCanonicalForm(chkmap)
        chkmap.map(('aac',), 'v')
        self.assertEqualDiff("'' InternalNode\n  'aaa' LeafNode\n      ('aaa',) 'v'\n  'aab' LeafNode\n      ('aab',) 'v'\n  'aac' LeafNode\n      ('aac',) 'v'\n", chkmap._dump_tree())
        self.assertCanonicalForm(chkmap)
        chkmap.map(('bbb',), 'v')
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'v'\n    'aab' LeafNode\n      ('aab',) 'v'\n    'aac' LeafNode\n      ('aac',) 'v'\n  'b' LeafNode\n      ('bbb',) 'v'\n", chkmap._dump_tree())
        self.assertCanonicalForm(chkmap)

    def test_map_splits_with_longer_key(self):
        if False:
            i = 10
            return i + 15
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(10)
        chkmap.map(('aaa',), 'v')
        chkmap.map(('aaaa',), 'v')
        self.assertCanonicalForm(chkmap)
        self.assertIsInstance(chkmap._root_node, InternalNode)

    def test_with_linefeed_in_key(self):
        if False:
            i = 10
            return i + 15
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(10)
        chkmap.map(('a\ra',), 'val1')
        chkmap.map(('a\rb',), 'val2')
        chkmap.map(('ac',), 'val3')
        self.assertCanonicalForm(chkmap)
        self.assertEqualDiff("'' InternalNode\n  'a\\r' InternalNode\n    'a\\ra' LeafNode\n      ('a\\ra',) 'val1'\n    'a\\rb' LeafNode\n      ('a\\rb',) 'val2'\n  'ac' LeafNode\n      ('ac',) 'val3'\n", chkmap._dump_tree())
        root_key = chkmap._save()
        chkmap = CHKMap(store, root_key)
        self.assertEqualDiff("'' InternalNode\n  'a\\r' InternalNode\n    'a\\ra' LeafNode\n      ('a\\ra',) 'val1'\n    'a\\rb' LeafNode\n      ('a\\rb',) 'val2'\n  'ac' LeafNode\n      ('ac',) 'val3'\n", chkmap._dump_tree())

    def test_deep_splitting(self):
        if False:
            print('Hello World!')
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(40)
        chkmap.map(('aaaaaaaa',), 'v')
        chkmap.map(('aaaaabaa',), 'v')
        self.assertEqualDiff("'' LeafNode\n      ('aaaaaaaa',) 'v'\n      ('aaaaabaa',) 'v'\n", chkmap._dump_tree())
        chkmap.map(('aaabaaaa',), 'v')
        chkmap.map(('aaababaa',), 'v')
        self.assertEqualDiff("'' InternalNode\n  'aaaa' LeafNode\n      ('aaaaaaaa',) 'v'\n      ('aaaaabaa',) 'v'\n  'aaab' LeafNode\n      ('aaabaaaa',) 'v'\n      ('aaababaa',) 'v'\n", chkmap._dump_tree())
        chkmap.map(('aaabacaa',), 'v')
        chkmap.map(('aaabadaa',), 'v')
        self.assertEqualDiff("'' InternalNode\n  'aaaa' LeafNode\n      ('aaaaaaaa',) 'v'\n      ('aaaaabaa',) 'v'\n  'aaab' InternalNode\n    'aaabaa' LeafNode\n      ('aaabaaaa',) 'v'\n    'aaabab' LeafNode\n      ('aaababaa',) 'v'\n    'aaabac' LeafNode\n      ('aaabacaa',) 'v'\n    'aaabad' LeafNode\n      ('aaabadaa',) 'v'\n", chkmap._dump_tree())
        chkmap.map(('aaababba',), 'val')
        chkmap.map(('aaababca',), 'val')
        self.assertEqualDiff("'' InternalNode\n  'aaaa' LeafNode\n      ('aaaaaaaa',) 'v'\n      ('aaaaabaa',) 'v'\n  'aaab' InternalNode\n    'aaabaa' LeafNode\n      ('aaabaaaa',) 'v'\n    'aaabab' InternalNode\n      'aaababa' LeafNode\n      ('aaababaa',) 'v'\n      'aaababb' LeafNode\n      ('aaababba',) 'val'\n      'aaababc' LeafNode\n      ('aaababca',) 'val'\n    'aaabac' LeafNode\n      ('aaabacaa',) 'v'\n    'aaabad' LeafNode\n      ('aaabadaa',) 'v'\n", chkmap._dump_tree())
        chkmap.map(('aaabDaaa',), 'v')
        self.assertEqualDiff("'' InternalNode\n  'aaaa' LeafNode\n      ('aaaaaaaa',) 'v'\n      ('aaaaabaa',) 'v'\n  'aaab' InternalNode\n    'aaabD' LeafNode\n      ('aaabDaaa',) 'v'\n    'aaaba' InternalNode\n      'aaabaa' LeafNode\n      ('aaabaaaa',) 'v'\n      'aaabab' InternalNode\n        'aaababa' LeafNode\n      ('aaababaa',) 'v'\n        'aaababb' LeafNode\n      ('aaababba',) 'val'\n        'aaababc' LeafNode\n      ('aaababca',) 'val'\n      'aaabac' LeafNode\n      ('aaabacaa',) 'v'\n      'aaabad' LeafNode\n      ('aaabadaa',) 'v'\n", chkmap._dump_tree())

    def test_map_collapses_if_size_changes(self):
        if False:
            print('Hello World!')
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(35)
        chkmap.map(('aaa',), 'v')
        chkmap.map(('aab',), 'very long value that splits')
        self.assertEqualDiff("'' InternalNode\n  'aaa' LeafNode\n      ('aaa',) 'v'\n  'aab' LeafNode\n      ('aab',) 'very long value that splits'\n", chkmap._dump_tree())
        self.assertCanonicalForm(chkmap)
        chkmap.map(('aab',), 'v')
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n      ('aab',) 'v'\n", chkmap._dump_tree())
        self.assertCanonicalForm(chkmap)

    def test_map_double_deep_collapses(self):
        if False:
            for i in range(10):
                print('nop')
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(40)
        chkmap.map(('aaa',), 'v')
        chkmap.map(('aab',), 'very long value that splits')
        chkmap.map(('abc',), 'v')
        self.assertEqualDiff("'' InternalNode\n  'aa' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'v'\n    'aab' LeafNode\n      ('aab',) 'very long value that splits'\n  'ab' LeafNode\n      ('abc',) 'v'\n", chkmap._dump_tree())
        chkmap.map(('aab',), 'v')
        self.assertCanonicalForm(chkmap)
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n      ('aab',) 'v'\n      ('abc',) 'v'\n", chkmap._dump_tree())

    def test_stable_unmap(self):
        if False:
            return 10
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(35)
        chkmap.map(('aaa',), 'v')
        chkmap.map(('aab',), 'v')
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n      ('aab',) 'v'\n", chkmap._dump_tree())
        chkmap.map(('aac',), 'v')
        self.assertEqualDiff("'' InternalNode\n  'aaa' LeafNode\n      ('aaa',) 'v'\n  'aab' LeafNode\n      ('aab',) 'v'\n  'aac' LeafNode\n      ('aac',) 'v'\n", chkmap._dump_tree())
        self.assertCanonicalForm(chkmap)
        chkmap.unmap(('aac',))
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n      ('aab',) 'v'\n", chkmap._dump_tree())
        self.assertCanonicalForm(chkmap)

    def test_unmap_double_deep(self):
        if False:
            print('Hello World!')
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(40)
        chkmap.map(('aaa',), 'v')
        chkmap.map(('aaab',), 'v')
        chkmap.map(('aab',), 'very long value')
        chkmap.map(('abc',), 'v')
        self.assertEqualDiff("'' InternalNode\n  'aa' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'v'\n      ('aaab',) 'v'\n    'aab' LeafNode\n      ('aab',) 'very long value'\n  'ab' LeafNode\n      ('abc',) 'v'\n", chkmap._dump_tree())
        chkmap.unmap(('aab',))
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n      ('aaab',) 'v'\n      ('abc',) 'v'\n", chkmap._dump_tree())

    def test_unmap_double_deep_non_empty_leaf(self):
        if False:
            print('Hello World!')
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(40)
        chkmap.map(('aaa',), 'v')
        chkmap.map(('aab',), 'long value')
        chkmap.map(('aabb',), 'v')
        chkmap.map(('abc',), 'v')
        self.assertEqualDiff("'' InternalNode\n  'aa' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'v'\n    'aab' LeafNode\n      ('aab',) 'long value'\n      ('aabb',) 'v'\n  'ab' LeafNode\n      ('abc',) 'v'\n", chkmap._dump_tree())
        chkmap.unmap(('aab',))
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n      ('aabb',) 'v'\n      ('abc',) 'v'\n", chkmap._dump_tree())

    def test_unmap_with_known_internal_node_doesnt_page(self):
        if False:
            for i in range(10):
                print('nop')
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(30)
        chkmap.map(('aaa',), 'v')
        chkmap.map(('aab',), 'v')
        chkmap.map(('aac',), 'v')
        chkmap.map(('abc',), 'v')
        chkmap.map(('acd',), 'v')
        self.assertEqualDiff("'' InternalNode\n  'aa' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'v'\n    'aab' LeafNode\n      ('aab',) 'v'\n    'aac' LeafNode\n      ('aac',) 'v'\n  'ab' LeafNode\n      ('abc',) 'v'\n  'ac' LeafNode\n      ('acd',) 'v'\n", chkmap._dump_tree())
        chkmap = CHKMap(store, chkmap._save())
        chkmap.map(('aad',), 'v')
        self.assertIsInstance(chkmap._root_node._items['aa'], InternalNode)
        self.assertIsInstance(chkmap._root_node._items['ab'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['ac'], StaticTuple)
        chkmap.unmap(('acd',))
        self.assertIsInstance(chkmap._root_node._items['aa'], InternalNode)
        self.assertIsInstance(chkmap._root_node._items['ab'], StaticTuple)

    def test_unmap_without_fitting_doesnt_page_in(self):
        if False:
            print('Hello World!')
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(20)
        chkmap.map(('aaa',), 'v')
        chkmap.map(('aab',), 'v')
        self.assertEqualDiff("'' InternalNode\n  'aaa' LeafNode\n      ('aaa',) 'v'\n  'aab' LeafNode\n      ('aab',) 'v'\n", chkmap._dump_tree())
        chkmap = CHKMap(store, chkmap._save())
        chkmap.map(('aac',), 'v')
        chkmap.map(('aad',), 'v')
        chkmap.map(('aae',), 'v')
        chkmap.map(('aaf',), 'v')
        self.assertIsInstance(chkmap._root_node._items['aaa'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aab'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aac'], LeafNode)
        self.assertIsInstance(chkmap._root_node._items['aad'], LeafNode)
        self.assertIsInstance(chkmap._root_node._items['aae'], LeafNode)
        self.assertIsInstance(chkmap._root_node._items['aaf'], LeafNode)
        chkmap.unmap(('aaf',))
        self.assertIsInstance(chkmap._root_node._items['aaa'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aab'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aac'], LeafNode)
        self.assertIsInstance(chkmap._root_node._items['aad'], LeafNode)
        self.assertIsInstance(chkmap._root_node._items['aae'], LeafNode)

    def test_unmap_pages_in_if_necessary(self):
        if False:
            while True:
                i = 10
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(30)
        chkmap.map(('aaa',), 'val')
        chkmap.map(('aab',), 'val')
        chkmap.map(('aac',), 'val')
        self.assertEqualDiff("'' InternalNode\n  'aaa' LeafNode\n      ('aaa',) 'val'\n  'aab' LeafNode\n      ('aab',) 'val'\n  'aac' LeafNode\n      ('aac',) 'val'\n", chkmap._dump_tree())
        root_key = chkmap._save()
        chkmap = CHKMap(store, root_key)
        chkmap.map(('aad',), 'v')
        self.assertIsInstance(chkmap._root_node._items['aaa'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aab'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aac'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aad'], LeafNode)
        chk_map.clear_cache()
        chkmap.unmap(('aad',))
        self.assertIsInstance(chkmap._root_node._items['aaa'], LeafNode)
        self.assertIsInstance(chkmap._root_node._items['aab'], LeafNode)
        self.assertIsInstance(chkmap._root_node._items['aac'], LeafNode)

    def test_unmap_pages_in_from_page_cache(self):
        if False:
            for i in range(10):
                print('nop')
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(30)
        chkmap.map(('aaa',), 'val')
        chkmap.map(('aab',), 'val')
        chkmap.map(('aac',), 'val')
        root_key = chkmap._save()
        chkmap = CHKMap(store, root_key)
        chkmap.map(('aad',), 'val')
        self.assertEqualDiff("'' InternalNode\n  'aaa' LeafNode\n      ('aaa',) 'val'\n  'aab' LeafNode\n      ('aab',) 'val'\n  'aac' LeafNode\n      ('aac',) 'val'\n  'aad' LeafNode\n      ('aad',) 'val'\n", chkmap._dump_tree())
        chkmap = CHKMap(store, root_key)
        chkmap.map(('aad',), 'v')
        self.assertIsInstance(chkmap._root_node._items['aaa'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aab'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aac'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aad'], LeafNode)
        aab_key = chkmap._root_node._items['aab']
        aab_bytes = chk_map._get_cache()[aab_key]
        aac_key = chkmap._root_node._items['aac']
        aac_bytes = chk_map._get_cache()[aac_key]
        chk_map.clear_cache()
        chk_map._get_cache()[aab_key] = aab_bytes
        chk_map._get_cache()[aac_key] = aac_bytes
        chkmap.unmap(('aad',))
        self.assertIsInstance(chkmap._root_node._items['aaa'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aab'], LeafNode)
        self.assertIsInstance(chkmap._root_node._items['aac'], LeafNode)

    def test_unmap_uses_existing_items(self):
        if False:
            print('Hello World!')
        store = self.get_chk_bytes()
        chkmap = CHKMap(store, None)
        chkmap._root_node.set_maximum_size(30)
        chkmap.map(('aaa',), 'val')
        chkmap.map(('aab',), 'val')
        chkmap.map(('aac',), 'val')
        root_key = chkmap._save()
        chkmap = CHKMap(store, root_key)
        chkmap.map(('aad',), 'val')
        chkmap.map(('aae',), 'val')
        chkmap.map(('aaf',), 'val')
        self.assertIsInstance(chkmap._root_node._items['aaa'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aab'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aac'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aad'], LeafNode)
        self.assertIsInstance(chkmap._root_node._items['aae'], LeafNode)
        self.assertIsInstance(chkmap._root_node._items['aaf'], LeafNode)
        chkmap.unmap(('aad',))
        self.assertIsInstance(chkmap._root_node._items['aaa'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aab'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aac'], StaticTuple)
        self.assertIsInstance(chkmap._root_node._items['aae'], LeafNode)
        self.assertIsInstance(chkmap._root_node._items['aaf'], LeafNode)

    def test_iter_changes_empty_ab(self):
        if False:
            print('Hello World!')
        basis = self._get_map({}, maximum_size=10)
        target = self._get_map({('a',): 'content here', ('b',): 'more content'}, chk_bytes=basis._store, maximum_size=10)
        self.assertEqual([(('a',), None, 'content here'), (('b',), None, 'more content')], sorted(list(target.iter_changes(basis))))

    def test_iter_changes_ab_empty(self):
        if False:
            for i in range(10):
                print('nop')
        basis = self._get_map({('a',): 'content here', ('b',): 'more content'}, maximum_size=10)
        target = self._get_map({}, chk_bytes=basis._store, maximum_size=10)
        self.assertEqual([(('a',), 'content here', None), (('b',), 'more content', None)], sorted(list(target.iter_changes(basis))))

    def test_iter_changes_empty_empty_is_empty(self):
        if False:
            while True:
                i = 10
        basis = self._get_map({}, maximum_size=10)
        target = self._get_map({}, chk_bytes=basis._store, maximum_size=10)
        self.assertEqual([], sorted(list(target.iter_changes(basis))))

    def test_iter_changes_ab_ab_is_empty(self):
        if False:
            print('Hello World!')
        basis = self._get_map({('a',): 'content here', ('b',): 'more content'}, maximum_size=10)
        target = self._get_map({('a',): 'content here', ('b',): 'more content'}, chk_bytes=basis._store, maximum_size=10)
        self.assertEqual([], sorted(list(target.iter_changes(basis))))

    def test_iter_changes_ab_ab_nodes_not_loaded(self):
        if False:
            i = 10
            return i + 15
        basis = self._get_map({('a',): 'content here', ('b',): 'more content'}, maximum_size=10)
        target = self._get_map({('a',): 'content here', ('b',): 'more content'}, chk_bytes=basis._store, maximum_size=10)
        list(target.iter_changes(basis))
        self.assertIsInstance(target._root_node, StaticTuple)
        self.assertIsInstance(basis._root_node, StaticTuple)

    def test_iter_changes_ab_ab_changed_values_shown(self):
        if False:
            i = 10
            return i + 15
        basis = self._get_map({('a',): 'content here', ('b',): 'more content'}, maximum_size=10)
        target = self._get_map({('a',): 'content here', ('b',): 'different content'}, chk_bytes=basis._store, maximum_size=10)
        result = sorted(list(target.iter_changes(basis)))
        self.assertEqual([(('b',), 'more content', 'different content')], result)

    def test_iter_changes_mixed_node_length(self):
        if False:
            for i in range(10):
                print('nop')
        basis_dict = {('aaa',): 'foo bar', ('aab',): 'common altered a', ('b',): 'foo bar b'}
        target_dict = {('aaa',): 'foo bar', ('aab',): 'common altered b', ('at',): 'foo bar t'}
        changes = [(('aab',), 'common altered a', 'common altered b'), (('at',), None, 'foo bar t'), (('b',), 'foo bar b', None)]
        basis = self._get_map(basis_dict, maximum_size=10)
        target = self._get_map(target_dict, maximum_size=10, chk_bytes=basis._store)
        self.assertEqual(changes, sorted(list(target.iter_changes(basis))))

    def test_iter_changes_common_pages_not_loaded(self):
        if False:
            return 10
        basis_dict = {('aaa',): 'foo bar', ('aab',): 'common altered a', ('b',): 'foo bar b'}
        target_dict = {('aaa',): 'foo bar', ('aab',): 'common altered b', ('at',): 'foo bar t'}
        basis = self._get_map(basis_dict, maximum_size=10)
        target = self._get_map(target_dict, maximum_size=10, chk_bytes=basis._store)
        basis_get = basis._store.get_record_stream

        def get_record_stream(keys, order, fulltext):
            if False:
                i = 10
                return i + 15
            if ('sha1:1adf7c0d1b9140ab5f33bb64c6275fa78b1580b7',) in keys:
                raise AssertionError("'aaa' pointer was followed %r" % keys)
            return basis_get(keys, order, fulltext)
        basis._store.get_record_stream = get_record_stream
        result = sorted(list(target.iter_changes(basis)))
        for change in result:
            if change[0] == ('aaa',):
                self.fail('Found unexpected change: %s' % change)

    def test_iter_changes_unchanged_keys_in_multi_key_leafs_ignored(self):
        if False:
            for i in range(10):
                print('nop')
        basis_dict = {('aaa',): 'foo bar', ('aab',): 'common altered a', ('b',): 'foo bar b'}
        target_dict = {('aaa',): 'foo bar', ('aab',): 'common altered b', ('at',): 'foo bar t'}
        changes = [(('aab',), 'common altered a', 'common altered b'), (('at',), None, 'foo bar t'), (('b',), 'foo bar b', None)]
        basis = self._get_map(basis_dict)
        target = self._get_map(target_dict, chk_bytes=basis._store)
        self.assertEqual(changes, sorted(list(target.iter_changes(basis))))

    def test_iteritems_empty(self):
        if False:
            i = 10
            return i + 15
        chk_bytes = self.get_chk_bytes()
        root_key = CHKMap.from_dict(chk_bytes, {})
        chkmap = CHKMap(chk_bytes, root_key)
        self.assertEqual([], list(chkmap.iteritems()))

    def test_iteritems_two_items(self):
        if False:
            return 10
        chk_bytes = self.get_chk_bytes()
        root_key = CHKMap.from_dict(chk_bytes, {'a': 'content here', 'b': 'more content'})
        chkmap = CHKMap(chk_bytes, root_key)
        self.assertEqual([(('a',), 'content here'), (('b',), 'more content')], sorted(list(chkmap.iteritems())))

    def test_iteritems_selected_one_of_two_items(self):
        if False:
            return 10
        chkmap = self._get_map({('a',): 'content here', ('b',): 'more content'})
        self.assertEqual({('a',): 'content here'}, self.to_dict(chkmap, [('a',)]))

    def test_iteritems_keys_prefixed_by_2_width_nodes(self):
        if False:
            return 10
        chkmap = self._get_map({('a', 'a'): 'content here', ('a', 'b'): 'more content', ('b', ''): 'boring content'}, maximum_size=10, key_width=2)
        self.assertEqual({('a', 'a'): 'content here', ('a', 'b'): 'more content'}, self.to_dict(chkmap, [('a',)]))

    def test_iteritems_keys_prefixed_by_2_width_nodes_hashed(self):
        if False:
            i = 10
            return i + 15
        search_key_func = chk_map.search_key_registry.get('hash-16-way')
        self.assertEqual('E8B7BE43\x00E8B7BE43', search_key_func(StaticTuple('a', 'a')))
        self.assertEqual('E8B7BE43\x0071BEEFF9', search_key_func(StaticTuple('a', 'b')))
        self.assertEqual('71BEEFF9\x0000000000', search_key_func(StaticTuple('b', '')))
        chkmap = self._get_map({('a', 'a'): 'content here', ('a', 'b'): 'more content', ('b', ''): 'boring content'}, maximum_size=10, key_width=2, search_key_func=search_key_func)
        self.assertEqual({('a', 'a'): 'content here', ('a', 'b'): 'more content'}, self.to_dict(chkmap, [('a',)]))

    def test_iteritems_keys_prefixed_by_2_width_one_leaf(self):
        if False:
            for i in range(10):
                print('nop')
        chkmap = self._get_map({('a', 'a'): 'content here', ('a', 'b'): 'more content', ('b', ''): 'boring content'}, key_width=2)
        self.assertEqual({('a', 'a'): 'content here', ('a', 'b'): 'more content'}, self.to_dict(chkmap, [('a',)]))

    def test___len__empty(self):
        if False:
            while True:
                i = 10
        chkmap = self._get_map({})
        self.assertEqual(0, len(chkmap))

    def test___len__2(self):
        if False:
            return 10
        chkmap = self._get_map({('foo',): 'bar', ('gam',): 'quux'})
        self.assertEqual(2, len(chkmap))

    def test_max_size_100_bytes_new(self):
        if False:
            while True:
                i = 10
        chkmap = self._get_map({('k1' * 50,): 'v1', ('k2' * 50,): 'v2'}, maximum_size=100)
        chkmap._ensure_root()
        self.assertEqual(100, chkmap._root_node.maximum_size)
        self.assertEqual(1, chkmap._root_node._key_width)
        self.assertEqual(2, len(chkmap._root_node._items))
        self.assertEqual('k', chkmap._root_node._compute_search_prefix())
        nodes = sorted(chkmap._root_node._items.items())
        ptr1 = nodes[0]
        ptr2 = nodes[1]
        self.assertEqual('k1', ptr1[0])
        self.assertEqual('k2', ptr2[0])
        node1 = chk_map._deserialise(chkmap._read_bytes(ptr1[1]), ptr1[1], None)
        self.assertIsInstance(node1, LeafNode)
        self.assertEqual(1, len(node1))
        self.assertEqual({('k1' * 50,): 'v1'}, self.to_dict(node1, chkmap._store))
        node2 = chk_map._deserialise(chkmap._read_bytes(ptr2[1]), ptr2[1], None)
        self.assertIsInstance(node2, LeafNode)
        self.assertEqual(1, len(node2))
        self.assertEqual({('k2' * 50,): 'v2'}, self.to_dict(node2, chkmap._store))
        self.assertEqual(2, len(chkmap))
        self.assertEqual({('k1' * 50,): 'v1', ('k2' * 50,): 'v2'}, self.to_dict(chkmap))

    def test_init_root_is_LeafNode_new(self):
        if False:
            while True:
                i = 10
        chk_bytes = self.get_chk_bytes()
        chkmap = CHKMap(chk_bytes, None)
        self.assertIsInstance(chkmap._root_node, LeafNode)
        self.assertEqual({}, self.to_dict(chkmap))
        self.assertEqual(0, len(chkmap))

    def test_init_and_save_new(self):
        if False:
            print('Hello World!')
        chk_bytes = self.get_chk_bytes()
        chkmap = CHKMap(chk_bytes, None)
        key = chkmap._save()
        leaf_node = LeafNode()
        self.assertEqual([key], leaf_node.serialise(chk_bytes))

    def test_map_first_item_new(self):
        if False:
            return 10
        chk_bytes = self.get_chk_bytes()
        chkmap = CHKMap(chk_bytes, None)
        chkmap.map(('foo,',), 'bar')
        self.assertEqual({('foo,',): 'bar'}, self.to_dict(chkmap))
        self.assertEqual(1, len(chkmap))
        key = chkmap._save()
        leaf_node = LeafNode()
        leaf_node.map(chk_bytes, ('foo,',), 'bar')
        self.assertEqual([key], leaf_node.serialise(chk_bytes))

    def test_unmap_last_item_root_is_leaf_new(self):
        if False:
            i = 10
            return i + 15
        chkmap = self._get_map({('k1' * 50,): 'v1', ('k2' * 50,): 'v2'})
        chkmap.unmap(('k1' * 50,))
        chkmap.unmap(('k2' * 50,))
        self.assertEqual(0, len(chkmap))
        self.assertEqual({}, self.to_dict(chkmap))
        key = chkmap._save()
        leaf_node = LeafNode()
        self.assertEqual([key], leaf_node.serialise(chkmap._store))

    def test__dump_tree(self):
        if False:
            print('Hello World!')
        chkmap = self._get_map({('aaa',): 'value1', ('aab',): 'value2', ('bbb',): 'value3'}, maximum_size=15)
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'value1'\n    'aab' LeafNode\n      ('aab',) 'value2'\n  'b' LeafNode\n      ('bbb',) 'value3'\n", chkmap._dump_tree())
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'value1'\n    'aab' LeafNode\n      ('aab',) 'value2'\n  'b' LeafNode\n      ('bbb',) 'value3'\n", chkmap._dump_tree())
        self.assertEqualDiff("'' InternalNode sha1:0690d471eb0a624f359797d0ee4672bd68f4e236\n  'a' InternalNode sha1:1514c35503da9418d8fd90c1bed553077cb53673\n    'aaa' LeafNode sha1:4cc5970454d40b4ce297a7f13ddb76f63b88fefb\n      ('aaa',) 'value1'\n    'aab' LeafNode sha1:1d68bc90914ef8a3edbcc8bb28b00cb4fea4b5e2\n      ('aab',) 'value2'\n  'b' LeafNode sha1:3686831435b5596515353364eab0399dc45d49e7\n      ('bbb',) 'value3'\n", chkmap._dump_tree(include_keys=True))

    def test__dump_tree_in_progress(self):
        if False:
            print('Hello World!')
        chkmap = self._get_map({('aaa',): 'value1', ('aab',): 'value2'}, maximum_size=10)
        chkmap.map(('bbb',), 'value3')
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'value1'\n    'aab' LeafNode\n      ('aab',) 'value2'\n  'b' LeafNode\n      ('bbb',) 'value3'\n", chkmap._dump_tree())
        self.assertEqualDiff("'' InternalNode None\n  'a' InternalNode sha1:6b0d881dd739a66f733c178b24da64395edfaafd\n    'aaa' LeafNode sha1:40b39a08d895babce17b20ae5f62d187eaa4f63a\n      ('aaa',) 'value1'\n    'aab' LeafNode sha1:ad1dc7c4e801302c95bf1ba7b20bc45e548cd51a\n      ('aab',) 'value2'\n  'b' LeafNode None\n      ('bbb',) 'value3'\n", chkmap._dump_tree(include_keys=True))

def _search_key_single(key):
    if False:
        for i in range(10):
            print('nop')
    'A search key function that maps all nodes to the same value'
    return 'value'

def _test_search_key(key):
    if False:
        while True:
            i = 10
    return 'test:' + '\x00'.join(key)

class TestMapSearchKeys(TestCaseWithStore):

    def test_default_chk_map_uses_flat_search_key(self):
        if False:
            while True:
                i = 10
        chkmap = chk_map.CHKMap(self.get_chk_bytes(), None)
        self.assertEqual('1', chkmap._search_key_func(('1',)))
        self.assertEqual('1\x002', chkmap._search_key_func(('1', '2')))
        self.assertEqual('1\x002\x003', chkmap._search_key_func(('1', '2', '3')))

    def test_search_key_is_passed_to_root_node(self):
        if False:
            print('Hello World!')
        chkmap = chk_map.CHKMap(self.get_chk_bytes(), None, search_key_func=_test_search_key)
        self.assertIs(_test_search_key, chkmap._search_key_func)
        self.assertEqual('test:1\x002\x003', chkmap._search_key_func(('1', '2', '3')))
        self.assertEqual('test:1\x002\x003', chkmap._root_node._search_key(('1', '2', '3')))

    def test_search_key_passed_via__ensure_root(self):
        if False:
            print('Hello World!')
        chk_bytes = self.get_chk_bytes()
        chkmap = chk_map.CHKMap(chk_bytes, None, search_key_func=_test_search_key)
        root_key = chkmap._save()
        chkmap = chk_map.CHKMap(chk_bytes, root_key, search_key_func=_test_search_key)
        chkmap._ensure_root()
        self.assertEqual('test:1\x002\x003', chkmap._root_node._search_key(('1', '2', '3')))

    def test_search_key_with_internal_node(self):
        if False:
            print('Hello World!')
        chk_bytes = self.get_chk_bytes()
        chkmap = chk_map.CHKMap(chk_bytes, None, search_key_func=_test_search_key)
        chkmap._root_node.set_maximum_size(10)
        chkmap.map(('1',), 'foo')
        chkmap.map(('2',), 'bar')
        chkmap.map(('3',), 'baz')
        self.assertEqualDiff("'' InternalNode\n  'test:1' LeafNode\n      ('1',) 'foo'\n  'test:2' LeafNode\n      ('2',) 'bar'\n  'test:3' LeafNode\n      ('3',) 'baz'\n", chkmap._dump_tree())
        root_key = chkmap._save()
        chkmap = chk_map.CHKMap(chk_bytes, root_key, search_key_func=_test_search_key)
        self.assertEqualDiff("'' InternalNode\n  'test:1' LeafNode\n      ('1',) 'foo'\n  'test:2' LeafNode\n      ('2',) 'bar'\n  'test:3' LeafNode\n      ('3',) 'baz'\n", chkmap._dump_tree())

    def test_search_key_16(self):
        if False:
            return 10
        chk_bytes = self.get_chk_bytes()
        chkmap = chk_map.CHKMap(chk_bytes, None, search_key_func=chk_map._search_key_16)
        chkmap._root_node.set_maximum_size(10)
        chkmap.map(('1',), 'foo')
        chkmap.map(('2',), 'bar')
        chkmap.map(('3',), 'baz')
        self.assertEqualDiff("'' InternalNode\n  '1' LeafNode\n      ('2',) 'bar'\n  '6' LeafNode\n      ('3',) 'baz'\n  '8' LeafNode\n      ('1',) 'foo'\n", chkmap._dump_tree())
        root_key = chkmap._save()
        chkmap = chk_map.CHKMap(chk_bytes, root_key, search_key_func=chk_map._search_key_16)
        self.assertEqual([(('1',), 'foo')], list(chkmap.iteritems([('1',)])))
        self.assertEqualDiff("'' InternalNode\n  '1' LeafNode\n      ('2',) 'bar'\n  '6' LeafNode\n      ('3',) 'baz'\n  '8' LeafNode\n      ('1',) 'foo'\n", chkmap._dump_tree())

    def test_search_key_255(self):
        if False:
            while True:
                i = 10
        chk_bytes = self.get_chk_bytes()
        chkmap = chk_map.CHKMap(chk_bytes, None, search_key_func=chk_map._search_key_255)
        chkmap._root_node.set_maximum_size(10)
        chkmap.map(('1',), 'foo')
        chkmap.map(('2',), 'bar')
        chkmap.map(('3',), 'baz')
        self.assertEqualDiff("'' InternalNode\n  '\\x1a' LeafNode\n      ('2',) 'bar'\n  'm' LeafNode\n      ('3',) 'baz'\n  '\\x83' LeafNode\n      ('1',) 'foo'\n", chkmap._dump_tree())
        root_key = chkmap._save()
        chkmap = chk_map.CHKMap(chk_bytes, root_key, search_key_func=chk_map._search_key_255)
        self.assertEqual([(('1',), 'foo')], list(chkmap.iteritems([('1',)])))
        self.assertEqualDiff("'' InternalNode\n  '\\x1a' LeafNode\n      ('2',) 'bar'\n  'm' LeafNode\n      ('3',) 'baz'\n  '\\x83' LeafNode\n      ('1',) 'foo'\n", chkmap._dump_tree())

    def test_search_key_collisions(self):
        if False:
            for i in range(10):
                print('nop')
        chkmap = chk_map.CHKMap(self.get_chk_bytes(), None, search_key_func=_search_key_single)
        chkmap._root_node.set_maximum_size(20)
        chkmap.map(('1',), 'foo')
        chkmap.map(('2',), 'bar')
        chkmap.map(('3',), 'baz')
        self.assertEqualDiff("'' LeafNode\n      ('1',) 'foo'\n      ('2',) 'bar'\n      ('3',) 'baz'\n", chkmap._dump_tree())

class TestLeafNode(TestCaseWithStore):

    def test_current_size_empty(self):
        if False:
            for i in range(10):
                print('nop')
        node = LeafNode()
        self.assertEqual(16, node._current_size())

    def test_current_size_size_changed(self):
        if False:
            for i in range(10):
                print('nop')
        node = LeafNode()
        node.set_maximum_size(10)
        self.assertEqual(17, node._current_size())

    def test_current_size_width_changed(self):
        if False:
            return 10
        node = LeafNode()
        node._key_width = 10
        self.assertEqual(17, node._current_size())

    def test_current_size_items(self):
        if False:
            print('Hello World!')
        node = LeafNode()
        base_size = node._current_size()
        node.map(None, ('foo bar',), 'baz')
        self.assertEqual(base_size + 14, node._current_size())

    def test_deserialise_empty(self):
        if False:
            i = 10
            return i + 15
        node = LeafNode.deserialise('chkleaf:\n10\n1\n0\n\n', ('sha1:1234',))
        self.assertEqual(0, len(node))
        self.assertEqual(10, node.maximum_size)
        self.assertEqual(('sha1:1234',), node.key())
        self.assertIs(None, node._search_prefix)
        self.assertIs(None, node._common_serialised_prefix)

    def test_deserialise_items(self):
        if False:
            print('Hello World!')
        node = LeafNode.deserialise('chkleaf:\n0\n1\n2\n\nfoo bar\x001\nbaz\nquux\x001\nblarh\n', ('sha1:1234',))
        self.assertEqual(2, len(node))
        self.assertEqual([(('foo bar',), 'baz'), (('quux',), 'blarh')], sorted(node.iteritems(None)))

    def test_deserialise_item_with_null_width_1(self):
        if False:
            i = 10
            return i + 15
        node = LeafNode.deserialise('chkleaf:\n0\n1\n2\n\nfoo\x001\nbar\x00baz\nquux\x001\nblarh\n', ('sha1:1234',))
        self.assertEqual(2, len(node))
        self.assertEqual([(('foo',), 'bar\x00baz'), (('quux',), 'blarh')], sorted(node.iteritems(None)))

    def test_deserialise_item_with_null_width_2(self):
        if False:
            while True:
                i = 10
        node = LeafNode.deserialise('chkleaf:\n0\n2\n2\n\nfoo\x001\x001\nbar\x00baz\nquux\x00\x001\nblarh\n', ('sha1:1234',))
        self.assertEqual(2, len(node))
        self.assertEqual([(('foo', '1'), 'bar\x00baz'), (('quux', ''), 'blarh')], sorted(node.iteritems(None)))

    def test_iteritems_selected_one_of_two_items(self):
        if False:
            for i in range(10):
                print('nop')
        node = LeafNode.deserialise('chkleaf:\n0\n1\n2\n\nfoo bar\x001\nbaz\nquux\x001\nblarh\n', ('sha1:1234',))
        self.assertEqual(2, len(node))
        self.assertEqual([(('quux',), 'blarh')], sorted(node.iteritems(None, [('quux',), ('qaz',)])))

    def test_deserialise_item_with_common_prefix(self):
        if False:
            while True:
                i = 10
        node = LeafNode.deserialise('chkleaf:\n0\n2\n2\nfoo\x00\n1\x001\nbar\x00baz\n2\x001\nblarh\n', ('sha1:1234',))
        self.assertEqual(2, len(node))
        self.assertEqual([(('foo', '1'), 'bar\x00baz'), (('foo', '2'), 'blarh')], sorted(node.iteritems(None)))
        self.assertIs(chk_map._unknown, node._search_prefix)
        self.assertEqual('foo\x00', node._common_serialised_prefix)

    def test_deserialise_multi_line(self):
        if False:
            i = 10
            return i + 15
        node = LeafNode.deserialise('chkleaf:\n0\n2\n2\nfoo\x00\n1\x002\nbar\nbaz\n2\x002\nblarh\n\n', ('sha1:1234',))
        self.assertEqual(2, len(node))
        self.assertEqual([(('foo', '1'), 'bar\nbaz'), (('foo', '2'), 'blarh\n')], sorted(node.iteritems(None)))
        self.assertIs(chk_map._unknown, node._search_prefix)
        self.assertEqual('foo\x00', node._common_serialised_prefix)

    def test_key_new(self):
        if False:
            return 10
        node = LeafNode()
        self.assertEqual(None, node.key())

    def test_key_after_map(self):
        if False:
            i = 10
            return i + 15
        node = LeafNode.deserialise('chkleaf:\n10\n1\n0\n\n', ('sha1:1234',))
        node.map(None, ('foo bar',), 'baz quux')
        self.assertEqual(None, node.key())

    def test_key_after_unmap(self):
        if False:
            i = 10
            return i + 15
        node = LeafNode.deserialise('chkleaf:\n0\n1\n2\n\nfoo bar\x001\nbaz\nquux\x001\nblarh\n', ('sha1:1234',))
        node.unmap(None, ('foo bar',))
        self.assertEqual(None, node.key())

    def test_map_exceeding_max_size_only_entry_new(self):
        if False:
            for i in range(10):
                print('nop')
        node = LeafNode()
        node.set_maximum_size(10)
        result = node.map(None, ('foo bar',), 'baz quux')
        self.assertEqual(('foo bar', [('', node)]), result)
        self.assertTrue(10 < node._current_size())

    def test_map_exceeding_max_size_second_entry_early_difference_new(self):
        if False:
            i = 10
            return i + 15
        node = LeafNode()
        node.set_maximum_size(10)
        node.map(None, ('foo bar',), 'baz quux')
        (prefix, result) = list(node.map(None, ('blue',), 'red'))
        self.assertEqual('', prefix)
        self.assertEqual(2, len(result))
        split_chars = set([result[0][0], result[1][0]])
        self.assertEqual(set(['f', 'b']), split_chars)
        nodes = dict(result)
        node = nodes['f']
        self.assertEqual({('foo bar',): 'baz quux'}, self.to_dict(node, None))
        self.assertEqual(10, node.maximum_size)
        self.assertEqual(1, node._key_width)
        node = nodes['b']
        self.assertEqual({('blue',): 'red'}, self.to_dict(node, None))
        self.assertEqual(10, node.maximum_size)
        self.assertEqual(1, node._key_width)

    def test_map_first(self):
        if False:
            return 10
        node = LeafNode()
        result = node.map(None, ('foo bar',), 'baz quux')
        self.assertEqual(('foo bar', [('', node)]), result)
        self.assertEqual({('foo bar',): 'baz quux'}, self.to_dict(node, None))
        self.assertEqual(1, len(node))

    def test_map_second(self):
        if False:
            for i in range(10):
                print('nop')
        node = LeafNode()
        node.map(None, ('foo bar',), 'baz quux')
        result = node.map(None, ('bingo',), 'bango')
        self.assertEqual(('', [('', node)]), result)
        self.assertEqual({('foo bar',): 'baz quux', ('bingo',): 'bango'}, self.to_dict(node, None))
        self.assertEqual(2, len(node))

    def test_map_replacement(self):
        if False:
            i = 10
            return i + 15
        node = LeafNode()
        node.map(None, ('foo bar',), 'baz quux')
        result = node.map(None, ('foo bar',), 'bango')
        self.assertEqual(('foo bar', [('', node)]), result)
        self.assertEqual({('foo bar',): 'bango'}, self.to_dict(node, None))
        self.assertEqual(1, len(node))

    def test_serialise_empty(self):
        if False:
            return 10
        store = self.get_chk_bytes()
        node = LeafNode()
        node.set_maximum_size(10)
        expected_key = ('sha1:f34c3f0634ea3f85953dffa887620c0a5b1f4a51',)
        self.assertEqual([expected_key], list(node.serialise(store)))
        self.assertEqual('chkleaf:\n10\n1\n0\n\n', self.read_bytes(store, expected_key))
        self.assertEqual(expected_key, node.key())

    def test_serialise_items(self):
        if False:
            return 10
        store = self.get_chk_bytes()
        node = LeafNode()
        node.set_maximum_size(10)
        node.map(None, ('foo bar',), 'baz quux')
        expected_key = ('sha1:f89fac7edfc6bdb1b1b54a556012ff0c646ef5e0',)
        self.assertEqual('foo bar', node._common_serialised_prefix)
        self.assertEqual([expected_key], list(node.serialise(store)))
        self.assertEqual('chkleaf:\n10\n1\n1\nfoo bar\n\x001\nbaz quux\n', self.read_bytes(store, expected_key))
        self.assertEqual(expected_key, node.key())

    def test_unique_serialised_prefix_empty_new(self):
        if False:
            return 10
        node = LeafNode()
        self.assertIs(None, node._compute_search_prefix())

    def test_unique_serialised_prefix_one_item_new(self):
        if False:
            return 10
        node = LeafNode()
        node.map(None, ('foo bar', 'baz'), 'baz quux')
        self.assertEqual('foo bar\x00baz', node._compute_search_prefix())

    def test_unmap_missing(self):
        if False:
            for i in range(10):
                print('nop')
        node = LeafNode()
        self.assertRaises(KeyError, node.unmap, None, ('foo bar',))

    def test_unmap_present(self):
        if False:
            while True:
                i = 10
        node = LeafNode()
        node.map(None, ('foo bar',), 'baz quux')
        result = node.unmap(None, ('foo bar',))
        self.assertEqual(node, result)
        self.assertEqual({}, self.to_dict(node, None))
        self.assertEqual(0, len(node))

    def test_map_maintains_common_prefixes(self):
        if False:
            for i in range(10):
                print('nop')
        node = LeafNode()
        node._key_width = 2
        node.map(None, ('foo bar', 'baz'), 'baz quux')
        self.assertEqual('foo bar\x00baz', node._search_prefix)
        self.assertEqual('foo bar\x00baz', node._common_serialised_prefix)
        node.map(None, ('foo bar', 'bing'), 'baz quux')
        self.assertEqual('foo bar\x00b', node._search_prefix)
        self.assertEqual('foo bar\x00b', node._common_serialised_prefix)
        node.map(None, ('fool', 'baby'), 'baz quux')
        self.assertEqual('foo', node._search_prefix)
        self.assertEqual('foo', node._common_serialised_prefix)
        node.map(None, ('foo bar', 'baz'), 'replaced')
        self.assertEqual('foo', node._search_prefix)
        self.assertEqual('foo', node._common_serialised_prefix)
        node.map(None, ('very', 'different'), 'value')
        self.assertEqual('', node._search_prefix)
        self.assertEqual('', node._common_serialised_prefix)

    def test_unmap_maintains_common_prefixes(self):
        if False:
            print('Hello World!')
        node = LeafNode()
        node._key_width = 2
        node.map(None, ('foo bar', 'baz'), 'baz quux')
        node.map(None, ('foo bar', 'bing'), 'baz quux')
        node.map(None, ('fool', 'baby'), 'baz quux')
        node.map(None, ('very', 'different'), 'value')
        self.assertEqual('', node._search_prefix)
        self.assertEqual('', node._common_serialised_prefix)
        node.unmap(None, ('very', 'different'))
        self.assertEqual('foo', node._search_prefix)
        self.assertEqual('foo', node._common_serialised_prefix)
        node.unmap(None, ('fool', 'baby'))
        self.assertEqual('foo bar\x00b', node._search_prefix)
        self.assertEqual('foo bar\x00b', node._common_serialised_prefix)
        node.unmap(None, ('foo bar', 'baz'))
        self.assertEqual('foo bar\x00bing', node._search_prefix)
        self.assertEqual('foo bar\x00bing', node._common_serialised_prefix)
        node.unmap(None, ('foo bar', 'bing'))
        self.assertEqual(None, node._search_prefix)
        self.assertEqual(None, node._common_serialised_prefix)

class TestInternalNode(TestCaseWithStore):

    def test_add_node_empty_new(self):
        if False:
            return 10
        node = InternalNode('fo')
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, ('foo',), 'bar')
        node.add_node('foo', child)
        self.assertEqual(3, node._node_width)
        self.assertEqual({('foo',): 'bar'}, self.to_dict(node, None))
        self.assertEqual(1, len(node))
        chk_bytes = self.get_chk_bytes()
        keys = list(node.serialise(chk_bytes))
        child_key = child.serialise(chk_bytes)[0]
        self.assertEqual([child_key, ('sha1:cf67e9997d8228a907c1f5bfb25a8bd9cd916fac',)], keys)
        bytes = self.read_bytes(chk_bytes, keys[1])
        node = chk_map._deserialise(bytes, keys[1], None)
        self.assertEqual(1, len(node))
        self.assertEqual({('foo',): 'bar'}, self.to_dict(node, chk_bytes))
        self.assertEqual(3, node._node_width)

    def test_add_node_resets_key_new(self):
        if False:
            i = 10
            return i + 15
        node = InternalNode('fo')
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, ('foo',), 'bar')
        node.add_node('foo', child)
        chk_bytes = self.get_chk_bytes()
        keys = list(node.serialise(chk_bytes))
        self.assertEqual(keys[1], node._key)
        node.add_node('fos', child)
        self.assertEqual(None, node._key)

    def test__iter_nodes_no_key_filter(self):
        if False:
            for i in range(10):
                print('nop')
        node = InternalNode('')
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, ('foo',), 'bar')
        node.add_node('f', child)
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, ('bar',), 'baz')
        node.add_node('b', child)
        for (child, node_key_filter) in node._iter_nodes(None, key_filter=None):
            self.assertEqual(None, node_key_filter)

    def test__iter_nodes_splits_key_filter(self):
        if False:
            i = 10
            return i + 15
        node = InternalNode('')
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, ('foo',), 'bar')
        node.add_node('f', child)
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, ('bar',), 'baz')
        node.add_node('b', child)
        key_filter = (('foo',), ('bar',), ('cat',))
        for (child, node_key_filter) in node._iter_nodes(None, key_filter=key_filter):
            self.assertEqual(1, len(node_key_filter))

    def test__iter_nodes_with_multiple_matches(self):
        if False:
            while True:
                i = 10
        node = InternalNode('')
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, ('foo',), 'val')
        child.map(None, ('fob',), 'val')
        node.add_node('f', child)
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, ('bar',), 'val')
        child.map(None, ('baz',), 'val')
        node.add_node('b', child)
        key_filter = (('foo',), ('fob',), ('bar',), ('baz',), ('ram',))
        for (child, node_key_filter) in node._iter_nodes(None, key_filter=key_filter):
            self.assertEqual(2, len(node_key_filter))

    def make_fo_fa_node(self):
        if False:
            print('Hello World!')
        node = InternalNode('f')
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, ('foo',), 'val')
        child.map(None, ('fob',), 'val')
        node.add_node('fo', child)
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, ('far',), 'val')
        child.map(None, ('faz',), 'val')
        node.add_node('fa', child)
        return node

    def test__iter_nodes_single_entry(self):
        if False:
            for i in range(10):
                print('nop')
        node = self.make_fo_fa_node()
        key_filter = [('foo',)]
        nodes = list(node._iter_nodes(None, key_filter=key_filter))
        self.assertEqual(1, len(nodes))
        self.assertEqual(key_filter, nodes[0][1])

    def test__iter_nodes_single_entry_misses(self):
        if False:
            print('Hello World!')
        node = self.make_fo_fa_node()
        key_filter = [('bar',)]
        nodes = list(node._iter_nodes(None, key_filter=key_filter))
        self.assertEqual(0, len(nodes))

    def test__iter_nodes_mixed_key_width(self):
        if False:
            i = 10
            return i + 15
        node = self.make_fo_fa_node()
        key_filter = [('foo', 'bar'), ('foo',), ('fo',), ('b',)]
        nodes = list(node._iter_nodes(None, key_filter=key_filter))
        self.assertEqual(1, len(nodes))
        matches = key_filter[:]
        matches.remove(('b',))
        self.assertEqual(sorted(matches), sorted(nodes[0][1]))

    def test__iter_nodes_match_all(self):
        if False:
            print('Hello World!')
        node = self.make_fo_fa_node()
        key_filter = [('foo', 'bar'), ('foo',), ('fo',), ('f',)]
        nodes = list(node._iter_nodes(None, key_filter=key_filter))
        self.assertEqual(2, len(nodes))

    def test__iter_nodes_fixed_widths_and_misses(self):
        if False:
            i = 10
            return i + 15
        node = self.make_fo_fa_node()
        key_filter = [('foo',), ('faa',), ('baz',)]
        nodes = list(node._iter_nodes(None, key_filter=key_filter))
        self.assertEqual(2, len(nodes))
        for (node, matches) in nodes:
            self.assertEqual(1, len(matches))

    def test_iteritems_empty_new(self):
        if False:
            for i in range(10):
                print('nop')
        node = InternalNode()
        self.assertEqual([], sorted(node.iteritems(None)))

    def test_iteritems_two_children(self):
        if False:
            print('Hello World!')
        node = InternalNode()
        leaf1 = LeafNode()
        leaf1.map(None, ('foo bar',), 'quux')
        leaf2 = LeafNode()
        leaf2.map(None, ('strange',), 'beast')
        node.add_node('f', leaf1)
        node.add_node('s', leaf2)
        self.assertEqual([(('foo bar',), 'quux'), (('strange',), 'beast')], sorted(node.iteritems(None)))

    def test_iteritems_two_children_partial(self):
        if False:
            return 10
        node = InternalNode()
        leaf1 = LeafNode()
        leaf1.map(None, ('foo bar',), 'quux')
        leaf2 = LeafNode()
        leaf2.map(None, ('strange',), 'beast')
        node.add_node('f', leaf1)
        node._items['f'] = None
        node.add_node('s', leaf2)
        self.assertEqual([(('strange',), 'beast')], sorted(node.iteritems(None, [('strange',), ('weird',)])))

    def test_iteritems_two_children_with_hash(self):
        if False:
            for i in range(10):
                print('nop')
        search_key_func = chk_map.search_key_registry.get('hash-255-way')
        node = InternalNode(search_key_func=search_key_func)
        leaf1 = LeafNode(search_key_func=search_key_func)
        leaf1.map(None, StaticTuple('foo bar'), 'quux')
        leaf2 = LeafNode(search_key_func=search_key_func)
        leaf2.map(None, StaticTuple('strange'), 'beast')
        self.assertEqual('¾F\x014', search_key_func(StaticTuple('foo bar')))
        self.assertEqual('\x85ú÷K', search_key_func(StaticTuple('strange')))
        node.add_node('¾', leaf1)
        node._items['¾'] = None
        node.add_node('\x85', leaf2)
        self.assertEqual([(('strange',), 'beast')], sorted(node.iteritems(None, [StaticTuple('strange'), StaticTuple('weird')])))

    def test_iteritems_partial_empty(self):
        if False:
            i = 10
            return i + 15
        node = InternalNode()
        self.assertEqual([], sorted(node.iteritems([('missing',)])))

    def test_map_to_new_child_new(self):
        if False:
            return 10
        chkmap = self._get_map({('k1',): 'foo', ('k2',): 'bar'}, maximum_size=10)
        chkmap._ensure_root()
        node = chkmap._root_node
        self.assertEqual(2, len([value for value in node._items.values() if type(value) is StaticTuple]))
        (prefix, nodes) = node.map(None, ('k3',), 'quux')
        self.assertEqual('k', prefix)
        self.assertEqual([('', node)], nodes)
        child = node._items['k3']
        self.assertIsInstance(child, LeafNode)
        self.assertEqual(1, len(child))
        self.assertEqual({('k3',): 'quux'}, self.to_dict(child, None))
        self.assertEqual(None, child._key)
        self.assertEqual(10, child.maximum_size)
        self.assertEqual(1, child._key_width)
        self.assertEqual(3, len(chkmap))
        self.assertEqual({('k1',): 'foo', ('k2',): 'bar', ('k3',): 'quux'}, self.to_dict(chkmap))
        keys = list(node.serialise(chkmap._store))
        child_key = child.serialise(chkmap._store)[0]
        self.assertEqual([child_key, keys[1]], keys)

    def test_map_to_child_child_splits_new(self):
        if False:
            for i in range(10):
                print('nop')
        chkmap = self._get_map({('k1',): 'foo', ('k22',): 'bar'}, maximum_size=10)
        self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' LeafNode\n      ('k22',) 'bar'\n", chkmap._dump_tree())
        chkmap = CHKMap(chkmap._store, chkmap._root_node)
        chkmap._ensure_root()
        node = chkmap._root_node
        self.assertEqual(2, len([value for value in node._items.values() if type(value) is StaticTuple]))
        (prefix, nodes) = node.map(chkmap._store, ('k23',), 'quux')
        self.assertEqual('k', prefix)
        self.assertEqual([('', node)], nodes)
        child = node._items['k2']
        self.assertIsInstance(child, InternalNode)
        self.assertEqual(2, len(child))
        self.assertEqual({('k22',): 'bar', ('k23',): 'quux'}, self.to_dict(child, None))
        self.assertEqual(None, child._key)
        self.assertEqual(10, child.maximum_size)
        self.assertEqual(1, child._key_width)
        self.assertEqual(3, child._node_width)
        self.assertEqual(3, len(chkmap))
        self.assertEqual({('k1',): 'foo', ('k22',): 'bar', ('k23',): 'quux'}, self.to_dict(chkmap))
        keys = list(node.serialise(chkmap._store))
        child_key = child._key
        k22_key = child._items['k22']._key
        k23_key = child._items['k23']._key
        self.assertEqual([k22_key, k23_key, child_key, node.key()], keys)
        self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' InternalNode\n    'k22' LeafNode\n      ('k22',) 'bar'\n    'k23' LeafNode\n      ('k23',) 'quux'\n", chkmap._dump_tree())

    def test__search_prefix_filter_with_hash(self):
        if False:
            while True:
                i = 10
        search_key_func = chk_map.search_key_registry.get('hash-16-way')
        node = InternalNode(search_key_func=search_key_func)
        node._key_width = 2
        node._node_width = 4
        self.assertEqual('E8B7BE43\x0071BEEFF9', search_key_func(StaticTuple('a', 'b')))
        self.assertEqual('E8B7', node._search_prefix_filter(StaticTuple('a', 'b')))
        self.assertEqual('E8B7', node._search_prefix_filter(StaticTuple('a')))

    def test_unmap_k23_from_k1_k22_k23_gives_k1_k22_tree_new(self):
        if False:
            print('Hello World!')
        chkmap = self._get_map({('k1',): 'foo', ('k22',): 'bar', ('k23',): 'quux'}, maximum_size=10)
        self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' InternalNode\n    'k22' LeafNode\n      ('k22',) 'bar'\n    'k23' LeafNode\n      ('k23',) 'quux'\n", chkmap._dump_tree())
        chkmap = CHKMap(chkmap._store, chkmap._root_node)
        chkmap._ensure_root()
        node = chkmap._root_node
        result = node.unmap(chkmap._store, ('k23',))
        child = node._items['k2']
        self.assertIsInstance(child, LeafNode)
        self.assertEqual(1, len(child))
        self.assertEqual({('k22',): 'bar'}, self.to_dict(child, None))
        self.assertEqual(2, len(chkmap))
        self.assertEqual({('k1',): 'foo', ('k22',): 'bar'}, self.to_dict(chkmap))
        keys = list(node.serialise(chkmap._store))
        self.assertEqual([keys[-1]], keys)
        chkmap = CHKMap(chkmap._store, keys[-1])
        self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' LeafNode\n      ('k22',) 'bar'\n", chkmap._dump_tree())

    def test_unmap_k1_from_k1_k22_k23_gives_k22_k23_tree_new(self):
        if False:
            print('Hello World!')
        chkmap = self._get_map({('k1',): 'foo', ('k22',): 'bar', ('k23',): 'quux'}, maximum_size=10)
        self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' InternalNode\n    'k22' LeafNode\n      ('k22',) 'bar'\n    'k23' LeafNode\n      ('k23',) 'quux'\n", chkmap._dump_tree())
        orig_root = chkmap._root_node
        chkmap = CHKMap(chkmap._store, orig_root)
        chkmap._ensure_root()
        node = chkmap._root_node
        k2_ptr = node._items['k2']
        result = node.unmap(chkmap._store, ('k1',))
        self.assertEqual(k2_ptr, result)
        chkmap = CHKMap(chkmap._store, orig_root)
        chkmap.unmap(('k1',))
        self.assertEqual(k2_ptr, chkmap._root_node)
        self.assertEqualDiff("'' InternalNode\n  'k22' LeafNode\n      ('k22',) 'bar'\n  'k23' LeafNode\n      ('k23',) 'quux'\n", chkmap._dump_tree())

class TestCHKMapDifference(TestCaseWithExampleMaps):

    def get_difference(self, new_roots, old_roots, search_key_func=None):
        if False:
            for i in range(10):
                print('nop')
        if search_key_func is None:
            search_key_func = chk_map._search_key_plain
        return chk_map.CHKMapDifference(self.get_chk_bytes(), new_roots, old_roots, search_key_func)

    def test__init__(self):
        if False:
            print('Hello World!')
        c_map = self.make_root_only_map()
        key1 = c_map.key()
        c_map.map(('aaa',), 'new aaa content')
        key2 = c_map._save()
        diff = self.get_difference([key2], [key1])
        self.assertEqual(set([key1]), diff._all_old_chks)
        self.assertEqual([], diff._old_queue)
        self.assertEqual([], diff._new_queue)

    def help__read_all_roots(self, search_key_func):
        if False:
            print('Hello World!')
        c_map = self.make_root_only_map(search_key_func=search_key_func)
        key1 = c_map.key()
        c_map.map(('aaa',), 'new aaa content')
        key2 = c_map._save()
        diff = self.get_difference([key2], [key1], search_key_func)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key2], root_results)
        self.assertEqual([(('aaa',), 'new aaa content')], diff._new_item_queue)
        self.assertEqual([], diff._new_queue)
        self.assertEqual([], diff._old_queue)

    def test__read_all_roots_plain(self):
        if False:
            print('Hello World!')
        self.help__read_all_roots(search_key_func=chk_map._search_key_plain)

    def test__read_all_roots_16(self):
        if False:
            while True:
                i = 10
        self.help__read_all_roots(search_key_func=chk_map._search_key_16)

    def test__read_all_roots_skips_known_old(self):
        if False:
            while True:
                i = 10
        c_map = self.make_one_deep_map(chk_map._search_key_plain)
        key1 = c_map.key()
        c_map2 = self.make_root_only_map(chk_map._search_key_plain)
        key2 = c_map2.key()
        diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([], root_results)

    def test__read_all_roots_prepares_queues(self):
        if False:
            i = 10
            return i + 15
        c_map = self.make_one_deep_map(chk_map._search_key_plain)
        key1 = c_map.key()
        c_map._dump_tree()
        key1_a = c_map._root_node._items['a'].key()
        c_map.map(('abb',), 'new abb content')
        key2 = c_map._save()
        key2_a = c_map._root_node._items['a'].key()
        diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key2], root_results)
        self.assertEqual([key2_a], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)
        self.assertEqual([key1_a], diff._old_queue)

    def test__read_all_roots_multi_new_prepares_queues(self):
        if False:
            i = 10
            return i + 15
        c_map = self.make_one_deep_map(chk_map._search_key_plain)
        key1 = c_map.key()
        c_map._dump_tree()
        key1_a = c_map._root_node._items['a'].key()
        key1_c = c_map._root_node._items['c'].key()
        c_map.map(('abb',), 'new abb content')
        key2 = c_map._save()
        key2_a = c_map._root_node._items['a'].key()
        key2_c = c_map._root_node._items['c'].key()
        c_map = chk_map.CHKMap(self.get_chk_bytes(), key1, chk_map._search_key_plain)
        c_map.map(('ccc',), 'new ccc content')
        key3 = c_map._save()
        key3_a = c_map._root_node._items['a'].key()
        key3_c = c_map._root_node._items['c'].key()
        diff = self.get_difference([key2, key3], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual(sorted([key2, key3]), sorted(root_results))
        self.assertEqual([key2_a, key3_c], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)
        self.assertEqual([key1_a, key1_c], diff._old_queue)

    def test__read_all_roots_different_depths(self):
        if False:
            for i in range(10):
                print('nop')
        c_map = self.make_two_deep_map(chk_map._search_key_plain)
        c_map._dump_tree()
        key1 = c_map.key()
        key1_a = c_map._root_node._items['a'].key()
        key1_c = c_map._root_node._items['c'].key()
        key1_d = c_map._root_node._items['d'].key()
        c_map2 = self.make_one_deep_two_prefix_map(chk_map._search_key_plain)
        c_map2._dump_tree()
        key2 = c_map2.key()
        key2_aa = c_map2._root_node._items['aa'].key()
        key2_ad = c_map2._root_node._items['ad'].key()
        diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key2], root_results)
        self.assertEqual([key1_a], diff._old_queue)
        self.assertEqual([key2_aa, key2_ad], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)
        diff = self.get_difference([key1], [key2], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key1], root_results)
        self.assertEqual([key2_aa, key2_ad], diff._old_queue)
        self.assertEqual([key1_a, key1_c, key1_d], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)

    def test__read_all_roots_different_depths_16(self):
        if False:
            i = 10
            return i + 15
        c_map = self.make_two_deep_map(chk_map._search_key_16)
        c_map._dump_tree()
        key1 = c_map.key()
        key1_2 = c_map._root_node._items['2'].key()
        key1_4 = c_map._root_node._items['4'].key()
        key1_C = c_map._root_node._items['C'].key()
        key1_F = c_map._root_node._items['F'].key()
        c_map2 = self.make_one_deep_two_prefix_map(chk_map._search_key_16)
        c_map2._dump_tree()
        key2 = c_map2.key()
        key2_F0 = c_map2._root_node._items['F0'].key()
        key2_F3 = c_map2._root_node._items['F3'].key()
        key2_F4 = c_map2._root_node._items['F4'].key()
        key2_FD = c_map2._root_node._items['FD'].key()
        diff = self.get_difference([key2], [key1], chk_map._search_key_16)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key2], root_results)
        self.assertEqual([key1_F], diff._old_queue)
        self.assertEqual(sorted([key2_F0, key2_F3, key2_F4, key2_FD]), sorted(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)
        diff = self.get_difference([key1], [key2], chk_map._search_key_16)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key1], root_results)
        self.assertEqual(sorted([key2_F0, key2_F3, key2_F4, key2_FD]), sorted(diff._old_queue))
        self.assertEqual(sorted([key1_2, key1_4, key1_C, key1_F]), sorted(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)

    def test__read_all_roots_mixed_depth(self):
        if False:
            while True:
                i = 10
        c_map = self.make_one_deep_two_prefix_map(chk_map._search_key_plain)
        c_map._dump_tree()
        key1 = c_map.key()
        key1_aa = c_map._root_node._items['aa'].key()
        key1_ad = c_map._root_node._items['ad'].key()
        c_map2 = self.make_one_deep_one_prefix_map(chk_map._search_key_plain)
        c_map2._dump_tree()
        key2 = c_map2.key()
        key2_a = c_map2._root_node._items['a'].key()
        key2_b = c_map2._root_node._items['b'].key()
        diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key2], root_results)
        self.assertEqual([], diff._old_queue)
        self.assertEqual([key2_b], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)
        diff = self.get_difference([key1], [key2], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key1], root_results)
        self.assertEqual([key2_a], diff._old_queue)
        self.assertEqual([key1_aa], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)

    def test__read_all_roots_yields_extra_deep_records(self):
        if False:
            i = 10
            return i + 15
        c_map = self.make_two_deep_map(chk_map._search_key_plain)
        c_map._dump_tree()
        key1 = c_map.key()
        key1_a = c_map._root_node._items['a'].key()
        c_map2 = self.get_map({('acc',): 'initial acc content', ('ace',): 'initial ace content'}, maximum_size=100)
        self.assertEqualDiff("'' LeafNode\n      ('acc',) 'initial acc content'\n      ('ace',) 'initial ace content'\n", c_map2._dump_tree())
        key2 = c_map2.key()
        diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key2], root_results)
        self.assertEqual([key1_a], diff._old_queue)
        self.assertEqual([(('acc',), 'initial acc content'), (('ace',), 'initial ace content')], diff._new_item_queue)

    def test__read_all_roots_multiple_targets(self):
        if False:
            for i in range(10):
                print('nop')
        c_map = self.make_root_only_map()
        key1 = c_map.key()
        c_map = self.make_one_deep_map()
        key2 = c_map.key()
        c_map._dump_tree()
        key2_c = c_map._root_node._items['c'].key()
        key2_d = c_map._root_node._items['d'].key()
        c_map.map(('ccc',), 'new ccc value')
        key3 = c_map._save()
        key3_c = c_map._root_node._items['c'].key()
        diff = self.get_difference([key2, key3], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual(sorted([key2, key3]), sorted(root_results))
        self.assertEqual([], diff._old_queue)
        self.assertEqual(sorted([key2_c, key3_c, key2_d]), sorted(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)

    def test__read_all_roots_no_old(self):
        if False:
            i = 10
            return i + 15
        c_map = self.make_two_deep_map()
        key1 = c_map.key()
        diff = self.get_difference([key1], [], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([], root_results)
        self.assertEqual([], diff._old_queue)
        self.assertEqual([key1], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)
        c_map2 = self.make_one_deep_map()
        key2 = c_map2.key()
        diff = self.get_difference([key1, key2], [], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([], root_results)
        self.assertEqual([], diff._old_queue)
        self.assertEqual(sorted([key1, key2]), sorted(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)

    def test__read_all_roots_no_old_16(self):
        if False:
            i = 10
            return i + 15
        c_map = self.make_two_deep_map(chk_map._search_key_16)
        key1 = c_map.key()
        diff = self.get_difference([key1], [], chk_map._search_key_16)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([], root_results)
        self.assertEqual([], diff._old_queue)
        self.assertEqual([key1], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)
        c_map2 = self.make_one_deep_map(chk_map._search_key_16)
        key2 = c_map2.key()
        diff = self.get_difference([key1, key2], [], chk_map._search_key_16)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([], root_results)
        self.assertEqual([], diff._old_queue)
        self.assertEqual(sorted([key1, key2]), sorted(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)

    def test__read_all_roots_multiple_old(self):
        if False:
            return 10
        c_map = self.make_two_deep_map()
        key1 = c_map.key()
        c_map._dump_tree()
        key1_a = c_map._root_node._items['a'].key()
        c_map.map(('ccc',), 'new ccc value')
        key2 = c_map._save()
        key2_a = c_map._root_node._items['a'].key()
        c_map.map(('add',), 'new add value')
        key3 = c_map._save()
        key3_a = c_map._root_node._items['a'].key()
        diff = self.get_difference([key3], [key1, key2], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key3], root_results)
        self.assertEqual([key1_a], diff._old_queue)
        self.assertEqual([key3_a], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)

    def test__process_next_old_batched_no_dupes(self):
        if False:
            return 10
        c_map = self.make_two_deep_map()
        key1 = c_map.key()
        c_map._dump_tree()
        key1_a = c_map._root_node._items['a'].key()
        key1_aa = c_map._root_node._items['a']._items['aa'].key()
        key1_ab = c_map._root_node._items['a']._items['ab'].key()
        key1_ac = c_map._root_node._items['a']._items['ac'].key()
        key1_ad = c_map._root_node._items['a']._items['ad'].key()
        c_map.map(('aaa',), 'new aaa value')
        key2 = c_map._save()
        key2_a = c_map._root_node._items['a'].key()
        key2_aa = c_map._root_node._items['a']._items['aa'].key()
        c_map.map(('acc',), 'new acc content')
        key3 = c_map._save()
        key3_a = c_map._root_node._items['a'].key()
        key3_ac = c_map._root_node._items['a']._items['ac'].key()
        diff = self.get_difference([key3], [key1, key2], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key3], root_results)
        self.assertEqual(sorted([key1_a, key2_a]), sorted(diff._old_queue))
        self.assertEqual([key3_a], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)
        diff._process_next_old()
        self.assertEqual(sorted([key1_aa, key1_ab, key1_ac, key1_ad, key2_aa]), sorted(diff._old_queue))

class TestIterInterestingNodes(TestCaseWithExampleMaps):

    def get_map_key(self, a_dict, maximum_size=10):
        if False:
            return 10
        c_map = self.get_map(a_dict, maximum_size=maximum_size)
        return c_map.key()

    def assertIterInteresting(self, records, items, interesting_keys, old_keys):
        if False:
            while True:
                i = 10
        'Check the result of iter_interesting_nodes.\n\n        Note that we no longer care how many steps are taken, etc, just that\n        the right contents are returned.\n\n        :param records: A list of record keys that should be yielded\n        :param items: A list of items (key,value) that should be yielded.\n        '
        store = self.get_chk_bytes()
        store._search_key_func = chk_map._search_key_plain
        iter_nodes = chk_map.iter_interesting_nodes(store, interesting_keys, old_keys)
        record_keys = []
        all_items = []
        for (record, new_items) in iter_nodes:
            if record is not None:
                record_keys.append(record.key)
            if new_items:
                all_items.extend(new_items)
        self.assertEqual(sorted(records), sorted(record_keys))
        self.assertEqual(sorted(items), sorted(all_items))

    def test_empty_to_one_keys(self):
        if False:
            print('Hello World!')
        target = self.get_map_key({('a',): 'content'})
        self.assertIterInteresting([target], [(('a',), 'content')], [target], [])

    def test_none_to_one_key(self):
        if False:
            i = 10
            return i + 15
        basis = self.get_map_key({})
        target = self.get_map_key({('a',): 'content'})
        self.assertIterInteresting([target], [(('a',), 'content')], [target], [basis])

    def test_one_to_none_key(self):
        if False:
            print('Hello World!')
        basis = self.get_map_key({('a',): 'content'})
        target = self.get_map_key({})
        self.assertIterInteresting([target], [], [target], [basis])

    def test_common_pages(self):
        if False:
            i = 10
            return i + 15
        basis = self.get_map_key({('a',): 'content', ('b',): 'content', ('c',): 'content'})
        target = self.get_map_key({('a',): 'content', ('b',): 'other content', ('c',): 'content'})
        target_map = CHKMap(self.get_chk_bytes(), target)
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('a',) 'content'\n  'b' LeafNode\n      ('b',) 'other content'\n  'c' LeafNode\n      ('c',) 'content'\n", target_map._dump_tree())
        b_key = target_map._root_node._items['b'].key()
        self.assertIterInteresting([target, b_key], [(('b',), 'other content')], [target], [basis])

    def test_common_sub_page(self):
        if False:
            i = 10
            return i + 15
        basis = self.get_map_key({('aaa',): 'common', ('c',): 'common'})
        target = self.get_map_key({('aaa',): 'common', ('aab',): 'new', ('c',): 'common'})
        target_map = CHKMap(self.get_chk_bytes(), target)
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'common'\n    'aab' LeafNode\n      ('aab',) 'new'\n  'c' LeafNode\n      ('c',) 'common'\n", target_map._dump_tree())
        a_key = target_map._root_node._items['a'].key()
        aab_key = target_map._root_node._items['a']._items['aab'].key()
        self.assertIterInteresting([target, a_key, aab_key], [(('aab',), 'new')], [target], [basis])

    def test_common_leaf(self):
        if False:
            while True:
                i = 10
        basis = self.get_map_key({})
        target1 = self.get_map_key({('aaa',): 'common'})
        target2 = self.get_map_key({('aaa',): 'common', ('bbb',): 'new'})
        target3 = self.get_map_key({('aaa',): 'common', ('aac',): 'other', ('bbb',): 'new'})
        target1_map = CHKMap(self.get_chk_bytes(), target1)
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'common'\n", target1_map._dump_tree())
        target2_map = CHKMap(self.get_chk_bytes(), target2)
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'common'\n  'b' LeafNode\n      ('bbb',) 'new'\n", target2_map._dump_tree())
        target3_map = CHKMap(self.get_chk_bytes(), target3)
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'common'\n    'aac' LeafNode\n      ('aac',) 'other'\n  'b' LeafNode\n      ('bbb',) 'new'\n", target3_map._dump_tree())
        aaa_key = target1_map._root_node.key()
        b_key = target2_map._root_node._items['b'].key()
        a_key = target3_map._root_node._items['a'].key()
        aac_key = target3_map._root_node._items['a']._items['aac'].key()
        self.assertIterInteresting([target1, target2, target3, a_key, aac_key, b_key], [(('aaa',), 'common'), (('bbb',), 'new'), (('aac',), 'other')], [target1, target2, target3], [basis])
        self.assertIterInteresting([target2, target3, a_key, aac_key, b_key], [(('bbb',), 'new'), (('aac',), 'other')], [target2, target3], [target1])
        self.assertIterInteresting([target1], [], [target1], [target3])

    def test_multiple_maps(self):
        if False:
            while True:
                i = 10
        basis1 = self.get_map_key({('aaa',): 'common', ('aab',): 'basis1'})
        basis2 = self.get_map_key({('bbb',): 'common', ('bbc',): 'basis2'})
        target1 = self.get_map_key({('aaa',): 'common', ('aac',): 'target1', ('bbb',): 'common'})
        target2 = self.get_map_key({('aaa',): 'common', ('bba',): 'target2', ('bbb',): 'common'})
        target1_map = CHKMap(self.get_chk_bytes(), target1)
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'common'\n    'aac' LeafNode\n      ('aac',) 'target1'\n  'b' LeafNode\n      ('bbb',) 'common'\n", target1_map._dump_tree())
        a_key = target1_map._root_node._items['a'].key()
        aac_key = target1_map._root_node._items['a']._items['aac'].key()
        target2_map = CHKMap(self.get_chk_bytes(), target2)
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'common'\n  'b' InternalNode\n    'bba' LeafNode\n      ('bba',) 'target2'\n    'bbb' LeafNode\n      ('bbb',) 'common'\n", target2_map._dump_tree())
        b_key = target2_map._root_node._items['b'].key()
        bba_key = target2_map._root_node._items['b']._items['bba'].key()
        self.assertIterInteresting([target1, target2, a_key, aac_key, b_key, bba_key], [(('aac',), 'target1'), (('bba',), 'target2')], [target1, target2], [basis1, basis2])

    def test_multiple_maps_overlapping_common_new(self):
        if False:
            return 10
        basis = self.get_map_key({('aaa',): 'left', ('abb',): 'right', ('ccc',): 'common'})
        left = self.get_map_key({('aaa',): 'left', ('abb',): 'right', ('ccc',): 'common', ('ddd',): 'change'})
        right = self.get_map_key({('abb',): 'right'})
        basis_map = CHKMap(self.get_chk_bytes(), basis)
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aa' LeafNode\n      ('aaa',) 'left'\n    'ab' LeafNode\n      ('abb',) 'right'\n  'c' LeafNode\n      ('ccc',) 'common'\n", basis_map._dump_tree())
        left_map = CHKMap(self.get_chk_bytes(), left)
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aa' LeafNode\n      ('aaa',) 'left'\n    'ab' LeafNode\n      ('abb',) 'right'\n  'c' LeafNode\n      ('ccc',) 'common'\n  'd' LeafNode\n      ('ddd',) 'change'\n", left_map._dump_tree())
        l_d_key = left_map._root_node._items['d'].key()
        right_map = CHKMap(self.get_chk_bytes(), right)
        self.assertEqualDiff("'' LeafNode\n      ('abb',) 'right'\n", right_map._dump_tree())
        self.assertIterInteresting([right, left, l_d_key], [(('ddd',), 'change')], [left, right], [basis])

    def test_multiple_maps_similar(self):
        if False:
            print('Hello World!')
        basis = self.get_map_key({('aaa',): 'unchanged', ('abb',): 'will change left', ('caa',): 'unchanged', ('cbb',): 'will change right'}, maximum_size=60)
        left = self.get_map_key({('aaa',): 'unchanged', ('abb',): 'changed left', ('caa',): 'unchanged', ('cbb',): 'will change right'}, maximum_size=60)
        right = self.get_map_key({('aaa',): 'unchanged', ('abb',): 'will change left', ('caa',): 'unchanged', ('cbb',): 'changed right'}, maximum_size=60)
        basis_map = CHKMap(self.get_chk_bytes(), basis)
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'unchanged'\n      ('abb',) 'will change left'\n  'c' LeafNode\n      ('caa',) 'unchanged'\n      ('cbb',) 'will change right'\n", basis_map._dump_tree())
        left_map = CHKMap(self.get_chk_bytes(), left)
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'unchanged'\n      ('abb',) 'changed left'\n  'c' LeafNode\n      ('caa',) 'unchanged'\n      ('cbb',) 'will change right'\n", left_map._dump_tree())
        l_a_key = left_map._root_node._items['a'].key()
        l_c_key = left_map._root_node._items['c'].key()
        right_map = CHKMap(self.get_chk_bytes(), right)
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'unchanged'\n      ('abb',) 'will change left'\n  'c' LeafNode\n      ('caa',) 'unchanged'\n      ('cbb',) 'changed right'\n", right_map._dump_tree())
        r_a_key = right_map._root_node._items['a'].key()
        r_c_key = right_map._root_node._items['c'].key()
        self.assertIterInteresting([right, left, l_a_key, r_c_key], [(('abb',), 'changed left'), (('cbb',), 'changed right')], [left, right], [basis])