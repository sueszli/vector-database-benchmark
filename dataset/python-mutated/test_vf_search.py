from bzrlib import graph as _mod_graph, tests, vf_search
from bzrlib.revision import NULL_REVISION
from bzrlib.tests.test_graph import TestGraphBase
ancestry_1 = {'rev1': [NULL_REVISION], 'rev2a': ['rev1'], 'rev2b': ['rev1'], 'rev3': ['rev2a'], 'rev4': ['rev3', 'rev2b']}
ancestry_2 = {'rev1a': [NULL_REVISION], 'rev2a': ['rev1a'], 'rev1b': [NULL_REVISION], 'rev3a': ['rev2a'], 'rev4a': ['rev3a']}
extended_history_shortcut = {'a': [NULL_REVISION], 'b': ['a'], 'c': ['b'], 'd': ['c'], 'e': ['d'], 'f': ['a', 'd']}

class TestSearchResultRefine(tests.TestCase):

    def make_graph(self, ancestors):
        if False:
            return 10
        return _mod_graph.Graph(_mod_graph.DictParentsProvider(ancestors))

    def test_refine(self):
        if False:
            for i in range(10):
                print('nop')
        g = self.make_graph({'tip': ['mid'], 'mid': ['base'], 'tag': ['base'], 'base': [NULL_REVISION], NULL_REVISION: []})
        result = vf_search.SearchResult(set(['tip', 'tag']), set([NULL_REVISION]), 4, set(['tip', 'mid', 'tag', 'base']))
        result = result.refine(set(['tip']), set(['mid']))
        recipe = result.get_recipe()
        self.assertEqual(set(['mid', 'tag']), recipe[1])
        self.assertEqual(set([NULL_REVISION, 'tip']), recipe[2])
        self.assertEqual(3, recipe[3])
        result = result.refine(set(['mid', 'tag', 'base']), set([NULL_REVISION]))
        recipe = result.get_recipe()
        self.assertEqual(set([]), recipe[1])
        self.assertEqual(set([NULL_REVISION, 'tip', 'tag', 'mid']), recipe[2])
        self.assertEqual(0, recipe[3])
        self.assertTrue(result.is_empty())

class TestSearchResultFromParentMap(TestGraphBase):

    def assertSearchResult(self, start_keys, stop_keys, key_count, parent_map, missing_keys=()):
        if False:
            print('Hello World!')
        (start, stop, count) = vf_search.search_result_from_parent_map(parent_map, missing_keys)
        self.assertEqual((sorted(start_keys), sorted(stop_keys), key_count), (sorted(start), sorted(stop), count))

    def test_no_parents(self):
        if False:
            i = 10
            return i + 15
        self.assertSearchResult([], [], 0, {})
        self.assertSearchResult([], [], 0, None)

    def test_ancestry_1(self):
        if False:
            while True:
                i = 10
        self.assertSearchResult(['rev4'], [NULL_REVISION], len(ancestry_1), ancestry_1)

    def test_ancestry_2(self):
        if False:
            print('Hello World!')
        self.assertSearchResult(['rev1b', 'rev4a'], [NULL_REVISION], len(ancestry_2), ancestry_2)
        self.assertSearchResult(['rev1b', 'rev4a'], [], len(ancestry_2) + 1, ancestry_2, missing_keys=[NULL_REVISION])

    def test_partial_search(self):
        if False:
            i = 10
            return i + 15
        parent_map = dict(((k, extended_history_shortcut[k]) for k in ['e', 'f']))
        self.assertSearchResult(['e', 'f'], ['d', 'a'], 2, parent_map)
        parent_map.update(((k, extended_history_shortcut[k]) for k in ['d', 'a']))
        self.assertSearchResult(['e', 'f'], ['c', NULL_REVISION], 4, parent_map)
        parent_map['c'] = extended_history_shortcut['c']
        self.assertSearchResult(['e', 'f'], ['b'], 6, parent_map, missing_keys=[NULL_REVISION])
        parent_map['b'] = extended_history_shortcut['b']
        self.assertSearchResult(['e', 'f'], [], 7, parent_map, missing_keys=[NULL_REVISION])

class TestLimitedSearchResultFromParentMap(TestGraphBase):

    def assertSearchResult(self, start_keys, stop_keys, key_count, parent_map, missing_keys, tip_keys, depth):
        if False:
            for i in range(10):
                print('nop')
        (start, stop, count) = vf_search.limited_search_result_from_parent_map(parent_map, missing_keys, tip_keys, depth)
        self.assertEqual((sorted(start_keys), sorted(stop_keys), key_count), (sorted(start), sorted(stop), count))

    def test_empty_ancestry(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertSearchResult([], [], 0, {}, (), ['tip-rev-id'], 10)

    def test_ancestry_1(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertSearchResult(['rev4'], ['rev1'], 4, ancestry_1, (), ['rev1'], 10)
        self.assertSearchResult(['rev2a', 'rev2b'], ['rev1'], 2, ancestry_1, (), ['rev1'], 1)

    def test_multiple_heads(self):
        if False:
            i = 10
            return i + 15
        self.assertSearchResult(['e', 'f'], ['a'], 5, extended_history_shortcut, (), ['a'], 10)
        self.assertSearchResult(['f'], ['a'], 4, extended_history_shortcut, (), ['a'], 1)
        self.assertSearchResult(['f'], ['a'], 4, extended_history_shortcut, (), ['a'], 2)

class TestPendingAncestryResultRefine(tests.TestCase):

    def make_graph(self, ancestors):
        if False:
            while True:
                i = 10
        return _mod_graph.Graph(_mod_graph.DictParentsProvider(ancestors))

    def test_refine(self):
        if False:
            print('Hello World!')
        g = self.make_graph({'tip': ['mid'], 'mid': ['base'], 'tag': ['base'], 'base': [NULL_REVISION], NULL_REVISION: []})
        result = vf_search.PendingAncestryResult(['tip', 'tag'], None)
        result = result.refine(set(['tip']), set(['mid']))
        self.assertEqual(set(['mid', 'tag']), result.heads)
        result = result.refine(set(['mid', 'tag', 'base']), set([NULL_REVISION]))
        self.assertEqual(set([NULL_REVISION]), result.heads)
        self.assertTrue(result.is_empty())

class TestPendingAncestryResultGetKeys(tests.TestCaseWithMemoryTransport):
    """Tests for bzrlib.graph.PendingAncestryResult."""

    def test_get_keys(self):
        if False:
            return 10
        builder = self.make_branch_builder('b')
        builder.start_series()
        builder.build_snapshot('rev-1', None, [('add', ('', 'root-id', 'directory', ''))])
        builder.build_snapshot('rev-2', ['rev-1'], [])
        builder.finish_series()
        repo = builder.get_branch().repository
        repo.lock_read()
        self.addCleanup(repo.unlock)
        result = vf_search.PendingAncestryResult(['rev-2'], repo)
        self.assertEqual(set(['rev-1', 'rev-2']), set(result.get_keys()))

    def test_get_keys_excludes_ghosts(self):
        if False:
            print('Hello World!')
        builder = self.make_branch_builder('b')
        builder.start_series()
        builder.build_snapshot('rev-1', None, [('add', ('', 'root-id', 'directory', ''))])
        builder.build_snapshot('rev-2', ['rev-1', 'ghost'], [])
        builder.finish_series()
        repo = builder.get_branch().repository
        repo.lock_read()
        self.addCleanup(repo.unlock)
        result = vf_search.PendingAncestryResult(['rev-2'], repo)
        self.assertEqual(sorted(['rev-1', 'rev-2']), sorted(result.get_keys()))

    def test_get_keys_excludes_null(self):
        if False:
            print('Hello World!')

        class StubGraph(object):

            def iter_ancestry(self, keys):
                if False:
                    print('Hello World!')
                return [(NULL_REVISION, ()), ('foo', (NULL_REVISION,))]
        result = vf_search.PendingAncestryResult(['rev-3'], None)
        result_keys = result._get_keys(StubGraph())
        self.assertEqual(set(['foo']), set(result_keys))