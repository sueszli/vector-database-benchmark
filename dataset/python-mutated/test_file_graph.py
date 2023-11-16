"""Tests for the per file graph API."""
from bzrlib.tests.per_repository import TestCaseWithRepository

class TestPerFileGraph(TestCaseWithRepository):

    def test_file_graph(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('a', 'contents')])
        tree.add(['a'], ['fileid'])
        revid1 = tree.commit('msg')
        self.build_tree_contents([('a', 'new contents')])
        revid2 = tree.commit('msg')
        self.addCleanup(tree.lock_read().unlock)
        graph = tree.branch.repository.get_file_graph()
        self.assertEqual({('fileid', revid2): (('fileid', revid1),), ('fileid', revid1): ()}, graph.get_parent_map([('fileid', revid2), ('fileid', revid1)]))