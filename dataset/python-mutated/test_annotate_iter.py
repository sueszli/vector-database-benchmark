"""Test that all Tree's implement .annotate_iter()"""
from bzrlib.tests.per_tree import TestCaseWithTree

class TestAnnotate(TestCaseWithTree):

    def get_simple_tree(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/one', 'first\ncontent\n')])
        tree.add(['one'], ['one-id'])
        tree.commit('one', rev_id='one')
        self.build_tree_contents([('tree/one', 'second\ncontent\n')])
        tree.commit('two', rev_id='two')
        return self._convert_tree(tree)

    def get_tree_with_ghost(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/one', 'first\ncontent\n')])
        tree.add(['one'], ['one-id'])
        tree.commit('one', rev_id='one')
        tree.set_parent_ids(['one', 'ghost-one'])
        self.build_tree_contents([('tree/one', 'second\ncontent\n')])
        tree.commit('two', rev_id='two')
        return self._convert_tree(tree)

    def test_annotate_simple(self):
        if False:
            print('Hello World!')
        tree = self.get_simple_tree()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual([('two', 'second\n'), ('one', 'content\n')], list(tree.annotate_iter('one-id')))

    def test_annotate_with_ghost(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.get_tree_with_ghost()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual([('two', 'second\n'), ('one', 'content\n')], list(tree.annotate_iter('one-id')))