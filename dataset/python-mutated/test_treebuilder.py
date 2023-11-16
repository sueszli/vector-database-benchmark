"""Tests for the TreeBuilder helper class."""
from bzrlib import errors, tests
from bzrlib.memorytree import MemoryTree
from bzrlib.tests import TestCaseWithTransport
from bzrlib.treebuilder import TreeBuilder

class FakeTree(object):
    """A pretend tree to test the calls made by TreeBuilder."""

    def __init__(self):
        if False:
            return 10
        self._calls = []

    def lock_tree_write(self):
        if False:
            print('Hello World!')
        self._calls.append('lock_tree_write')

    def unlock(self):
        if False:
            return 10
        self._calls.append('unlock')

class TestFakeTree(TestCaseWithTransport):

    def testFakeTree(self):
        if False:
            i = 10
            return i + 15
        'Check that FakeTree works as required for the TreeBuilder tests.'
        tree = FakeTree()
        self.assertEqual([], tree._calls)
        tree.lock_tree_write()
        self.assertEqual(['lock_tree_write'], tree._calls)
        tree.unlock()
        self.assertEqual(['lock_tree_write', 'unlock'], tree._calls)

class TestTreeBuilderMemoryTree(tests.TestCaseWithMemoryTransport):

    def test_create(self):
        if False:
            i = 10
            return i + 15
        builder = TreeBuilder()

    def test_start_tree_locks_write(self):
        if False:
            while True:
                i = 10
        builder = TreeBuilder()
        tree = FakeTree()
        builder.start_tree(tree)
        self.assertEqual(['lock_tree_write'], tree._calls)

    def test_start_tree_when_started_fails(self):
        if False:
            i = 10
            return i + 15
        builder = TreeBuilder()
        tree = FakeTree()
        builder.start_tree(tree)
        self.assertRaises(errors.AlreadyBuilding, builder.start_tree, tree)

    def test_finish_tree_not_started_errors(self):
        if False:
            i = 10
            return i + 15
        builder = TreeBuilder()
        self.assertRaises(errors.NotBuilding, builder.finish_tree)

    def test_finish_tree_unlocks(self):
        if False:
            i = 10
            return i + 15
        builder = TreeBuilder()
        tree = FakeTree()
        builder.start_tree(tree)
        builder.finish_tree()
        self.assertEqual(['lock_tree_write', 'unlock'], tree._calls)

    def test_build_tree_not_started_errors(self):
        if False:
            for i in range(10):
                print('nop')
        builder = TreeBuilder()
        self.assertRaises(errors.NotBuilding, builder.build, 'foo')

    def test_build_tree(self):
        if False:
            print('Hello World!')
        'Test building works using a MemoryTree.'
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        builder = TreeBuilder()
        builder.start_tree(tree)
        builder.build(['foo', 'bar/', 'bar/file'])
        self.assertEqual('contents of foo\n', tree.get_file(tree.path2id('foo')).read())
        self.assertEqual('contents of bar/file\n', tree.get_file(tree.path2id('bar/file')).read())
        builder.finish_tree()