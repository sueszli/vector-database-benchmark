"""Test that all Tree's implement get_symlink_target"""
import os
from bzrlib import osutils, tests
from bzrlib.tests import per_tree
from bzrlib.tests import features

class TestGetSymlinkTarget(per_tree.TestCaseWithTree):

    def get_tree_with_symlinks(self):
        if False:
            print('Hello World!')
        self.requireFeature(features.SymlinkFeature)
        tree = self.make_branch_and_tree('tree')
        os.symlink('foo', 'tree/link')
        os.symlink('../bar', 'tree/rel_link')
        os.symlink('/baz/bing', 'tree/abs_link')
        tree.add(['link', 'rel_link', 'abs_link'], ['link-id', 'rel-link-id', 'abs-link-id'])
        return self._convert_tree(tree)

    def test_get_symlink_target(self):
        if False:
            return 10
        tree = self.get_tree_with_symlinks()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual('foo', tree.get_symlink_target('link-id'))
        self.assertEqual('../bar', tree.get_symlink_target('rel-link-id'))
        self.assertEqual('/baz/bing', tree.get_symlink_target('abs-link-id'))
        self.assertEqual('foo', tree.get_symlink_target('link-id', 'link'))

    def test_get_unicode_symlink_target(self):
        if False:
            i = 10
            return i + 15
        self.requireFeature(features.SymlinkFeature)
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('tree')
        target = u'targ€t'
        os.symlink(target, u'tree/β_link'.encode(osutils._fs_enc))
        tree.add([u'β_link'], ['link-id'])
        tree.lock_read()
        self.addCleanup(tree.unlock)
        actual = tree.get_symlink_target('link-id')
        self.assertEqual(target, actual)