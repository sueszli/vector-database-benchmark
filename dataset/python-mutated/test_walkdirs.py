"""Tests for the generic Tree.walkdirs interface."""
import os
from bzrlib import tests
from bzrlib.osutils import has_symlinks
from bzrlib.tests.per_tree import TestCaseWithTree

class TestWalkdirs(TestCaseWithTree):

    def get_all_subdirs_expected(self, tree, symlinks):
        if False:
            i = 10
            return i + 15
        dirblocks = [(('', tree.path2id('')), [('0file', '0file', 'file', None, '2file', 'file'), ('1top-dir', '1top-dir', 'directory', None, '1top-dir', 'directory'), (u'2utfሴfile', u'2utfሴfile', 'file', None, u'0utfሴfile'.encode('utf8'), 'file')]), (('1top-dir', '1top-dir'), [('1top-dir/0file-in-1topdir', '0file-in-1topdir', 'file', None, '1file-in-1topdir', 'file'), ('1top-dir/1dir-in-1topdir', '1dir-in-1topdir', 'directory', None, '0dir-in-1topdir', 'directory')]), (('1top-dir/1dir-in-1topdir', '0dir-in-1topdir'), [])]
        if symlinks:
            dirblocks[0][1].append(('symlink', 'symlink', 'symlink', None, 'symlink', 'symlink'))
        return dirblocks

    def test_walkdir_root(self):
        if False:
            i = 10
            return i + 15
        tree = self.get_tree_with_subdirs_and_all_supported_content_types(has_symlinks())
        tree.lock_read()
        expected_dirblocks = self.get_all_subdirs_expected(tree, has_symlinks())
        result = []
        for (dirinfo, block) in tree.walkdirs():
            newblock = []
            for row in block:
                if row[4] is not None:
                    newblock.append(row[0:3] + (None,) + row[4:])
                else:
                    newblock.append(row)
            result.append((dirinfo, newblock))
        tree.unlock()
        for (pos, item) in enumerate(expected_dirblocks):
            self.assertEqual(item, result[pos])
        self.assertEqual(len(expected_dirblocks), len(result))

    def test_walkdir_subtree(self):
        if False:
            while True:
                i = 10
        tree = self.get_tree_with_subdirs_and_all_supported_content_types(has_symlinks())
        result = []
        tree.lock_read()
        expected_dirblocks = self.get_all_subdirs_expected(tree, has_symlinks())[1:]
        for (dirinfo, block) in tree.walkdirs('1top-dir'):
            newblock = []
            for row in block:
                if row[4] is not None:
                    newblock.append(row[0:3] + (None,) + row[4:])
                else:
                    newblock.append(row)
            result.append((dirinfo, newblock))
        tree.unlock()
        for (pos, item) in enumerate(expected_dirblocks):
            self.assertEqual(item, result[pos])
        self.assertEqual(len(expected_dirblocks), len(result))

    def test_walkdir_versioned_kind(self):
        if False:
            for i in range(10):
                print('nop')
        work_tree = self.make_branch_and_tree('tree')
        work_tree.set_root_id('tree-root')
        self.build_tree(['tree/file', 'tree/dir/'])
        work_tree.add(['file', 'dir'], ['file-id', 'dir-id'])
        os.unlink('tree/file')
        os.rmdir('tree/dir')
        tree = self._convert_tree(work_tree)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        if tree.path2id('file') is None:
            raise tests.TestNotApplicable('Tree type cannot represent dangling ids.')
        expected = [(('', 'tree-root'), [('dir', 'dir', 'unknown', None, 'dir-id', 'directory'), ('file', 'file', 'unknown', None, 'file-id', 'file')]), (('dir', 'dir-id'), [])]
        self.assertEqual(expected, list(tree.walkdirs()))