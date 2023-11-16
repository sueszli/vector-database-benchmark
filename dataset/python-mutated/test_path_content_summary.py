"""Test that all Trees implement path_content_summary."""
import os
from bzrlib import osutils, tests, transform
from bzrlib.tests import features, per_tree
from bzrlib.tests.features import SymlinkFeature

class TestPathContentSummary(per_tree.TestCaseWithTree):

    def _convert_tree(self, tree):
        if False:
            while True:
                i = 10
        result = per_tree.TestCaseWithTree._convert_tree(self, tree)
        result.lock_read()
        self.addCleanup(result.unlock)
        return result

    def check_content_summary_size(self, tree, summary, expected_size):
        if False:
            while True:
                i = 10
        returned_size = summary[1]
        if returned_size == expected_size or (tree.supports_content_filtering() and returned_size is None):
            pass
        else:
            self.fail('invalid size in summary: %r' % (returned_size,))

    def test_symlink_content_summary(self):
        if False:
            print('Hello World!')
        self.requireFeature(SymlinkFeature)
        tree = self.make_branch_and_tree('tree')
        os.symlink('target', 'tree/path')
        tree.add(['path'])
        summary = self._convert_tree(tree).path_content_summary('path')
        self.assertEqual(('symlink', None, None, 'target'), summary)

    def test_unicode_symlink_content_summary(self):
        if False:
            return 10
        self.requireFeature(features.SymlinkFeature)
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('tree')
        os.symlink('target', u'tree/β-path'.encode(osutils._fs_enc))
        tree.add([u'β-path'])
        summary = self._convert_tree(tree).path_content_summary(u'β-path')
        self.assertEqual(('symlink', None, None, 'target'), summary)

    def test_unicode_symlink_target_summary(self):
        if False:
            while True:
                i = 10
        self.requireFeature(features.SymlinkFeature)
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('tree')
        os.symlink(u'tree/β-path'.encode(osutils._fs_enc), 'tree/link')
        tree.add(['link'])
        summary = self._convert_tree(tree).path_content_summary('link')
        self.assertEqual(('symlink', None, None, u'tree/β-path'), summary)

    def test_missing_content_summary(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree')
        summary = self._convert_tree(tree).path_content_summary('path')
        self.assertEqual(('missing', None, None, None), summary)

    def test_file_content_summary_executable(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/path'])
        tree.add(['path'])
        tt = transform.TreeTransform(tree)
        self.addCleanup(tt.finalize)
        tt.set_executability(True, tt.trans_id_tree_path('path'))
        tt.apply()
        summary = self._convert_tree(tree).path_content_summary('path')
        self.assertEqual(4, len(summary))
        self.assertEqual('file', summary[0])
        self.check_content_summary_size(tree, summary, 22)
        self.assertEqual(True, summary[2])
        self.assertSubset((summary[3],), (None, '0c352290ae1c26ca7f97d5b2906c4624784abd60'))

    def test_file_content_summary_not_versioned(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/path'])
        tree = self._convert_tree(tree)
        summary = tree.path_content_summary('path')
        self.assertEqual(4, len(summary))
        if isinstance(tree, (per_tree.DirStateRevisionTree, per_tree.RevisionTree)):
            self.assertEqual('missing', summary[0])
            self.assertIs(None, summary[2])
            self.assertIs(None, summary[3])
        elif isinstance(tree, transform._PreviewTree):
            self.expectFailure('PreviewTree returns "missing" for unversionedfiles', self.assertEqual, 'file', summary[0])
            self.assertEqual('file', summary[0])
        else:
            self.assertEqual('file', summary[0])
            self.check_content_summary_size(tree, summary, 22)
            self.assertEqual(False, summary[2])
        self.assertSubset((summary[3],), (None, '0c352290ae1c26ca7f97d5b2906c4624784abd60'))

    def test_file_content_summary_non_exec(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/path'])
        tree.add(['path'])
        summary = self._convert_tree(tree).path_content_summary('path')
        self.assertEqual(4, len(summary))
        self.assertEqual('file', summary[0])
        self.check_content_summary_size(tree, summary, 22)
        self.assertEqual(False, summary[2])
        self.assertSubset((summary[3],), (None, '0c352290ae1c26ca7f97d5b2906c4624784abd60'))

    def test_dir_content_summary(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/path/'])
        tree.add(['path'])
        summary = self._convert_tree(tree).path_content_summary('path')
        self.assertEqual(('directory', None, None, None), summary)

    def test_tree_content_summary(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree')
        if not tree.branch.repository._format.supports_tree_reference:
            raise tests.TestNotApplicable('Tree references not supported.')
        subtree = self.make_branch_and_tree('tree/path')
        tree.add(['path'])
        summary = self._convert_tree(tree).path_content_summary('path')
        self.assertEqual(4, len(summary))
        self.assertEqual('tree-reference', summary[0])