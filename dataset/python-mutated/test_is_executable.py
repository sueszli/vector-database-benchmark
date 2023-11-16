from bzrlib.tests import per_tree
from bzrlib.tests.features import SymlinkFeature

class TestIsExecutable(per_tree.TestCaseWithTree):

    def test_is_executable_dir(self):
        if False:
            i = 10
            return i + 15
        tree = self.get_tree_with_subdirs_and_all_supported_content_types(False)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual(False, tree.is_executable('1top-dir'))

    def test_is_executable_symlink(self):
        if False:
            return 10
        self.requireFeature(SymlinkFeature)
        tree = self.get_tree_with_subdirs_and_all_content_types()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual(False, tree.is_executable('symlink'))