from bzrlib.osutils import basename
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

class TestIsControlFilename(TestCaseWithWorkingTree):

    def validate_tree_is_controlfilename(self, tree):
        if False:
            print('Hello World!')
        "check that 'tree' obeys the contract for is_control_filename."
        bzrdirname = basename(tree.bzrdir.transport.base[:-1])
        self.assertTrue(tree.is_control_filename(bzrdirname))
        self.assertTrue(tree.is_control_filename(bzrdirname + '/subdir'))
        self.assertFalse(tree.is_control_filename('dir/' + bzrdirname))
        self.assertFalse(tree.is_control_filename('dir/' + bzrdirname + '/sub'))

    def test_dotbzr_is_control_in_cwd(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        self.validate_tree_is_controlfilename(tree)

    def test_dotbzr_is_control_in_subdir(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('subdir')
        self.validate_tree_is_controlfilename(tree)