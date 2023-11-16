"""Test for 'bzr mv'"""
import os
import bzrlib.branch
from bzrlib import osutils, workingtree
from bzrlib.tests import TestCaseWithTransport
from bzrlib.tests.features import CaseInsensitiveFilesystemFeature, SymlinkFeature, UnicodeFilenameFeature

class TestMove(TestCaseWithTransport):

    def assertMoved(self, from_path, to_path):
        if False:
            return 10
        'Assert that to_path is existing and versioned but from_path not. '
        self.assertPathDoesNotExist(from_path)
        self.assertNotInWorkingTree(from_path)
        self.assertPathExists(to_path)
        self.assertInWorkingTree(to_path)

    def test_mv_modes(self):
        if False:
            return 10
        'Test two modes of operation for mv'
        tree = self.make_branch_and_tree('.')
        files = self.build_tree(['a', 'c', 'subdir/'])
        tree.add(['a', 'c', 'subdir'])
        self.run_bzr('mv a b')
        self.assertMoved('a', 'b')
        self.run_bzr('mv b subdir')
        self.assertMoved('b', 'subdir/b')
        self.run_bzr('mv subdir/b a')
        self.assertMoved('subdir/b', 'a')
        self.run_bzr('mv a c subdir')
        self.assertMoved('a', 'subdir/a')
        self.assertMoved('c', 'subdir/c')
        self.run_bzr('mv subdir/a subdir/newa')
        self.assertMoved('subdir/a', 'subdir/newa')

    def test_mv_unversioned(self):
        if False:
            for i in range(10):
                print('nop')
        self.build_tree(['unversioned.txt'])
        self.run_bzr_error(['^bzr: ERROR: Could not rename unversioned.txt => elsewhere. .*unversioned.txt is not versioned\\.$'], 'mv unversioned.txt elsewhere')

    def test_mv_nonexisting(self):
        if False:
            print('Hello World!')
        self.run_bzr_error(['^bzr: ERROR: Could not rename doesnotexist => somewhereelse. .*doesnotexist is not versioned\\.$'], 'mv doesnotexist somewhereelse')

    def test_mv_unqualified(self):
        if False:
            while True:
                i = 10
        self.run_bzr_error(['^bzr: ERROR: missing file argument$'], 'mv')

    def test_mv_invalid(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.build_tree(['test.txt', 'sub1/'])
        tree.add(['test.txt'])
        self.run_bzr_error(['^bzr: ERROR: Could not move to sub1: sub1 is not versioned\\.$'], 'mv test.txt sub1')
        self.run_bzr_error(['^bzr: ERROR: Could not move test.txt => .*hello.txt: sub1 is not versioned\\.$'], 'mv test.txt sub1/hello.txt')

    def test_mv_dirs(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.build_tree(['hello.txt', 'sub1/'])
        tree.add(['hello.txt', 'sub1'])
        self.run_bzr('mv sub1 sub2')
        self.assertMoved('sub1', 'sub2')
        self.run_bzr('mv hello.txt sub2')
        self.assertMoved('hello.txt', 'sub2/hello.txt')
        self.build_tree(['sub1/'])
        tree.add(['sub1'])
        self.run_bzr('mv sub2/hello.txt sub1')
        self.assertMoved('sub2/hello.txt', 'sub1/hello.txt')
        self.run_bzr('mv sub2 sub1')
        self.assertMoved('sub2', 'sub1/sub2')

    def test_mv_relative(self):
        if False:
            for i in range(10):
                print('nop')
        self.build_tree(['sub1/', 'sub1/sub2/', 'sub1/hello.txt'])
        tree = self.make_branch_and_tree('.')
        tree.add(['sub1', 'sub1/sub2', 'sub1/hello.txt'])
        self.run_bzr('mv ../hello.txt .', working_dir='sub1/sub2')
        self.assertPathExists('sub1/sub2/hello.txt')
        self.run_bzr('mv sub2/hello.txt .', working_dir='sub1')
        self.assertMoved('sub1/sub2/hello.txt', 'sub1/hello.txt')

    def test_mv_change_case_file(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        self.build_tree(['test.txt'])
        tree.add(['test.txt'])
        self.run_bzr('mv test.txt Test.txt')
        shape = sorted(os.listdir(u'.'))
        self.assertEqual(['.bzr', 'Test.txt'], shape)
        self.assertInWorkingTree('Test.txt')
        self.assertNotInWorkingTree('test.txt')

    def test_mv_change_case_dir(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        self.build_tree(['foo/'])
        tree.add(['foo'])
        self.run_bzr('mv foo Foo')
        shape = sorted(os.listdir(u'.'))
        self.assertEqual(['.bzr', 'Foo'], shape)
        self.assertInWorkingTree('Foo')
        self.assertNotInWorkingTree('foo')

    def test_mv_change_case_dir_w_files(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        self.build_tree(['foo/', 'foo/bar'])
        tree.add(['foo'])
        self.run_bzr('mv foo Foo')
        shape = sorted(os.listdir(u'.'))
        self.assertEqual(['.bzr', 'Foo'], shape)
        self.assertInWorkingTree('Foo')
        self.assertNotInWorkingTree('foo')

    def test_mv_file_to_wrong_case_dir(self):
        if False:
            for i in range(10):
                print('nop')
        self.requireFeature(CaseInsensitiveFilesystemFeature)
        tree = self.make_branch_and_tree('.')
        self.build_tree(['foo/', 'bar'])
        tree.add(['foo', 'bar'])
        (out, err) = self.run_bzr('mv bar Foo', retcode=3)
        self.assertEqual('', out)
        self.assertEqual('bzr: ERROR: Could not move to Foo: Foo is not versioned.\n', err)

    def test_mv_smoke_aliases(self):
        if False:
            i = 10
            return i + 15
        self.build_tree(['a'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a'])
        self.run_bzr('move a b')
        self.run_bzr('rename b a')

    def test_mv_no_root(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('.')
        self.run_bzr_error(['bzr: ERROR: can not move root of branch'], 'mv . a')

    def test_mv_through_symlinks(self):
        if False:
            print('Hello World!')
        self.requireFeature(SymlinkFeature)
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'a/b'])
        os.symlink('a', 'c')
        os.symlink('.', 'd')
        tree.add(['a', 'a/b', 'c'], ['a-id', 'b-id', 'c-id'])
        self.run_bzr('mv c/b b')
        tree = workingtree.WorkingTree.open('.')
        self.assertEqual('b-id', tree.path2id('b'))

    def test_mv_already_moved_file(self):
        if False:
            return 10
        'Test bzr mv original_file to moved_file.\n\n        Tests if a file which has allready been moved by an external tool,\n        is handled correctly by bzr mv.\n        Setup: a is in the working tree, b does not exist.\n        User does: mv a b; bzr mv a b\n        '
        self.build_tree(['a'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a'])
        osutils.rename('a', 'b')
        self.run_bzr('mv a b')
        self.assertMoved('a', 'b')

    def test_mv_already_moved_file_to_versioned_target(self):
        if False:
            i = 10
            return i + 15
        'Test bzr mv existing_file to versioned_file.\n\n        Tests if an attempt to move an existing versioned file\n        to another versiond file will fail.\n        Setup: a and b are in the working tree.\n        User does: rm b; mv a b; bzr mv a b\n        '
        self.build_tree(['a', 'b'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a', 'b'])
        os.remove('b')
        osutils.rename('a', 'b')
        self.run_bzr_error(['^bzr: ERROR: Could not move a => b. b is already versioned\\.$'], 'mv a b')
        self.assertPathDoesNotExist('a')
        self.assertPathExists('b')

    def test_mv_already_moved_file_into_subdir(self):
        if False:
            return 10
        'Test bzr mv original_file to versioned_directory/file.\n\n        Tests if a file which has already been moved into a versioned\n        directory by an external tool, is handled correctly by bzr mv.\n        Setup: a and sub/ are in the working tree.\n        User does: mv a sub/a; bzr mv a sub/a\n        '
        self.build_tree(['a', 'sub/'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a', 'sub'])
        osutils.rename('a', 'sub/a')
        self.run_bzr('mv a sub/a')
        self.assertMoved('a', 'sub/a')

    def test_mv_already_moved_file_into_unversioned_subdir(self):
        if False:
            for i in range(10):
                print('nop')
        'Test bzr mv original_file to unversioned_directory/file.\n\n        Tests if an attempt to move an existing versioned file\n        into an unversioned directory will fail.\n        Setup: a is in the working tree, sub/ is not.\n        User does: mv a sub/a; bzr mv a sub/a\n        '
        self.build_tree(['a', 'sub/'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a'])
        osutils.rename('a', 'sub/a')
        self.run_bzr_error(['^bzr: ERROR: Could not move a => a: sub is not versioned\\.$'], 'mv a sub/a')
        self.assertPathDoesNotExist('a')
        self.assertPathExists('sub/a')

    def test_mv_already_moved_files_into_subdir(self):
        if False:
            print('Hello World!')
        'Test bzr mv original_files to versioned_directory.\n\n        Tests if files which has already been moved into a versioned\n        directory by an external tool, is handled correctly by bzr mv.\n        Setup: a1, a2, sub are in the working tree.\n        User does: mv a1 sub/.; bzr mv a1 a2 sub\n        '
        self.build_tree(['a1', 'a2', 'sub/'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a1', 'a2', 'sub'])
        osutils.rename('a1', 'sub/a1')
        self.run_bzr('mv a1 a2 sub')
        self.assertMoved('a1', 'sub/a1')
        self.assertMoved('a2', 'sub/a2')

    def test_mv_already_moved_files_into_unversioned_subdir(self):
        if False:
            while True:
                i = 10
        'Test bzr mv original_file to unversioned_directory.\n\n        Tests if an attempt to move existing versioned file\n        into an unversioned directory will fail.\n        Setup: a1, a2 are in the working tree, sub is not.\n        User does: mv a1 sub/.; bzr mv a1 a2 sub\n        '
        self.build_tree(['a1', 'a2', 'sub/'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a1', 'a2'])
        osutils.rename('a1', 'sub/a1')
        self.run_bzr_error(['^bzr: ERROR: Could not move to sub. sub is not versioned\\.$'], 'mv a1 a2 sub')
        self.assertPathDoesNotExist('a1')
        self.assertPathExists('sub/a1')
        self.assertPathExists('a2')
        self.assertPathDoesNotExist('sub/a2')

    def test_mv_already_moved_file_forcing_after(self):
        if False:
            return 10
        'Test bzr mv versioned_file to unversioned_file.\n\n        Tests if an attempt to move an existing versioned file to an existing\n        unversioned file will fail, informing the user to use the --after\n        option to force this.\n        Setup: a is in the working tree, b not versioned.\n        User does: mv a b; touch a; bzr mv a b\n        '
        self.build_tree(['a', 'b'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a'])
        osutils.rename('a', 'b')
        self.build_tree(['a'])
        self.run_bzr_error(['^bzr: ERROR: Could not rename a => b because both files exist. \\(Use --after to tell bzr about a rename that has already happened\\)$'], 'mv a b')
        self.assertPathExists('a')
        self.assertPathExists('b')

    def test_mv_already_moved_file_using_after(self):
        if False:
            i = 10
            return i + 15
        'Test bzr mv --after versioned_file to unversioned_file.\n\n        Tests if an existing versioned file can be forced to move to an\n        existing unversioned file using the --after option. With the result\n        that bazaar considers the unversioned_file to be moved from\n        versioned_file and versioned_file will become unversioned.\n        Setup: a is in the working tree and b exists.\n        User does: mv a b; touch a; bzr mv a b --after\n        Resulting in a => b and a is unknown.\n        '
        self.build_tree(['a', 'b'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a'])
        osutils.rename('a', 'b')
        self.build_tree(['a'])
        self.run_bzr('mv a b --after')
        self.assertPathExists('a')
        self.assertNotInWorkingTree('a')
        self.assertPathExists('b')
        self.assertInWorkingTree('b')

    def test_mv_already_moved_files_forcing_after(self):
        if False:
            i = 10
            return i + 15
        'Test bzr mv versioned_files to directory/unversioned_file.\n\n        Tests if an attempt to move an existing versioned file to an existing\n        unversioned file in some other directory will fail, informing the user\n        to use the --after option to force this.\n\n        Setup: a1, a2, sub are versioned and in the working tree,\n               sub/a1, sub/a2 are in working tree.\n        User does: mv a* sub; touch a1; touch a2; bzr mv a1 a2 sub\n        '
        self.build_tree(['a1', 'a2', 'sub/', 'sub/a1', 'sub/a2'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a1', 'a2', 'sub'])
        osutils.rename('a1', 'sub/a1')
        osutils.rename('a2', 'sub/a2')
        self.build_tree(['a1'])
        self.build_tree(['a2'])
        self.run_bzr_error(['^bzr: ERROR: Could not rename a1 => sub/a1 because both files exist. \\(Use --after to tell bzr about a rename that has already happened\\)$'], 'mv a1 a2 sub')
        self.assertPathExists('a1')
        self.assertPathExists('a2')
        self.assertPathExists('sub/a1')
        self.assertPathExists('sub/a2')

    def test_mv_already_moved_files_using_after(self):
        if False:
            return 10
        'Test bzr mv --after versioned_file to directory/unversioned_file.\n\n        Tests if an existing versioned file can be forced to move to an\n        existing unversioned file in some other directory using the --after\n        option. With the result that bazaar considers\n        directory/unversioned_file to be moved from versioned_file and\n        versioned_file will become unversioned.\n\n        Setup: a1, a2, sub are versioned and in the working tree,\n               sub/a1, sub/a2 are in working tree.\n        User does: mv a* sub; touch a1; touch a2; bzr mv a1 a2 sub --after\n        '
        self.build_tree(['a1', 'a2', 'sub/', 'sub/a1', 'sub/a2'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a1', 'a2', 'sub'])
        osutils.rename('a1', 'sub/a1')
        osutils.rename('a2', 'sub/a2')
        self.build_tree(['a1'])
        self.build_tree(['a2'])
        self.run_bzr('mv a1 a2 sub --after')
        self.assertPathExists('a1')
        self.assertPathExists('a2')
        self.assertPathExists('sub/a1')
        self.assertPathExists('sub/a2')
        self.assertInWorkingTree('sub/a1')
        self.assertInWorkingTree('sub/a2')

    def test_mv_already_moved_directory(self):
        if False:
            for i in range(10):
                print('nop')
        'Use `bzr mv a b` to mark a directory as renamed.\n\n        https://bugs.launchpad.net/bzr/+bug/107967/\n        '
        self.build_tree(['a/', 'c/'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a', 'c'])
        osutils.rename('a', 'b')
        osutils.rename('c', 'd')
        self.run_bzr('mv a b')
        self.assertPathDoesNotExist('a')
        self.assertNotInWorkingTree('a')
        self.assertPathExists('b')
        self.assertInWorkingTree('b')
        self.run_bzr('mv --after c d')
        self.assertPathDoesNotExist('c')
        self.assertNotInWorkingTree('c')
        self.assertPathExists('d')
        self.assertInWorkingTree('d')

    def make_abcd_tree(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/a', 'tree/c'])
        tree.add(['a', 'c'])
        tree.commit('record old names')
        osutils.rename('tree/a', 'tree/b')
        osutils.rename('tree/c', 'tree/d')
        return tree

    def test_mv_auto(self):
        if False:
            print('Hello World!')
        self.make_abcd_tree()
        (out, err) = self.run_bzr('mv --auto', working_dir='tree')
        self.assertEqual(out, '')
        self.assertEqual(err, 'a => b\nc => d\n')
        tree = workingtree.WorkingTree.open('tree')
        self.assertIsNot(None, tree.path2id('b'))
        self.assertIsNot(None, tree.path2id('d'))

    def test_mv_auto_one_path(self):
        if False:
            while True:
                i = 10
        self.make_abcd_tree()
        (out, err) = self.run_bzr('mv --auto tree')
        self.assertEqual(out, '')
        self.assertEqual(err, 'a => b\nc => d\n')
        tree = workingtree.WorkingTree.open('tree')
        self.assertIsNot(None, tree.path2id('b'))
        self.assertIsNot(None, tree.path2id('d'))

    def test_mv_auto_two_paths(self):
        if False:
            return 10
        self.make_abcd_tree()
        (out, err) = self.run_bzr('mv --auto tree tree2', retcode=3)
        self.assertEqual('bzr: ERROR: Only one path may be specified to --auto.\n', err)

    def test_mv_auto_dry_run(self):
        if False:
            while True:
                i = 10
        self.make_abcd_tree()
        (out, err) = self.run_bzr('mv --auto --dry-run', working_dir='tree')
        self.assertEqual(out, '')
        self.assertEqual(err, 'a => b\nc => d\n')
        tree = workingtree.WorkingTree.open('tree')
        self.assertIsNot(None, tree.path2id('a'))
        self.assertIsNot(None, tree.path2id('c'))

    def test_mv_no_auto_dry_run(self):
        if False:
            i = 10
            return i + 15
        self.make_abcd_tree()
        (out, err) = self.run_bzr('mv c d --dry-run', working_dir='tree', retcode=3)
        self.assertEqual('bzr: ERROR: --dry-run requires --auto.\n', err)

    def test_mv_auto_after(self):
        if False:
            return 10
        self.make_abcd_tree()
        (out, err) = self.run_bzr('mv --auto --after', working_dir='tree', retcode=3)
        self.assertEqual('bzr: ERROR: --after cannot be specified with --auto.\n', err)

    def test_mv_quiet(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        self.build_tree(['aaa'])
        tree.add(['aaa'])
        (out, err) = self.run_bzr('mv --quiet aaa bbb')
        self.assertEqual(out, '')
        self.assertEqual(err, '')

    def test_mv_readonly_lightweight_checkout(self):
        if False:
            print('Hello World!')
        branch = self.make_branch('foo')
        branch = bzrlib.branch.Branch.open(self.get_readonly_url('foo'))
        tree = branch.create_checkout('tree', lightweight=True)
        self.build_tree(['tree/path'])
        tree.add('path')
        self.run_bzr(['mv', 'tree/path', 'tree/path2'])

    def test_mv_unversioned_non_ascii(self):
        if False:
            while True:
                i = 10
        'Clear error on mv of an unversioned non-ascii file, see lp:707954'
        self.requireFeature(UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('.')
        self.build_tree([u'§'])
        (out, err) = self.run_bzr_error(['Could not rename', 'not versioned'], ['mv', u'§', 'b'])

    def test_mv_removed_non_ascii(self):
        if False:
            i = 10
            return i + 15
        'Clear error on mv of a removed non-ascii file, see lp:898541'
        self.requireFeature(UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('.')
        self.build_tree([u'§'])
        tree.add([u'§'])
        tree.commit(u'Adding §')
        os.remove(u'§')
        (out, err) = self.run_bzr_error(['Could not rename', 'not exist'], ['mv', u'§', 'b'])