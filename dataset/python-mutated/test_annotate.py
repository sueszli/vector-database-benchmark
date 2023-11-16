"""Black-box tests for bzr.

These check that it behaves properly when it's invoked through the regular
command-line interface. This doesn't actually run a new interpreter but
rather starts again from the run_bzr function.
"""
from bzrlib import config, tests
from bzrlib.tests.matchers import ContainsNoVfsCalls
from bzrlib.urlutils import joinpath

class TestAnnotate(tests.TestCaseWithTransport):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestAnnotate, self).setUp()
        wt = self.make_branch_and_tree('.')
        b = wt.branch
        self.build_tree_contents([('hello.txt', 'my helicopter\n'), ('nomail.txt', 'nomail\n')])
        wt.add(['hello.txt'])
        self.revision_id_1 = wt.commit('add hello', committer='test@user', timestamp=1165960000.0, timezone=0)
        wt.add(['nomail.txt'])
        self.revision_id_2 = wt.commit('add nomail', committer='no mail', timestamp=1165970000.0, timezone=0)
        self.build_tree_contents([('hello.txt', 'my helicopter\nyour helicopter\n')])
        self.revision_id_3 = wt.commit('mod hello', committer='user@test', timestamp=1166040000.0, timezone=0)
        self.build_tree_contents([('hello.txt', 'my helicopter\nyour helicopter\nall of\nour helicopters\n')])
        self.revision_id_4 = wt.commit('mod hello', committer='user@test', timestamp=1166050000.0, timezone=0)

    def test_help_annotate(self):
        if False:
            i = 10
            return i + 15
        'Annotate command exists'
        (out, err) = self.run_bzr('--no-plugins annotate --help')

    def test_annotate_cmd(self):
        if False:
            while True:
                i = 10
        (out, err) = self.run_bzr('annotate hello.txt')
        self.assertEqual('', err)
        self.assertEqualDiff('1   test@us | my helicopter\n3   user@te | your helicopter\n4   user@te | all of\n            | our helicopters\n', out)

    def test_annotate_cmd_full(self):
        if False:
            for i in range(10):
                print('nop')
        (out, err) = self.run_bzr('annotate hello.txt --all')
        self.assertEqual('', err)
        self.assertEqualDiff('1   test@us | my helicopter\n3   user@te | your helicopter\n4   user@te | all of\n4   user@te | our helicopters\n', out)

    def test_annotate_cmd_long(self):
        if False:
            print('Hello World!')
        (out, err) = self.run_bzr('annotate hello.txt --long')
        self.assertEqual('', err)
        self.assertEqualDiff('1   test@user 20061212 | my helicopter\n3   user@test 20061213 | your helicopter\n4   user@test 20061213 | all of\n                       | our helicopters\n', out)

    def test_annotate_cmd_show_ids(self):
        if False:
            return 10
        (out, err) = self.run_bzr('annotate hello.txt --show-ids')
        max_len = max([len(self.revision_id_1), len(self.revision_id_3), len(self.revision_id_4)])
        self.assertEqual('', err)
        self.assertEqualDiff('%*s | my helicopter\n%*s | your helicopter\n%*s | all of\n%*s | our helicopters\n' % (max_len, self.revision_id_1, max_len, self.revision_id_3, max_len, self.revision_id_4, max_len, ''), out)

    def test_no_mail(self):
        if False:
            for i in range(10):
                print('nop')
        (out, err) = self.run_bzr('annotate nomail.txt')
        self.assertEqual('', err)
        self.assertEqualDiff('2   no mail | nomail\n', out)

    def test_annotate_cmd_revision(self):
        if False:
            i = 10
            return i + 15
        (out, err) = self.run_bzr('annotate hello.txt -r1')
        self.assertEqual('', err)
        self.assertEqualDiff('1   test@us | my helicopter\n', out)

    def test_annotate_cmd_revision3(self):
        if False:
            return 10
        (out, err) = self.run_bzr('annotate hello.txt -r3')
        self.assertEqual('', err)
        self.assertEqualDiff('1   test@us | my helicopter\n3   user@te | your helicopter\n', out)

    def test_annotate_cmd_unknown_revision(self):
        if False:
            while True:
                i = 10
        (out, err) = self.run_bzr('annotate hello.txt -r 10', retcode=3)
        self.assertEqual('', out)
        self.assertContainsRe(err, "Requested revision: '10' does not exist")

    def test_annotate_cmd_two_revisions(self):
        if False:
            return 10
        (out, err) = self.run_bzr('annotate hello.txt -r1..2', retcode=3)
        self.assertEqual('', out)
        self.assertEqual('bzr: ERROR: bzr annotate --revision takes exactly one revision identifier\n', err)

class TestSimpleAnnotate(tests.TestCaseWithTransport):
    """Annotate tests with no complex setup."""

    def _setup_edited_file(self, relpath='.'):
        if False:
            i = 10
            return i + 15
        'Create a tree with a locally edited file.'
        tree = self.make_branch_and_tree(relpath)
        file_relpath = joinpath(relpath, 'file')
        self.build_tree_contents([(file_relpath, 'foo\ngam\n')])
        tree.add('file')
        tree.commit('add file', committer='test@host', rev_id='rev1')
        self.build_tree_contents([(file_relpath, 'foo\nbar\ngam\n')])
        return tree

    def test_annotate_cmd_revspec_branch(self):
        if False:
            i = 10
            return i + 15
        tree = self._setup_edited_file('trunk')
        tree.branch.create_checkout(self.get_url('work'), lightweight=True)
        (out, err) = self.run_bzr(['annotate', 'file', '-r', 'branch:../trunk'], working_dir='work')
        self.assertEqual('', err)
        self.assertEqual('1   test@ho | foo\n            | gam\n', out)

    def test_annotate_edited_file(self):
        if False:
            i = 10
            return i + 15
        tree = self._setup_edited_file()
        self.overrideEnv('BZR_EMAIL', 'current@host2')
        (out, err) = self.run_bzr('annotate file')
        self.assertEqual('1   test@ho | foo\n2?  current | bar\n1   test@ho | gam\n', out)

    def test_annotate_edited_file_no_default(self):
        if False:
            i = 10
            return i + 15
        self.overrideEnv('EMAIL', None)
        self.overrideEnv('BZR_EMAIL', None)
        self.overrideAttr(config, '_auto_user_id', lambda : (None, None))
        tree = self._setup_edited_file()
        (out, err) = self.run_bzr('annotate file')
        self.assertEqual('1   test@ho | foo\n2?  local u | bar\n1   test@ho | gam\n', out)

    def test_annotate_edited_file_show_ids(self):
        if False:
            while True:
                i = 10
        tree = self._setup_edited_file()
        self.overrideEnv('BZR_EMAIL', 'current@host2')
        (out, err) = self.run_bzr('annotate file --show-ids')
        self.assertEqual('    rev1 | foo\ncurrent: | bar\n    rev1 | gam\n', out)

    def _create_merged_file(self):
        if False:
            while True:
                i = 10
        'Create a file with a pending merge and local edit.'
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('file', 'foo\ngam\n')])
        tree.add('file')
        tree.commit('add file', rev_id='rev1', committer='test@host')
        self.build_tree_contents([('file', 'foo\nbar\ngam\n')])
        tree.commit('right', rev_id='rev1.1.1', committer='test@host')
        tree.pull(tree.branch, True, 'rev1')
        self.build_tree_contents([('file', 'foo\nbaz\ngam\n')])
        tree.commit('left', rev_id='rev2', committer='test@host')
        tree.merge_from_branch(tree.branch, 'rev1.1.1')
        self.build_tree_contents([('file', 'local\nfoo\nbar\nbaz\ngam\n')])
        return tree

    def test_annotated_edited_merged_file_revnos(self):
        if False:
            print('Hello World!')
        wt = self._create_merged_file()
        (out, err) = self.run_bzr(['annotate', 'file'])
        email = config.extract_email_address(wt.branch.get_config_stack().get('email'))
        self.assertEqual('3?    %-7s | local\n1     test@ho | foo\n1.1.1 test@ho | bar\n2     test@ho | baz\n1     test@ho | gam\n' % email[:7], out)

    def test_annotated_edited_merged_file_ids(self):
        if False:
            for i in range(10):
                print('nop')
        self._create_merged_file()
        (out, err) = self.run_bzr(['annotate', 'file', '--show-ids'])
        self.assertEqual('current: | local\n    rev1 | foo\nrev1.1.1 | bar\n    rev2 | baz\n    rev1 | gam\n', out)

    def test_annotate_empty_file(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('empty', '')])
        tree.add('empty')
        tree.commit('add empty file')
        (out, err) = self.run_bzr(['annotate', 'empty'])
        self.assertEqual('', out)

    def test_annotate_removed_file(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('empty', '')])
        tree.add('empty')
        tree.commit('add empty file')
        tree.remove('empty')
        tree.commit('remove empty file')
        (out, err) = self.run_bzr(['annotate', '-r1', 'empty'])
        self.assertEqual('', out)

    def test_annotate_empty_file_show_ids(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('empty', '')])
        tree.add('empty')
        tree.commit('add empty file')
        (out, err) = self.run_bzr(['annotate', '--show-ids', 'empty'])
        self.assertEqual('', out)

    def test_annotate_nonexistant_file(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        self.build_tree(['file'])
        tree.add(['file'])
        tree.commit('add a file')
        (out, err) = self.run_bzr(['annotate', 'doesnotexist'], retcode=3)
        self.assertEqual('', out)
        self.assertEqual('bzr: ERROR: doesnotexist is not versioned.\n', err)

    def test_annotate_without_workingtree(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('empty', '')])
        tree.add('empty')
        tree.commit('add empty file')
        bzrdir = tree.branch.bzrdir
        bzrdir.destroy_workingtree()
        self.assertFalse(bzrdir.has_workingtree())
        (out, err) = self.run_bzr(['annotate', 'empty'])
        self.assertEqual('', out)

    def test_annotate_directory(self):
        if False:
            print('Hello World!')
        'Test --directory option'
        wt = self.make_branch_and_tree('a')
        self.build_tree_contents([('a/hello.txt', 'my helicopter\n')])
        wt.add(['hello.txt'])
        wt.commit('commit', committer='test@user')
        (out, err) = self.run_bzr(['annotate', '-d', 'a', 'hello.txt'])
        self.assertEqualDiff('1   test@us | my helicopter\n', out)

class TestSmartServerAnnotate(tests.TestCaseWithTransport):

    def test_simple_annotate(self):
        if False:
            i = 10
            return i + 15
        self.setup_smart_server_with_call_log()
        wt = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/hello.txt', 'my helicopter\n')])
        wt.add(['hello.txt'])
        wt.commit('commit', committer='test@user')
        self.reset_smart_call_log()
        (out, err) = self.run_bzr(['annotate', '-d', self.get_url('branch'), 'hello.txt'])
        self.assertLength(16, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.expectFailure('annotate accesses inventories, which require VFS access', self.assertThat, self.hpss_calls, ContainsNoVfsCalls)