"""Black-box tests for bzr dpush."""
from bzrlib import branch, tests
from bzrlib.tests import script, test_foreign
from bzrlib.tests.blackbox import test_push
from bzrlib.tests.scenarios import load_tests_apply_scenarios
load_tests = load_tests_apply_scenarios

class TestDpush(tests.TestCaseWithTransport):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestDpush, self).setUp()
        test_foreign.register_dummy_foreign_for_test(self)

    def make_dummy_builder(self, relpath):
        if False:
            return 10
        builder = self.make_branch_builder(relpath, format=test_foreign.DummyForeignVcsDirFormat())
        builder.build_snapshot('revid', None, [('add', ('', 'TREE_ROOT', 'directory', None)), ('add', ('foo', 'fooid', 'file', 'bar'))])
        return builder

    def test_dpush_native(self):
        if False:
            print('Hello World!')
        target_tree = self.make_branch_and_tree('dp')
        source_tree = self.make_branch_and_tree('dc')
        (output, error) = self.run_bzr('dpush -d dc dp', retcode=3)
        self.assertEqual('', output)
        self.assertContainsRe(error, 'in the same VCS, lossy push not necessary. Please use regular push.')

    def test_dpush(self):
        if False:
            i = 10
            return i + 15
        branch = self.make_dummy_builder('d').get_branch()
        dc = branch.bzrdir.sprout('dc', force_new_repo=True)
        self.build_tree(('dc/foo', 'blaaaa'))
        dc.open_workingtree().commit('msg')
        script.run_script(self, '\n            $ bzr dpush -d dc d\n            2>Doing on-the-fly conversion from DummyForeignVcsRepositoryFormat() to RepositoryFormat2a().\n            2>This may take some time. Upgrade the repositories to the same format for better performance.\n            2>Pushed up to revision 2.\n            $ bzr status dc\n            ')

    def test_dpush_new(self):
        if False:
            while True:
                i = 10
        b = self.make_dummy_builder('d').get_branch()
        dc = b.bzrdir.sprout('dc', force_new_repo=True)
        self.build_tree_contents([('dc/foofile', 'blaaaa')])
        dc_tree = dc.open_workingtree()
        dc_tree.add('foofile')
        dc_tree.commit('msg')
        script.run_script(self, '\n            $ bzr dpush -d dc d\n            2>Doing on-the-fly conversion from DummyForeignVcsRepositoryFormat() to RepositoryFormat2a().\n            2>This may take some time. Upgrade the repositories to the same format for better performance.\n            2>Pushed up to revision 2.\n            $ bzr revno dc\n            2\n            $ bzr status dc\n            ')

    def test_dpush_wt_diff(self):
        if False:
            i = 10
            return i + 15
        b = self.make_dummy_builder('d').get_branch()
        dc = b.bzrdir.sprout('dc', force_new_repo=True)
        self.build_tree_contents([('dc/foofile', 'blaaaa')])
        dc_tree = dc.open_workingtree()
        dc_tree.add('foofile')
        newrevid = dc_tree.commit('msg')
        self.build_tree_contents([('dc/foofile', 'blaaaal')])
        script.run_script(self, '\n            $ bzr dpush -d dc d --no-strict\n            2>Doing on-the-fly conversion from DummyForeignVcsRepositoryFormat() to RepositoryFormat2a().\n            2>This may take some time. Upgrade the repositories to the same format for better performance.\n            2>Pushed up to revision 2.\n            ')
        self.assertFileEqual('blaaaal', 'dc/foofile')
        script.run_script(self, '\n            $ bzr status dc\n            modified:\n              foofile\n            ')

    def test_diverged(self):
        if False:
            while True:
                i = 10
        builder = self.make_dummy_builder('d')
        b = builder.get_branch()
        dc = b.bzrdir.sprout('dc', force_new_repo=True)
        dc_tree = dc.open_workingtree()
        self.build_tree_contents([('dc/foo', 'bar')])
        dc_tree.commit('msg1')
        builder.build_snapshot('revid2', None, [('modify', ('fooid', 'blie'))])
        (output, error) = self.run_bzr('dpush -d dc d', retcode=3)
        self.assertEqual(output, '')
        self.assertContainsRe(error, 'have diverged')

class TestDpushStrictMixin(object):

    def setUp(self):
        if False:
            while True:
                i = 10
        test_foreign.register_dummy_foreign_for_test(self)
        self.foreign = self.make_branch('to', format=test_foreign.DummyForeignVcsDirFormat())

    def set_config_push_strict(self, value):
        if False:
            for i in range(10):
                print('nop')
        br = branch.Branch.open('local')
        br.get_config_stack().set('dpush_strict', value)
    _default_command = ['dpush', '../to']

class TestDpushStrictWithoutChanges(TestDpushStrictMixin, test_push.TestPushStrictWithoutChanges):

    def setUp(self):
        if False:
            return 10
        test_push.TestPushStrictWithoutChanges.setUp(self)
        TestDpushStrictMixin.setUp(self)

class TestDpushStrictWithChanges(TestDpushStrictMixin, test_push.TestPushStrictWithChanges):
    scenarios = test_push.strict_push_change_scenarios
    _changes_type = None

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        test_push.TestPushStrictWithChanges.setUp(self)
        TestDpushStrictMixin.setUp(self)

    def test_push_with_revision(self):
        if False:
            for i in range(10):
                print('nop')
        raise tests.TestNotApplicable('dpush does not handle --revision')