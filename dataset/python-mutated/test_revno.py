"""Black-box tests for bzr revno.
"""
import os
from bzrlib import tests
from bzrlib.tests.matchers import ContainsNoVfsCalls

class TestRevno(tests.TestCaseWithTransport):

    def test_revno(self):
        if False:
            while True:
                i = 10

        def bzr(*args, **kwargs):
            if False:
                print('Hello World!')
            return self.run_bzr(*args, **kwargs)[0]
        os.mkdir('a')
        os.chdir('a')
        bzr('init')
        self.assertEqual(int(bzr('revno')), 0)
        with open('foo', 'wb') as f:
            f.write('foo\n')
        bzr('add foo')
        bzr('commit -m foo')
        self.assertEqual(int(bzr('revno')), 1)
        os.mkdir('baz')
        bzr('add baz')
        bzr('commit -m baz')
        self.assertEqual(int(bzr('revno')), 2)
        os.chdir('..')
        self.assertEqual(int(bzr('revno a')), 2)
        self.assertEqual(int(bzr('revno a/baz')), 2)

    def test_revno_tree(self):
        if False:
            return 10
        wt = self.make_branch_and_tree('branch')
        checkout = wt.branch.create_checkout('checkout', lightweight=True)
        self.build_tree(['branch/file'])
        wt.add(['file'])
        wt.commit('mkfile')
        (out, err) = self.run_bzr('revno checkout')
        self.assertEqual('', err)
        self.assertEqual('1\n', out)
        (out, err) = self.run_bzr('revno --tree checkout')
        self.assertEqual('', err)
        self.assertEqual('0\n', out)

    def test_revno_tree_no_tree(self):
        if False:
            while True:
                i = 10
        b = self.make_branch('branch')
        (out, err) = self.run_bzr('revno --tree branch', retcode=3)
        self.assertEqual('', out)
        self.assertEqual('bzr: ERROR: No WorkingTree exists for "branch".\n', err)

    def test_dotted_revno_tree(self):
        if False:
            i = 10
            return i + 15
        builder = self.make_branch_builder('branch')
        builder.start_series()
        builder.build_snapshot('A-id', None, [('add', ('', 'root-id', 'directory', None)), ('add', ('file', 'file-id', 'file', 'content\n'))])
        builder.build_snapshot('B-id', ['A-id'], [])
        builder.build_snapshot('C-id', ['A-id', 'B-id'], [])
        builder.finish_series()
        b = builder.get_branch()
        co_b = b.create_checkout('checkout_b', lightweight=True, revision_id='B-id')
        (out, err) = self.run_bzr('revno checkout_b')
        self.assertEqual('', err)
        self.assertEqual('2\n', out)
        (out, err) = self.run_bzr('revno --tree checkout_b')
        self.assertEqual('', err)
        self.assertEqual('1.1.1\n', out)

    def test_stale_revno_tree(self):
        if False:
            while True:
                i = 10
        builder = self.make_branch_builder('branch')
        builder.start_series()
        builder.build_snapshot('A-id', None, [('add', ('', 'root-id', 'directory', None)), ('add', ('file', 'file-id', 'file', 'content\n'))])
        builder.build_snapshot('B-id', ['A-id'], [])
        builder.build_snapshot('C-id', ['A-id'], [])
        builder.finish_series()
        b = builder.get_branch()
        co_b = b.create_checkout('checkout_b', lightweight=True, revision_id='B-id')
        (out, err) = self.run_bzr('revno checkout_b')
        self.assertEqual('', err)
        self.assertEqual('2\n', out)
        (out, err) = self.run_bzr('revno --tree checkout_b')
        self.assertEqual('', err)
        self.assertEqual('???\n', out)

    def test_revno_with_revision(self):
        if False:
            i = 10
            return i + 15
        wt = self.make_branch_and_tree('.')
        revid1 = wt.commit('rev1')
        revid2 = wt.commit('rev2')
        (out, err) = self.run_bzr('revno -r-2 .')
        self.assertEqual('1\n', out)
        (out, err) = self.run_bzr('revno -rrevid:%s .' % revid1)
        self.assertEqual('1\n', out)

    def test_revno_and_tree_mutually_exclusive(self):
        if False:
            i = 10
            return i + 15
        wt = self.make_branch_and_tree('.')
        (out, err) = self.run_bzr('revno -r-2 --tree .', retcode=3)
        self.assertEqual('', out)
        self.assertEqual('bzr: ERROR: --tree and --revision can not be used together\n', err)

class TestSmartServerRevno(tests.TestCaseWithTransport):

    def test_simple_branch_revno(self):
        if False:
            print('Hello World!')
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', 'thecontents')])
        t.add('foo')
        revid = t.commit('message')
        self.reset_smart_call_log()
        (out, err) = self.run_bzr(['revno', self.get_url('branch')])
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)
        self.assertLength(1, self.hpss_connections)
        self.assertLength(6, self.hpss_calls)

    def test_simple_branch_revno_lookup(self):
        if False:
            while True:
                i = 10
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', 'thecontents')])
        t.add('foo')
        revid1 = t.commit('message')
        revid2 = t.commit('message')
        self.reset_smart_call_log()
        (out, err) = self.run_bzr(['revno', '-rrevid:' + revid1, self.get_url('branch')])
        self.assertLength(5, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)