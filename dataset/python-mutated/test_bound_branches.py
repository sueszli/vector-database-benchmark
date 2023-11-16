"""Tests of bound branches (binding, unbinding, commit, etc) command."""
from bzrlib import branch, controldir, errors, tests
from bzrlib.tests import script

class TestBoundBranches(tests.TestCaseWithTransport):

    def create_branches(self):
        if False:
            i = 10
            return i + 15
        base_tree = self.make_branch_and_tree('base')
        base_tree.lock_write()
        self.build_tree(['base/a', 'base/b'])
        base_tree.add(['a', 'b'])
        base_tree.commit('init')
        base_tree.unlock()
        child_tree = base_tree.branch.create_checkout('child')
        self.check_revno(1, 'child')
        d = controldir.ControlDir.open('child')
        self.assertNotEqual(None, d.open_branch().get_master_branch())
        return (base_tree, child_tree)

    def check_revno(self, val, loc='.'):
        if False:
            i = 10
            return i + 15
        self.assertEqual(val, controldir.ControlDir.open(loc).open_branch().last_revision_info()[0])

    def test_simple_binding(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('base')
        self.build_tree(['base/a', 'base/b'])
        tree.add('a', 'b')
        tree.commit(message='init')
        tree.bzrdir.sprout('child')
        self.run_bzr('bind ../base', working_dir='child')
        d = controldir.ControlDir.open('child')
        self.assertNotEqual(None, d.open_branch().get_master_branch())
        self.run_bzr('unbind', working_dir='child')
        self.assertEqual(None, d.open_branch().get_master_branch())
        self.run_bzr('unbind', retcode=3, working_dir='child')

    def test_bind_branch6(self):
        if False:
            print('Hello World!')
        branch1 = self.make_branch('branch1', format='dirstate-tags')
        error = self.run_bzr('bind', retcode=3, working_dir='branch1')[1]
        self.assertEndsWith(error, 'No location supplied and no previous location known\n')

    def setup_rebind(self, format):
        if False:
            while True:
                i = 10
        branch1 = self.make_branch('branch1')
        branch2 = self.make_branch('branch2', format=format)
        branch2.bind(branch1)
        branch2.unbind()

    def test_rebind_branch6(self):
        if False:
            i = 10
            return i + 15
        self.setup_rebind('dirstate-tags')
        self.run_bzr('bind', working_dir='branch2')
        b = branch.Branch.open('branch2')
        self.assertEndsWith(b.get_bound_location(), '/branch1/')

    def test_rebind_branch5(self):
        if False:
            while True:
                i = 10
        self.setup_rebind('knit')
        error = self.run_bzr('bind', retcode=3, working_dir='branch2')[1]
        self.assertEndsWith(error, 'No location supplied.  This format does not remember old locations.\n')

    def test_bound_commit(self):
        if False:
            while True:
                i = 10
        child_tree = self.create_branches()[1]
        self.build_tree_contents([('child/a', 'new contents')])
        child_tree.commit(message='child')
        self.check_revno(2, 'child')
        self.check_revno(2, 'base')

    def test_bound_fail(self):
        if False:
            i = 10
            return i + 15
        (base_tree, child_tree) = self.create_branches()
        self.build_tree_contents([('base/a', 'new base contents\n'), ('child/b', 'new b child contents\n')])
        base_tree.commit(message='base')
        self.check_revno(2, 'base')
        self.check_revno(1, 'child')
        self.assertRaises(errors.BoundBranchOutOfDate, child_tree.commit, message='child')
        self.check_revno(1, 'child')
        child_tree.update()
        self.check_revno(2, 'child')
        child_tree.commit(message='child')
        self.check_revno(3, 'child')
        self.check_revno(3, 'base')

    def test_double_binding(self):
        if False:
            for i in range(10):
                print('nop')
        child_tree = self.create_branches()[1]
        child_tree.bzrdir.sprout('child2')
        self.run_bzr('bind ../child', working_dir='child2')
        child2_tree = controldir.ControlDir.open('child2').open_workingtree()
        self.assertRaises(errors.CommitToDoubleBoundBranch, child2_tree.commit, message='child2', allow_pointless=True)

    def test_unbinding(self):
        if False:
            for i in range(10):
                print('nop')
        (base_tree, child_tree) = self.create_branches()
        self.build_tree_contents([('base/a', 'new base contents\n'), ('child/b', 'new b child contents\n')])
        base_tree.commit(message='base')
        self.check_revno(2, 'base')
        self.check_revno(1, 'child')
        self.run_bzr('commit -m child', retcode=3, working_dir='child')
        self.check_revno(1, 'child')
        self.run_bzr('unbind', working_dir='child')
        child_tree = child_tree.bzrdir.open_workingtree()
        child_tree.commit(message='child')
        self.check_revno(2, 'child')

    def test_commit_remote_bound(self):
        if False:
            i = 10
            return i + 15
        (base_tree, child_tree) = self.create_branches()
        base_tree.bzrdir.sprout('newbase')
        self.run_bzr('bind ../newbase', working_dir='base')
        self.run_bzr('commit -m failure --unchanged', retcode=3, working_dir='child')

    def test_pull_updates_both(self):
        if False:
            i = 10
            return i + 15
        base_tree = self.create_branches()[0]
        newchild_tree = base_tree.bzrdir.sprout('newchild').open_workingtree()
        self.build_tree_contents([('newchild/b', 'newchild b contents\n')])
        newchild_tree.commit(message='newchild')
        self.check_revno(2, 'newchild')
        self.run_bzr('pull ../newchild', working_dir='child')
        self.check_revno(2, 'child')
        self.check_revno(2, 'base')

    def test_pull_local_updates_local(self):
        if False:
            print('Hello World!')
        base_tree = self.create_branches()[0]
        newchild_tree = base_tree.bzrdir.sprout('newchild').open_workingtree()
        self.build_tree_contents([('newchild/b', 'newchild b contents\n')])
        newchild_tree.commit(message='newchild')
        self.check_revno(2, 'newchild')
        self.run_bzr('pull ../newchild --local', working_dir='child')
        self.check_revno(2, 'child')
        self.check_revno(1, 'base')

    def test_bind_diverged(self):
        if False:
            return 10
        (base_tree, child_tree) = self.create_branches()
        base_branch = base_tree.branch
        child_branch = child_tree.branch
        self.run_bzr('unbind', working_dir='child')
        child_tree = child_tree.bzrdir.open_workingtree()
        child_tree.commit(message='child', allow_pointless=True)
        self.check_revno(2, 'child')
        self.check_revno(1, 'base')
        base_tree.commit(message='base', allow_pointless=True)
        self.check_revno(2, 'base')
        self.run_bzr('bind ../base', working_dir='child')
        child_tree = child_tree.bzrdir.open_workingtree()
        child_tree.update()
        child_tree.commit(message='merged')
        self.check_revno(3, 'child')
        self.assertEqual(child_tree.branch.last_revision(), base_tree.branch.last_revision())

    def test_bind_parent_ahead(self):
        if False:
            i = 10
            return i + 15
        base_tree = self.create_branches()[0]
        self.run_bzr('unbind', working_dir='child')
        base_tree.commit(message='base', allow_pointless=True)
        self.check_revno(1, 'child')
        self.run_bzr('bind ../base', working_dir='child')
        self.check_revno(1, 'child')
        self.run_bzr('unbind', working_dir='child')
        base_tree.commit(message='base 3', allow_pointless=True)
        base_tree.commit(message='base 4', allow_pointless=True)
        base_tree.commit(message='base 5', allow_pointless=True)
        self.check_revno(5, 'base')
        self.check_revno(1, 'child')
        self.run_bzr('bind ../base', working_dir='child')
        self.check_revno(1, 'child')

    def test_bind_child_ahead(self):
        if False:
            return 10
        child_tree = self.create_branches()[1]
        self.run_bzr('unbind', working_dir='child')
        child_tree = child_tree.bzrdir.open_workingtree()
        child_tree.commit(message='child', allow_pointless=True)
        self.check_revno(2, 'child')
        self.check_revno(1, 'base')
        self.run_bzr('bind ../base', working_dir='child')
        self.check_revno(1, 'base')
        self.run_bzr('unbind', working_dir='child')
        child_tree.commit(message='child 3', allow_pointless=True)
        child_tree.commit(message='child 4', allow_pointless=True)
        child_tree.commit(message='child 5', allow_pointless=True)
        self.check_revno(5, 'child')
        self.check_revno(1, 'base')
        self.run_bzr('bind ../base', working_dir='child')
        self.check_revno(1, 'base')

    def test_bind_fail_if_missing(self):
        if False:
            while True:
                i = 10
        'We should not be able to bind to a missing branch.'
        tree = self.make_branch_and_tree('tree_1')
        tree.commit('dummy commit')
        self.run_bzr_error(['Not a branch.*no-such-branch/'], ['bind', '../no-such-branch'], working_dir='tree_1')
        self.assertIs(None, tree.branch.get_bound_location())

    def test_commit_after_merge(self):
        if False:
            print('Hello World!')
        (base_tree, child_tree) = self.create_branches()
        other_tree = child_tree.bzrdir.sprout('other').open_workingtree()
        other_branch = other_tree.branch
        self.build_tree_contents([('other/c', 'file c\n')])
        other_tree.add('c')
        other_tree.commit(message='adding c')
        new_rev_id = other_branch.last_revision()
        child_tree.merge_from_branch(other_branch)
        self.assertPathExists('child/c')
        self.assertEqual([new_rev_id], child_tree.get_parent_ids()[1:])
        self.assertTrue(child_tree.branch.repository.has_revision(new_rev_id))
        self.assertFalse(base_tree.branch.repository.has_revision(new_rev_id))
        self.run_bzr(['commit', '-m', 'merge other'], working_dir='child')
        self.check_revno(2, 'child')
        self.check_revno(2, 'base')
        self.assertTrue(base_tree.branch.repository.has_revision(new_rev_id))

    def test_pull_overwrite(self):
        if False:
            for i in range(10):
                print('nop')
        child_tree = self.create_branches()[1]
        other_tree = child_tree.bzrdir.sprout('other').open_workingtree()
        self.build_tree_contents([('other/a', 'new contents\n')])
        other_tree.commit(message='changed a')
        self.check_revno(2, 'other')
        self.build_tree_contents([('other/a', 'new contents\nand then some\n')])
        other_tree.commit(message='another a')
        self.check_revno(3, 'other')
        self.build_tree_contents([('other/a', 'new contents\nand then some\nand some more\n')])
        other_tree.commit('yet another a')
        self.check_revno(4, 'other')
        self.build_tree_contents([('child/a', 'also changed a\n')])
        child_tree.commit(message='child modified a')
        self.check_revno(2, 'child')
        self.check_revno(2, 'base')
        self.run_bzr('pull --overwrite ../other', working_dir='child')
        self.check_revno(4, 'child')
        self.check_revno(4, 'base')

    def test_bind_directory(self):
        if False:
            i = 10
            return i + 15
        'Test --directory option'
        tree = self.make_branch_and_tree('base')
        self.build_tree(['base/a', 'base/b'])
        tree.add('a', 'b')
        tree.commit(message='init')
        branch = tree.branch
        tree.bzrdir.sprout('child')
        self.run_bzr('bind --directory=child base')
        d = controldir.ControlDir.open('child')
        self.assertNotEqual(None, d.open_branch().get_master_branch())
        self.run_bzr('unbind -d child')
        self.assertEqual(None, d.open_branch().get_master_branch())
        self.run_bzr('unbind --directory child', retcode=3)

class TestBind(script.TestCaseWithTransportAndScript):

    def test_bind_when_bound(self):
        if False:
            return 10
        self.run_script('\n$ bzr init trunk\n...\n$ bzr init copy\n...\n$ cd copy\n$ bzr bind ../trunk\n$ bzr bind\n2>bzr: ERROR: Branch is already bound\n')

    def test_bind_before_bound(self):
        if False:
            return 10
        self.run_script('\n$ bzr init trunk\n...\n$ cd trunk\n$ bzr bind\n2>bzr: ERROR: No location supplied and no previous location known\n')