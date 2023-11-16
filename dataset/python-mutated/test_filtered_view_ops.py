"""Tests that an enabled view is reported and impacts expected commands."""
from bzrlib import osutils, tests

class TestViewFileOperations(tests.TestCaseWithTransport):

    def make_abc_tree_with_ab_view(self):
        if False:
            print('Hello World!')
        wt = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b', 'c'])
        wt.views.set_view('my', ['a', 'b'])
        return wt

    def test_view_on_status(self):
        if False:
            print('Hello World!')
        wt = self.make_abc_tree_with_ab_view()
        (out, err) = self.run_bzr('status')
        self.assertEqual('Ignoring files outside view. View is a, b\n', err)
        self.assertEqual('unknown:\n  a\n  b\n', out)

    def test_view_on_status_selected(self):
        if False:
            return 10
        wt = self.make_abc_tree_with_ab_view()
        (out, err) = self.run_bzr('status a')
        self.assertEqual('', err)
        self.assertEqual('unknown:\n  a\n', out)
        (out, err) = self.run_bzr('status c', retcode=3)
        self.assertEqual('bzr: ERROR: Specified file "c" is outside the current view: a, b\n', err)
        self.assertEqual('', out)

    def test_view_on_add(self):
        if False:
            i = 10
            return i + 15
        wt = self.make_abc_tree_with_ab_view()
        (out, err) = self.run_bzr('add')
        self.assertEqual('Ignoring files outside view. View is a, b\n', err)
        self.assertEqual('adding a\nadding b\n', out)

    def test_view_on_add_selected(self):
        if False:
            for i in range(10):
                print('nop')
        wt = self.make_abc_tree_with_ab_view()
        (out, err) = self.run_bzr('add a')
        self.assertEqual('', err)
        self.assertEqual('adding a\n', out)
        (out, err) = self.run_bzr('add c', retcode=3)
        self.assertEqual('bzr: ERROR: Specified file "c" is outside the current view: a, b\n', err)
        self.assertEqual('', out)

    def test_view_on_diff(self):
        if False:
            i = 10
            return i + 15
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        (out, err) = self.run_bzr('diff', retcode=1)
        self.assertEqual('*** Ignoring files outside view. View is a, b\n', err)

    def test_view_on_diff_selected(self):
        if False:
            i = 10
            return i + 15
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        (out, err) = self.run_bzr('diff a', retcode=1)
        self.assertEqual('', err)
        self.assertStartsWith(out, "=== added file 'a'\n")
        (out, err) = self.run_bzr('diff c', retcode=3)
        self.assertEqual('bzr: ERROR: Specified file "c" is outside the current view: a, b\n', err)
        self.assertEqual('', out)

    def test_view_on_commit(self):
        if False:
            while True:
                i = 10
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        (out, err) = self.run_bzr('commit -m "testing commit"')
        err_lines = err.splitlines()
        self.assertEqual('Ignoring files outside view. View is a, b', err_lines[0])
        self.assertStartsWith(err_lines[1], 'Committing to:')
        self.assertEqual('added a', err_lines[2])
        self.assertEqual('added b', err_lines[3])
        self.assertEqual('Committed revision 1.', err_lines[4])
        self.assertEqual('', out)

    def test_view_on_commit_selected(self):
        if False:
            i = 10
            return i + 15
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        (out, err) = self.run_bzr('commit -m "file in view" a')
        err_lines = err.splitlines()
        self.assertStartsWith(err_lines[0], 'Committing to:')
        self.assertEqual('added a', err_lines[1])
        self.assertEqual('Committed revision 1.', err_lines[2])
        self.assertEqual('', out)
        (out, err) = self.run_bzr('commit -m "file out of view" c', retcode=3)
        self.assertEqual('bzr: ERROR: Specified file "c" is outside the current view: a, b\n', err)
        self.assertEqual('', out)

    def test_view_on_remove_selected(self):
        if False:
            while True:
                i = 10
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        (out, err) = self.run_bzr('remove --keep a')
        self.assertEqual('removed a\n', err)
        self.assertEqual('', out)
        (out, err) = self.run_bzr('remove --keep c', retcode=3)
        self.assertEqual('bzr: ERROR: Specified file "c" is outside the current view: a, b\n', err)
        self.assertEqual('', out)

    def test_view_on_revert(self):
        if False:
            for i in range(10):
                print('nop')
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        (out, err) = self.run_bzr('revert')
        err_lines = err.splitlines()
        self.assertEqual('Ignoring files outside view. View is a, b', err_lines[0])
        self.assertEqual('-   a', err_lines[1])
        self.assertEqual('-   b', err_lines[2])
        self.assertEqual('', out)

    def test_view_on_revert_selected(self):
        if False:
            for i in range(10):
                print('nop')
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        (out, err) = self.run_bzr('revert a')
        self.assertEqual('-   a\n', err)
        self.assertEqual('', out)
        (out, err) = self.run_bzr('revert c', retcode=3)
        self.assertEqual('bzr: ERROR: Specified file "c" is outside the current view: a, b\n', err)
        self.assertEqual('', out)

    def test_view_on_ls(self):
        if False:
            while True:
                i = 10
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        (out, err) = self.run_bzr('ls')
        out_lines = out.splitlines()
        self.assertEqual('Ignoring files outside view. View is a, b\n', err)
        self.assertEqual('a', out_lines[0])
        self.assertEqual('b', out_lines[1])

class TestViewTreeOperations(tests.TestCaseWithTransport):

    def make_abc_tree_and_clone_with_ab_view(self):
        if False:
            for i in range(10):
                print('nop')
        wt1 = self.make_branch_and_tree('tree_1')
        self.build_tree(['tree_1/a', 'tree_1/b', 'tree_1/c'])
        wt1.add(['a', 'b', 'c'])
        wt1.commit('adding a b c')
        wt2 = wt1.bzrdir.sprout('tree_2').open_workingtree()
        wt2.views.set_view('my', ['a', 'b'])
        self.build_tree_contents([('tree_1/a', 'changed a\n'), ('tree_1/c', 'changed c\n')])
        wt1.commit('changing a c')
        return (wt1, wt2)

    def test_view_on_pull(self):
        if False:
            i = 10
            return i + 15
        (tree_1, tree_2) = self.make_abc_tree_and_clone_with_ab_view()
        (out, err) = self.run_bzr('pull -d tree_2 tree_1')
        self.assertEqualDiff("Operating on whole tree but only reporting on 'my' view.\n M  a\nAll changes applied successfully.\n", err)
        self.assertEqualDiff('Now on revision 2.\n', out)

    def test_view_on_update(self):
        if False:
            print('Hello World!')
        (tree_1, tree_2) = self.make_abc_tree_and_clone_with_ab_view()
        self.run_bzr('bind ../tree_1', working_dir='tree_2')
        (out, err) = self.run_bzr('update', working_dir='tree_2')
        self.assertEqualDiff("Operating on whole tree but only reporting on 'my' view.\n M  a\nAll changes applied successfully.\nUpdated to revision 2 of branch %s\n" % osutils.pathjoin(self.test_dir, 'tree_1'), err)
        self.assertEqual('', out)

    def test_view_on_merge(self):
        if False:
            print('Hello World!')
        (tree_1, tree_2) = self.make_abc_tree_and_clone_with_ab_view()
        (out, err) = self.run_bzr('merge -d tree_2 tree_1')
        self.assertEqualDiff("Operating on whole tree but only reporting on 'my' view.\n M  a\nAll changes applied successfully.\n", err)
        self.assertEqual('', out)