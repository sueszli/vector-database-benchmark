import os
from bzrlib.errors import BzrCommandError, NoSuchRevision
from bzrlib.tests import TestCaseWithTransport
from bzrlib.workingtree import WorkingTree

class TestRevisionInfo(TestCaseWithTransport):

    def check_output(self, output, *args):
        if False:
            i = 10
            return i + 15
        'Verify that the expected output matches what bzr says.\n\n        The output is supplied first, so that you can supply a variable\n        number of arguments to bzr.\n        '
        self.assertEqual(self.run_bzr(*args)[0], output)

    def test_revision_info(self):
        if False:
            print('Hello World!')
        "Test that 'bzr revision-info' reports the correct thing."
        wt = self.make_branch_and_tree('.')
        wt.commit('Commit one', rev_id='a@r-0-1')
        wt.commit('Commit two', rev_id='a@r-0-1.1.1')
        wt.set_parent_ids(['a@r-0-1', 'a@r-0-1.1.1'])
        wt.branch.set_last_revision_info(1, 'a@r-0-1')
        wt.commit('Commit three', rev_id='a@r-0-2')
        wt.bzrdir.destroy_workingtree()
        values = {'1': '1 a@r-0-1\n', '1.1.1': '1.1.1 a@r-0-1.1.1\n', '2': '2 a@r-0-2\n'}
        self.check_output(values['2'], 'revision-info')
        self.check_output(values['1'], 'revision-info 1')
        self.check_output(values['1.1.1'], 'revision-info 1.1.1')
        self.check_output(values['2'], 'revision-info 2')
        self.check_output(values['1'] + values['2'], 'revision-info 1 2')
        self.check_output('    ' + values['1'] + values['1.1.1'] + '    ' + values['2'], 'revision-info 1 1.1.1 2')
        self.check_output(values['2'] + values['1'], 'revision-info 2 1')
        self.check_output(values['1'], 'revision-info -r 1')
        self.check_output(values['1.1.1'], 'revision-info --revision 1.1.1')
        self.check_output(values['2'], 'revision-info -r 2')
        self.check_output(values['1'] + values['2'], 'revision-info -r 1..2')
        self.check_output('    ' + values['1'] + values['1.1.1'] + '    ' + values['2'], 'revision-info -r 1..1.1.1..2')
        self.check_output(values['2'] + values['1'], 'revision-info -r 2..1')
        self.check_output(values['1'], 'revision-info -r revid:a@r-0-1')
        self.check_output(values['1.1.1'], 'revision-info --revision revid:a@r-0-1.1.1')

    def test_revision_info_explicit_branch_dir(self):
        if False:
            print('Hello World!')
        "Test that 'bzr revision-info' honors the '-d' option."
        wt = self.make_branch_and_tree('branch')
        wt.commit('Commit one', rev_id='a@r-0-1')
        self.check_output('1 a@r-0-1\n', 'revision-info -d branch')

    def test_revision_info_tree(self):
        if False:
            for i in range(10):
                print('nop')
        wt = self.make_branch_and_tree('branch')
        wt.commit('Commit one', rev_id='a@r-0-1')
        wt.branch.create_checkout('checkout', lightweight=True)
        wt.commit('Commit two', rev_id='a@r-0-2')
        self.check_output('2 a@r-0-2\n', 'revision-info -d checkout')
        self.check_output('1 a@r-0-1\n', 'revision-info --tree -d checkout')

    def test_revision_info_tree_no_working_tree(self):
        if False:
            while True:
                i = 10
        b = self.make_branch('branch')
        (out, err) = self.run_bzr('revision-info --tree -d branch', retcode=3)
        self.assertEqual('', out)
        self.assertEqual('bzr: ERROR: No WorkingTree exists for "branch".\n', err)

    def test_revision_info_not_in_history(self):
        if False:
            for i in range(10):
                print('nop')
        builder = self.make_branch_builder('branch')
        builder.start_series()
        builder.build_snapshot('A-id', None, [('add', ('', 'root-id', 'directory', None))])
        builder.build_snapshot('B-id', ['A-id'], [])
        builder.build_snapshot('C-id', ['A-id'], [])
        builder.finish_series()
        self.check_output('  1 A-id\n??? B-id\n  2 C-id\n', 'revision-info -d branch revid:A-id revid:B-id revid:C-id')