"""Black-box tests for 'bzr inventory'."""
import os
from bzrlib.tests import TestCaseWithTransport

class TestInventory(TestCaseWithTransport):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestInventory, self).setUp()
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b/', 'b/c'])
        tree.add(['a', 'b', 'b/c'], ['a-id', 'b-id', 'c-id'])
        tree.commit('init', rev_id='one')
        self.tree = tree

    def assertInventoryEqual(self, expected, args=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "Test that the output of 'bzr inventory' is as expected.\n\n        Any arguments supplied will be passed to run_bzr.\n        "
        command = 'inventory'
        if args is not None:
            command += ' ' + args
        (out, err) = self.run_bzr(command, **kwargs)
        self.assertEqual(expected, out)
        self.assertEqual('', err)

    def test_inventory(self):
        if False:
            i = 10
            return i + 15
        self.assertInventoryEqual('a\nb\nb/c\n')

    def test_inventory_kind(self):
        if False:
            i = 10
            return i + 15
        self.assertInventoryEqual('a\nb/c\n', '--kind file')
        self.assertInventoryEqual('b\n', '--kind directory')

    def test_inventory_show_ids(self):
        if False:
            while True:
                i = 10
        expected = ''.join(('%-50s %s\n' % (path, file_id) for (path, file_id) in [('a', 'a-id'), ('b', 'b-id'), ('b/c', 'c-id')]))
        self.assertInventoryEqual(expected, '--show-ids')

    def test_inventory_specific_files(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertInventoryEqual('a\n', 'a')
        self.assertInventoryEqual('b\nb/c\n', 'b b/c')
        self.assertInventoryEqual('b\nb/c\n', 'b')

    def test_inventory_mixed(self):
        if False:
            return 10
        'Test that we get expected results when mixing parameters'
        a_line = '%-50s %s\n' % ('a', 'a-id')
        b_line = '%-50s %s\n' % ('b', 'b-id')
        c_line = '%-50s %s\n' % ('b/c', 'c-id')
        self.assertInventoryEqual('', '--kind directory a')
        self.assertInventoryEqual(a_line + c_line, '--kind file --show-ids')
        self.assertInventoryEqual(c_line, '--kind file --show-ids b b/c')

    def test_in_subdir(self):
        if False:
            i = 10
            return i + 15
        os.chdir('b')
        self.assertInventoryEqual('a\nb\nb/c\n')
        self.assertInventoryEqual('b\nb/c\n', '.')

    def test_inventory_revision(self):
        if False:
            while True:
                i = 10
        self.build_tree(['b/d', 'e'])
        self.tree.add(['b/d', 'e'], ['d-id', 'e-id'])
        self.tree.commit('add files')
        self.tree.rename_one('b/d', 'd')
        self.tree.commit('rename b/d => d')
        self.assertInventoryEqual('a\nb\nb/c\n', '-r 1')
        self.assertInventoryEqual('a\nb\nb/c\nb/d\ne\n', '-r 2')
        self.assertInventoryEqual('b/d\n', '-r 2 b/d')
        self.assertInventoryEqual('b/d\n', '-r 2 d')
        self.tree.rename_one('e', 'b/e')
        self.tree.commit('rename e => b/e')
        self.assertInventoryEqual('b\nb/c\nb/d\ne\n', '-r 2 b')

    def test_missing_file(self):
        if False:
            while True:
                i = 10
        self.run_bzr_error(['Path\\(s\\) are not versioned: no-such-file'], 'inventory no-such-file')