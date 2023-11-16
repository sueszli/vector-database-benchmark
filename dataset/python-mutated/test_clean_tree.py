"""Tests of the 'bzr clean-tree' command."""
import os
from bzrlib import ignores
from bzrlib.tests import TestCaseWithTransport
from bzrlib.tests.script import run_script

class TestBzrTools(TestCaseWithTransport):

    @staticmethod
    def touch(filename):
        if False:
            for i in range(10):
                print('nop')
        my_file = open(filename, 'wb')
        try:
            my_file.write('')
        finally:
            my_file.close()

    def test_clean_tree(self):
        if False:
            print('Hello World!')
        self.run_bzr('init')
        self.run_bzr('ignore *~')
        self.run_bzr('ignore *.pyc')
        self.touch('name')
        self.touch('name~')
        self.assertPathExists('name~')
        self.touch('name.pyc')
        self.run_bzr('clean-tree --force')
        self.assertPathExists('name~')
        self.assertPathDoesNotExist('name')
        self.touch('name')
        self.run_bzr('clean-tree --detritus --force')
        self.assertPathExists('name')
        self.assertPathDoesNotExist('name~')
        self.assertPathExists('name.pyc')
        self.run_bzr('clean-tree --ignored --force')
        self.assertPathExists('name')
        self.assertPathDoesNotExist('name.pyc')
        self.run_bzr('clean-tree --unknown --force')
        self.assertPathDoesNotExist('name')
        self.touch('name')
        self.touch('name~')
        self.touch('name.pyc')
        self.run_bzr('clean-tree --unknown --ignored --force')
        self.assertPathDoesNotExist('name')
        self.assertPathDoesNotExist('name~')
        self.assertPathDoesNotExist('name.pyc')

    def test_clean_tree_nested_bzrdir(self):
        if False:
            i = 10
            return i + 15
        wt1 = self.make_branch_and_tree('.')
        wt2 = self.make_branch_and_tree('foo')
        wt3 = self.make_branch_and_tree('bar')
        ignores.tree_ignores_add_patterns(wt1, ['./foo'])
        self.run_bzr(['clean-tree', '--unknown', '--force'])
        self.assertPathExists('foo')
        self.assertPathExists('bar')
        self.run_bzr(['clean-tree', '--ignored', '--force'])
        self.assertPathExists('foo')
        self.assertPathExists('bar')

    def test_clean_tree_directory(self):
        if False:
            return 10
        'Test --directory option'
        tree = self.make_branch_and_tree('a')
        self.build_tree(['a/added', 'a/unknown', 'a/ignored'])
        tree.add('added')
        self.run_bzr('clean-tree -d a --unknown --ignored --force')
        self.assertPathDoesNotExist('a/unknown')
        self.assertPathDoesNotExist('a/ignored')
        self.assertPathExists('a/added')

    def test_clean_tree_interactive(self):
        if False:
            i = 10
            return i + 15
        wt = self.make_branch_and_tree('.')
        self.touch('bar')
        self.touch('foo')
        run_script(self, '\n        $ bzr clean-tree\n        bar\n        foo\n        2>Are you sure you wish to delete these? ([y]es, [n]o): no\n        <n\n        Canceled\n        ')
        self.assertPathExists('bar')
        self.assertPathExists('foo')
        run_script(self, '\n        $ bzr clean-tree\n        bar\n        foo\n        2>Are you sure you wish to delete these? ([y]es, [n]o): yes\n        <y\n        2>deleting paths:\n        2>  bar\n        2>  foo\n        ')
        self.assertPathDoesNotExist('bar')
        self.assertPathDoesNotExist('foo')