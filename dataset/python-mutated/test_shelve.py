import os
from bzrlib import shelf
from bzrlib.tests import TestCaseWithTransport
from bzrlib.tests.script import ScriptRunner

class TestShelveList(TestCaseWithTransport):

    def test_no_shelved_changes(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        err = self.run_bzr('shelve --list')[1]
        self.assertEqual('No shelved changes.\n', err)

    def make_creator(self, tree):
        if False:
            i = 10
            return i + 15
        creator = shelf.ShelfCreator(tree, tree.basis_tree(), [])
        self.addCleanup(creator.finalize)
        return creator

    def test_shelve_one(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        creator = self.make_creator(tree)
        shelf_id = tree.get_shelf_manager().shelve_changes(creator, 'Foo')
        (out, err) = self.run_bzr('shelve --list', retcode=1)
        self.assertEqual('', err)
        self.assertEqual('  1: Foo\n', out)

    def test_shelve_list_via_directory(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree')
        creator = self.make_creator(tree)
        shelf_id = tree.get_shelf_manager().shelve_changes(creator, 'Foo')
        (out, err) = self.run_bzr('shelve -d tree --list', retcode=1)
        self.assertEqual('', err)
        self.assertEqual('  1: Foo\n', out)

    def test_shelve_no_message(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('.')
        creator = self.make_creator(tree)
        shelf_id = tree.get_shelf_manager().shelve_changes(creator)
        (out, err) = self.run_bzr('shelve --list', retcode=1)
        self.assertEqual('', err)
        self.assertEqual('  1: <no message>\n', out)

    def test_shelf_order(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        creator = self.make_creator(tree)
        tree.get_shelf_manager().shelve_changes(creator, 'Foo')
        creator = self.make_creator(tree)
        tree.get_shelf_manager().shelve_changes(creator, 'Bar')
        (out, err) = self.run_bzr('shelve --list', retcode=1)
        self.assertEqual('', err)
        self.assertEqual('  2: Bar\n  1: Foo\n', out)

    def test_shelve_destroy(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        self.build_tree(['file'])
        tree.add('file')
        self.run_bzr('shelve --all --destroy')
        self.assertPathDoesNotExist('file')
        self.assertIs(None, tree.get_shelf_manager().last_shelf())

    def test_unshelve_keep(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        tree.commit('make root')
        self.build_tree(['file'])
        sr = ScriptRunner()
        sr.run_script(self, '\n$ bzr add file\nadding file\n$ bzr shelve --all -m Foo\n2>Selected changes:\n2>-D  file\n2>Changes shelved with id "1".\n$ bzr shelve --list\n  1: Foo\n$ bzr unshelve --keep\n2>Using changes with id "1".\n2>Message: Foo\n2>+N  file\n2>All changes applied successfully.\n$ bzr shelve --list\n  1: Foo\n$ cat file\ncontents of file\n')

class TestUnshelvePreview(TestCaseWithTransport):

    def test_non_ascii(self):
        if False:
            return 10
        'Test that we can show a non-ascii diff that would result from unshelving'
        init_content = u'Initial: Изнач\n'.encode('utf-8')
        more_content = u'More: Ещё\n'.encode('utf-8')
        next_content = init_content + more_content
        diff_part = '@@ -1,1 +1,2 @@\n %s+%s' % (init_content, more_content)
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('a_file', init_content)])
        tree.add('a_file')
        tree.commit(message='committed')
        self.build_tree_contents([('a_file', next_content)])
        self.run_bzr(['shelve', '--all'])
        (out, err) = self.run_bzr(['unshelve', '--preview'], encoding='latin-1')
        self.assertContainsString(out, diff_part)

class TestShelveRelpath(TestCaseWithTransport):

    def test_shelve_in_subdir(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/file', 'tree/dir/'])
        tree.add('file')
        os.chdir('tree/dir')
        self.run_bzr('shelve --all ../file')

    def test_shelve_via_directory(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/file', 'tree/dir/'])
        tree.add('file')
        self.run_bzr('shelve -d tree/dir --all ../file')

class TestShelveUnshelve(TestCaseWithTransport):

    def test_directory(self):
        if False:
            i = 10
            return i + 15
        'Test --directory option'
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/a', 'initial\n')])
        tree.add('a')
        tree.commit(message='committed')
        self.build_tree_contents([('tree/a', 'initial\nmore\n')])
        self.run_bzr('shelve -d tree --all')
        self.assertFileEqual('initial\n', 'tree/a')
        self.run_bzr('unshelve --directory tree')
        self.assertFileEqual('initial\nmore\n', 'tree/a')