from cStringIO import StringIO
from bzrlib import add, inventory, osutils, tests

class AddCustomIDAction(add.AddAction):

    def __call__(self, inv, parent_ie, path, kind):
        if False:
            for i in range(10):
                print('nop')
        file_id = osutils.safe_file_id(kind + '-' + path.replace('/', '%'), warn=False)
        if self.should_print:
            self._to_file.write('added %s with id %s\n' % (path, file_id))
        return file_id

class TestAddFrom(tests.TestCaseWithTransport):
    """Tests for AddFromBaseAction"""

    def make_base_tree(self):
        if False:
            return 10
        self.base_tree = self.make_branch_and_tree('base')
        self.build_tree(['base/a', 'base/b', 'base/dir/', 'base/dir/a', 'base/dir/subdir/', 'base/dir/subdir/b'])
        self.base_tree.add(['a', 'b', 'dir', 'dir/a', 'dir/subdir', 'dir/subdir/b'])
        self.base_tree.commit('creating initial tree.')

    def add_helper(self, base_tree, base_path, new_tree, file_list, should_print=False):
        if False:
            i = 10
            return i + 15
        to_file = StringIO()
        base_tree.lock_read()
        try:
            new_tree.lock_write()
            try:
                action = add.AddFromBaseAction(base_tree, base_path, to_file=to_file, should_print=should_print)
                new_tree.smart_add(file_list, action=action)
            finally:
                new_tree.unlock()
        finally:
            base_tree.unlock()
        return to_file.getvalue()

    def test_copy_all(self):
        if False:
            i = 10
            return i + 15
        self.make_base_tree()
        new_tree = self.make_branch_and_tree('new')
        files = ['a', 'b', 'dir/', 'dir/a', 'dir/subdir/', 'dir/subdir/b']
        self.build_tree(['new/' + fn for fn in files])
        self.add_helper(self.base_tree, '', new_tree, ['new'])
        for fn in files:
            base_file_id = self.base_tree.path2id(fn)
            new_file_id = new_tree.path2id(fn)
            self.assertEqual(base_file_id, new_file_id)

    def test_copy_from_dir(self):
        if False:
            while True:
                i = 10
        self.make_base_tree()
        new_tree = self.make_branch_and_tree('new')
        self.build_tree(['new/a', 'new/b', 'new/c', 'new/subdir/', 'new/subdir/b', 'new/subdir/d'])
        new_tree.set_root_id(self.base_tree.get_root_id())
        self.add_helper(self.base_tree, 'dir', new_tree, ['new'])
        self.assertEqual(self.base_tree.path2id('a'), new_tree.path2id('a'))
        self.assertEqual(self.base_tree.path2id('b'), new_tree.path2id('b'))
        self.assertEqual(self.base_tree.path2id('dir/subdir'), new_tree.path2id('subdir'))
        self.assertEqual(self.base_tree.path2id('dir/subdir/b'), new_tree.path2id('subdir/b'))
        c_id = new_tree.path2id('c')
        self.assertNotEqual(None, c_id)
        self.base_tree.lock_read()
        self.addCleanup(self.base_tree.unlock)
        self.assertFalse(self.base_tree.has_id(c_id))
        d_id = new_tree.path2id('subdir/d')
        self.assertNotEqual(None, d_id)
        self.assertFalse(self.base_tree.has_id(d_id))

    def test_copy_existing_dir(self):
        if False:
            return 10
        self.make_base_tree()
        new_tree = self.make_branch_and_tree('new')
        self.build_tree(['new/subby/', 'new/subby/a', 'new/subby/b'])
        subdir_file_id = self.base_tree.path2id('dir/subdir')
        new_tree.add(['subby'], [subdir_file_id])
        self.add_helper(self.base_tree, '', new_tree, ['new'])
        self.assertEqual(self.base_tree.path2id('dir/subdir/b'), new_tree.path2id('subby/b'))
        a_id = new_tree.path2id('subby/a')
        self.assertNotEqual(None, a_id)
        self.base_tree.lock_read()
        self.addCleanup(self.base_tree.unlock)
        self.assertFalse(self.base_tree.has_id(a_id))

class TestAddActions(tests.TestCase):

    def test_quiet(self):
        if False:
            return 10
        self.run_action('')

    def test__print(self):
        if False:
            print('Hello World!')
        self.run_action('adding path\n')

    def run_action(self, output):
        if False:
            i = 10
            return i + 15
        inv = inventory.Inventory()
        stdout = StringIO()
        action = add.AddAction(to_file=stdout, should_print=bool(output))
        self.apply_redirected(None, stdout, None, action, inv, None, 'path', 'file')
        self.assertEqual(stdout.getvalue(), output)