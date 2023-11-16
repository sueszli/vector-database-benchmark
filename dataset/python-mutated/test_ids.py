from bzrlib import errors
from bzrlib.tests.per_tree import TestCaseWithTree

class IdTests(TestCaseWithTree):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(IdTests, self).setUp()
        work_a = self.make_branch_and_tree('wta')
        self.build_tree(['wta/bla', 'wta/dir/', 'wta/dir/file'])
        work_a.add(['bla', 'dir', 'dir/file'], ['bla-id', 'dir-id', 'file-id'])
        work_a.commit('add files')
        self.tree_a = self.workingtree_to_test_tree(work_a)

    def test_path2id(self):
        if False:
            return 10
        self.assertEqual('bla-id', self.tree_a.path2id('bla'))
        self.assertEqual('dir-id', self.tree_a.path2id('dir'))
        self.assertIs(None, self.tree_a.path2id('idontexist'))

    def test_path2id_list(self):
        if False:
            print('Hello World!')
        self.assertEqual('bla-id', self.tree_a.path2id(['bla']))
        self.assertEqual('dir-id', self.tree_a.path2id(['dir']))
        self.assertEqual('file-id', self.tree_a.path2id(['dir', 'file']))
        self.assertEqual(self.tree_a.get_root_id(), self.tree_a.path2id([]))
        self.assertIs(None, self.tree_a.path2id(['idontexist']))
        self.assertIs(None, self.tree_a.path2id(['dir', 'idontexist']))

    def test_id2path(self):
        if False:
            while True:
                i = 10
        self.addCleanup(self.tree_a.lock_read().unlock)
        self.assertEqual('bla', self.tree_a.id2path('bla-id'))
        self.assertEqual('dir', self.tree_a.id2path('dir-id'))
        self.assertEqual('dir/file', self.tree_a.id2path('file-id'))
        self.assertRaises(errors.NoSuchId, self.tree_a.id2path, 'nonexistant')