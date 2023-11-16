from bzrlib.tests import TestNotApplicable
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree
import bzrlib.xml6

class TestBasisInventory(TestCaseWithWorkingTree):

    def test_create(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.workingtree_format, bzrlib.workingtree_4.DirStateWorkingTreeFormat):
            raise TestNotApplicable('not applicable to %r' % (self.workingtree_format,))
        t = self.make_branch_and_tree('.')
        b = t.branch
        with open('a', 'wb') as f:
            f.write('a\n')
        t.add('a')
        t.commit('a', rev_id='r1')
        self.assertTrue(t._transport.has('basis-inventory-cache'))
        basis_inv = t.basis_tree().root_inventory
        self.assertEqual('r1', basis_inv.revision_id)
        store_inv = b.repository.get_inventory('r1')
        self.assertEqual([], store_inv._make_delta(basis_inv))
        with open('b', 'wb') as f:
            f.write('b\n')
        t.add('b')
        t.commit('b', rev_id='r2')
        self.assertTrue(t._transport.has('basis-inventory-cache'))
        basis_inv_txt = t.read_basis_inventory()
        basis_inv = bzrlib.xml7.serializer_v7.read_inventory_from_string(basis_inv_txt)
        self.assertEqual('r2', basis_inv.revision_id)
        store_inv = b.repository.get_inventory('r2')
        self.assertEqual([], store_inv._make_delta(basis_inv))

    def test_wrong_format(self):
        if False:
            for i in range(10):
                print('nop')
        'WorkingTree.basis safely ignores junk basis inventories'
        if isinstance(self.workingtree_format, bzrlib.workingtree_4.DirStateWorkingTreeFormat):
            raise TestNotApplicable('not applicable to %r' % (self.workingtree_format,))
        t = self.make_branch_and_tree('.')
        b = t.branch
        with open('a', 'wb') as f:
            f.write('a\n')
        t.add('a')
        t.commit('a', rev_id='r1')
        t._transport.put_bytes('basis-inventory-cache', 'booga')
        t.basis_tree()
        t._transport.put_bytes('basis-inventory-cache', '<xml/>')
        t.basis_tree()
        t._transport.put_bytes('basis-inventory-cache', '<inventory />')
        t.basis_tree()
        t._transport.put_bytes('basis-inventory-cache', '<inventory format="pi"/>')
        t.basis_tree()