"""Tests for the Branch.create_checkout"""
from bzrlib.tests import per_branch

class TestCreateCheckout(per_branch.TestCaseWithBranch):

    def test_checkout_format_lightweight(self):
        if False:
            return 10
        'Make sure the new light checkout uses the desired branch format.'
        a_branch = self.make_branch('branch')
        tree = a_branch.create_checkout('checkout', lightweight=True)
        expected_format = a_branch._get_checkout_format(lightweight=True)
        self.assertEqual(expected_format.get_branch_format().network_name(), tree.branch._format.network_name())

    def test_checkout_format_heavyweight(self):
        if False:
            i = 10
            return i + 15
        'Make sure the new heavy checkout uses the desired branch format.'
        a_branch = self.make_branch('branch')
        tree = a_branch.create_checkout('checkout', lightweight=False)
        expected_format = a_branch._get_checkout_format(lightweight=False)
        self.assertEqual(expected_format.get_branch_format().network_name(), tree.branch._format.network_name())

    def test_create_revision_checkout(self):
        if False:
            print('Hello World!')
        'Test that we can create a checkout from an earlier revision.'
        tree1 = self.make_branch_and_tree('base')
        self.build_tree(['base/a'])
        tree1.add(['a'], ['a-id'])
        tree1.commit('first', rev_id='rev-1')
        self.build_tree(['base/b'])
        tree1.add(['b'], ['b-id'])
        tree1.commit('second', rev_id='rev-2')
        tree2 = tree1.branch.create_checkout('checkout', revision_id='rev-1')
        self.assertEqual('rev-1', tree2.last_revision())
        self.assertPathExists('checkout/a')
        self.assertPathDoesNotExist('checkout/b')

    def test_create_lightweight_checkout(self):
        if False:
            while True:
                i = 10
        'We should be able to make a lightweight checkout.'
        tree1 = self.make_branch_and_tree('base')
        tree2 = tree1.branch.create_checkout('checkout', lightweight=True)
        self.assertNotEqual(tree1.basedir, tree2.basedir)
        self.assertEqual(tree1.branch.base, tree2.branch.base)

    def test_create_checkout_exists(self):
        if False:
            return 10
        "We shouldn't fail if the directory already exists."
        tree1 = self.make_branch_and_tree('base')
        self.build_tree(['checkout/'])
        tree2 = tree1.branch.create_checkout('checkout', lightweight=True)