from odoo.tests import common

class TestDelegation(common.TransactionCase):

    def setUp(self):
        if False:
            return 10
        super(TestDelegation, self).setUp()
        env = self.env
        record = env['delegation.parent'].create({'child0_id': env['delegation.child0'].create({'field_0': 0}).id, 'child1_id': env['delegation.child1'].create({'field_1': 1}).id})
        self.record = record

    def test_delegating_record(self):
        if False:
            i = 10
            return i + 15
        env = self.env
        record = self.record
        self.assertEqual(record.field_0, 0)
        self.assertEqual(record.field_1, 1)

    def test_swap_child(self):
        if False:
            return 10
        env = self.env
        record = self.record
        record.write({'child0_id': env['delegation.child0'].create({'field_0': 42}).id})
        self.assertEqual(record.field_0, 42)

    def test_write(self):
        if False:
            for i in range(10):
                print('nop')
        record = self.record
        record.write({'field_1': 4})
        self.assertEqual(record.field_1, 4)
        self.assertEqual(record.child1_id.field_1, 4)