from odoo.exceptions import AccessError
from odoo.tests.common import TransactionCase

class TestRules(TransactionCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestRules, self).setUp()
        self.id1 = self.env['test_access_right.some_obj'].create({'val': 1}).id
        self.id2 = self.env['test_access_right.some_obj'].create({'val': -1}).id
        self.env['ir.rule'].create({'name': 'Forbid negatives', 'model_id': self.browse_ref('test_access_rights.model_test_access_right_some_obj').id, 'domain_force': "[('val', '>', 0)]"})

    def test_basic_access(self):
        if False:
            while True:
                i = 10
        env = self.env(user=self.browse_ref('base.public_user'))
        browse2 = env['test_access_right.some_obj'].browse(self.id2)
        browse1 = env['test_access_right.some_obj'].browse(self.id1)
        self.assertEqual(browse1.val, 1)
        with self.assertRaises(AccessError):
            self.assertEqual(browse2.val, -1)

    def test_many2many(self):
        if False:
            while True:
                i = 10
        ' Test assignment of many2many field where rules apply. '
        ids = [self.id1, self.id2]
        container_admin = self.env['test_access_right.container'].create({'some_ids': [(6, 0, ids)]})
        self.assertItemsEqual(container_admin.some_ids.ids, ids)
        container_user = container_admin.sudo(self.browse_ref('base.public_user'))
        self.assertItemsEqual(container_user.some_ids.ids, [self.id1])
        container_user.write({'some_ids': [(6, 0, ids)]})
        self.assertItemsEqual(container_user.some_ids.ids, [self.id1])
        self.assertItemsEqual(container_admin.some_ids.ids, ids)
        container_user.write({'some_ids': [(5,)]})
        self.assertItemsEqual(container_user.some_ids.ids, [])
        self.assertItemsEqual(container_admin.some_ids.ids, [self.id2])