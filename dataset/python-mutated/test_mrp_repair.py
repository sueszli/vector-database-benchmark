from odoo.addons.account.tests.account_test_classes import AccountingTestCase

class TestMrpRepair(AccountingTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestMrpRepair, self).setUp()
        self.MrpRepair = self.env['mrp.repair']
        self.ResUsers = self.env['res.users']
        self.MrpRepairMakeInvoice = self.env['mrp.repair.make_invoice']
        self.res_group_user = self.env.ref('stock.group_stock_user')
        self.res_group_manager = self.env.ref('stock.group_stock_manager')
        self.mrp_repair_rmrp0 = self.env.ref('mrp_repair.mrp_repair_rmrp0')
        self.mrp_repair_rmrp1 = self.env.ref('mrp_repair.mrp_repair_rmrp1')
        self.mrp_repair_rmrp2 = self.env.ref('mrp_repair.mrp_repair_rmrp2')
        self.res_mrp_repair_user = self.ResUsers.create({'name': 'MRP User', 'login': 'maru', 'password': 'maru', 'email': 'mrp_repair_user@yourcompany.com', 'groups_id': [(6, 0, [self.res_group_user.id])]})
        self.res_mrp_repair_manager = self.ResUsers.create({'name': 'MRP Manager', 'login': 'marm', 'password': 'marm', 'email': 'mrp_repair_manager@yourcompany.com', 'groups_id': [(6, 0, [self.res_group_manager.id])]})

    def test_00_mrp_repair_afterinv(self):
        if False:
            i = 10
            return i + 15
        self.mrp_repair_rmrp0.sudo(self.res_mrp_repair_user.id).action_repair_confirm()
        self.assertEqual(self.mrp_repair_rmrp0.state, 'confirmed', 'Mrp repair order should be in "Confirmed" state.')
        self.mrp_repair_rmrp0.action_repair_start()
        self.assertEqual(self.mrp_repair_rmrp0.state, 'under_repair', 'Mrp repair order should be in "Under_repair" state.')
        self.mrp_repair_rmrp0.action_repair_end()
        mrp_make_invoice = self.MrpRepairMakeInvoice.create({'group': True})
        context = {'active_model': 'mrp_repair', 'active_ids': [self.mrp_repair_rmrp0.id], 'active_id': self.mrp_repair_rmrp0.id}
        mrp_make_invoice.with_context(context).make_invoices()
        self.assertEqual(len(self.mrp_repair_rmrp0.invoice_id), 1, 'No invoice exists for this repair order')

    def test_01_mrp_repair_b4inv(self):
        if False:
            for i in range(10):
                print('nop')
        self.mrp_repair_rmrp2.sudo(self.res_mrp_repair_user.id).action_repair_confirm()
        self.mrp_repair_rmrp2.action_repair_invoice_create()
        self.assertEqual(len(self.mrp_repair_rmrp2.invoice_id), 1, 'No invoice exists for this repair order')
        self.mrp_repair_rmrp2.action_repair_start()
        self.mrp_repair_rmrp2.action_repair_end()

    def test_02_mrp_repair_noneinv(self):
        if False:
            return 10
        self.mrp_repair_rmrp1.sudo(self.res_mrp_repair_user.id).action_repair_confirm()
        self.mrp_repair_rmrp1.action_repair_start()
        self.assertEqual(self.mrp_repair_rmrp1.state, 'under_repair', 'Mrp repair order should be in "Under_repair" state.')
        self.mrp_repair_rmrp1.action_repair_end()
        self.assertNotEqual(len(self.mrp_repair_rmrp1.invoice_id), 1, 'Invoice should not exist for this repair order')

    def test_03_mrp_repair_fee(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.mrp_repair_rmrp1.amount_total, 100, 'Amount_total should be 100')
        product_assembly = self.env.ref('product.product_product_5')
        product_uom_hour = self.env.ref('product.product_uom_hour')
        self.MrpRepairFee = self.env['mrp.repair.fee']
        self.MrpRepairFee.create({'name': 'PC Assemble + Custom (PC on Demand)', 'product_id': product_assembly.id, 'product_uom_qty': 1.0, 'product_uom': product_uom_hour.id, 'price_unit': 12.0, 'to_invoice': True, 'repair_id': self.mrp_repair_rmrp1.id})
        self.assertEqual(self.mrp_repair_rmrp1.amount_total, 112, 'Amount_total should be 100')