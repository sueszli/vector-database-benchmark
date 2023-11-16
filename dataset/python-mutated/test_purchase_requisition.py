from odoo.tests import common
from odoo.tools import float_compare

class TestPurchaseRequisition(common.TransactionCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestPurchaseRequisition, self).setUp()
        self.product_09_id = self.ref('product.product_product_9')
        self.product_09_uom_id = self.ref('product.product_uom_unit')
        self.product_13_id = self.ref('product.product_product_13')
        self.res_partner_1_id = self.ref('base.res_partner_1')
        self.res_company_id = self.ref('base.main_company')
        self.ResUser = self.env['res.users']
        self.res_users_purchase_requisition_manager = self.ResUser.create({'company_id': self.res_company_id, 'name': 'Purchase requisition Manager', 'login': 'prm', 'email': 'requisition_manager@yourcompany.com'})
        self.res_users_purchase_requisition_manager.group_id = self.ref('purchase.group_purchase_manager')
        self.res_users_purchase_requisition_user = self.ResUser.create({'company_id': self.res_company_id, 'name': 'Purchase requisition User', 'login': 'pru', 'email': 'requisition_user@yourcompany.com'})
        self.res_users_purchase_requisition_user.group_id = self.ref('purchase.group_purchase_user')
        self.requisition1 = self.env['purchase.requisition'].create({'line_ids': [(0, 0, {'product_id': self.product_09_id, 'product_qty': 10.0, 'product_uom_id': self.product_09_uom_id})]})

    def test_00_purchase_requisition_users(self):
        if False:
            return 10
        self.assertTrue(self.res_users_purchase_requisition_manager, 'Manager Should be created')
        self.assertTrue(self.res_users_purchase_requisition_user, 'User Should be created')

    def test_01_cancel_purchase_requisition(self):
        if False:
            while True:
                i = 10
        self.requisition1.sudo(self.res_users_purchase_requisition_user.id).action_cancel()
        self.assertEqual(self.requisition1.state, 'cancel', 'Requisition should be in cancelled state.')
        self.requisition1.sudo(self.res_users_purchase_requisition_user.id).action_draft()
        self.requisition1.sudo(self.res_users_purchase_requisition_user.id).copy()

    def test_02_purchase_requisition(self):
        if False:
            return 10
        procurement_product_hdd3 = self.env['make.procurement'].create({'product_id': self.product_13_id, 'qty': 15, 'uom_id': self.ref('product.product_uom_unit'), 'warehouse_id': self.ref('stock.warehouse0')})
        procurement_product_hdd3.make_procurement()
        ProcurementOrder = self.env['procurement.order']
        ProcurementOrder.run_scheduler()
        procurements = ProcurementOrder.search([('requisition_id', '!=', False)])
        for procurement in procurements:
            requisition = procurement.requisition_id
            self.assertEqual(requisition.date_end, procurement.date_planned, 'End date is not correspond.')
            self.assertEqual(len(requisition.line_ids), 1, 'Requisition Lines should be one.')
            line = requisition.line_ids[0]
            self.assertEqual(line.product_id.id, procurement.product_id.id, 'Product is not correspond.')
            self.assertEqual(line.product_uom_id.id, procurement.product_uom.id, 'UOM is not correspond.')
            self.assertEqual(float_compare(line.product_qty, procurement.product_qty, precision_digits=2), 0, 'Quantity is not correspond.')
        self.requisition1.sudo(self.res_users_purchase_requisition_user.id).action_in_progress()
        self.requisition1.sudo(self.res_users_purchase_requisition_user.id).action_open()
        PurchaseOrder = self.env['purchase.order']
        purchase_order = PurchaseOrder.new({'partner_id': self.res_partner_1_id, 'requisition_id': self.requisition1.id})
        purchase_order._onchange_requisition_id()
        po_dict = purchase_order._convert_to_write({name: purchase_order[name] for name in purchase_order._cache})
        self.po_requisition = PurchaseOrder.create(po_dict)
        self.assertEqual(len(self.po_requisition.order_line), 1, 'Purchase order should have one line')