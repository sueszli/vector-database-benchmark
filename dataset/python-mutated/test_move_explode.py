from odoo.tests import common

class TestMoveExplode(common.TransactionCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestMoveExplode, self).setUp()
        self.SaleOrderLine = self.env['sale.order.line']
        self.SaleOrder = self.env['sale.order']
        self.MrpBom = self.env['mrp.bom']
        self.Product = self.env['product.product']
        self.product_bom = self.env.ref('product.product_product_5')
        self.bom = self.env.ref('mrp.mrp_bom_kit')
        self.partner = self.env.ref('base.res_partner_1')
        self.template = self.env.ref('product.product_product_3_product_template')
        self.product_bom_prop = self.env.ref('product.product_product_5')
        self.product_A = self.env.ref('product.product_product_11')
        self.product_B = self.env.ref('product.product_product_12')
        self.pricelist = self.env.ref('product.list0')

    def test_00_sale_move_explode(self):
        if False:
            return 10
        'check that when creating a sale order with a product that has a phantom BoM, move explode into content of the\n            BoM'
        so_vals = {'partner_id': self.partner.id, 'partner_invoice_id': self.partner.id, 'partner_shipping_id': self.partner.id, 'order_line': [(0, 0, {'name': self.product_bom.name, 'product_id': self.product_bom.id, 'product_uom_qty': 1, 'product_uom': self.product_bom.uom_id.id})], 'pricelist_id': self.pricelist.id}
        self.so = self.SaleOrder.create(so_vals)
        self.so.action_confirm()
        move_ids = self.so.picking_ids.mapped('move_lines').ids