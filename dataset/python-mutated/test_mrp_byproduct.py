from odoo.tests import common

class TestMrpByProduct(common.TransactionCase):

    def setUp(self):
        if False:
            return 10
        super(TestMrpByProduct, self).setUp()
        self.MrpBom = self.env['mrp.bom']
        self.warehouse = self.env.ref('stock.warehouse0')
        route_manufacture = self.warehouse.manufacture_pull_id.route_id.id
        route_mto = self.warehouse.mto_pull_id.route_id.id
        self.uom_unit_id = self.ref('product.product_uom_unit')

        def create_product(name, route_ids=[]):
            if False:
                while True:
                    i = 10
            return self.env['product.product'].create({'name': name, 'type': 'product', 'route_ids': route_ids})
        self.product_a = create_product('Product A', route_ids=[(6, 0, [route_manufacture, route_mto])])
        self.product_b = create_product('Product B', route_ids=[(6, 0, [route_manufacture, route_mto])])
        self.product_c_id = create_product('Product C', route_ids=[]).id

    def test_00_mrp_byproduct(self):
        if False:
            print('Hello World!')
        ' Test by product with production order.'
        bom_product_b = self.MrpBom.create({'product_tmpl_id': self.product_b.product_tmpl_id.id, 'product_qty': 1.0, 'type': 'normal', 'product_uom_id': self.uom_unit_id, 'bom_line_ids': [(0, 0, {'product_id': self.product_c_id, 'product_uom_id': self.uom_unit_id, 'product_qty': 2})]})
        bom_product_a = self.MrpBom.create({'product_tmpl_id': self.product_a.product_tmpl_id.id, 'product_qty': 1.0, 'type': 'normal', 'product_uom_id': self.uom_unit_id, 'bom_line_ids': [(0, 0, {'product_id': self.product_c_id, 'product_uom_id': self.uom_unit_id, 'product_qty': 2})], 'sub_products': [(0, 0, {'product_id': self.product_b.id, 'product_uom_id': self.uom_unit_id, 'product_qty': 1})]})
        mnf_product_a = self.env['mrp.production'].create({'product_id': self.product_a.id, 'product_qty': 2.0, 'product_uom_id': self.uom_unit_id, 'bom_id': bom_product_a.id})
        context = {'active_model': 'mrp.production', 'active_ids': [mnf_product_a.id], 'active_id': mnf_product_a.id}
        self.assertEqual(mnf_product_a.state, 'confirmed', 'Production order should be in state confirmed')
        moves = mnf_product_a.move_raw_ids | mnf_product_a.move_finished_ids
        self.assertTrue(moves, 'No moves are created !')
        product_consume = self.env['mrp.product.produce'].with_context(context).create({'product_qty': 2.0})
        self.assertEqual(len(mnf_product_a.move_raw_ids), 1, 'Wrong consume move on production order.')
        product_consume.do_produce()
        consume_move_c = mnf_product_a.move_raw_ids
        by_product_move = mnf_product_a.move_finished_ids.filtered(lambda x: x.product_id.id == self.product_b.id)
        self.assertEqual(consume_move_c.product_uom_qty, 4, 'Wrong consumed quantity of product c.')
        self.assertEqual(by_product_move.product_uom_qty, 2, 'Wrong produced quantity of sub product.')
        mnf_product_a.post_inventory()
        self.assertFalse(any((move.state != 'done' for move in moves)), 'Moves are not done!')