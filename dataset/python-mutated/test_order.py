from datetime import datetime, timedelta
from odoo.fields import Datetime as Dt
from odoo.addons.mrp.tests.common import TestMrpCommon

class TestMrpOrder(TestMrpCommon):

    def test_access_rights_manager(self):
        if False:
            for i in range(10):
                print('nop')
        man_order = self.env['mrp.production'].sudo(self.user_mrp_manager).create({'name': 'Stick-0', 'product_id': self.product_4.id, 'product_uom_id': self.product_4.uom_id.id, 'product_qty': 5.0, 'bom_id': self.bom_1.id, 'location_src_id': self.location_1.id, 'location_dest_id': self.warehouse_1.wh_output_stock_loc_id.id})
        man_order.action_cancel()
        self.assertEqual(man_order.state, 'cancel', 'Production order should be in cancel state.')
        man_order.unlink()

    def test_access_rights_user(self):
        if False:
            for i in range(10):
                print('nop')
        man_order = self.env['mrp.production'].sudo(self.user_mrp_user).create({'name': 'Stick-0', 'product_id': self.product_4.id, 'product_uom_id': self.product_4.uom_id.id, 'product_qty': 5.0, 'bom_id': self.bom_1.id, 'location_src_id': self.location_1.id, 'location_dest_id': self.warehouse_1.wh_output_stock_loc_id.id})
        man_order.action_cancel()
        self.assertEqual(man_order.state, 'cancel', 'Production order should be in cancel state.')
        man_order.unlink()

    def test_basic(self):
        if False:
            while True:
                i = 10
        ' Basic order test: no routing (thus no workorders), no lot '
        inventory = self.env['stock.inventory'].create({'name': 'Initial inventory', 'filter': 'partial', 'line_ids': [(0, 0, {'product_id': self.product_1.id, 'product_uom_id': self.product_1.uom_id.id, 'product_qty': 500, 'location_id': self.warehouse_1.lot_stock_id.id}), (0, 0, {'product_id': self.product_2.id, 'product_uom_id': self.product_2.uom_id.id, 'product_qty': 500, 'location_id': self.warehouse_1.lot_stock_id.id})]})
        inventory.action_done()
        test_date_planned = datetime.now() - timedelta(days=1)
        test_quantity = 2.0
        self.bom_1.routing_id = False
        man_order = self.env['mrp.production'].sudo(self.user_mrp_user).create({'name': 'Stick-0', 'product_id': self.product_4.id, 'product_uom_id': self.product_4.uom_id.id, 'product_qty': test_quantity, 'bom_id': self.bom_1.id, 'date_planned_start': test_date_planned, 'location_src_id': self.location_1.id, 'location_dest_id': self.warehouse_1.wh_output_stock_loc_id.id})
        self.assertEqual(man_order.state, 'confirmed', 'Production order should be in confirmed state.')
        production_move = man_order.move_finished_ids
        self.assertEqual(production_move.date, Dt.to_string(test_date_planned))
        self.assertEqual(production_move.product_id, self.product_4)
        self.assertEqual(production_move.product_uom, man_order.product_uom_id)
        self.assertEqual(production_move.product_qty, man_order.product_qty)
        self.assertEqual(production_move.location_id, self.product_4.property_stock_production)
        self.assertEqual(production_move.location_dest_id, man_order.location_dest_id)
        for move in man_order.move_raw_ids:
            self.assertEqual(move.date, Dt.to_string(test_date_planned))
        first_move = man_order.move_raw_ids.filtered(lambda move: move.product_id == self.product_2)
        self.assertEqual(first_move.product_qty, test_quantity / self.bom_1.product_qty * self.product_4.uom_id.factor_inv * 2)
        first_move = man_order.move_raw_ids.filtered(lambda move: move.product_id == self.product_1)
        self.assertEqual(first_move.product_qty, test_quantity / self.bom_1.product_qty * self.product_4.uom_id.factor_inv * 4)
        qty_wizard = self.env['change.production.qty'].create({'mo_id': man_order.id, 'product_qty': 3.0})
        produce_wizard = self.env['mrp.product.produce'].sudo(self.user_mrp_user).with_context({'active_id': man_order.id, 'active_ids': [man_order.id]}).create({'product_qty': 1.0})
        produce_wizard.do_produce()
        man_order.button_mark_done()
        self.assertEqual(man_order.state, 'done', 'Production order should be in done state.')

    def test_explode_from_order(self):
        if False:
            return 10
        self.workcenter_1.write({'capacity': 1, 'time_start': 0, 'time_stop': 0, 'time_efficiency': 100})
        self.operation_1.write({'time_cycle_manual': 20})
        (self.operation_2 | self.operation_3).write({'time_cycle_manual': 10})
        man_order = self.env['mrp.production'].create({'name': 'MO-Test', 'product_id': self.product_6.id, 'product_uom_id': self.product_6.uom_id.id, 'product_qty': 48, 'bom_id': self.bom_3.id})
        self.env['stock.change.product.qty'].create({'product_id': self.product_1.id, 'new_quantity': 0.0, 'location_id': self.warehouse_1.lot_stock_id.id}).change_product_qty()
        (self.product_2 | self.product_4).write({'tracking': 'none'})
        man_order.action_assign()
        self.assertEqual(man_order.availability, 'waiting', 'Production order should be in waiting state.')
        self.assertEqual(len(man_order.move_raw_ids), 4, 'Consume material lines are not generated proper.')
        product_2_consume_moves = man_order.move_raw_ids.filtered(lambda x: x.product_id == self.product_2)
        product_3_consume_moves = man_order.move_raw_ids.filtered(lambda x: x.product_id == self.product_3)
        product_4_consume_moves = man_order.move_raw_ids.filtered(lambda x: x.product_id == self.product_4)
        product_5_consume_moves = man_order.move_raw_ids.filtered(lambda x: x.product_id == self.product_5)
        consume_qty_2 = product_2_consume_moves.product_uom_qty
        self.assertEqual(consume_qty_2, 24.0, 'Consume material quantity of Wood should be 24 instead of %s' % str(consume_qty_2))
        consume_qty_3 = product_3_consume_moves.product_uom_qty
        self.assertEqual(consume_qty_3, 12.0, 'Consume material quantity of Stone should be 12 instead of %s' % str(consume_qty_3))
        self.assertEqual(len(product_4_consume_moves), 2, 'Consume move are not generated proper.')
        for consume_moves in product_4_consume_moves:
            consume_qty_4 = consume_moves.product_uom_qty
            self.assertIn(consume_qty_4, [8.0, 16.0], 'Consume material quantity of Stick should be 8 or 16 instead of %s' % str(consume_qty_4))
        self.assertFalse(product_5_consume_moves, 'Move should not create for phantom bom')
        lot_product_2 = self.env['stock.production.lot'].create({'product_id': self.product_2.id})
        lot_product_4 = self.env['stock.production.lot'].create({'product_id': self.product_4.id})
        inventory = self.env['stock.inventory'].create({'name': 'Inventory For Product C', 'filter': 'partial', 'line_ids': [(0, 0, {'product_id': self.product_2.id, 'product_uom_id': self.product_2.uom_id.id, 'product_qty': 30, 'prod_lot_id': lot_product_2.id, 'location_id': self.ref('stock.stock_location_14')}), (0, 0, {'product_id': self.product_3.id, 'product_uom_id': self.product_3.uom_id.id, 'product_qty': 60, 'location_id': self.ref('stock.stock_location_14')}), (0, 0, {'product_id': self.product_4.id, 'product_uom_id': self.product_4.uom_id.id, 'product_qty': 60, 'prod_lot_id': lot_product_4.id, 'location_id': self.ref('stock.stock_location_14')})]})
        inventory.prepare_inventory()
        inventory.action_done()
        man_order.action_assign()
        self.assertEqual(man_order.availability, 'assigned', 'Production order should be in assigned state.')
        man_order.button_plan()
        workorders = man_order.workorder_ids
        kit_wo = man_order.workorder_ids.filtered(lambda wo: wo.operation_id == self.operation_1)
        door_wo_1 = man_order.workorder_ids.filtered(lambda wo: wo.operation_id == self.operation_2)
        door_wo_2 = man_order.workorder_ids.filtered(lambda wo: wo.operation_id == self.operation_3)
        for workorder in workorders:
            self.assertEqual(workorder.workcenter_id, self.workcenter_1, 'Workcenter does not match.')
        self.assertEqual(kit_wo.state, 'ready', 'Workorder should be in ready state.')
        self.assertEqual(door_wo_1.state, 'ready', 'Workorder should be in ready state.')
        self.assertEqual(door_wo_2.state, 'pending', 'Workorder should be in pending state.')
        self.assertEqual(kit_wo.duration_expected, 80, 'Workorder duration should be 80 instead of %s.' % str(kit_wo.duration_expected))
        self.assertEqual(door_wo_1.duration_expected, 20, 'Workorder duration should be 20 instead of %s.' % str(door_wo_1.duration_expected))
        self.assertEqual(door_wo_2.duration_expected, 20, 'Workorder duration should be 20 instead of %s.' % str(door_wo_2.duration_expected))
        kit_wo.button_start()
        finished_lot = self.env['stock.production.lot'].create({'product_id': man_order.product_id.id})
        kit_wo.write({'final_lot_id': finished_lot.id, 'qty_producing': 48})
        kit_wo.record_production()
        self.assertEqual(kit_wo.state, 'done', 'Workorder should be in done state.')
        finished_lot = self.env['stock.production.lot'].create({'product_id': man_order.product_id.id})
        door_wo_1.write({'final_lot_id': finished_lot.id, 'qty_producing': 48})
        door_wo_1.record_production()
        self.assertEqual(door_wo_1.state, 'done', 'Workorder should be in done state.')
        self.assertEqual(door_wo_2.state, 'ready', 'Workorder should be in ready state.')
        door_wo_2.record_production()
        self.assertEqual(door_wo_2.state, 'done', 'Workorder should be in done state.')

    def test_production_avialability(self):
        if False:
            print('Hello World!')
        '\n            Test availability of production order.\n        '
        self.bom_3.bom_line_ids.filtered(lambda x: x.product_id == self.product_5).unlink()
        self.bom_3.bom_line_ids.filtered(lambda x: x.product_id == self.product_4).unlink()
        production_2 = self.env['mrp.production'].create({'name': 'MO-Test001', 'product_id': self.product_6.id, 'product_qty': 5.0, 'bom_id': self.bom_3.id, 'product_uom_id': self.product_6.uom_id.id})
        production_2.action_assign()
        self.assertEqual(production_2.availability, 'waiting', 'Production order should be availability for waiting state')
        inventory_wizard = self.env['stock.change.product.qty'].create({'product_id': self.product_2.id, 'new_quantity': 2.0})
        inventory_wizard.change_product_qty()
        production_2.action_assign()
        self.assertEqual(production_2.availability, 'partially_available', 'Production order should be availability for partially available state')
        inventory_wizard = self.env['stock.change.product.qty'].create({'product_id': self.product_2.id, 'new_quantity': 5.0})
        inventory_wizard.change_product_qty()
        production_2.action_assign()
        self.assertEqual(production_2.availability, 'assigned', 'Production order should be availability for assigned state')

    def test_empty_routing(self):
        if False:
            for i in range(10):
                print('nop')
        ' Check what happens when you work with an empty routing'
        routing = self.env['mrp.routing'].create({'name': 'Routing without operations', 'location_id': self.warehouse_1.wh_input_stock_loc_id.id})
        self.bom_3.routing_id = routing.id
        production = self.env['mrp.production'].create({'name': 'MO test', 'product_id': self.product_6.id, 'product_qty': 3, 'bom_id': self.bom_3.id, 'product_uom_id': self.product_6.uom_id.id})
        self.assertEqual(production.routing_id.id, False, 'The routing field should be empty on the mo')
        self.assertEqual(production.move_raw_ids[0].location_id.id, self.warehouse_1.wh_input_stock_loc_id.id, 'Raw moves start location should have altered.')

    def test_multiple_post_inventory(self):
        if False:
            i = 10
            return i + 15
        ' Check the consumed quants of the produced quants when intermediate calls to `post_inventory` during a MO.'
        unit = self.ref('product.product_uom_unit')
        custom_laptop = self.env.ref('product.product_product_27')
        custom_laptop.tracking = 'none'
        product_charger = self.env['product.product'].create({'name': 'Charger', 'type': 'product', 'uom_id': unit, 'uom_po_id': unit})
        product_keybord = self.env['product.product'].create({'name': 'Usb Keybord', 'type': 'product', 'uom_id': unit, 'uom_po_id': unit})
        bom_custom_laptop = self.env['mrp.bom'].create({'product_tmpl_id': custom_laptop.product_tmpl_id.id, 'product_qty': 1, 'product_uom_id': unit, 'bom_line_ids': [(0, 0, {'product_id': product_charger.id, 'product_qty': 1, 'product_uom_id': unit}), (0, 0, {'product_id': product_keybord.id, 'product_qty': 1, 'product_uom_id': unit})]})
        source_location_id = self.ref('stock.stock_location_14')
        inventory = self.env['stock.inventory'].create({'name': 'Inventory Product Table', 'filter': 'partial', 'line_ids': [(0, 0, {'product_id': product_charger.id, 'product_uom_id': product_charger.uom_id.id, 'product_qty': 2, 'location_id': source_location_id}), (0, 0, {'product_id': product_keybord.id, 'product_uom_id': product_keybord.uom_id.id, 'product_qty': 2, 'location_id': source_location_id})]})
        inventory.action_done()
        mo_custom_laptop = self.env['mrp.production'].create({'product_id': custom_laptop.id, 'product_qty': 2, 'product_uom_id': unit, 'bom_id': bom_custom_laptop.id})
        mo_custom_laptop.action_assign()
        self.assertEqual(mo_custom_laptop.availability, 'assigned')
        context = {'active_ids': [mo_custom_laptop.id], 'active_id': mo_custom_laptop.id}
        custom_laptop_produce = self.env['mrp.product.produce'].with_context(context).create({'product_qty': 1.0})
        custom_laptop_produce.do_produce()
        mo_custom_laptop.post_inventory()
        first_move = mo_custom_laptop.move_finished_ids.filtered(lambda mo: mo.state == 'done')
        self.assertEquals(sum(first_move.quant_ids.mapped('consumed_quant_ids').mapped('qty')), 2)
        second_move = mo_custom_laptop.move_finished_ids.filtered(lambda mo: mo.state == 'confirmed')
        context = {'active_ids': [mo_custom_laptop.id], 'active_id': mo_custom_laptop.id}
        custom_laptop_produce = self.env['mrp.product.produce'].with_context(context).create({'product_qty': 1.0})
        custom_laptop_produce.do_produce()
        mo_custom_laptop.post_inventory()
        self.assertEquals(sum(second_move.quant_ids.mapped('consumed_quant_ids').mapped('qty')), 2)

    def test_rounding(self):
        if False:
            print('Hello World!')
        ' In previous versions we had rounding and efficiency fields.  We check if we can still do the same, but with only the rounding on the UoM'
        self.product_6.uom_id.rounding = 1.0
        bom_eff = self.env['mrp.bom'].create({'product_id': self.product_6.id, 'product_tmpl_id': self.product_6.product_tmpl_id.id, 'product_qty': 1, 'product_uom_id': self.product_6.uom_id.id, 'type': 'normal', 'bom_line_ids': [(0, 0, {'product_id': self.product_2.id, 'product_qty': 2.03}), (0, 0, {'product_id': self.product_8.id, 'product_qty': 4.16})]})
        production = self.env['mrp.production'].create({'name': 'MO efficiency test', 'product_id': self.product_6.id, 'product_qty': 20, 'bom_id': bom_eff.id, 'product_uom_id': self.product_6.uom_id.id})
        self.assertEqual(production.move_raw_ids[0].product_qty, 41, 'The quantity should be rounded up')
        self.assertEqual(production.move_raw_ids[1].product_qty, 84, 'The quantity should be rounded up')
        produce_wizard = self.env['mrp.product.produce'].with_context({'active_id': production.id, 'active_ids': [production.id]}).create({'product_qty': 8})
        produce_wizard.do_produce()
        self.assertEqual(production.move_raw_ids[0].quantity_done, 16, 'Should use half-up rounding when producing')
        self.assertEqual(production.move_raw_ids[1].quantity_done, 34, 'Should use half-up rounding when producing')