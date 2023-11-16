from datetime import datetime, timedelta
from odoo import fields
from odoo.tests import common

class TestsStockCalendar(common.TransactionCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestsStockCalendar, self).setUp()
        self.stock_warehouse0_id = self.ref('stock.warehouse0')
        self.purchase_route_warehouse0_buy_id = self.ref('purchase.route_warehouse0_buy')
        self.stock_picking_type_out_id = self.ref('stock.picking_type_out')
        self.stock_location_id = self.ref('stock.stock_location_stock')
        self.stock_location_customer_id = self.ref('stock.stock_location_customers')
        self.res_partner_id = self.env['res.partner'].create({'name': 'Supplier', 'supplier': 1})
        self.resource_calendar_id = self.env['resource.calendar'].create({'name': 'Calendar', 'attendance_ids': [(0, 0, {'name': 'Thursday', 'dayofweek': '3', 'hour_from': 8, 'hour_to': 9})]})
        self.calendar_product_id = self.env['product.product'].create({'name': 'Calendar Product', 'seller_ids': [(0, 0, {'name': self.res_partner_id.id, 'delay': 1})], 'orderpoint_ids': [(0, 0, {'name': 'Product A Truck', 'calendar_id': self.resource_calendar_id.id, 'product_min_qty': 0, 'product_max_qty': 10, 'warehouse_id': self.stock_warehouse0_id})]})
        self.pick_out_calendar = self._create_stock_picking('Delivery order for procurement', self.calendar_product_id.name, 3.0)
        self.pick_out_calendar2 = self._create_stock_picking('Delivery order for procurement2', 'stock_move_2', 4.0)
        self.pick_out_calendar3 = self._create_stock_picking('Delivery order for procurement3', 'stock_move_3', 11.0)

    def _create_stock_picking(self, pickname, movelinename, productqty):
        if False:
            while True:
                i = 10
        return self.env['stock.picking'].create({'name': pickname, 'partner_id': self.res_partner_id.id, 'picking_type_id': self.stock_picking_type_out_id, 'location_id': self.stock_location_id, 'location_dest_id': self.stock_location_customer_id, 'move_lines': [(0, 0, {'name': movelinename, 'product_id': self.calendar_product_id.id, 'product_uom': self.calendar_product_id.uom_id.id, 'product_uom_qty': productqty, 'location_id': self.stock_location_id, 'location_dest_id': self.stock_location_customer_id, 'procure_method': 'make_to_stock'})]})

    def test_stock_calendar(self):
        if False:
            i = 10
            return i + 15
        self.calendar_product_id.write({'route_ids': [(4, self.purchase_route_warehouse0_buy_id)]})
        today8 = datetime.now() + timedelta(days=7)
        today21 = datetime.now() + timedelta(days=21)
        self.pick_out_calendar2.move_lines.write({'date': fields.Datetime.to_string(today8), 'date_expected': fields.Datetime.to_string(today8)})
        self.pick_out_calendar3.move_lines.write({'date_expected': fields.Datetime.to_string(today21), 'date': fields.Datetime.to_string(today21)})
        self.pick_out_calendar.action_confirm()
        self.pick_out_calendar2.action_confirm()
        self.pick_out_calendar3.action_confirm()
        Procurementorder = self.env['procurement.order']
        Procurementorder.run_scheduler()
        procurement = Procurementorder.search([('product_id', '=', self.calendar_product_id.id)], limit=1)
        self.assertEqual(len(procurement), 1, 'should have one procurement')
        self.assertEqual(procurement.product_qty, 17, 'It should have taken the two first pickings into account for the virtual stock for the orderpoint, not the third')
        self.assertEqual(fields.Datetime.from_string(procurement.next_delivery_date).weekday(), 3, 'The next delivery date should be on a Thursday')
        purchase_line_id_date_planned = fields.Datetime.from_string(procurement.purchase_line_id.date_planned).weekday()
        self.assertEqual(purchase_line_id_date_planned, 3, 'Check it has been put on the purchase line also, got %d' % purchase_line_id_date_planned)