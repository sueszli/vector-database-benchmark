from datetime import timedelta
from odoo.tests import common
from odoo import fields

class TestSaleOrderDates(common.TransactionCase):

    def test_sale_order_requested_date(self):
        if False:
            i = 10
            return i + 15
        new_order = self.env.ref('sale.sale_order_6').copy({'requested_date': '2010-07-12'})
        new_order.action_confirm()
        security_delay = timedelta(days=new_order.company_id.security_lead)
        requested_date = fields.Datetime.from_string(new_order.requested_date)
        right_date = fields.Datetime.to_string(requested_date - security_delay)
        for line in new_order.order_line:
            self.assertNotEqual(len(line.procurement_ids), 0, 'No Procurement was created')
            procurement = line.procurement_ids[0]
            self.assertEqual(procurement.date_planned, right_date, 'The planned date for the Procurement Order is wrong')
            self.assertNotEqual(len(procurement.move_ids), 0, 'No Move was created')
            self.assertEqual(procurement.move_ids[0].date_expected, right_date, 'The expected date for the Stock Move is wrong')