from odoo import api, models

class SaleOrderLine(models.Model):
    _inherit = 'sale.order.line'

    @api.multi
    def _action_procurement_create(self):
        if False:
            return 10
        res = super(SaleOrderLine, self)._action_procurement_create()
        orders = list(set((x.order_id for x in self)))
        for order in orders:
            reassign = order.picking_ids.filtered(lambda x: x.state == 'confirmed' or (x.state in ['partially_available', 'waiting'] and (not x.printed)))
            if reassign:
                reassign.do_unreserve()
                reassign.action_assign()
        return res