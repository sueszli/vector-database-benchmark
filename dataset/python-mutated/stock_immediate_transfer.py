from odoo import api, fields, models, _
from odoo.exceptions import UserError

class StockImmediateTransfer(models.TransientModel):
    _name = 'stock.immediate.transfer'
    _description = 'Immediate Transfer'
    pick_id = fields.Many2one('stock.picking')

    @api.model
    def default_get(self, fields):
        if False:
            i = 10
            return i + 15
        res = super(StockImmediateTransfer, self).default_get(fields)
        if not res.get('pick_id') and self._context.get('active_id'):
            res['pick_id'] = self._context['active_id']
        return res

    @api.multi
    def process(self):
        if False:
            for i in range(10):
                print('nop')
        self.ensure_one()
        if self.pick_id.state == 'draft':
            self.pick_id.action_confirm()
            if self.pick_id.state != 'assigned':
                self.pick_id.action_assign()
                if self.pick_id.state != 'assigned':
                    raise UserError(_("Could not reserve all requested products. Please use the 'Mark as Todo' button to handle the reservation manually."))
        for pack in self.pick_id.pack_operation_ids:
            if pack.product_qty > 0:
                pack.write({'qty_done': pack.product_qty})
            else:
                pack.unlink()
        self.pick_id.do_transfer()