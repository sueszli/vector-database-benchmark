from odoo import api, fields, models

class PosDiscount(models.TransientModel):
    _name = 'pos.discount'
    _description = 'Add a Global Discount'
    discount = fields.Float(string='Discount (%)', required=True, digits=(16, 2), default=5)

    @api.multi
    def apply_discount(self):
        if False:
            return 10
        self.ensure_one()
        for order in self.env['pos.order'].browse(self.env.context.get('active_id', False)):
            order.lines.write({'discount': self.discount})