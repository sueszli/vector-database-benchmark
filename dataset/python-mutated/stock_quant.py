from odoo import api, fields, models

class StockQuant(models.Model):
    _inherit = 'stock.quant'
    removal_date = fields.Datetime(related='lot_id.removal_date', store=True)

    @api.model
    def _quants_removal_get_order(self, removal_strategy):
        if False:
            while True:
                i = 10
        if removal_strategy == 'fefo':
            return 'removal_date, in_date, id'
        return super(StockQuant, self)._quants_removal_get_order(removal_strategy=removal_strategy)