from odoo import fields, models

class SaleReport(models.Model):
    _inherit = 'sale.report'
    warehouse_id = fields.Many2one('stock.warehouse', 'Warehouse', readonly=True)

    def _select(self):
        if False:
            for i in range(10):
                print('nop')
        return super(SaleReport, self)._select() + ', s.warehouse_id as warehouse_id'

    def _group_by(self):
        if False:
            print('Hello World!')
        return super(SaleReport, self)._group_by() + ', s.warehouse_id'