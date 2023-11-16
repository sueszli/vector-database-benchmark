from odoo import api, fields, models
from odoo.tools.sql import drop_view_if_exists

class ReportStockLinesDate(models.Model):
    _name = 'report.stock.lines.date'
    _description = 'Dates of Inventories and latest Moves'
    _auto = False
    _order = 'date'
    id = fields.Integer('Product Id', readonly=True)
    product_id = fields.Many2one('product.product', 'Product', readonly=True, index=True)
    date = fields.Datetime('Date of latest Inventory', readonly=True)
    move_date = fields.Datetime('Date of latest Stock Move', readonly=True)
    active = fields.Boolean('Active', readonly=True)

    @api.model_cr
    def init(self):
        if False:
            print('Hello World!')
        drop_view_if_exists(self._cr, 'report_stock_lines_date')
        self._cr.execute("\n            create or replace view report_stock_lines_date as (\n                select\n                p.id as id,\n                p.id as product_id,\n                max(s.date) as date,\n                max(m.date) as move_date,\n                p.active as active\n            from\n                product_product p\n                    left join (\n                        stock_inventory_line l\n                        inner join stock_inventory s on (l.inventory_id=s.id and s.state = 'done')\n                    ) on (p.id=l.product_id)\n                    left join stock_move m on (m.product_id=p.id and m.state = 'done')\n                group by p.id\n            )")