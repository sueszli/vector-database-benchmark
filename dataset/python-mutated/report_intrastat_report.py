from odoo import fields, models
from odoo.tools.sql import drop_view_if_exists

class ReportIntrastat(models.Model):
    _name = 'report.intrastat'
    _description = 'Intrastat report'
    _auto = False
    name = fields.Char(string='Year', readonly=True)
    month = fields.Selection([('01', 'January'), ('02', 'February'), ('03', 'March'), ('04', 'April'), ('05', 'May'), ('06', 'June'), ('07', 'July'), ('08', 'August'), ('09', 'September'), ('10', 'October'), ('11', 'November'), ('12', 'December')], readonly=True)
    supply_units = fields.Float(string='Supply Units', readonly=True)
    ref = fields.Char(string='Source document', readonly=True)
    code = fields.Char(string='Country code', readonly=True)
    intrastat_id = fields.Many2one('report.intrastat.code', string='Intrastat code', readonly=True)
    weight = fields.Float(string='Weight', readonly=True)
    value = fields.Float(string='Value', readonly=True, digits=0)
    type = fields.Selection([('import', 'Import'), ('export', 'Export')], string='Type')
    currency_id = fields.Many2one('res.currency', string='Currency', readonly=True)
    company_id = fields.Many2one('res.company', string='Company', readonly=True)

    def init(self):
        if False:
            print('Hello World!')
        drop_view_if_exists(self.env.cr, self._table)
        self.env.cr.execute("\n            create or replace view report_intrastat as (\n                select\n                    to_char(inv.date_invoice, 'YYYY') as name,\n                    to_char(inv.date_invoice, 'MM') as month,\n                    min(inv_line.id) as id,\n                    intrastat.id as intrastat_id,\n                    upper(inv_country.code) as code,\n                    sum(case when inv_line.price_unit is not null\n                            then inv_line.price_unit * inv_line.quantity\n                            else 0\n                        end) as value,\n                    sum(\n                        case when uom.category_id != puom.category_id then (pt.weight * inv_line.quantity)\n                        else (pt.weight * inv_line.quantity * uom.factor) end\n                    ) as weight,\n                    sum(\n                        case when uom.category_id != puom.category_id then inv_line.quantity\n                        else (inv_line.quantity * uom.factor) end\n                    ) as supply_units,\n\n                    inv.currency_id as currency_id,\n                    inv.number as ref,\n                    case when inv.type in ('out_invoice','in_refund')\n                        then 'export'\n                        else 'import'\n                        end as type,\n                    inv.company_id as company_id\n                from\n                    account_invoice inv\n                    left join account_invoice_line inv_line on inv_line.invoice_id=inv.id\n                    left join (product_template pt\n                        left join product_product pp on (pp.product_tmpl_id = pt.id))\n                    on (inv_line.product_id = pp.id)\n                    left join product_uom uom on uom.id=inv_line.uom_id\n                    left join product_uom puom on puom.id = pt.uom_id\n                    left join report_intrastat_code intrastat on pt.intrastat_id = intrastat.id\n                    left join (res_partner inv_address\n                        left join res_country inv_country on (inv_country.id = inv_address.country_id))\n                    on (inv_address.id = inv.partner_id)\n                where\n                    inv.state in ('open','paid')\n                    and inv_line.product_id is not null\n                    and inv_country.intrastat=true\n                group by to_char(inv.date_invoice, 'YYYY'), to_char(inv.date_invoice, 'MM'),intrastat.id,inv.type,pt.intrastat_id, inv_country.code,inv.number,  inv.currency_id, inv.company_id\n            )")