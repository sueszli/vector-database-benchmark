from odoo import api, fields, models, tools

class PurchaseReport(models.Model):
    _name = 'purchase.report'
    _description = 'Purchases Orders'
    _auto = False
    _order = 'date_order desc, price_total desc'
    date_order = fields.Datetime('Order Date', readonly=True, help='Date on which this document has been created', oldname='date')
    state = fields.Selection([('draft', 'Draft RFQ'), ('sent', 'RFQ Sent'), ('to approve', 'To Approve'), ('purchase', 'Purchase Order'), ('done', 'Done'), ('cancel', 'Cancelled')], 'Order Status', readonly=True)
    product_id = fields.Many2one('product.product', 'Product', readonly=True)
    picking_type_id = fields.Many2one('stock.warehouse', 'Warehouse', readonly=True)
    partner_id = fields.Many2one('res.partner', 'Vendor', readonly=True)
    date_approve = fields.Date('Date Approved', readonly=True)
    product_uom = fields.Many2one('product.uom', 'Reference Unit of Measure', required=True)
    company_id = fields.Many2one('res.company', 'Company', readonly=True)
    currency_id = fields.Many2one('res.currency', 'Currency', readonly=True)
    user_id = fields.Many2one('res.users', 'Responsible', readonly=True)
    delay = fields.Float('Days to Validate', digits=(16, 2), readonly=True)
    delay_pass = fields.Float('Days to Deliver', digits=(16, 2), readonly=True)
    unit_quantity = fields.Float('Product Quantity', readonly=True, oldname='quantity')
    price_total = fields.Float('Total Price', readonly=True)
    price_average = fields.Float('Average Price', readonly=True, group_operator='avg')
    negociation = fields.Float('Purchase-Standard Price', readonly=True, group_operator='avg')
    price_standard = fields.Float('Products Value', readonly=True, group_operator='sum')
    nbr_lines = fields.Integer('# of Lines', readonly=True, oldname='nbr')
    category_id = fields.Many2one('product.category', 'Product Category', readonly=True)
    product_tmpl_id = fields.Many2one('product.template', 'Product Template', readonly=True)
    country_id = fields.Many2one('res.country', 'Partner Country', readonly=True)
    fiscal_position_id = fields.Many2one('account.fiscal.position', string='Fiscal Position', oldname='fiscal_position', readonly=True)
    account_analytic_id = fields.Many2one('account.analytic.account', 'Analytic Account', readonly=True)
    commercial_partner_id = fields.Many2one('res.partner', 'Commercial Entity', readonly=True)
    weight = fields.Float('Gross Weight', readonly=True)
    volume = fields.Float('Volume', readonly=True)

    @api.model_cr
    def init(self):
        if False:
            i = 10
            return i + 15
        tools.drop_view_if_exists(self._cr, 'purchase_report')
        self._cr.execute("\n            create view purchase_report as (\n                WITH currency_rate as (%s)\n                select\n                    min(l.id) as id,\n                    s.date_order as date_order,\n                    s.state,\n                    s.date_approve,\n                    s.dest_address_id,\n                    spt.warehouse_id as picking_type_id,\n                    s.partner_id as partner_id,\n                    s.create_uid as user_id,\n                    s.company_id as company_id,\n                    s.fiscal_position_id as fiscal_position_id,\n                    l.product_id,\n                    p.product_tmpl_id,\n                    t.categ_id as category_id,\n                    s.currency_id,\n                    t.uom_id as product_uom,\n                    sum(l.product_qty/u.factor*u2.factor) as unit_quantity,\n                    extract(epoch from age(s.date_approve,s.date_order))/(24*60*60)::decimal(16,2) as delay,\n                    extract(epoch from age(l.date_planned,s.date_order))/(24*60*60)::decimal(16,2) as delay_pass,\n                    count(*) as nbr_lines,\n                    sum(l.price_unit / COALESCE(cr.rate, 1.0) * l.product_qty)::decimal(16,2) as price_total,\n                    avg(100.0 * (l.price_unit / COALESCE(cr.rate,1.0) * l.product_qty) / NULLIF(ip.value_float*l.product_qty/u.factor*u2.factor, 0.0))::decimal(16,2) as negociation,\n                    sum(ip.value_float*l.product_qty/u.factor*u2.factor)::decimal(16,2) as price_standard,\n                    (sum(l.product_qty * l.price_unit / COALESCE(cr.rate, 1.0))/NULLIF(sum(l.product_qty/u.factor*u2.factor),0.0))::decimal(16,2) as price_average,\n                    partner.country_id as country_id,\n                    partner.commercial_partner_id as commercial_partner_id,\n                    analytic_account.id as account_analytic_id,\n                    sum(p.weight * l.product_qty/u.factor*u2.factor) as weight,\n                    sum(p.volume * l.product_qty/u.factor*u2.factor) as volume\n                from purchase_order_line l\n                    join purchase_order s on (l.order_id=s.id)\n                    join res_partner partner on s.partner_id = partner.id\n                        left join product_product p on (l.product_id=p.id)\n                            left join product_template t on (p.product_tmpl_id=t.id)\n                            LEFT JOIN ir_property ip ON (ip.name='standard_price' AND ip.res_id=CONCAT('product.template,',t.id) AND ip.company_id=s.company_id)\n                    left join product_uom u on (u.id=l.product_uom)\n                    left join product_uom u2 on (u2.id=t.uom_id)\n                    left join stock_picking_type spt on (spt.id=s.picking_type_id)\n                    left join account_analytic_account analytic_account on (l.account_analytic_id = analytic_account.id)\n                    left join currency_rate cr on (cr.currency_id = s.currency_id and\n                        cr.company_id = s.company_id and\n                        cr.date_start <= coalesce(s.date_order, now()) and\n                        (cr.date_end is null or cr.date_end > coalesce(s.date_order, now())))\n                group by\n                    s.company_id,\n                    s.create_uid,\n                    s.partner_id,\n                    u.factor,\n                    s.currency_id,\n                    l.price_unit,\n                    s.date_approve,\n                    l.date_planned,\n                    l.product_uom,\n                    s.dest_address_id,\n                    s.fiscal_position_id,\n                    l.product_id,\n                    p.product_tmpl_id,\n                    t.categ_id,\n                    s.date_order,\n                    s.state,\n                    spt.warehouse_id,\n                    u.uom_type,\n                    u.category_id,\n                    t.uom_id,\n                    u.id,\n                    u2.factor,\n                    partner.country_id,\n                    partner.commercial_partner_id,\n                    analytic_account.id\n            )\n        " % self.env['res.currency']._select_companies_rates())