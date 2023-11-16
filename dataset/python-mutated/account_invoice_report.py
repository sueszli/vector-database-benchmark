from odoo import tools
from odoo import models, fields, api

class AccountInvoiceReport(models.Model):
    _name = 'account.invoice.report'
    _description = 'Invoices Statistics'
    _auto = False
    _rec_name = 'date'

    @api.multi
    @api.depends('currency_id', 'date', 'price_total', 'price_average', 'residual')
    def _compute_amounts_in_user_currency(self):
        if False:
            return 10
        'Compute the amounts in the currency of the user\n        '
        context = dict(self._context or {})
        user_currency_id = self.env.user.company_id.currency_id
        currency_rate_id = self.env['res.currency.rate'].search([('rate', '=', 1), '|', ('company_id', '=', self.env.user.company_id.id), ('company_id', '=', False)], limit=1)
        base_currency_id = currency_rate_id.currency_id
        ctx = context.copy()
        for record in self:
            ctx['date'] = record.date
            record.user_currency_price_total = base_currency_id.with_context(ctx).compute(record.price_total, user_currency_id)
            record.user_currency_price_average = base_currency_id.with_context(ctx).compute(record.price_average, user_currency_id)
            record.user_currency_residual = base_currency_id.with_context(ctx).compute(record.residual, user_currency_id)
    date = fields.Date(readonly=True)
    product_id = fields.Many2one('product.product', string='Product', readonly=True)
    product_qty = fields.Float(string='Product Quantity', readonly=True)
    uom_name = fields.Char(string='Reference Unit of Measure', readonly=True)
    payment_term_id = fields.Many2one('account.payment.term', string='Payment Terms', oldname='payment_term', readonly=True)
    fiscal_position_id = fields.Many2one('account.fiscal.position', oldname='fiscal_position', string='Fiscal Position', readonly=True)
    currency_id = fields.Many2one('res.currency', string='Currency', readonly=True)
    categ_id = fields.Many2one('product.category', string='Product Category', readonly=True)
    journal_id = fields.Many2one('account.journal', string='Journal', readonly=True)
    partner_id = fields.Many2one('res.partner', string='Partner', readonly=True)
    commercial_partner_id = fields.Many2one('res.partner', string='Partner Company', help='Commercial Entity')
    company_id = fields.Many2one('res.company', string='Company', readonly=True)
    user_id = fields.Many2one('res.users', string='Salesperson', readonly=True)
    price_total = fields.Float(string='Total Without Tax', readonly=True)
    user_currency_price_total = fields.Float(string='Total Without Tax', compute='_compute_amounts_in_user_currency', digits=0)
    price_average = fields.Float(string='Average Price', readonly=True, group_operator='avg')
    user_currency_price_average = fields.Float(string='Average Price', compute='_compute_amounts_in_user_currency', digits=0)
    currency_rate = fields.Float(string='Currency Rate', readonly=True, group_operator='avg')
    nbr = fields.Integer(string='# of Lines', readonly=True)
    type = fields.Selection([('out_invoice', 'Customer Invoice'), ('in_invoice', 'Vendor Bill'), ('out_refund', 'Customer Refund'), ('in_refund', 'Vendor Refund')], readonly=True)
    state = fields.Selection([('draft', 'Draft'), ('proforma', 'Pro-forma'), ('proforma2', 'Pro-forma'), ('open', 'Open'), ('paid', 'Done'), ('cancel', 'Cancelled')], string='Invoice Status', readonly=True)
    date_due = fields.Date(string='Due Date', readonly=True)
    account_id = fields.Many2one('account.account', string='Account', readonly=True, domain=[('deprecated', '=', False)])
    account_line_id = fields.Many2one('account.account', string='Account Line', readonly=True, domain=[('deprecated', '=', False)])
    partner_bank_id = fields.Many2one('res.partner.bank', string='Bank Account', readonly=True)
    residual = fields.Float(string='Total Residual', readonly=True)
    user_currency_residual = fields.Float(string='Total Residual', compute='_compute_amounts_in_user_currency', digits=0)
    country_id = fields.Many2one('res.country', string='Country of the Partner Company')
    weight = fields.Float(string='Gross Weight', readonly=True)
    volume = fields.Float(string='Volume', readonly=True)
    _order = 'date desc'
    _depends = {'account.invoice': ['account_id', 'amount_total_company_signed', 'commercial_partner_id', 'company_id', 'currency_id', 'date_due', 'date_invoice', 'fiscal_position_id', 'journal_id', 'partner_bank_id', 'partner_id', 'payment_term_id', 'residual', 'state', 'type', 'user_id'], 'account.invoice.line': ['account_id', 'invoice_id', 'price_subtotal', 'product_id', 'quantity', 'uom_id', 'account_analytic_id'], 'product.product': ['product_tmpl_id'], 'product.template': ['categ_id'], 'product.uom': ['category_id', 'factor', 'name', 'uom_type'], 'res.currency.rate': ['currency_id', 'name'], 'res.partner': ['country_id']}

    def _select(self):
        if False:
            print('Hello World!')
        select_str = '\n            SELECT sub.id, sub.date, sub.product_id, sub.partner_id, sub.country_id, sub.account_analytic_id,\n                sub.payment_term_id, sub.uom_name, sub.currency_id, sub.journal_id,\n                sub.fiscal_position_id, sub.user_id, sub.company_id, sub.nbr, sub.type, sub.state,\n                sub.weight, sub.volume,\n                sub.categ_id, sub.date_due, sub.account_id, sub.account_line_id, sub.partner_bank_id,\n                sub.product_qty, sub.price_total as price_total, sub.price_average as price_average,\n                COALESCE(cr.rate, 1) as currency_rate, sub.residual as residual, sub.commercial_partner_id as commercial_partner_id\n        '
        return select_str

    def _sub_select(self):
        if False:
            for i in range(10):
                print('nop')
        select_str = '\n                SELECT ail.id AS id,\n                    ai.date_invoice AS date,\n                    ail.product_id, ai.partner_id, ai.payment_term_id, ail.account_analytic_id,\n                    u2.name AS uom_name,\n                    ai.currency_id, ai.journal_id, ai.fiscal_position_id, ai.user_id, ai.company_id,\n                    1 AS nbr,\n                    ai.type, ai.state, pt.categ_id, ai.date_due, ai.account_id, ail.account_id AS account_line_id,\n                    ai.partner_bank_id,\n                    SUM ((invoice_type.sign * ail.quantity) / u.factor * u2.factor) AS product_qty,\n                    SUM(ail.price_subtotal_signed) AS price_total,\n                    SUM(ABS(ail.price_subtotal_signed)) / CASE\n                            WHEN SUM(ail.quantity / u.factor * u2.factor) <> 0::numeric\n                               THEN SUM(ail.quantity / u.factor * u2.factor)\n                               ELSE 1::numeric\n                            END AS price_average,\n                    ai.residual_company_signed / (SELECT count(*) FROM account_invoice_line l where invoice_id = ai.id) *\n                    count(*) * invoice_type.sign AS residual,\n                    ai.commercial_partner_id as commercial_partner_id,\n                    partner.country_id,\n                    SUM(pr.weight * (invoice_type.sign*ail.quantity) / u.factor * u2.factor) AS weight,\n                    SUM(pr.volume * (invoice_type.sign*ail.quantity) / u.factor * u2.factor) AS volume\n        '
        return select_str

    def _from(self):
        if False:
            i = 10
            return i + 15
        from_str = "\n                FROM account_invoice_line ail\n                JOIN account_invoice ai ON ai.id = ail.invoice_id\n                JOIN res_partner partner ON ai.commercial_partner_id = partner.id\n                LEFT JOIN product_product pr ON pr.id = ail.product_id\n                left JOIN product_template pt ON pt.id = pr.product_tmpl_id\n                LEFT JOIN product_uom u ON u.id = ail.uom_id\n                LEFT JOIN product_uom u2 ON u2.id = pt.uom_id\n                JOIN (\n                    -- Temporary table to decide if the qty should be added or retrieved (Invoice vs Refund) \n                    SELECT id,(CASE\n                         WHEN ai.type::text = ANY (ARRAY['out_refund'::character varying::text, 'in_invoice'::character varying::text])\n                            THEN -1\n                            ELSE 1\n                        END) AS sign\n                    FROM account_invoice ai\n                ) AS invoice_type ON invoice_type.id = ai.id\n        "
        return from_str

    def _group_by(self):
        if False:
            return 10
        group_by_str = '\n                GROUP BY ail.id, ail.product_id, ail.account_analytic_id, ai.date_invoice, ai.id,\n                    ai.partner_id, ai.payment_term_id, u2.name, u2.id, ai.currency_id, ai.journal_id,\n                    ai.fiscal_position_id, ai.user_id, ai.company_id, ai.type, invoice_type.sign, ai.state, pt.categ_id,\n                    ai.date_due, ai.account_id, ail.account_id, ai.partner_bank_id, ai.residual_company_signed,\n                    ai.amount_total_company_signed, ai.commercial_partner_id, partner.country_id\n        '
        return group_by_str

    @api.model_cr
    def init(self):
        if False:
            for i in range(10):
                print('nop')
        tools.drop_view_if_exists(self.env.cr, self._table)
        self.env.cr.execute('CREATE or REPLACE VIEW %s as (\n            WITH currency_rate AS (%s)\n            %s\n            FROM (\n                %s %s %s\n            ) AS sub\n            LEFT JOIN currency_rate cr ON\n                (cr.currency_id = sub.currency_id AND\n                 cr.company_id = sub.company_id AND\n                 cr.date_start <= COALESCE(sub.date, NOW()) AND\n                 (cr.date_end IS NULL OR cr.date_end > COALESCE(sub.date, NOW())))\n        )' % (self._table, self.env['res.currency']._select_companies_rates(), self._select(), self._sub_select(), self._from(), self._group_by()))