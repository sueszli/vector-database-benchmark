from odoo import api, fields, models, tools
STATE = [('none', 'Non Member'), ('canceled', 'Cancelled Member'), ('old', 'Old Member'), ('waiting', 'Waiting Member'), ('invoiced', 'Invoiced Member'), ('free', 'Free Member'), ('paid', 'Paid Member')]

class ReportMembership(models.Model):
    """Membership Analysis"""
    _name = 'report.membership'
    _description = __doc__
    _auto = False
    _rec_name = 'start_date'
    start_date = fields.Date(string='Start Date', readonly=True)
    date_to = fields.Date(string='End Date', readonly=True, help='End membership date')
    num_waiting = fields.Integer(string='# Waiting', readonly=True)
    num_invoiced = fields.Integer(string='# Invoiced', readonly=True)
    num_paid = fields.Integer(string='# Paid', readonly=True)
    tot_pending = fields.Float(string='Pending Amount', digits=0, readonly=True)
    tot_earned = fields.Float(string='Earned Amount', digits=0, readonly=True)
    partner_id = fields.Many2one('res.partner', string='Member', readonly=True)
    associate_member_id = fields.Many2one('res.partner', string='Associate Member', readonly=True)
    membership_id = fields.Many2one('product.product', string='Membership Product', readonly=True)
    membership_state = fields.Selection(STATE, string='Current Membership State', readonly=True)
    user_id = fields.Many2one('res.users', string='Salesperson', readonly=True)
    company_id = fields.Many2one('res.company', string='Company', readonly=True)
    quantity = fields.Integer(readonly=True)

    @api.model_cr
    def init(self):
        if False:
            while True:
                i = 10
        'Create the view'
        tools.drop_view_if_exists(self._cr, self._table)
        self._cr.execute("\n        CREATE OR REPLACE VIEW %s AS (\n        SELECT\n        MIN(id) AS id,\n        partner_id,\n        count(membership_id) as quantity,\n        user_id,\n        membership_state,\n        associate_member_id,\n        membership_amount,\n        date_to,\n        start_date,\n        COUNT(num_waiting) AS num_waiting,\n        COUNT(num_invoiced) AS num_invoiced,\n        COUNT(num_paid) AS num_paid,\n        SUM(tot_pending) AS tot_pending,\n        SUM(tot_earned) AS tot_earned,\n        membership_id,\n        company_id\n        FROM\n        (SELECT\n            MIN(p.id) AS id,\n            p.id AS partner_id,\n            p.user_id AS user_id,\n            p.membership_state AS membership_state,\n            p.associate_member AS associate_member_id,\n            p.membership_amount AS membership_amount,\n            p.membership_stop AS date_to,\n            p.membership_start AS start_date,\n            CASE WHEN ml.state = 'waiting'  THEN ml.id END AS num_waiting,\n            CASE WHEN ml.state = 'invoiced' THEN ml.id END AS num_invoiced,\n            CASE WHEN ml.state = 'paid'     THEN ml.id END AS num_paid,\n            CASE WHEN ml.state IN ('waiting', 'invoiced') THEN SUM(il.price_subtotal) ELSE 0 END AS tot_pending,\n            CASE WHEN ml.state = 'paid' OR p.membership_state = 'old' THEN SUM(il.price_subtotal) ELSE 0 END AS tot_earned,\n            ml.membership_id AS membership_id,\n            p.company_id AS company_id\n            FROM res_partner p\n            LEFT JOIN membership_membership_line ml ON (ml.partner = p.id)\n            LEFT JOIN account_invoice_line il ON (ml.account_invoice_line = il.id)\n            LEFT JOIN account_invoice ai ON (il.invoice_id = ai.id)\n            WHERE p.membership_state != 'none' and p.active = 'true'\n            GROUP BY\n              p.id,\n              p.user_id,\n              p.membership_state,\n              p.associate_member,\n              p.membership_amount,\n              p.membership_start,\n              ml.membership_id,\n              p.company_id,\n              ml.state,\n              ml.id\n        ) AS foo\n        GROUP BY\n            start_date,\n            date_to,\n            partner_id,\n            user_id,\n            membership_id,\n            company_id,\n            membership_state,\n            associate_member_id,\n            membership_amount\n        )" % (self._table,))