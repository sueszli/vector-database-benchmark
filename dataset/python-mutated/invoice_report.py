from odoo import fields, models

class AccountInvoiceReport(models.Model):
    _inherit = 'account.invoice.report'
    team_id = fields.Many2one('crm.team', string='Sales Team')

    def _select(self):
        if False:
            while True:
                i = 10
        return super(AccountInvoiceReport, self)._select() + ', sub.team_id as team_id'

    def _sub_select(self):
        if False:
            return 10
        return super(AccountInvoiceReport, self)._sub_select() + ', ai.team_id as team_id'

    def _group_by(self):
        if False:
            return 10
        return super(AccountInvoiceReport, self)._group_by() + ', ai.team_id'