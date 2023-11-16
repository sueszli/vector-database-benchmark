from odoo import api, fields, models
from odoo.tools.sql import drop_view_if_exists

class PaymentAdviceReport(models.Model):
    _name = 'payment.advice.report'
    _description = 'Payment Advice Analysis'
    _auto = False
    name = fields.Char(readonly=True)
    date = fields.Date(readonly=True)
    year = fields.Char(readonly=True)
    month = fields.Selection([('01', 'January'), ('02', 'February'), ('03', 'March'), ('04', 'April'), ('05', 'May'), ('06', 'June'), ('07', 'July'), ('08', 'August'), ('09', 'September'), ('10', 'October'), ('11', 'November'), ('12', 'December')], readonly=True)
    day = fields.Char(readonly=True)
    state = fields.Selection([('draft', 'Draft'), ('confirm', 'Confirmed'), ('cancel', 'Cancelled')], string='Status', index=True, readonly=True)
    employee_id = fields.Many2one('hr.employee', string='Employee', readonly=True)
    nbr = fields.Integer(string='# Payment Lines', readonly=True)
    number = fields.Char(readonly=True)
    bysal = fields.Float(string='By Salary', readonly=True)
    bank_id = fields.Many2one('res.bank', string='Bank', readonly=True)
    company_id = fields.Many2one('res.company', string='Company', readonly=True)
    cheque_nos = fields.Char(string='Cheque Numbers', readonly=True)
    neft = fields.Boolean(string='NEFT Transaction', readonly=True)
    ifsc_code = fields.Char(string='IFSC Code', readonly=True)
    employee_bank_no = fields.Char(string='Employee Bank Account', required=True)

    @api.model_cr
    def init(self):
        if False:
            return 10
        drop_view_if_exists(self.env.cr, self._table)
        self.env.cr.execute("\n            create or replace view payment_advice_report as (\n                select\n                    min(l.id) as id,\n                    sum(l.bysal) as bysal,\n                    p.name,\n                    p.state,\n                    p.date,\n                    p.number,\n                    p.company_id,\n                    p.bank_id,\n                    p.chaque_nos as cheque_nos,\n                    p.neft,\n                    l.employee_id,\n                    l.ifsc_code,\n                    l.name as employee_bank_no,\n                    to_char(p.date, 'YYYY') as year,\n                    to_char(p.date, 'MM') as month,\n                    to_char(p.date, 'YYYY-MM-DD') as day,\n                    1 as nbr\n                from\n                    hr_payroll_advice as p\n                    left join hr_payroll_advice_line as l on (p.id=l.advice_id)\n                where\n                    l.employee_id IS NOT NULL\n                group by\n                    p.number,p.name,p.date,p.state,p.company_id,p.bank_id,p.chaque_nos,p.neft,\n                    l.employee_id,l.advice_id,l.bysal,l.ifsc_code, l.name\n            )\n        ")