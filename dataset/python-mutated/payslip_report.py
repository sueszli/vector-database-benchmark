from odoo import api, fields, models
from odoo.tools.sql import drop_view_if_exists

class PayslipReport(models.Model):
    _name = 'payslip.report'
    _description = 'Payslip Analysis'
    _auto = False
    name = fields.Char(readonly=True)
    date_from = fields.Date(string='Date From', readonly=True)
    date_to = fields.Date(string='Date To', readonly=True)
    year = fields.Char(size=4, readonly=True)
    month = fields.Selection([('01', 'January'), ('02', 'February'), ('03', 'March'), ('04', 'April'), ('05', 'May'), ('06', 'June'), ('07', 'July'), ('08', 'August'), ('09', 'September'), ('10', 'October'), ('11', 'November'), ('12', 'December')], readonly=True)
    day = fields.Char(size=128, readonly=True)
    state = fields.Selection([('draft', 'Draft'), ('done', 'Done'), ('cancel', 'Rejected')], string='Status', readonly=True)
    employee_id = fields.Many2one('hr.employee', string='Employee', readonly=True)
    nbr = fields.Integer(string='# Payslip lines', readonly=True)
    number = fields.Char(readonly=True)
    struct_id = fields.Many2one('hr.payroll.structure', string='Structure', readonly=True)
    company_id = fields.Many2one('res.company', string='Company', readonly=True)
    paid = fields.Boolean(string='Made Payment Order ? ', readonly=True)
    total = fields.Float(readonly=True)
    category_id = fields.Many2one('hr.salary.rule.category', string='Category', readonly=True)

    @api.model_cr
    def init(self):
        if False:
            while True:
                i = 10
        drop_view_if_exists(self.env.cr, self._table)
        self.env.cr.execute("\n            create or replace view payslip_report as (\n                select\n                    min(l.id) as id,\n                    l.name,\n                    p.struct_id,\n                    p.state,\n                    p.date_from,\n                    p.date_to,\n                    p.number,\n                    p.company_id,\n                    p.paid,\n                    l.category_id,\n                    l.employee_id,\n                    sum(l.total) as total,\n                    to_char(p.date_from, 'YYYY') as year,\n                    to_char(p.date_from, 'MM') as month,\n                    to_char(p.date_from, 'YYYY-MM-DD') as day,\n                    to_char(p.date_to, 'YYYY') as to_year,\n                    to_char(p.date_to, 'MM') as to_month,\n                    to_char(p.date_to, 'YYYY-MM-DD') as to_day,\n                    1 AS nbr\n                from\n                    hr_payslip as p\n                    left join hr_payslip_line as l on (p.id=l.slip_id)\n                where\n                    l.employee_id IS NOT NULL\n                group by\n                    p.number,l.name,p.date_from,p.date_to,p.state,p.company_id,p.paid,\n                    l.employee_id,p.struct_id,l.category_id\n            )\n        ")