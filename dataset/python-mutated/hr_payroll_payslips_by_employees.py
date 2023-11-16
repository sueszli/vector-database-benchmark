from odoo import api, models

class HrPayslipEmployees(models.TransientModel):
    _inherit = 'hr.payslip.employees'

    @api.multi
    def compute_sheet(self):
        if False:
            return 10
        journal_id = False
        if self.env.context.get('active_id'):
            journal_id = self.env['hr.payslip.run'].browse(self.env.context.get('active_id')).journal_id.id
        return super(HrPayslipEmployees, self.with_context(journal_id=journal_id)).compute_sheet()