from odoo import api, fields, models

class HrExpenseRefuseWizard(models.TransientModel):
    _name = 'hr.expense.refuse.wizard'
    _description = 'Hr Expense refuse Reason wizard'
    description = fields.Char(string='Reason', required=True)

    @api.multi
    def expense_refuse_reason(self):
        if False:
            print('Hello World!')
        self.ensure_one()
        context = dict(self._context or {})
        active_ids = context.get('active_ids', [])
        expense_sheet = self.env['hr.expense.sheet'].browse(active_ids)
        expense_sheet.refuse_expenses(self.description)
        return {'type': 'ir.actions.act_window_close'}