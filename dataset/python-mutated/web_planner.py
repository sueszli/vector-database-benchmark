from odoo import api, models

class PlannerHrExpense(models.Model):
    _inherit = 'web.planner'

    @api.model
    def _get_planner_application(self):
        if False:
            while True:
                i = 10
        planner = super(PlannerHrExpense, self)._get_planner_application()
        planner.append(['planner_hr_expense', 'Expense Planner'])
        return planner

    @api.model
    def _prepare_planner_hr_expense_data(self):
        if False:
            i = 10
            return i + 15
        alias_record = self.env.ref('hr_expense.mail_alias_expense')
        return {'alias_domain': alias_record.alias_domain, 'alias_name': alias_record.alias_name}