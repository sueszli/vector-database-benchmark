from .common import TestAccountBudgetCommon
from odoo.fields import Date
import datetime

class TestAccountBudget(TestAccountBudgetCommon):

    def test_account_budget(self):
        if False:
            while True:
                i = 10
        budget = self.env['crossovered.budget'].create({'date_from': Date.from_string('%s-01-01' % (datetime.datetime.now().year + 1)), 'date_to': Date.from_string('%s-12-31' % (datetime.datetime.now().year + 1)), 'name': 'Budget %s' % (datetime.datetime.now().year + 1), 'state': 'draft'})
        self.env['crossovered.budget.lines'].create({'crossovered_budget_id': budget.id, 'analytic_account_id': self.ref('analytic.analytic_partners_camp_to_camp'), 'date_from': Date.from_string('%s-01-01' % (datetime.datetime.now().year + 1)), 'date_to': Date.from_string('%s-12-31' % (datetime.datetime.now().year + 1)), 'general_budget_id': self.account_budget_post_purchase0.id, 'planned_amount': 10000.0})
        self.env['crossovered.budget.lines'].create({'crossovered_budget_id': budget.id, 'analytic_account_id': self.ref('analytic.analytic_our_super_product'), 'date_from': Date.from_string('%s-09-01' % (datetime.datetime.now().year + 1)), 'date_to': Date.from_string('%s-09-30' % (datetime.datetime.now().year + 1)), 'general_budget_id': self.account_budget_post_sales0.id, 'planned_amount': 400000.0})
        self.assertEqual(budget.state, 'draft')
        budget.action_budget_confirm()
        self.assertEqual(budget.state, 'confirm')
        budget.action_budget_validate()
        self.assertEqual(budget.state, 'validate')
        budget.action_budget_done()
        self.assertEqual(budget.state, 'done')