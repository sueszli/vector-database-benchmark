from odoo.addons.account.tests.account_test_classes import AccountingTestCase

class TestAccountValidateAccount(AccountingTestCase):

    def test_account_validate_account(self):
        if False:
            for i in range(10):
                print('nop')
        account_move_line = self.env['account.move.line']
        account_cash = self.env['account.account'].search([('user_type_id.type', '=', 'liquidity')], limit=1)
        journal = self.env['account.journal'].search([('type', '=', 'bank')], limit=1)
        company_id = self.env['res.users'].browse(self.env.uid).company_id.id
        move = self.env['account.move'].create({'name': '/', 'ref': '2011010', 'journal_id': journal.id, 'state': 'draft', 'company_id': company_id})
        account_move_line.create({'account_id': account_cash.id, 'name': 'Basic Computer', 'move_id': move.id})
        account_move_line.create({'account_id': account_cash.id, 'name': 'Basic Computer', 'move_id': move.id})
        self.assertTrue(move.state == 'draft', 'Initially account move state is Draft')
        validate_account_move = self.env['validate.account.move'].with_context(active_ids=move.id).create({})
        validate_account_move.with_context({'active_ids': [move.id]}).validate_move()
        self.assertTrue(move.state == 'posted', 'Initially account move state is Posted')