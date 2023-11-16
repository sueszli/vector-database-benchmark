from odoo.api import Environment
import odoo.tests

class TestUi(odoo.tests.HttpCase):

    def test_01_pos_basic_order(self):
        if False:
            while True:
                i = 10
        cr = self.registry.cursor()
        assert cr == self.registry.test_cr
        env = Environment(cr, self.uid, {})
        journal_obj = env['account.journal']
        account_obj = env['account.account']
        main_company = env.ref('base.main_company')
        main_pos_config = env.ref('point_of_sale.pos_config_main')
        account_receivable = account_obj.create({'code': 'X1012', 'name': 'Account Receivable - Test', 'user_type_id': env.ref('account.data_account_type_receivable').id, 'reconcile': True})
        field = self.env['ir.model.fields'].search([('name', '=', 'property_account_receivable_id'), ('model', '=', 'res.partner'), ('relation', '=', 'account.account')], limit=1)
        env['ir.property'].create({'name': 'property_account_receivable_id', 'company_id': main_company.id, 'fields_id': field.id, 'value': 'account.account,' + str(account_receivable.id)})
        main_company.currency_id = env.ref('base.USD')
        test_sale_journal = journal_obj.create({'name': 'Sale Journal - Test', 'code': 'TSJ', 'type': 'sale', 'company_id': main_company.id})
        env['product.pricelist'].search([]).write(dict(currency_id=main_company.currency_id.id))
        main_pos_config.write({'journal_id': test_sale_journal.id, 'invoice_journal_id': test_sale_journal.id, 'journal_ids': [(0, 0, {'name': 'Cash Journal - Test', 'code': 'TSC', 'type': 'cash', 'company_id': main_company.id, 'journal_user': True})]})
        main_pos_config.open_session_cb()
        env['ir.module.module'].search([('name', '=', 'point_of_sale')], limit=1).state = 'installed'
        cr.release()
        self.phantom_js('/pos/web', "odoo.__DEBUG__.services['web_tour.tour'].run('pos_basic_order')", "odoo.__DEBUG__.services['web_tour.tour'].tours.pos_basic_order.ready", login='admin')
        for order in env['pos.order'].search([]):
            self.assertEqual(order.state, 'paid', 'Validated order has payment of ' + str(order.amount_paid) + ' and total of ' + str(order.amount_total))