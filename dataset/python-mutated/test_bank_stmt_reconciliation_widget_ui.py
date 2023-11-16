from odoo.tests import HttpCase

class TestUi(HttpCase):
    post_install = True
    at_install = False

    def test_01_admin_bank_statement_reconciliation(self):
        if False:
            for i in range(10):
                print('nop')
        self.phantom_js('/', "odoo.__DEBUG__.services['web.Tour'].run('bank_statement_reconciliation', 'test')", "odoo.__DEBUG__.services['web.Tour'].tours.bank_statement_reconciliation", login='admin')