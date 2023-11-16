from odoo.addons.account.tests.account_test_classes import AccountingTestCase
from odoo.exceptions import Warning

class TestAccountSupplierInvoice(AccountingTestCase):

    def test_supplier_invoice(self):
        if False:
            print('Hello World!')
        tax = self.env['account.tax'].create({'name': 'Tax 10.0', 'amount': 10.0, 'amount_type': 'fixed'})
        analytic_account = self.env['account.analytic.account'].create({'name': 'test account'})
        invoice_account = self.env['account.account'].search([('user_type_id', '=', self.env.ref('account.data_account_type_receivable').id)], limit=1).id
        invoice_line_account = self.env['account.account'].search([('user_type_id', '=', self.env.ref('account.data_account_type_expenses').id)], limit=1).id
        invoice = self.env['account.invoice'].create({'partner_id': self.env.ref('base.res_partner_2').id, 'account_id': invoice_account, 'type': 'in_invoice'})
        self.env['account.invoice.line'].create({'product_id': self.env.ref('product.product_product_4').id, 'quantity': 1.0, 'price_unit': 100.0, 'invoice_id': invoice.id, 'name': 'product that cost 100', 'account_id': invoice_line_account, 'invoice_line_tax_ids': [(6, 0, [tax.id])], 'account_analytic_id': analytic_account.id})
        self.assertTrue(invoice.state == 'draft', 'Initially vendor bill state is Draft')
        invoice.action_invoice_open()
        with self.assertRaises(Warning):
            invoice.move_id.button_cancel()

    def test_supplier_invoice2(self):
        if False:
            while True:
                i = 10
        tax_fixed = self.env['account.tax'].create({'sequence': 10, 'name': 'Tax 10.0 (Fixed)', 'amount': 10.0, 'amount_type': 'fixed', 'include_base_amount': True})
        tax_percent_included_base_incl = self.env['account.tax'].create({'sequence': 20, 'name': 'Tax 50.0% (Percentage of Price Tax Included)', 'amount': 50.0, 'amount_type': 'division', 'include_base_amount': True})
        tax_percentage = self.env['account.tax'].create({'sequence': 30, 'name': 'Tax 20.0% (Percentage of Price)', 'amount': 20.0, 'amount_type': 'percent', 'include_base_amount': False})
        analytic_account = self.env['account.analytic.account'].create({'name': 'test account'})
        invoice_account = self.env['account.account'].search([('user_type_id', '=', self.env.ref('account.data_account_type_receivable').id)], limit=1).id
        invoice_line_account = self.env['account.account'].search([('user_type_id', '=', self.env.ref('account.data_account_type_expenses').id)], limit=1).id
        invoice = self.env['account.invoice'].create({'partner_id': self.env.ref('base.res_partner_2').id, 'account_id': invoice_account, 'type': 'in_invoice'})
        invoice_line = self.env['account.invoice.line'].create({'product_id': self.env.ref('product.product_product_4').id, 'quantity': 5.0, 'price_unit': 100.0, 'invoice_id': invoice.id, 'name': 'product that cost 100', 'account_id': invoice_line_account, 'invoice_line_tax_ids': [(6, 0, [tax_fixed.id, tax_percent_included_base_incl.id, tax_percentage.id])], 'account_analytic_id': analytic_account.id})
        invoice.compute_taxes()
        self.assertTrue(invoice.state == 'draft', 'Initially vendor bill state is Draft')
        invoice.action_invoice_open()
        invoice_tax = invoice.tax_line_ids.sorted(key=lambda r: r.sequence)
        self.assertEquals(invoice_tax.mapped('amount'), [50.0, 550.0, 220.0])
        self.assertEquals(invoice_tax.mapped('base'), [500.0, 550.0, 1100.0])
        with self.assertRaises(Warning):
            invoice.move_id.button_cancel()