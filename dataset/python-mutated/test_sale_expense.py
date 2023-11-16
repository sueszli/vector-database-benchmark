from odoo.addons.sale.tests.test_sale_common import TestSale

class TestSaleExpense(TestSale):

    def test_sale_expense(self):
        if False:
            i = 10
            return i + 15
        ' Test the behaviour of sales orders when managing expenses '
        self.env.ref('product.list0').currency_id = self.env.ref('base.main_company').currency_id
        prod = self.env.ref('product.product_product_1')
        so = self.env['sale.order'].create({'partner_id': self.partner.id, 'partner_invoice_id': self.partner.id, 'partner_shipping_id': self.partner.id, 'order_line': [(0, 0, {'name': prod.name, 'product_id': prod.id, 'product_uom_qty': 2, 'product_uom': prod.uom_id.id, 'price_unit': prod.list_price})], 'pricelist_id': self.env.ref('product.list0').id})
        so._compute_tax_id()
        so.action_confirm()
        so._create_analytic_account()
        init_price = so.amount_total
        prod_exp_1 = self.env.ref('hr_expense.air_ticket')
        company = self.env.ref('base.main_company')
        journal = self.env['account.journal'].create({'name': 'Purchase Journal - Test', 'code': 'HRTPJ', 'type': 'purchase', 'company_id': company.id})
        account_payable = self.env['account.account'].create({'code': 'X1111', 'name': 'HR Expense - Test Payable Account', 'user_type_id': self.env.ref('account.data_account_type_payable').id, 'reconcile': True})
        employee = self.env['hr.employee'].create({'name': 'Test employee', 'user_id': self.user.id, 'address_home_id': self.user.partner_id.id})
        self.user.partner_id.property_account_payable_id = account_payable.id
        sheet = self.env['hr.expense.sheet'].create({'name': 'Expense for John Smith', 'employee_id': employee.id, 'journal_id': journal.id})
        exp = self.env['hr.expense'].create({'name': 'Air Travel', 'product_id': prod_exp_1.id, 'analytic_account_id': so.project_id.id, 'unit_amount': 621.54, 'employee_id': employee.id, 'sheet_id': sheet.id})
        sheet.approve_expense_sheets()
        sheet.action_sheet_move_create()
        self.assertTrue(prod_exp_1 in map(lambda so: so.product_id, so.order_line), 'Sale Expense: expense product should be in so')
        sol = so.order_line.filtered(lambda sol: sol.product_id.id == prod_exp_1.id)
        self.assertEqual((sol.price_unit, sol.qty_delivered), (621.54, 1.0), 'Sale Expense: error when invoicing an expense at cost')
        self.assertEqual(so.amount_total, init_price, 'Sale Expense: price of so not updated after adding expense')
        init_price = so.amount_total
        prod_exp_2 = self.env.ref('hr_expense.car_travel')
        sheet = self.env['hr.expense.sheet'].create({'name': 'Expense for John Smith', 'employee_id': employee.id, 'journal_id': journal.id})
        exp = self.env['hr.expense'].create({'name': 'Car Travel', 'product_id': prod_exp_2.id, 'analytic_account_id': so.project_id.id, 'product_uom_id': self.env.ref('product.product_uom_km').id, 'unit_amount': 0.15, 'quantity': 100, 'employee_id': employee.id, 'sheet_id': sheet.id})
        sheet.approve_expense_sheets()
        sheet.action_sheet_move_create()
        self.assertTrue(prod_exp_2 in map(lambda so: so.product_id, so.order_line), 'Sale Expense: expense product should be in so')
        sol = so.order_line.filtered(lambda sol: sol.product_id.id == prod_exp_2.id)
        self.assertEqual((sol.price_unit, sol.qty_delivered), (prod_exp_2.list_price, 100.0), 'Sale Expense: error when invoicing an expense at cost')
        self.assertEqual(so.amount_total, init_price, 'Sale Expense: price of so not updated after adding expense')
        inv_id = so.action_invoice_create()
        inv = self.env['account.invoice'].browse(inv_id)
        self.assertEqual(inv.amount_untaxed, 621.54 + prod_exp_2.list_price * 100.0, 'Sale Expense: invoicing of expense is wrong')