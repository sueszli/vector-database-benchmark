import time
import odoo
from odoo import fields
from odoo.tools import float_compare, mute_logger
from odoo.addons.point_of_sale.tests.common import TestPointOfSaleCommon

@odoo.tests.common.at_install(False)
@odoo.tests.common.post_install(True)
class TestPointOfSaleFlow(TestPointOfSaleCommon):

    def test_register_open(self):
        if False:
            return 10
        '\n            In order to test the Point of Sale module, I will open all cash registers through the wizard\n            '
        self.env['pos.open.statement'].create({}).open_statement()

    def test_order_to_payment(self):
        if False:
            for i in range(10):
                print('nop')
        '\n            In order to test the Point of Sale in module, I will do a full flow from the sale to the payment and invoicing.\n            I will use two products, one with price including a 10% tax, the other one with 5% tax excluded from the price.\n        '
        self.pos_config.open_session_cb()
        self.pos_order_pos0 = self.PosOrder.create({'company_id': self.company_id, 'pricelist_id': self.partner1.property_product_pricelist.id, 'partner_id': self.partner1.id, 'lines': [(0, 0, {'name': 'OL/0001', 'product_id': self.product3.id, 'price_unit': 450, 'discount': 0.0, 'qty': 2.0, 'tax_ids': [(6, 0, self.product3.taxes_id.ids)]}), (0, 0, {'name': 'OL/0002', 'product_id': self.product4.id, 'price_unit': 300, 'discount': 0.0, 'qty': 3.0, 'tax_ids': [(6, 0, self.product4.taxes_id.ids)]})]})
        self.assertLess(abs(self.pos_order_pos0.amount_total - (450 * 2 + 300 * 3 * 1.05)), 0.01, 'The order has a wrong amount, tax included.')
        self.assertLess(abs(self.pos_order_pos0.amount_tax - (900 - 450 * 2 / 1.1 + 300 * 0.05 * 3)), 0.01, 'The order has a wrong tax amount.')
        self.pos_discount_0 = self.env['pos.discount'].create({'discount': 5.0})
        context = {'active_model': 'pos.order', 'active_ids': [self.pos_order_pos0.id], 'active_id': self.pos_order_pos0.id}
        self.pos_discount_0.with_context(context).apply_discount()
        self.assertLess(abs(self.pos_order_pos0.amount_total - (450 * 2 + 300 * 3 * 1.05) * 0.95), 0.01, 'The order has a wrong total including tax and discounts')
        context_make_payment = {'active_ids': [self.pos_order_pos0.id], 'active_id': self.pos_order_pos0.id}
        self.pos_make_payment_0 = self.PosMakePayment.with_context(context_make_payment).create({'amount': 100.0})
        context_payment = {'active_id': self.pos_order_pos0.id}
        self.pos_make_payment_0.with_context(context_payment).check()
        self.assertEqual(self.pos_order_pos0.state, 'draft', 'Order should be in draft state.')
        defs = self.pos_make_payment_0.with_context({'active_id': self.pos_order_pos0.id}).default_get(['amount'])
        self.assertLess(abs(defs['amount'] - ((450 * 2 + 300 * 3 * 1.05) * 0.95 - 100.0)), 0.01, 'The remaining balance is incorrect.')
        context_make_payment = {'active_ids': [self.pos_order_pos0.id], 'active_id': self.pos_order_pos0.id}
        self.pos_make_payment_1 = self.PosMakePayment.with_context(context_make_payment).create({'amount': (450 * 2 + 300 * 3 * 1.05) * 0.95 - 100.0})
        self.pos_make_payment_1.with_context(context_make_payment).check()
        self.assertEqual(self.pos_order_pos0.state, 'paid', 'Order should be in paid state.')
        self.pos_order_pos0._create_account_move_line()
        self.assertTrue(self.pos_order_pos0.account_move, 'Journal entry has not been attached to Pos order.')

    def test_order_to_picking(self):
        if False:
            while True:
                i = 10
        '\n            In order to test the Point of Sale in module, I will do three orders from the sale to the payment,\n            invoicing + picking, but will only check the picking consistency in the end.\n\n            TODO: Check the negative picking after changing the picking relation to One2many (also for a mixed use case),\n            check the quantity, the locations and return picking logic\n        '
        self.pos_config.open_session_cb()
        self.pos_order_pos1 = self.PosOrder.create({'company_id': self.company_id, 'pricelist_id': self.partner1.property_product_pricelist.id, 'partner_id': self.partner1.id, 'lines': [(0, 0, {'name': 'OL/0001', 'product_id': self.product3.id, 'price_unit': 450, 'discount': 0.0, 'qty': 2.0, 'tax_ids': [(6, 0, self.product3.taxes_id.ids)]}), (0, 0, {'name': 'OL/0002', 'product_id': self.product4.id, 'price_unit': 300, 'discount': 0.0, 'qty': 3.0, 'tax_ids': [(6, 0, self.product4.taxes_id.ids)]})]})
        context_make_payment = {'active_ids': [self.pos_order_pos1.id], 'active_id': self.pos_order_pos1.id}
        self.pos_make_payment_2 = self.PosMakePayment.with_context(context_make_payment).create({'amount': 1845})
        context_payment = {'active_id': self.pos_order_pos1.id}
        self.pos_make_payment_2.with_context(context_payment).check()
        self.assertEqual(self.pos_order_pos1.state, 'paid', 'Order should be in paid state.')
        self.pos_order_pos1._create_account_move_line()
        self.assertTrue(self.pos_order_pos1.account_move, 'Journal entry has not been attached to Pos order.')
        self.pos_order_pos1.create_picking()
        self.assertEqual(self.pos_order_pos1.picking_id.state, 'done', 'Picking should be in done state.')
        self.assertEqual(self.pos_order_pos1.picking_id.move_lines.mapped('state'), ['done', 'done'], 'Move Lines should be in done state.')
        self.pos_order_pos2 = self.PosOrder.create({'company_id': self.company_id, 'pricelist_id': self.partner1.property_product_pricelist.id, 'partner_id': self.partner1.id, 'lines': [(0, 0, {'name': 'OL/0003', 'product_id': self.product3.id, 'price_unit': 450, 'discount': 0.0, 'qty': -2.0, 'tax_ids': [(6, 0, self.product3.taxes_id.ids)]}), (0, 0, {'name': 'OL/0004', 'product_id': self.product4.id, 'price_unit': 300, 'discount': 0.0, 'qty': -3.0, 'tax_ids': [(6, 0, self.product4.taxes_id.ids)]})]})
        context_make_payment = {'active_ids': [self.pos_order_pos2.id], 'active_id': self.pos_order_pos2.id}
        self.pos_make_payment_3 = self.PosMakePayment.with_context(context_make_payment).create({'amount': -1845})
        context_payment = {'active_id': self.pos_order_pos2.id}
        self.pos_make_payment_3.with_context(context_payment).check()
        self.assertEqual(self.pos_order_pos2.state, 'paid', 'Order should be in paid state.')
        self.pos_order_pos2._create_account_move_line()
        self.assertTrue(self.pos_order_pos2.account_move, 'Journal entry has not been attached to PoS order.')
        self.pos_order_pos2.create_picking()
        self.assertEqual(self.pos_order_pos2.picking_id.state, 'done', 'Picking should be in done state.')
        self.assertEqual(self.pos_order_pos2.picking_id.move_lines.mapped('state'), ['done', 'done'], 'Move Lines should be in done state.')
        self.pos_order_pos3 = self.PosOrder.create({'company_id': self.company_id, 'pricelist_id': self.partner1.property_product_pricelist.id, 'partner_id': self.partner1.id, 'lines': [(0, 0, {'name': 'OL/0005', 'product_id': self.product3.id, 'price_unit': 450, 'discount': 0.0, 'qty': -2.0, 'tax_ids': [(6, 0, self.product3.taxes_id.ids)]}), (0, 0, {'name': 'OL/0006', 'product_id': self.product4.id, 'price_unit': 300, 'discount': 0.0, 'qty': 3.0, 'tax_ids': [(6, 0, self.product4.taxes_id.ids)]})]})
        context_make_payment = {'active_ids': [self.pos_order_pos3.id], 'active_id': self.pos_order_pos3.id}
        self.pos_make_payment_4 = self.PosMakePayment.with_context(context_make_payment).create({'amount': 45})
        context_payment = {'active_id': self.pos_order_pos3.id}
        self.pos_make_payment_4.with_context(context_payment).check()
        self.assertEqual(self.pos_order_pos3.state, 'paid', 'Order should be in paid state.')
        self.pos_order_pos3._create_account_move_line()
        self.assertTrue(self.pos_order_pos3.account_move, 'Journal entry has not been attached to PoS order.')
        self.pos_order_pos3.create_picking()
        self.assertEqual(self.pos_order_pos3.picking_id.state, 'done', 'Picking should be in done state.')
        self.assertEqual(self.pos_order_pos3.picking_id.move_lines.mapped('state'), ['done'], 'Move Lines should be in done state.')

    def test_order_to_invoice(self):
        if False:
            return 10
        self.pos_order_pos1 = self.PosOrder.create({'company_id': self.company_id, 'partner_id': self.partner1.id, 'pricelist_id': self.partner1.property_product_pricelist.id, 'lines': [(0, 0, {'name': 'OL/0001', 'product_id': self.product3.id, 'price_unit': 450, 'discount': 5.0, 'qty': 2.0, 'tax_ids': [(6, 0, self.product3.taxes_id.ids)]}), (0, 0, {'name': 'OL/0002', 'product_id': self.product4.id, 'price_unit': 300, 'discount': 5.0, 'qty': 3.0, 'tax_ids': [(6, 0, self.product4.taxes_id.ids)]})]})
        context_make_payment = {'active_ids': [self.pos_order_pos1.id], 'active_id': self.pos_order_pos1.id}
        self.pos_make_payment = self.PosMakePayment.with_context(context_make_payment).create({'amount': (450 * 2 + 300 * 3 * 1.05) * 0.95})
        context_payment = {'active_id': self.pos_order_pos1.id}
        self.pos_make_payment.with_context(context_payment).check()
        self.assertEqual(self.pos_order_pos1.state, 'paid', 'Order should be in paid state.')
        self.assertFalse(self.pos_order_pos1.invoice_id, 'Invoice should not be attached to order.')
        self.invoice = self.pos_order_pos1.action_pos_order_invoice()
        self.amount_total = self.pos_order_pos1.amount_total
        self.assertEqual(float_compare(self.amount_total, 1752.75, precision_digits=2), 0, 'Invoice not correct')
        'In order to test the reports on Bank Statement defined in point_of_sale module, I create a bank statement line, confirm it and print the reports'
        context_journal = {'journal_type': 'bank'}
        self.assertTrue(self.AccountBankStatement.with_context(context_journal)._default_journal(), 'Journal has not been selected')
        journal = self.env['account.journal'].create({'name': 'Bank Test', 'code': 'BNKT', 'type': 'bank', 'company_id': self.company_id})
        account_statement = self.AccountBankStatement.create({'balance_start': 0.0, 'balance_end_real': 0.0, 'date': time.strftime('%Y-%m-%d'), 'journal_id': journal.id, 'company_id': self.company_id, 'name': 'pos session test'})
        account_statement_line = self.AccountBankStatementLine.create({'amount': 1000, 'partner_id': self.partner4.id, 'statement_id': account_statement.id, 'name': 'EXT001'})
        account_statement.write({'balance_end_real': 1000.0})
        new_aml_dicts = [{'account_id': self.partner4.property_account_receivable_id.id, 'name': 'EXT001', 'credit': 1000.0, 'debit': 0.0}]
        account_statement_line.process_reconciliations([{'new_aml_dicts': new_aml_dicts}])
        self.AccountBankStatement.button_confirm_bank()

    def test_create_from_ui(self):
        if False:
            i = 10
            return i + 15
        '\n        Simulation of sales coming from the interface, even after closing the session\n        '
        FROMPRODUCT = object()

        def compute_tax(product, price, taxes=FROMPRODUCT, qty=1):
            if False:
                i = 10
                return i + 15
            if taxes is FROMPRODUCT:
                taxes = product.taxes_id
            currency = self.pos_config.pricelist_id.currency_id
            taxes = taxes.compute_all(price, currency, qty, product=product)['taxes']
            untax = price * qty
            return (untax, sum((tax.get('amount', 0.0) for tax in taxes)))
        self.pos_config.open_session_cb()
        current_session = self.pos_config.current_session_id
        num_starting_orders = len(current_session.order_ids)
        (untax, atax) = compute_tax(self.carotte, 0.9)
        carrot_order = {'data': {'amount_paid': untax + atax, 'amount_return': 0, 'amount_tax': atax, 'amount_total': untax + atax, 'creation_date': fields.Datetime.now(), 'fiscal_position_id': False, 'lines': [[0, 0, {'discount': 0, 'id': 42, 'pack_lot_ids': [], 'price_unit': 0.9, 'product_id': self.carotte.id, 'qty': 1, 'tax_ids': [(6, 0, self.carotte.taxes_id.ids)]}]], 'name': 'Order 00042-003-0014', 'partner_id': False, 'pos_session_id': current_session.id, 'sequence_number': 2, 'statement_ids': [[0, 0, {'account_id': self.env.user.partner_id.property_account_receivable_id.id, 'amount': untax + atax, 'journal_id': self.pos_config.journal_ids[0].id, 'name': fields.Datetime.now(), 'statement_id': current_session.statement_ids[0].id}]], 'uid': '00042-003-0014', 'user_id': self.env.uid}, 'id': '00042-003-0014', 'to_invoice': False}
        (untax, atax) = compute_tax(self.courgette, 1.2)
        zucchini_order = {'data': {'amount_paid': untax + atax, 'amount_return': 0, 'amount_tax': atax, 'amount_total': untax + atax, 'creation_date': fields.Datetime.now(), 'fiscal_position_id': False, 'lines': [[0, 0, {'discount': 0, 'id': 3, 'pack_lot_ids': [], 'price_unit': 1.2, 'product_id': self.courgette.id, 'qty': 1, 'tax_ids': [(6, 0, self.courgette.taxes_id.ids)]}]], 'name': 'Order 00043-003-0014', 'partner_id': False, 'pos_session_id': current_session.id, 'sequence_number': self.pos_config.journal_id.id, 'statement_ids': [[0, 0, {'account_id': self.env.user.partner_id.property_account_receivable_id.id, 'amount': untax + atax, 'journal_id': self.pos_config.journal_ids[0].id, 'name': fields.Datetime.now(), 'statement_id': current_session.statement_ids[0].id}]], 'uid': '00043-003-0014', 'user_id': self.env.uid}, 'id': '00043-003-0014', 'to_invoice': False}
        (untax, atax) = compute_tax(self.onions, 1.28)
        onions_order = {'data': {'amount_paid': untax + atax, 'amount_return': 0, 'amount_tax': atax, 'amount_total': untax + atax, 'creation_date': fields.Datetime.now(), 'fiscal_position_id': False, 'lines': [[0, 0, {'discount': 0, 'id': 3, 'pack_lot_ids': [], 'price_unit': 1.28, 'product_id': self.onions.id, 'qty': 1, 'tax_ids': [[6, False, self.onions.taxes_id.ids]]}]], 'name': 'Order 00044-003-0014', 'partner_id': False, 'pos_session_id': current_session.id, 'sequence_number': self.pos_config.journal_id.id, 'statement_ids': [[0, 0, {'account_id': self.env.user.partner_id.property_account_receivable_id.id, 'amount': untax + atax, 'journal_id': self.pos_config.journal_ids[0].id, 'name': fields.Datetime.now(), 'statement_id': current_session.statement_ids[0].id}]], 'uid': '00044-003-0014', 'user_id': self.env.uid}, 'id': '00044-003-0014', 'to_invoice': False}
        self.PosOrder.create_from_ui([carrot_order])
        self.assertEqual(num_starting_orders + 1, len(current_session.order_ids), 'Submitted order not encoded')
        self.PosOrder.create_from_ui([carrot_order])
        self.assertEqual(num_starting_orders + 1, len(current_session.order_ids), 'Resubmitted order was not skipped')
        current_session.action_pos_session_closing_control()
        self.assertEqual(current_session.state, 'closed', 'Session was not properly closed')
        self.assertFalse(self.pos_config.current_session_id, 'Current session not properly recomputed')
        with mute_logger('odoo.addons.point_of_sale.models.pos_order'):
            self.PosOrder.create_from_ui([zucchini_order, onions_order])
        rescue_session = self.PosSession.search([('config_id', '=', self.pos_config.id), ('state', '=', 'opened')])
        self.assertEqual(len(rescue_session), 1, 'One (and only one) rescue session should be created for orphan orders')
        self.assertIn('(RESCUE FOR %s)' % current_session.name, rescue_session.name, 'Rescue session is not linked to the previous one')
        self.assertEqual(len(rescue_session.order_ids), 2, 'Rescue session does not contain both orders')
        rescue_session.action_pos_session_closing_control()
        self.assertEqual(rescue_session.state, 'closed', 'Rescue session was not properly closed')