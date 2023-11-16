from odoo import api, models, _
from odoo.exceptions import UserError
from odoo.tools import float_is_zero

class AccountPartialReconcileCashBasis(models.Model):
    _inherit = 'account.partial.reconcile'

    def _get_tax_cash_basis_lines(self, value_before_reconciliation):
        if False:
            while True:
                i = 10
        tax_group = {}
        total_by_cash_basis_account = {}
        line_to_create = []
        move_date = self.debit_move_id.date
        for move in (self.debit_move_id.move_id, self.credit_move_id.move_id):
            if move_date < move.date:
                move_date = move.date
            for line in move.line_ids:
                currency_id = line.currency_id or line.company_id.currency_id
                matched_percentage = value_before_reconciliation[move.id]
                amount = currency_id.round(line.credit_cash_basis - line.debit_cash_basis - (line.credit - line.debit) * matched_percentage)
                if not float_is_zero(amount, precision_rounding=currency_id.rounding) and (not line.tax_exigible):
                    if line.tax_line_id and line.tax_line_id.use_cash_basis:
                        acc = line.account_id.id
                        if tax_group.get(acc, False):
                            tax_group[acc] += amount
                        else:
                            tax_group[acc] = amount
                        acc = line.tax_line_id.cash_basis_account.id
                        if not acc:
                            raise UserError(_('Please configure a Tax Received Account for tax %s') % line.tax_line_id.name)
                        key = (acc, line.tax_line_id.id)
                        if key in total_by_cash_basis_account:
                            total_by_cash_basis_account[key] += amount
                        else:
                            total_by_cash_basis_account[key] = amount
                    if any([tax.use_cash_basis for tax in line.tax_ids]):
                        for tax in line.tax_ids:
                            line_to_create.append((0, 0, {'name': '/', 'debit': currency_id.round(line.debit_cash_basis - line.debit * matched_percentage), 'credit': currency_id.round(line.credit_cash_basis - line.credit * matched_percentage), 'account_id': line.account_id.id, 'tax_ids': [(6, 0, [tax.id])], 'tax_exigible': True}))
                            line_to_create.append((0, 0, {'name': '/', 'credit': currency_id.round(line.debit_cash_basis - line.debit * matched_percentage), 'debit': currency_id.round(line.credit_cash_basis - line.credit * matched_percentage), 'account_id': line.account_id.id, 'tax_exigible': True}))
        for (k, v) in tax_group.items():
            line_to_create.append((0, 0, {'name': '/', 'debit': v if v > 0 else 0.0, 'credit': abs(v) if v < 0 else 0.0, 'account_id': k, 'tax_exigible': True}))
        for (key, v) in total_by_cash_basis_account.items():
            (k, tax_id) = key
            if not self.company_id.currency_id.is_zero(v):
                line_to_create.append((0, 0, {'name': '/', 'debit': abs(v) if v < 0 else 0.0, 'credit': v if v > 0 else 0.0, 'account_id': k, 'tax_line_id': tax_id, 'tax_exigible': True}))
        return (line_to_create, move_date)

    def create_tax_cash_basis_entry(self, value_before_reconciliation):
        if False:
            print('Hello World!')
        (line_to_create, move_date) = self._get_tax_cash_basis_lines(value_before_reconciliation)
        if len(line_to_create) > 0:
            if not self.company_id.tax_cash_basis_journal_id:
                raise UserError(_('There is no tax cash basis journal defined for this company: "%s" \nConfigure it in Accounting/Configuration/Settings') % self.company_id.name)
            move_vals = {'journal_id': self.company_id.tax_cash_basis_journal_id.id, 'line_ids': line_to_create, 'tax_cash_basis_rec_id': self.id}
            if move_date > self.company_id.period_lock_date:
                move_vals['date'] = move_date
            move = self.env['account.move'].create(move_vals)
            move.post()

    @api.model
    def create(self, vals):
        if False:
            i = 10
            return i + 15
        aml = []
        if vals.get('debit_move_id', False):
            aml.append(vals['debit_move_id'])
        if vals.get('credit_move_id', False):
            aml.append(vals['credit_move_id'])
        lines = self.env['account.move.line'].browse(aml)
        value_before_reconciliation = {}
        for line in lines:
            if not value_before_reconciliation.get(line.move_id.id, False):
                value_before_reconciliation[line.move_id.id] = line.move_id.matched_percentage
        res = super(AccountPartialReconcileCashBasis, self).create(vals)
        res.create_tax_cash_basis_entry(value_before_reconciliation)
        return res

    @api.multi
    def unlink(self):
        if False:
            i = 10
            return i + 15
        move = self.env['account.move'].search([('tax_cash_basis_rec_id', 'in', self._ids)])
        move.reverse_moves()
        super(AccountPartialReconcileCashBasis, self).unlink()