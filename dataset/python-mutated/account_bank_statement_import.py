import base64
from odoo import api, fields, models, _
from odoo.exceptions import UserError
from odoo.addons.base.res.res_bank import sanitize_account_number
import logging
_logger = logging.getLogger(__name__)

class AccountBankStatementLine(models.Model):
    _inherit = 'account.bank.statement.line'
    unique_import_id = fields.Char(string='Import ID', readonly=True, copy=False)
    _sql_constraints = [('unique_import_id', 'unique (unique_import_id)', 'A bank account transactions can be imported only once !')]

class AccountBankStatementImport(models.TransientModel):
    _name = 'account.bank.statement.import'
    _description = 'Import Bank Statement'
    data_file = fields.Binary(string='Bank Statement File', required=True, help='Get you bank statements in electronic format from your bank and select them here.')
    filename = fields.Char()

    @api.multi
    def import_file(self):
        if False:
            for i in range(10):
                print('nop')
        ' Process the file chosen in the wizard, create bank statement(s) and go to reconciliation. '
        self.ensure_one()
        (currency_code, account_number, stmts_vals) = self.with_context(active_id=self.ids[0])._parse_file(base64.b64decode(self.data_file))
        self._check_parsed_data(stmts_vals)
        (currency, journal) = self._find_additional_data(currency_code, account_number)
        if not journal:
            return self.with_context(active_id=self.ids[0])._journal_creation_wizard(currency, account_number)
        if not journal.default_debit_account_id or not journal.default_credit_account_id:
            raise UserError(_('You have to set a Default Debit Account and a Default Credit Account for the journal: %s') % (journal.name,))
        stmts_vals = self._complete_stmts_vals(stmts_vals, journal, account_number)
        (statement_ids, notifications) = self._create_bank_statements(stmts_vals)
        journal.bank_statements_source = 'file_import'
        action = self.env.ref('account.action_bank_reconcile_bank_statements')
        return {'name': action.name, 'tag': action.tag, 'context': {'statement_ids': statement_ids, 'notifications': notifications}, 'type': 'ir.actions.client'}

    def _journal_creation_wizard(self, currency, account_number):
        if False:
            for i in range(10):
                print('nop')
        ' Calls a wizard that allows the user to carry on with journal creation '
        return {'name': _('Journal Creation'), 'type': 'ir.actions.act_window', 'res_model': 'account.bank.statement.import.journal.creation', 'view_type': 'form', 'view_mode': 'form', 'target': 'new', 'context': {'statement_import_transient_id': self.env.context['active_id'], 'default_bank_acc_number': account_number, 'default_name': _('Bank') + ' ' + account_number, 'default_currency_id': currency and currency.id or False, 'default_type': 'bank'}}

    def _parse_file(self, data_file):
        if False:
            while True:
                i = 10
        " Each module adding a file support must extends this method. It processes the file if it can, returns super otherwise, resulting in a chain of responsability.\n            This method parses the given file and returns the data required by the bank statement import process, as specified below.\n            rtype: triplet (if a value can't be retrieved, use None)\n                - currency code: string (e.g: 'EUR')\n                    The ISO 4217 currency code, case insensitive\n                - account number: string (e.g: 'BE1234567890')\n                    The number of the bank account which the statement belongs to\n                - bank statements data: list of dict containing (optional items marked by o) :\n                    - 'name': string (e.g: '000000123')\n                    - 'date': date (e.g: 2013-06-26)\n                    -o 'balance_start': float (e.g: 8368.56)\n                    -o 'balance_end_real': float (e.g: 8888.88)\n                    - 'transactions': list of dict containing :\n                        - 'name': string (e.g: 'KBC-INVESTERINGSKREDIET 787-5562831-01')\n                        - 'date': date\n                        - 'amount': float\n                        - 'unique_import_id': string\n                        -o 'account_number': string\n                            Will be used to find/create the res.partner.bank in odoo\n                        -o 'note': string\n                        -o 'partner_name': string\n                        -o 'ref': string\n        "
        raise UserError(_('Could not make sense of the given file.\nDid you install the module to support this type of file ?'))

    def _check_parsed_data(self, stmts_vals):
        if False:
            for i in range(10):
                print('nop')
        ' Basic and structural verifications '
        if len(stmts_vals) == 0:
            raise UserError(_("This file doesn't contain any statement."))
        no_st_line = True
        for vals in stmts_vals:
            if vals['transactions'] and len(vals['transactions']) > 0:
                no_st_line = False
                break
        if no_st_line:
            raise UserError(_("This file doesn't contain any transaction."))

    def _check_journal_bank_account(self, journal, account_number):
        if False:
            print('Hello World!')
        return journal.bank_account_id.sanitized_acc_number == account_number

    def _find_additional_data(self, currency_code, account_number):
        if False:
            while True:
                i = 10
        " Look for a res.currency and account.journal using values extracted from the\n            statement and make sure it's consistent.\n        "
        company_currency = self.env.user.company_id.currency_id
        journal_obj = self.env['account.journal']
        currency = None
        sanitized_account_number = sanitize_account_number(account_number)
        if currency_code:
            currency = self.env['res.currency'].search([('name', '=ilike', currency_code)], limit=1)
            if not currency:
                raise UserError(_("No currency found matching '%s'.") % currency_code)
            if currency == company_currency:
                currency = False
        journal = journal_obj.browse(self.env.context.get('journal_id', []))
        if account_number:
            if journal and (not journal.bank_account_id):
                journal.set_bank_account(account_number)
            elif not journal:
                journal = journal_obj.search([('bank_account_id.sanitized_acc_number', '=', sanitized_account_number)])
            elif not self._check_journal_bank_account(journal, sanitized_account_number):
                raise UserError(_('The account of this statement (%s) is not the same as the journal (%s).') % (account_number, journal.bank_account_id.acc_number))
        if journal:
            journal_currency = journal.currency_id
            if currency is None:
                currency = journal_currency
            if currency and currency != journal_currency:
                statement_cur_code = not currency and company_currency.name or currency.name
                journal_cur_code = not journal_currency and company_currency.name or journal_currency.name
                raise UserError(_('The currency of the bank statement (%s) is not the same as the currency of the journal (%s) !') % (statement_cur_code, journal_cur_code))
        if not journal and (not account_number):
            raise UserError(_('Cannot find in which journal import this statement. Please manually select a journal.'))
        return (currency, journal)

    def _complete_stmts_vals(self, stmts_vals, journal, account_number):
        if False:
            return 10
        for st_vals in stmts_vals:
            st_vals['journal_id'] = journal.id
            if not st_vals.get('reference'):
                st_vals['reference'] = self.filename
            if st_vals.get('number'):
                st_vals['name'] = journal.sequence_id.with_context(ir_sequence_date=st_vals.get('date')).get_next_char(st_vals['number'])
                del st_vals['number']
            for line_vals in st_vals['transactions']:
                unique_import_id = line_vals.get('unique_import_id')
                if unique_import_id:
                    sanitized_account_number = sanitize_account_number(account_number)
                    line_vals['unique_import_id'] = (sanitized_account_number and sanitized_account_number + '-' or '') + str(journal.id) + '-' + unique_import_id
                if not line_vals.get('bank_account_id'):
                    partner_id = False
                    bank_account_id = False
                    identifying_string = line_vals.get('account_number')
                    if identifying_string:
                        partner_bank = self.env['res.partner.bank'].search([('acc_number', '=', identifying_string)], limit=1)
                        if partner_bank:
                            bank_account_id = partner_bank.id
                            partner_id = partner_bank.partner_id.id
                        else:
                            bank_account_id = self.env['res.partner.bank'].create({'acc_number': line_vals['account_number']}).id
                    line_vals['partner_id'] = partner_id
                    line_vals['bank_account_id'] = bank_account_id
        return stmts_vals

    def _create_bank_statements(self, stmts_vals):
        if False:
            while True:
                i = 10
        ' Create new bank statements from imported values, filtering out already imported transactions, and returns data used by the reconciliation widget '
        BankStatement = self.env['account.bank.statement']
        BankStatementLine = self.env['account.bank.statement.line']
        statement_ids = []
        ignored_statement_lines_import_ids = []
        for st_vals in stmts_vals:
            filtered_st_lines = []
            for line_vals in st_vals['transactions']:
                if 'unique_import_id' not in line_vals or not line_vals['unique_import_id'] or (not bool(BankStatementLine.sudo().search([('unique_import_id', '=', line_vals['unique_import_id'])], limit=1))):
                    if line_vals['amount'] != 0:
                        filtered_st_lines.append(line_vals)
                else:
                    ignored_statement_lines_import_ids.append(line_vals['unique_import_id'])
                    if 'balance_start' in st_vals:
                        st_vals['balance_start'] += line_vals['amount']
            if len(filtered_st_lines) > 0:
                st_vals.pop('transactions', None)
                for line_vals in filtered_st_lines:
                    line_vals.pop('account_number', None)
                st_vals['line_ids'] = [[0, False, line] for line in filtered_st_lines]
                statement_ids.append(BankStatement.create(st_vals).id)
        if len(statement_ids) == 0:
            raise UserError(_('You have already imported that file.'))
        notifications = []
        num_ignored = len(ignored_statement_lines_import_ids)
        if num_ignored > 0:
            notifications += [{'type': 'warning', 'message': _('%d transactions had already been imported and were ignored.') % num_ignored if num_ignored > 1 else _('1 transaction had already been imported and was ignored.'), 'details': {'name': _('Already imported items'), 'model': 'account.bank.statement.line', 'ids': BankStatementLine.search([('unique_import_id', 'in', ignored_statement_lines_import_ids)]).ids}}]
        return (statement_ids, notifications)