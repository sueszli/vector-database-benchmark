from datetime import datetime
import time
from odoo import api, models
from odoo.tools import DEFAULT_SERVER_DATE_FORMAT

class ReportPartnerLedger(models.AbstractModel):
    _name = 'report.account.report_partnerledger'

    def _lines(self, data, partner):
        if False:
            print('Hello World!')
        full_account = []
        currency = self.env['res.currency']
        query_get_data = self.env['account.move.line'].with_context(data['form'].get('used_context', {}))._query_get()
        reconcile_clause = '' if data['form']['reconciled'] else ' AND "account_move_line".reconciled = false '
        params = [partner.id, tuple(data['computed']['move_state']), tuple(data['computed']['account_ids'])] + query_get_data[2]
        query = '\n            SELECT "account_move_line".id, "account_move_line".date, j.code, acc.code as a_code, acc.name as a_name, "account_move_line".ref, m.name as move_name, "account_move_line".name, "account_move_line".debit, "account_move_line".credit, "account_move_line".amount_currency,"account_move_line".currency_id, c.symbol AS currency_code\n            FROM ' + query_get_data[0] + '\n            LEFT JOIN account_journal j ON ("account_move_line".journal_id = j.id)\n            LEFT JOIN account_account acc ON ("account_move_line".account_id = acc.id)\n            LEFT JOIN res_currency c ON ("account_move_line".currency_id=c.id)\n            LEFT JOIN account_move m ON (m.id="account_move_line".move_id)\n            WHERE "account_move_line".partner_id = %s\n                AND m.state IN %s\n                AND "account_move_line".account_id IN %s AND ' + query_get_data[1] + reconcile_clause + '\n                ORDER BY "account_move_line".date'
        self.env.cr.execute(query, tuple(params))
        res = self.env.cr.dictfetchall()
        sum = 0.0
        lang_code = self.env.context.get('lang') or 'en_US'
        lang = self.env['res.lang']
        lang_id = lang._lang_get(lang_code)
        date_format = lang_id.date_format
        for r in res:
            r['date'] = datetime.strptime(r['date'], DEFAULT_SERVER_DATE_FORMAT).strftime(date_format)
            r['displayed_name'] = '-'.join((r[field_name] for field_name in ('move_name', 'ref', 'name') if r[field_name] not in (None, '', '/')))
            sum += r['debit'] - r['credit']
            r['progress'] = sum
            r['currency_id'] = currency.browse(r.get('currency_id'))
            full_account.append(r)
        return full_account

    def _sum_partner(self, data, partner, field):
        if False:
            print('Hello World!')
        if field not in ['debit', 'credit', 'debit - credit']:
            return
        result = 0.0
        query_get_data = self.env['account.move.line'].with_context(data['form'].get('used_context', {}))._query_get()
        reconcile_clause = '' if data['form']['reconciled'] else ' AND "account_move_line".reconciled = false '
        params = [partner.id, tuple(data['computed']['move_state']), tuple(data['computed']['account_ids'])] + query_get_data[2]
        query = 'SELECT sum(' + field + ')\n                FROM ' + query_get_data[0] + ', account_move AS m\n                WHERE "account_move_line".partner_id = %s\n                    AND m.id = "account_move_line".move_id\n                    AND m.state IN %s\n                    AND account_id IN %s\n                    AND ' + query_get_data[1] + reconcile_clause
        self.env.cr.execute(query, tuple(params))
        contemp = self.env.cr.fetchone()
        if contemp is not None:
            result = contemp[0] or 0.0
        return result

    @api.model
    def render_html(self, docids, data=None):
        if False:
            i = 10
            return i + 15
        data['computed'] = {}
        obj_partner = self.env['res.partner']
        query_get_data = self.env['account.move.line'].with_context(data['form'].get('used_context', {}))._query_get()
        data['computed']['move_state'] = ['draft', 'posted']
        if data['form'].get('target_move', 'all') == 'posted':
            data['computed']['move_state'] = ['posted']
        result_selection = data['form'].get('result_selection', 'customer')
        if result_selection == 'supplier':
            data['computed']['ACCOUNT_TYPE'] = ['payable']
        elif result_selection == 'customer':
            data['computed']['ACCOUNT_TYPE'] = ['receivable']
        else:
            data['computed']['ACCOUNT_TYPE'] = ['payable', 'receivable']
        self.env.cr.execute('\n            SELECT a.id\n            FROM account_account a\n            WHERE a.internal_type IN %s\n            AND NOT a.deprecated', (tuple(data['computed']['ACCOUNT_TYPE']),))
        data['computed']['account_ids'] = [a for (a,) in self.env.cr.fetchall()]
        params = [tuple(data['computed']['move_state']), tuple(data['computed']['account_ids'])] + query_get_data[2]
        reconcile_clause = '' if data['form']['reconciled'] else ' AND "account_move_line".reconciled = false '
        query = '\n            SELECT DISTINCT "account_move_line".partner_id\n            FROM ' + query_get_data[0] + ', account_account AS account, account_move AS am\n            WHERE "account_move_line".partner_id IS NOT NULL\n                AND "account_move_line".account_id = account.id\n                AND am.id = "account_move_line".move_id\n                AND am.state IN %s\n                AND "account_move_line".account_id IN %s\n                AND NOT account.deprecated\n                AND ' + query_get_data[1] + reconcile_clause
        self.env.cr.execute(query, tuple(params))
        partner_ids = [res['partner_id'] for res in self.env.cr.dictfetchall()]
        partners = obj_partner.browse(partner_ids)
        partners = sorted(partners, key=lambda x: (x.ref, x.name))
        docargs = {'doc_ids': partner_ids, 'doc_model': self.env['res.partner'], 'data': data, 'docs': partners, 'time': time, 'lines': self._lines, 'sum_partner': self._sum_partner}
        return self.env['report'].render('account.report_partnerledger', docargs)