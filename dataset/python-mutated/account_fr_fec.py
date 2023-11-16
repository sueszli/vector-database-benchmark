import base64
import csv
import StringIO
from odoo import api, fields, models, _
from odoo.exceptions import Warning

class AccountFrFec(models.TransientModel):
    _name = 'account.fr.fec'
    _description = 'Ficher Echange Informatise'
    date_from = fields.Date(string='Start Date', required=True)
    date_to = fields.Date(string='End Date', required=True)
    fec_data = fields.Binary('FEC File', readonly=True)
    filename = fields.Char(string='Filename', size=256, readonly=True)
    export_type = fields.Selection([('official', 'Official FEC report (posted entries only)'), ('nonofficial', 'Non-official FEC report (posted and unposted entries)')], string='Export Type', required=True, default='official')

    def do_query_unaffected_earnings(self):
        if False:
            while True:
                i = 10
        ' Compute the sum of ending balances for all accounts that are of a type that does not bring forward the balance in new fiscal years.\n            This is needed because we have to display only one line for the initial balance of all expense/revenue accounts in the FEC.\n        '
        sql_query = "\n        SELECT\n            'OUV' AS JournalCode,\n            'Balance initiale' AS JournalLib,\n            'Balance initiale PL' AS EcritureNum,\n            %s AS EcritureDate,\n            '120/129' AS CompteNum,\n            'Benefice (perte) reporte(e)' AS CompteLib,\n            '' AS CompAuxNum,\n            '' AS CompAuxLib,\n            '-' AS PieceRef,\n            %s AS PieceDate,\n            '/' AS EcritureLib,\n            replace(CASE WHEN COALESCE(sum(aml.balance), 0) <= 0 THEN '0,00' ELSE to_char(SUM(aml.balance), '999999999999999D99') END, '.', ',') AS Debit,\n            replace(CASE WHEN COALESCE(sum(aml.balance), 0) >= 0 THEN '0,00' ELSE to_char(-SUM(aml.balance), '999999999999999D99') END, '.', ',') AS Credit,\n            '' AS EcritureLet,\n            '' AS DateLet,\n            %s AS ValidDate,\n            '' AS Montantdevise,\n            '' AS Idevise\n        FROM\n            account_move_line aml\n            LEFT JOIN account_move am ON am.id=aml.move_id\n            JOIN account_account aa ON aa.id = aml.account_id\n            LEFT JOIN account_account_type aat ON aa.user_type_id = aat.id\n        WHERE\n            am.date < %s\n            AND am.company_id = %s\n            AND aat.include_initial_balance = 'f'\n            AND (aml.debit != 0 OR aml.credit != 0)\n        "
        if self.export_type == 'official':
            sql_query += "\n            AND am.state = 'posted'\n            "
        company = self.env.user.company_id
        formatted_date_from = self.date_from.replace('-', '')
        self._cr.execute(sql_query, (formatted_date_from, formatted_date_from, formatted_date_from, self.date_from, company.id))
        listrow = []
        row = self._cr.fetchone()
        listrow = list(row)
        return listrow

    @api.multi
    def generate_fec(self):
        if False:
            for i in range(10):
                print('nop')
        self.ensure_one()
        header = ['JournalCode', 'JournalLib', 'EcritureNum', 'EcritureDate', 'CompteNum', 'CompteLib', 'CompAuxNum', 'CompAuxLib', 'PieceRef', 'PieceDate', 'EcritureLib', 'Debit', 'Credit', 'EcritureLet', 'DateLet', 'ValidDate', 'Montantdevise', 'Idevise']
        company = self.env.user.company_id
        if not company.vat:
            raise Warning(_('Missing VAT number for company %s') % company.name)
        if company.vat[0:2] != 'FR':
            raise Warning(_('FEC is for French companies only !'))
        fecfile = StringIO.StringIO()
        w = csv.writer(fecfile, delimiter='|')
        w.writerow(header)
        unaffected_earnings_xml_ref = self.env.ref('account.data_unaffected_earnings')
        unaffected_earnings_line = True
        if unaffected_earnings_xml_ref:
            unaffected_earnings_results = self.do_query_unaffected_earnings()
            unaffected_earnings_line = False
        sql_query = "\n        SELECT\n            'OUV' AS JournalCode,\n            'Balance initiale' AS JournalLib,\n            'Balance initiale ' || MIN(aa.name) AS EcritureNum,\n            %s AS EcritureDate,\n            MIN(aa.code) AS CompteNum,\n            replace(MIN(aa.name), '|', '/') AS CompteLib,\n            '' AS CompAuxNum,\n            '' AS CompAuxLib,\n            '-' AS PieceRef,\n            %s AS PieceDate,\n            '/' AS EcritureLib,\n            replace(CASE WHEN sum(aml.balance) <= 0 THEN '0,00' ELSE to_char(SUM(aml.balance), '999999999999999D99') END, '.', ',') AS Debit,\n            replace(CASE WHEN sum(aml.balance) >= 0 THEN '0,00' ELSE to_char(-SUM(aml.balance), '999999999999999D99') END, '.', ',') AS Credit,\n            '' AS EcritureLet,\n            '' AS DateLet,\n            %s AS ValidDate,\n            '' AS Montantdevise,\n            '' AS Idevise,\n            MIN(aa.id) AS CompteID\n        FROM\n            account_move_line aml\n            LEFT JOIN account_move am ON am.id=aml.move_id\n            JOIN account_account aa ON aa.id = aml.account_id\n            LEFT JOIN account_account_type aat ON aa.user_type_id = aat.id\n        WHERE\n            am.date < %s\n            AND am.company_id = %s\n            AND aat.include_initial_balance = 't'\n            AND (aml.debit != 0 OR aml.credit != 0)\n        "
        if self.export_type == 'official':
            sql_query += "\n            AND am.state = 'posted'\n            "
        sql_query += '\n        GROUP BY aml.account_id\n        HAVING sum(aml.balance) != 0\n        '
        formatted_date_from = self.date_from.replace('-', '')
        self._cr.execute(sql_query, (formatted_date_from, formatted_date_from, formatted_date_from, self.date_from, company.id))
        for row in self._cr.fetchall():
            listrow = list(row)
            account_id = listrow.pop()
            if not unaffected_earnings_line:
                account = self.env['account.account'].browse(account_id)
                if account.user_type_id.id == self.env.ref('account.data_unaffected_earnings').id:
                    unaffected_earnings_line = True
                    current_amount = float(listrow[11].replace(',', '.')) - float(listrow[12].replace(',', '.'))
                    unaffected_earnings_amount = float(unaffected_earnings_results[11].replace(',', '.')) - float(unaffected_earnings_results[12].replace(',', '.'))
                    listrow_amount = current_amount + unaffected_earnings_amount
                    if listrow_amount > 0:
                        listrow[11] = str(listrow_amount).replace('.', ',')
                        listrow[12] = '0,00'
                    else:
                        listrow[11] = '0,00'
                        listrow[12] = str(-listrow_amount).replace('.', ',')
            w.writerow([s.encode('utf-8') for s in listrow])
        if not unaffected_earnings_line and unaffected_earnings_results and (unaffected_earnings_results[11] != '0,00' or unaffected_earnings_results[12] != '0,00'):
            unaffected_earnings_account = self.env['account.account'].search([('user_type_id', '=', self.env.ref('account.data_unaffected_earnings').id)], limit=1)
            if unaffected_earnings_account:
                unaffected_earnings_results[4] = unaffected_earnings_account.code
                unaffected_earnings_results[5] = unaffected_earnings_account.name
            w.writerow([s.encode('utf-8') for s in unaffected_earnings_results])
        sql_query = "\n        SELECT\n            replace(aj.code, '|', '/') AS JournalCode,\n            replace(aj.name, '|', '/') AS JournalLib,\n            replace(am.name, '|', '/') AS EcritureNum,\n            TO_CHAR(am.date, 'YYYYMMDD') AS EcritureDate,\n            aa.code AS CompteNum,\n            replace(aa.name, '|', '/') AS CompteLib,\n            CASE WHEN rp.ref IS null OR rp.ref = ''\n            THEN COALESCE('ID ' || rp.id, '')\n            ELSE rp.ref\n            END\n            AS CompAuxNum,\n            COALESCE(replace(rp.name, '|', '/'), '') AS CompAuxLib,\n            CASE WHEN am.ref IS null OR am.ref = ''\n            THEN '-'\n            ELSE replace(am.ref, '|', '/')\n            END\n            AS PieceRef,\n            TO_CHAR(am.date, 'YYYYMMDD') AS PieceDate,\n            CASE WHEN aml.name IS NULL THEN '/' ELSE replace(aml.name, '|', '/') END AS EcritureLib,\n            replace(CASE WHEN aml.debit = 0 THEN '0,00' ELSE to_char(aml.debit, '999999999999999D99') END, '.', ',') AS Debit,\n            replace(CASE WHEN aml.credit = 0 THEN '0,00' ELSE to_char(aml.credit, '999999999999999D99') END, '.', ',') AS Credit,\n            CASE WHEN rec.name IS NULL THEN '' ELSE rec.name END AS EcritureLet,\n            CASE WHEN aml.full_reconcile_id IS NULL THEN '' ELSE TO_CHAR(rec.create_date, 'YYYYMMDD') END AS DateLet,\n            TO_CHAR(am.date, 'YYYYMMDD') AS ValidDate,\n            CASE\n                WHEN aml.amount_currency IS NULL OR aml.amount_currency = 0 THEN ''\n                ELSE replace(to_char(aml.amount_currency, '999999999999999D99'), '.', ',')\n            END AS Montantdevise,\n            CASE WHEN aml.currency_id IS NULL THEN '' ELSE rc.name END AS Idevise\n        FROM\n            account_move_line aml\n            LEFT JOIN account_move am ON am.id=aml.move_id\n            LEFT JOIN res_partner rp ON rp.id=aml.partner_id\n            JOIN account_journal aj ON aj.id = am.journal_id\n            JOIN account_account aa ON aa.id = aml.account_id\n            LEFT JOIN res_currency rc ON rc.id = aml.currency_id\n            LEFT JOIN account_full_reconcile rec ON rec.id = aml.full_reconcile_id\n        WHERE\n            am.date >= %s\n            AND am.date <= %s\n            AND am.company_id = %s\n            AND (aml.debit != 0 OR aml.credit != 0)\n        "
        if self.export_type == 'official':
            sql_query += "\n            AND am.state = 'posted'\n            "
        sql_query += '\n        ORDER BY\n            am.date,\n            am.name,\n            aml.id\n        '
        self._cr.execute(sql_query, (self.date_from, self.date_to, company.id))
        for row in self._cr.fetchall():
            listrow = list(row)
            w.writerow([s.encode('utf-8') for s in listrow])
        siren = company.vat[4:13]
        end_date = self.date_to.replace('-', '')
        suffix = ''
        if self.export_type == 'nonofficial':
            suffix = '-NONOFFICIAL'
        fecvalue = fecfile.getvalue()
        self.write({'fec_data': base64.encodestring(fecvalue), 'filename': '%sFEC%s%s.csv' % (siren, end_date, suffix)})
        fecfile.close()
        action = {'name': 'FEC', 'type': 'ir.actions.act_url', 'url': 'web/content/?model=account.fr.fec&id=' + str(self.id) + '&filename_field=filename&field=fec_data&download=true&filename=' + self.filename, 'target': 'self'}
        return action