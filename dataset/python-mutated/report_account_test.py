import datetime
from odoo import api, models, _
from odoo.tools.safe_eval import safe_eval

class ReportAssertAccount(models.AbstractModel):
    _name = 'report.account_test.report_accounttest'

    @api.model
    def execute_code(self, code_exec):
        if False:
            while True:
                i = 10

        def reconciled_inv():
            if False:
                return 10
            '\n            returns the list of invoices that are set as reconciled = True\n            '
            return self.env['account.invoice'].search([('reconciled', '=', True)]).ids

        def order_columns(item, cols=None):
            if False:
                while True:
                    i = 10
            '\n            This function is used to display a dictionary as a string, with its columns in the order chosen.\n\n            :param item: dict\n            :param cols: list of field names\n            :returns: a list of tuples (fieldname: value) in a similar way that would dict.items() do except that the\n                returned values are following the order given by cols\n            :rtype: [(key, value)]\n            '
            if cols is None:
                cols = item.keys()
            return [(col, item.get(col)) for col in cols if col in item.keys()]
        localdict = {'cr': self.env.cr, 'uid': self.env.uid, 'reconciled_inv': reconciled_inv, 'result': None, 'column_order': None, '_': _}
        safe_eval(code_exec, localdict, mode='exec', nocopy=True)
        result = localdict['result']
        column_order = localdict.get('column_order', None)
        if not isinstance(result, (tuple, list, set)):
            result = [result]
        if not result:
            result = [_('The test was passed successfully')]
        else:

            def _format(item):
                if False:
                    print('Hello World!')
                if isinstance(item, dict):
                    return ', '.join(['%s: %s' % (tup[0], tup[1]) for tup in order_columns(item, column_order)])
                else:
                    return item
            result = [_format(rec) for rec in result]
        return result

    @api.model
    def render_html(self, docids, data=None):
        if False:
            return 10
        Report = self.env['report']
        report = Report._get_report_from_name('account_test.report_accounttest')
        records = self.env['accounting.assert.test'].browse(self.ids)
        docargs = {'doc_ids': self._ids, 'doc_model': report.model, 'docs': records, 'data': data, 'execute_code': self.execute_code, 'datetime': datetime}
        return Report.render('account_test.report_accounttest', docargs)