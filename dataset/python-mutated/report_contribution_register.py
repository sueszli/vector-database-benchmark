from datetime import datetime
from dateutil.relativedelta import relativedelta
from odoo import api, fields, models

class ContributionRegisterReport(models.AbstractModel):
    _name = 'report.hr_payroll.report_contributionregister'

    def _get_payslip_lines(self, register_ids, date_from, date_to):
        if False:
            print('Hello World!')
        result = {}
        self.env.cr.execute("\n            SELECT pl.id from hr_payslip_line as pl\n            LEFT JOIN hr_payslip AS hp on (pl.slip_id = hp.id)\n            WHERE (hp.date_from >= %s) AND (hp.date_to <= %s)\n            AND pl.register_id in %s\n            AND hp.state = 'done'\n            ORDER BY pl.slip_id, pl.sequence", (date_from, date_to, tuple(register_ids)))
        line_ids = [x[0] for x in self.env.cr.fetchall()]
        for line in self.env['hr.payslip.line'].browse(line_ids):
            result.setdefault(line.register_id.id, self.env['hr.payslip.line'])
            result[line.register_id.id] += line
        return result

    @api.model
    def render_html(self, docids, data=None):
        if False:
            while True:
                i = 10
        register_ids = self.env.context.get('active_ids', [])
        contrib_registers = self.env['hr.contribution.register'].browse(register_ids)
        date_from = data['form'].get('date_from', fields.Date.today())
        date_to = data['form'].get('date_to', str(datetime.now() + relativedelta(months=+1, day=1, days=-1))[:10])
        lines_data = self._get_payslip_lines(register_ids, date_from, date_to)
        lines_total = {}
        for register in contrib_registers:
            lines = lines_data.get(register.id)
            lines_total[register.id] = lines and sum(lines.mapped('total')) or 0.0
        docargs = {'doc_ids': register_ids, 'doc_model': 'hr.contribution.register', 'docs': contrib_registers, 'data': data, 'lines_data': lines_data, 'lines_total': lines_total}
        return self.env['report'].render('hr_payroll.report_contributionregister', docargs)