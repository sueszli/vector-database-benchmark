import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from odoo.report import render_report
from odoo.tools import config, test_reports
from odoo.addons.hr_payroll.tests.common import TestPayslipBase

class TestPayslipFlow(TestPayslipBase):

    def test_00_payslip_flow(self):
        if False:
            print('Hello World!')
        ' Testing payslip flow and report printing '
        richard_payslip = self.env['hr.payslip'].create({'name': 'Payslip of Richard', 'employee_id': self.richard_emp.id})
        payslip_input = self.env['hr.payslip.input'].search([('payslip_id', '=', richard_payslip.id)])
        payslip_input.write({'amount': 5.0})
        self.assertEqual(richard_payslip.state, 'draft', 'State not changed!')
        context = {'lang': 'en_US', 'tz': False, 'active_model': 'ir.ui.menu', 'department_id': False, 'section_id': False, 'active_ids': [self.ref('hr_payroll.menu_department_tree')], 'active_id': self.ref('hr_payroll.menu_department_tree')}
        richard_payslip.with_context(context).compute_sheet()
        richard_payslip.action_payslip_done()
        self.assertEqual(richard_payslip.state, 'done', 'State not changed!')
        richard_payslip.refund_sheet()
        payslip_refund = self.env['hr.payslip'].search([('name', 'like', 'Refund: ' + richard_payslip.name), ('credit_note', '=', True)])
        self.assertTrue(bool(payslip_refund), 'Payslip not refunded!')
        payslip_run = self.env['hr.payslip.run'].create({'date_end': '2011-09-30', 'date_start': '2011-09-01', 'name': 'Payslip for Employee'})
        payslip_employee = self.env['hr.payslip.employees'].create({'employee_ids': [(4, self.richard_emp.ids)]})
        payslip_employee.with_context(active_id=payslip_run.id).compute_sheet()
        self.env['payslip.lines.contribution.register'].create({'date_from': '2011-09-30', 'date_to': '2011-09-01'})
        (data, format) = render_report(self.env.cr, self.env.uid, richard_payslip.ids, 'hr_payroll.report_payslip', {}, {})
        if config.get('test_report_directory'):
            file(os.path.join(config['test_report_directory'], 'hr_payroll-payslip.' + format), 'wb+').write(data)
        (data, format) = render_report(self.env.cr, self.env.uid, richard_payslip.ids, 'hr_payroll.report_payslipdetails', {}, {})
        if config.get('test_report_directory'):
            file(os.path.join(config['test_report_directory'], 'hr_payroll-payslipdetails.' + format), 'wb+').write(data)
        context = {'model': 'hr.contribution.register', 'active_ids': [self.ref('hr_payroll.hr_houserent_register')]}
        test_reports.try_report_action(self.env.cr, self.env.uid, 'action_payslip_lines_contribution_register', context=context, our_module='hr_payroll')