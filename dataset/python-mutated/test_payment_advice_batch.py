from odoo.addons.l10n_in_hr_payroll.tests.common import TestPaymentAdviceBase

class TestPaymentAdviceBatch(TestPaymentAdviceBase):

    def test_00_payment_advice_batch_flow(self):
        if False:
            while True:
                i = 10
        payslip_run = self.PayslipRun.create({'name': 'Payslip Batch'})
        payslip_employee = self.PayslipEmployee.create({'employee_ids': [(4, self.rahul_emp.ids)]})
        payslip_employee.with_context(active_id=payslip_run.id).compute_sheet()
        self.assertEqual(payslip_run.state, 'draft')
        payslip_run.write({'state': 'close'})
        self.assertEqual(payslip_run.state, 'close')
        payslip_run.create_advice()
        advice_ids = self.Advice.search([('batch_id', '=', payslip_run.id)])
        self.assertTrue(bool(advice_ids), 'Advice is not created from Payslip Batch.')