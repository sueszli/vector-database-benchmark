from odoo.tests.common import TransactionCase
import time

class TestHrTimesheetSheet(TransactionCase):
    """Test for hr_timesheet_sheet.sheet"""

    def setUp(self):
        if False:
            return 10
        super(TestHrTimesheetSheet, self).setUp()
        self.attendance = self.env['hr.attendance']
        self.timesheet_sheet = self.env['hr_timesheet_sheet.sheet']
        self.test_employee = self.browse_ref('hr.employee_qdp')
        self.company = self.browse_ref('base.main_company')
        self.company.timesheet_max_difference = 1.0

    def test_hr_timesheet_sheet(self):
        if False:
            while True:
                i = 10
        self.test_timesheet_sheet = self.timesheet_sheet.create({'date_from': time.strftime('%Y-%m-11'), 'date_to': time.strftime('%Y-%m-17'), 'name': 'Gilles Gravie', 'state': 'new', 'user_id': self.browse_ref('base.user_demo').id, 'employee_id': self.test_employee.id})
        self.attendance.create({'employee_id': self.test_employee.id, 'check_in': time.strftime('%Y-%m-11 09:12:37'), 'check_out': time.strftime('%Y-%m-11 17:30:00')})
        self.test_timesheet_sheet.write({'timesheet_ids': [(0, 0, {'project_id': self.browse_ref('project.project_project_2').id, 'date': time.strftime('%Y-%m-11'), 'name': 'Develop yaml for hr module(1)', 'user_id': self.browse_ref('base.user_demo').id, 'unit_amount': 6.0, 'amount': -90.0, 'product_id': self.browse_ref('product.product_product_1').id})]})
        try:
            self.test_timesheet_sheet.action_timesheet_confirm()
        except Exception:
            pass
        self.test_timesheet_sheet.write({'timesheet_ids': [(0, 0, {'project_id': self.browse_ref('project.project_project_2').id, 'date': time.strftime('%Y-%m-11'), 'name': 'Develop yaml for hr module(2)', 'user_id': self.browse_ref('base.user_demo').id, 'unit_amount': 2.0, 'amount': -90.0, 'product_id': self.browse_ref('product.product_product_1').id})]})
        self.test_timesheet_sheet.invalidate_cache(['total_attendance', 'total_timesheet', 'total_difference'])
        self.test_timesheet_sheet.action_timesheet_confirm()
        assert self.test_timesheet_sheet.state == 'confirm'
        self.test_timesheet_sheet.write({'state': 'done'})
        assert self.test_timesheet_sheet.state == 'done'