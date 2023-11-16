from odoo.tests.common import TransactionCase
import time

class TestHrAttendance(TransactionCase):
    """Tests for attendance date ranges validity"""

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestHrAttendance, self).setUp()
        self.attendance = self.env['hr.attendance']
        self.test_employee = self.browse_ref('hr.employee_qdp')

    def test_attendance_in_before_out(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(Exception):
            self.my_attend = self.attendance.create({'employee_id': self.test_employee.id, 'check_in': time.strftime('%Y-%m-10 12:00'), 'check_out': time.strftime('%Y-%m-10 11:00')})

    def test_attendance_no_check_out(self):
        if False:
            for i in range(10):
                print('nop')
        self.attendance.create({'employee_id': self.test_employee.id, 'check_in': time.strftime('%Y-%m-10 10:00')})
        with self.assertRaises(Exception):
            self.attendance.create({'employee_id': self.test_employee.id, 'check_in': time.strftime('%Y-%m-10 11:00')})

    def test_attendance_1(self):
        if False:
            print('Hello World!')
        with self.assertRaises(Exception):
            self.attendance.create({'employee_id': self.test_employee.id, 'check_in': time.strftime('%Y-%m-10 08:30'), 'check_out': time.strftime('%Y-%m-10 09:30')})

    def test_attendance_2(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(Exception):
            self.attendance.create({'employee_id': self.test_employee.id, 'check_in': time.strftime('%Y-%m-10 07:30'), 'check_out': time.strftime('%Y-%m-10 08:30')})

    def test_attendance_3(self):
        if False:
            return 10
        with self.assertRaises(Exception):
            self.attendance.create({'employee_id': self.test_employee.id, 'check_in': time.strftime('%Y-%m-10 07:30'), 'check_out': time.strftime('%Y-%m-10 09:30')})

    def test_attendance_4(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(Exception):
            self.attendance.create({'employee_id': self.test_employee.id, 'check_in': time.strftime('%Y-%m-10 08:15'), 'check_out': time.strftime('%Y-%m-10 08:45')})

    def test_attendance_5(self):
        if False:
            i = 10
            return i + 15
        self.attendance.create({'employee_id': self.test_employee.id, 'check_in': time.strftime('%Y-%m-10 10:00')})
        with self.assertRaises(Exception):
            self.attendance.create({'employee_id': self.test_employee.id, 'check_in': time.strftime('%Y-%m-10 09:30'), 'check_out': time.strftime('%Y-%m-10 10:30')})

    def test_new_attendances(self):
        if False:
            i = 10
            return i + 15
        self.attendance.create({'employee_id': self.test_employee.id, 'check_in': time.strftime('%Y-%m-10 11:00'), 'check_out': time.strftime('%Y-%m-10 12:00')})
        open_attendance = self.attendance.create({'employee_id': self.test_employee.id, 'check_in': time.strftime('%Y-%m-10 10:00')})
        with self.assertRaises(Exception):
            open_attendance.write({'check_out': time.strftime('%Y-%m-10 11:30')})