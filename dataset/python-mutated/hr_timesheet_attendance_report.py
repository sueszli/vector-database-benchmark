from odoo import api, fields, models

class TimesheetAttendance(models.Model):
    _name = 'hr.timesheet.attendance.report'
    _auto = False
    user_id = fields.Many2one('res.users')
    date = fields.Date()
    total_timesheet = fields.Float()
    total_attendance = fields.Float()
    total_difference = fields.Float()

    @api.model_cr
    def init(self):
        if False:
            for i in range(10):
                print('nop')
        self._cr.execute("CREATE OR REPLACE VIEW %s AS (\n            SELECT\n                max(id) AS id,\n                t.user_id,\n                t.date,\n                coalesce(sum(t.attendance), 0) AS total_attendance,\n                coalesce(sum(t.timesheet), 0) AS total_timesheet,\n                coalesce(sum(t.attendance), 0) - coalesce(sum(t.timesheet), 0) as total_difference\n            FROM (\n                SELECT\n                    -hr_attendance.id AS id,\n                    resource_resource.user_id AS user_id,\n                    hr_attendance.worked_hours AS attendance,\n                    NULL AS timesheet,\n                    date_trunc('day', hr_attendance.check_in) AS date\n                FROM hr_attendance\n                LEFT JOIN hr_employee ON hr_employee.id = hr_attendance.employee_id\n                LEFT JOIN resource_resource on resource_resource.id = hr_employee.resource_id\n            UNION ALL\n                SELECT\n                    ts.id AS id,\n                    ts.user_id AS user_id,\n                    NULL AS attendance,\n                    ts.unit_amount AS timesheet,\n                    date_trunc('day', ts.date) AS date\n                FROM account_analytic_line AS ts\n            ) AS t\n            GROUP BY t.user_id, t.date\n            ORDER BY t.date\n        )\n        " % self._table)