import babel.dates
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from odoo.fields import Datetime
from odoo.tools import float_compare
from odoo.addons.resource.tests.common import TestResourceCommon
from odoo.tests import TransactionCase

class TestResource(TestResourceCommon):

    def test_00_intervals(self):
        if False:
            return 10
        intervals = [(Datetime.from_string('2013-02-04 09:00:00'), Datetime.from_string('2013-02-04 11:00:00')), (Datetime.from_string('2013-02-04 08:00:00'), Datetime.from_string('2013-02-04 12:00:00')), (Datetime.from_string('2013-02-04 11:00:00'), Datetime.from_string('2013-02-04 14:00:00')), (Datetime.from_string('2013-02-04 17:00:00'), Datetime.from_string('2013-02-04 21:00:00')), (Datetime.from_string('2013-02-03 08:00:00'), Datetime.from_string('2013-02-03 10:00:00')), (Datetime.from_string('2013-02-04 18:00:00'), Datetime.from_string('2013-02-04 19:00:00'))]
        cleaned_intervals = self.ResourceCalendar.interval_clean(intervals)
        self.assertEqual(len(cleaned_intervals), 3, 'resource_calendar: wrong interval cleaning')
        self.assertEqual(cleaned_intervals[0][0], Datetime.from_string('2013-02-03 08:00:00'), 'resource_calendar: wrong interval cleaning')
        self.assertEqual(cleaned_intervals[0][1], Datetime.from_string('2013-02-03 10:00:00'), 'resource_calendar: wrong interval cleaning')
        self.assertEqual(cleaned_intervals[1][0], Datetime.from_string('2013-02-04 08:00:00'), 'resource_calendar: wrong interval cleaning')
        self.assertEqual(cleaned_intervals[1][1], Datetime.from_string('2013-02-04 14:00:00'), 'resource_calendar: wrong interval cleaning')
        self.assertEqual(cleaned_intervals[2][0], Datetime.from_string('2013-02-04 17:00:00'), 'resource_calendar: wrong interval cleaning')
        self.assertEqual(cleaned_intervals[2][1], Datetime.from_string('2013-02-04 21:00:00'), 'resource_calendar: wrong interval cleaning')
        working_interval = (Datetime.from_string('2013-02-04 08:00:00'), Datetime.from_string('2013-02-04 18:00:00'))
        result = self.ResourceCalendar.interval_remove_leaves(working_interval, intervals)
        self.assertEqual(len(result), 1, 'resource_calendar: wrong leave removal from interval')
        self.assertEqual(result[0][0], Datetime.from_string('2013-02-04 14:00:00'), 'resource_calendar: wrong leave removal from interval')
        self.assertEqual(result[0][1], Datetime.from_string('2013-02-04 17:00:00'), 'resource_calendar: wrong leave removal from interval')
        result = self.ResourceCalendar.interval_schedule_hours(cleaned_intervals, 5.5)
        self.assertEqual(len(result), 2, 'resource_calendar: wrong hours scheduling in interval')
        self.assertEqual(result[0][0], Datetime.from_string('2013-02-03 08:00:00'), 'resource_calendar: wrong leave removal from interval')
        self.assertEqual(result[0][1], Datetime.from_string('2013-02-03 10:00:00'), 'resource_calendar: wrong leave removal from interval')
        self.assertEqual(result[1][0], Datetime.from_string('2013-02-04 08:00:00'), 'resource_calendar: wrong leave removal from interval')
        self.assertEqual(result[1][1], Datetime.from_string('2013-02-04 11:30:00'), 'resource_calendar: wrong leave removal from interval')
        cleaned_intervals.reverse()
        result = self.ResourceCalendar.interval_schedule_hours(cleaned_intervals, 5.5, remove_at_end=False)
        self.assertEqual(len(result), 2, 'resource_calendar: wrong hours scheduling in interval')
        self.assertEqual(result[0][0], Datetime.from_string('2013-02-04 17:00:00'), 'resource_calendar: wrong leave removal from interval')
        self.assertEqual(result[0][1], Datetime.from_string('2013-02-04 21:00:00'), 'resource_calendar: wrong leave removal from interval')
        self.assertEqual(result[1][0], Datetime.from_string('2013-02-04 12:30:00'), 'resource_calendar: wrong leave removal from interval')
        self.assertEqual(result[1][1], Datetime.from_string('2013-02-04 14:00:00'), 'resource_calendar: wrong leave removal from interval')

    def test_10_calendar_basics(self):
        if False:
            for i in range(10):
                print('nop')
        ' Testing basic method of resource.calendar '
        date = self.calendar.get_next_day(day_date=self.date1.date())
        self.assertEqual(date, self.date2.date(), 'resource_calendar: wrong next day computing')
        date = self.calendar.get_next_day(day_date=self.date2.date())
        self.assertEqual(date, self.date1.date() + relativedelta(days=7), 'resource_calendar: wrong next day computing')
        date = self.calendar.get_next_day(day_date=self.date2.date() + relativedelta(days=1))
        self.assertEqual(date, self.date1.date() + relativedelta(days=7), 'resource_calendar: wrong next day computing')
        date = self.calendar.get_next_day(day_date=self.date1.date() + relativedelta(days=-1))
        self.assertEqual(date, self.date1.date(), 'resource_calendar: wrong next day computing')
        date = self.calendar.get_previous_day(day_date=self.date1.date())
        self.assertEqual(date, self.date2.date() + relativedelta(days=-7), 'resource_calendar: wrong previous day computing')
        date = self.calendar.get_previous_day(day_date=self.date2.date())
        self.assertEqual(date, self.date1.date(), 'resource_calendar: wrong previous day computing')
        date = self.calendar.get_previous_day(day_date=self.date2.date() + relativedelta(days=1))
        self.assertEqual(date, self.date2.date(), 'resource_calendar: wrong previous day computing')
        date = self.calendar.get_previous_day(day_date=self.date1.date() + relativedelta(days=-1))
        self.assertEqual(date, self.date2.date() + relativedelta(days=-7), 'resource_calendar: wrong previous day computing')
        weekdays = self.calendar.get_weekdays()
        self.assertEqual(weekdays, [1, 4], 'resource_calendar: wrong weekdays computing')

    def test_20_calendar_working_intervals(self):
        if False:
            return 10
        ' Testing working intervals computing method of resource.calendar '
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date1)
        self.assertEqual(len(intervals), 1, 'resource_calendar: wrong working intervals')
        self.assertEqual(intervals[0][0], Datetime.from_string('2013-02-12 09:08:07'), 'resource_calendar: wrong working intervals')
        self.assertEqual(intervals[0][1], Datetime.from_string('2013-02-12 16:00:00'), 'resource_calendar: wrong working intervals')
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date2)
        self.assertEqual(len(intervals), 2, 'resource_calendar: wrong working intervals')
        self.assertEqual(intervals[0][0], Datetime.from_string('2013-02-15 10:11:12'), 'resource_calendar: wrong working intervals')
        self.assertEqual(intervals[0][1], Datetime.from_string('2013-02-15 13:00:00'), 'resource_calendar: wrong working intervals')
        self.assertEqual(intervals[1][0], Datetime.from_string('2013-02-15 16:00:00'), 'resource_calendar: wrong working intervals')
        self.assertEqual(intervals[1][1], Datetime.from_string('2013-02-15 23:00:00'), 'resource_calendar: wrong working intervals')
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date1.replace(hour=0), compute_leaves=True)
        self.assertEqual(len(intervals), 1, 'resource_calendar: wrong working intervals')
        self.assertEqual(intervals[0][0], Datetime.from_string('2013-02-12 08:00:00'), 'resource_calendar: wrong working intervals')
        self.assertEqual(intervals[0][1], Datetime.from_string('2013-02-12 16:00:00'), 'resource_calendar: wrong working intervals')
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date1.replace(hour=8) + relativedelta(days=7), end_dt=self.date1.replace(hour=15, minute=45, second=30) + relativedelta(days=7), compute_leaves=True)
        self.assertEqual(len(intervals), 2, 'resource_calendar: wrong working intervals')
        self.assertEqual(intervals[0][0], Datetime.from_string('2013-02-19 08:08:07'), 'resource_calendar: wrong working intervals')
        self.assertEqual(intervals[0][1], Datetime.from_string('2013-02-19 09:00:00'), 'resource_calendar: wrong working intervals')
        self.assertEqual(intervals[1][0], Datetime.from_string('2013-02-19 12:00:00'), 'resource_calendar: wrong working intervals')
        self.assertEqual(intervals[1][1], Datetime.from_string('2013-02-19 15:45:30'), 'resource_calendar: wrong working intervals')

    def test_21_calendar_working_intervals_limited_attendances(self):
        if False:
            for i in range(10):
                print('nop')
        ' Test attendances limited in time. '
        self.env['resource.calendar.attendance'].browse(self.att3_id).write({'date_from': self.date2 + relativedelta(days=7), 'date_to': False})
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date2)
        self.assertEqual(intervals, [(Datetime.from_string('2013-02-15 10:11:12'), Datetime.from_string('2013-02-15 13:00:00'))])
        self.env['resource.calendar.attendance'].browse(self.att3_id).write({'date_from': False, 'date_to': self.date2 - relativedelta(days=7)})
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date2)
        self.assertEqual(intervals, [(Datetime.from_string('2013-02-15 10:11:12'), Datetime.from_string('2013-02-15 13:00:00'))])
        self.env['resource.calendar.attendance'].browse(self.att3_id).write({'date_from': self.date2 + relativedelta(days=7), 'date_to': self.date2 - relativedelta(days=7)})
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date2)
        self.assertEqual(intervals, [(Datetime.from_string('2013-02-15 10:11:12'), Datetime.from_string('2013-02-15 13:00:00'))])
        self.env['resource.calendar.attendance'].browse(self.att3_id).write({'date_from': self.date2, 'date_to': self.date2})
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date2)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0], (Datetime.from_string('2013-02-15 10:11:12'), Datetime.from_string('2013-02-15 13:00:00')))
        self.assertEqual(intervals[1], (Datetime.from_string('2013-02-15 16:00:00'), Datetime.from_string('2013-02-15 23:00:00')))

    def test_30_calendar_working_days(self):
        if False:
            while True:
                i = 10
        ' Testing calendar hours computation on a working day '
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date1.replace(hour=10, minute=30, second=0))
        self.assertEqual(len(intervals), 1, 'resource_calendar: wrong working interval / day computing')
        self.assertEqual(intervals[0][0], Datetime.from_string('2013-02-12 10:30:00'), 'resource_calendar: wrong working interval / day computing')
        self.assertEqual(intervals[0][1], Datetime.from_string('2013-02-12 16:00:00'), 'resource_calendar: wrong working interval / day computing')
        wh = self.calendar.get_working_hours_of_date(start_dt=self.date1.replace(hour=10, minute=30, second=0))
        self.assertEqual(wh, 5.5, 'resource_calendar: wrong working interval / day time computing')
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date1.replace(hour=7, minute=0, second=0) + relativedelta(days=7))
        self.assertEqual(len(intervals), 1, 'resource_calendar: wrong working interval/day computing')
        self.assertEqual(intervals[0][0], Datetime.from_string('2013-02-19 08:00:00'), 'resource_calendar: wrong working interval / day computing')
        self.assertEqual(intervals[0][1], Datetime.from_string('2013-02-19 16:00:00'), 'resource_calendar: wrong working interval / day computing')
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date1.replace(hour=7, minute=0, second=0) + relativedelta(days=7), compute_leaves=True)
        self.assertEqual(len(intervals), 2, 'resource_calendar: wrong working interval/day computing')
        self.assertEqual(intervals[0][0], Datetime.from_string('2013-02-19 08:00:00'), 'resource_calendar: wrong working interval / day computing')
        self.assertEqual(intervals[0][1], Datetime.from_string('2013-02-19 09:00:00'), 'resource_calendar: wrong working interval / day computing')
        self.assertEqual(intervals[1][0], Datetime.from_string('2013-02-19 12:00:00'), 'resource_calendar: wrong working interval / day computing')
        self.assertEqual(intervals[1][1], Datetime.from_string('2013-02-19 16:00:00'), 'resource_calendar: wrong working interval / day computing')
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date1.replace(hour=7, minute=0, second=0) + relativedelta(days=14), compute_leaves=True)
        self.assertEqual(len(intervals), 1, 'resource_calendar: wrong working interval/day computing')
        self.assertEqual(intervals[0][0], Datetime.from_string('2013-02-26 08:00:00'), 'resource_calendar: wrong working interval / day computing')
        self.assertEqual(intervals[0][1], Datetime.from_string('2013-02-26 16:00:00'), 'resource_calendar: wrong working interval / day computing')
        intervals = self.calendar.get_working_intervals_of_day(start_dt=self.date1.replace(hour=7, minute=0, second=0) + relativedelta(days=14), compute_leaves=True, resource_id=self.resource1_id)
        self.assertEqual(len(intervals), 0, 'resource_calendar: wrong working interval/day computing')

    def test_40_calendar_hours_scheduling(self):
        if False:
            i = 10
            return i + 15
        ' Testing calendar hours scheduling '
        res = self.calendar.schedule_hours(-40, day_dt=self.date1.replace(minute=0, second=0))
        self.assertEqual(res[-1][0], Datetime.from_string('2013-02-12 08:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-1][1], Datetime.from_string('2013-02-12 09:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-2][0], Datetime.from_string('2013-02-08 16:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-2][1], Datetime.from_string('2013-02-08 23:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-3][0], Datetime.from_string('2013-02-08 08:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-3][1], Datetime.from_string('2013-02-08 13:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-4][0], Datetime.from_string('2013-02-05 08:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-4][1], Datetime.from_string('2013-02-05 16:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-5][0], Datetime.from_string('2013-02-01 16:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-5][1], Datetime.from_string('2013-02-01 23:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-6][0], Datetime.from_string('2013-02-01 08:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-6][1], Datetime.from_string('2013-02-01 13:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-7][0], Datetime.from_string('2013-01-29 09:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[-7][1], Datetime.from_string('2013-01-29 16:00:00'), 'resource_calendar: wrong hours scheduling')
        td = timedelta()
        for item in res:
            td += item[1] - item[0]
        self.assertEqual(seconds(td) / 3600.0, 40.0, 'resource_calendar: wrong hours scheduling')
        res = self.calendar.schedule_hours_get_date(-40, day_dt=self.date1.replace(minute=0, second=0))
        self.assertEqual(res, Datetime.from_string('2013-01-29 09:00:00'))
        res = self.calendar.schedule_hours(40, day_dt=self.date1.replace(minute=0, second=0))
        self.assertEqual(res[0][0], Datetime.from_string('2013-02-12 09:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[0][1], Datetime.from_string('2013-02-12 16:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[1][0], Datetime.from_string('2013-02-15 08:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[1][1], Datetime.from_string('2013-02-15 13:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[2][0], Datetime.from_string('2013-02-15 16:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[2][1], Datetime.from_string('2013-02-15 23:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[3][0], Datetime.from_string('2013-02-19 08:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[3][1], Datetime.from_string('2013-02-19 16:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[4][0], Datetime.from_string('2013-02-22 08:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[4][1], Datetime.from_string('2013-02-22 13:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[5][0], Datetime.from_string('2013-02-22 16:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[5][1], Datetime.from_string('2013-02-22 23:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[6][0], Datetime.from_string('2013-02-26 08:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[6][1], Datetime.from_string('2013-02-26 09:00:00'), 'resource_calendar: wrong hours scheduling')
        td = timedelta()
        for item in res:
            td += item[1] - item[0]
        self.assertEqual(seconds(td) / 3600.0, 40.0, 'resource_calendar: wrong hours scheduling')
        res = self.calendar.schedule_hours_get_date(40, day_dt=self.date1.replace(minute=0, second=0))
        self.assertEqual(res, Datetime.from_string('2013-02-26 09:00:00'))
        res = self.calendar.schedule_hours(40, day_dt=self.date1.replace(minute=0, second=0), compute_leaves=True, resource_id=self.resource1_id)
        self.assertEqual(res[0][0], Datetime.from_string('2013-02-12 09:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[0][1], Datetime.from_string('2013-02-12 16:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[1][0], Datetime.from_string('2013-02-15 08:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[1][1], Datetime.from_string('2013-02-15 13:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[2][0], Datetime.from_string('2013-02-15 16:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[2][1], Datetime.from_string('2013-02-15 23:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[3][0], Datetime.from_string('2013-02-19 08:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[3][1], Datetime.from_string('2013-02-19 09:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[4][0], Datetime.from_string('2013-02-19 12:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[4][1], Datetime.from_string('2013-02-19 16:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[5][0], Datetime.from_string('2013-02-22 08:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[5][1], Datetime.from_string('2013-02-22 09:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[6][0], Datetime.from_string('2013-02-22 16:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[6][1], Datetime.from_string('2013-02-22 23:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[7][0], Datetime.from_string('2013-03-01 11:30:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[7][1], Datetime.from_string('2013-03-01 13:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[8][0], Datetime.from_string('2013-03-01 16:00:00'), 'resource_calendar: wrong hours scheduling')
        self.assertEqual(res[8][1], Datetime.from_string('2013-03-01 22:30:00'), 'resource_calendar: wrong hours scheduling')
        td = timedelta()
        for item in res:
            td += item[1] - item[0]
        self.assertEqual(seconds(td) / 3600.0, 40.0, 'resource_calendar: wrong hours scheduling')
        res = self.calendar._interval_hours_get(self.date1.replace(hour=6, minute=0), self.date2.replace(hour=23, minute=0) + relativedelta(days=7), resource_id=self.resource1_id, exclude_leaves=True)
        self.assertEqual(res, 40.0, 'resource_calendar: wrong _interval_hours_get compatibility computation')
        res = self.calendar.get_working_hours(self.date1.replace(hour=6, minute=0), self.date2.replace(hour=23, minute=0) + relativedelta(days=7), compute_leaves=False, resource_id=self.resource1_id)
        self.assertEqual(res, 40.0, 'resource_calendar: wrong get_working_hours computation')
        res = self.calendar._interval_hours_get(self.date1.replace(hour=6, minute=0), self.date2.replace(hour=23, minute=0) + relativedelta(days=7), resource_id=self.resource1_id, exclude_leaves=False)
        self.assertEqual(res, 33.0, 'resource_calendar: wrong _interval_hours_get compatibility computation')
        res = self.calendar.get_working_hours(self.date1.replace(hour=6, minute=0), self.date2.replace(hour=23, minute=0) + relativedelta(days=7), compute_leaves=True, resource_id=self.resource1_id)
        self.assertEqual(res, 33.0, 'resource_calendar: wrong get_working_hours computation')
        res = self.ResourceCalendar.with_context(self.context).get_working_hours(self.date1.replace(hour=6, minute=0), self.date2.replace(hour=23, minute=0), compute_leaves=True, resource_id=self.resource1_id, default_interval=(8, 16))
        self.assertEqual(res, 32.0, 'resource_calendar: wrong get_working_hours computation')
        self.att0_0_id = self.ResourceAttendance.with_context(self.context).create({'name': 'Att0', 'dayofweek': '0', 'hour_from': 7.5, 'hour_to': 12.5, 'calendar_id': self.calendar.id})
        self.att0_1_id = self.ResourceAttendance.with_context(self.context).create({'name': 'Att0', 'dayofweek': '0', 'hour_from': 13, 'hour_to': 14, 'calendar_id': self.calendar.id})
        date1 = Datetime.from_string('2013-02-11 07:30:00')
        date2 = Datetime.from_string('2013-02-11 14:00:00')
        res = self.calendar.get_working_hours(date1, date2, compute_leaves=False, resource_id=self.resource1_id)
        self.assertEqual(res, 6, 'resource_calendar: wrong get_working_hours computation')

    def test_45_calendar_hours_scheduling_minutes(self):
        if False:
            return 10
        ' Testing minutes computation in calendar hours scheduling '
        res = self.calendar.schedule_hours_get_date(-39, day_dt=self.date1.replace(minute=25, second=20))
        self.assertEqual(res, Datetime.from_string('2013-01-29 10:25:20'))

    def test_50_calendar_schedule_days(self):
        if False:
            i = 10
            return i + 15
        ' Testing calendar days scheduling '
        res = self.calendar.schedule_days_get_date(5, day_date=self.date1)
        self.assertEqual(res.date(), Datetime.from_string('2013-02-26 00:00:00').date(), 'resource_calendar: wrong days scheduling')
        res = self.calendar.schedule_days_get_date(-2, day_date=self.date1)
        self.assertEqual(res.date(), Datetime.from_string('2013-02-08 00:00:00').date(), 'resource_calendar: wrong days scheduling')
        res = self.calendar.schedule_days_get_date(5, day_date=self.date1, compute_leaves=True, resource_id=self.resource1_id)
        self.assertEqual(res.date(), Datetime.from_string('2013-03-01 00:00:00').date(), 'resource_calendar: wrong days scheduling')
        res = self.ResourceCalendar.with_context(self.context).schedule_days_get_date(5, day_date=self.date1, default_interval=(8, 16))
        self.assertEqual(res, Datetime.from_string('2013-02-16 16:00:00'), 'resource_calendar: wrong days scheduling')

    def test_60_project(self):
        if False:
            while True:
                i = 10
        resources = self.env.ref('resource.resource_analyst') + self.env.ref('resource.resource_designer') + self.env.ref('resource.resource_developer')
        resources.write({'calendar_id': self.ref('resource.timesheet_group1'), 'resource_type': 'user'})
        now = datetime.now()
        dt = now - timedelta(days=now.weekday())
        for resource in resources:
            result = resource.calendar_id.working_hours_on_day(dt)
            self.assertEqual(float_compare(result, 8.0, precision_digits=2), 0, 'Wrong calculation of day work hour availability of the Resource (found %d).' % result)
        now = datetime.now()
        dt = now - timedelta(days=now.weekday()) + timedelta(days=3)
        vals = {'resource_id': self.ref('resource.resource_developer'), 'calendar_id': self.ref('resource.timesheet_group1'), 'date_from': dt.strftime('%Y-%m-%d 09:00:00'), 'date_to': dt.strftime('%Y-%m-%d 18:00:00')}
        self.env.ref('resource.resource_dummyleave').write(vals)
        now = datetime.now()
        dt_from = now - relativedelta(days=now.weekday(), hour=8, minute=30)
        dt_to = dt_from + relativedelta(days=6, hour=17)
        hours = self.env.ref('resource.timesheet_group1').interval_hours_get(dt_from, dt_to, resource=self.ref('resource.resource_developer'))
        self.assertGreater(hours, 27, 'Invalid Total Week working hour calculated, got %r, expected > 27' % hours)
        now = datetime.now()
        work_intreval = self.env.ref('resource.timesheet_group1').interval_min_get(now, 20.0, resource=self.ref('resource.resource_designer'))
        self.assertGreaterEqual(len(work_intreval), 5, 'Wrong Schedule Calculated')

    def test_70_duplicate_resource(self):
        if False:
            return 10
        resource_id = self.env.ref('resource.resource_analyst').copy()
        self.assertTrue(resource_id, 'Unable to Duplicate Resource')

    def test_80_resource_schedule_tz(self):
        if False:
            while True:
                i = 10
        tz_context = dict(tz='Australia/Sydney')
        self.env.user.with_context(tz_context).write({'tz': 'Australia/Sydney'})
        calendar = self.calendar.with_context(tz_context)
        self.env['resource.calendar.attendance'].create({'name': 'Day3 - 1', 'dayofweek': '3', 'hour_from': 8, 'hour_to': 12, 'calendar_id': calendar.id})
        self.env['resource.calendar.attendance'].create({'name': 'Day3 - 2', 'dayofweek': '3', 'hour_from': 13, 'hour_to': 17, 'calendar_id': calendar.id})
        hours = 1.0 / 60.0
        start_dt = Datetime.from_string('2013-02-14 21:00:00')
        res = calendar.schedule_hours(hours, start_dt)
        self.assertEqual([(start_dt, start_dt.replace(minute=1))], res, 'resource_calendar: wrong schedule_hours computation')
        start_dt = Datetime.from_string('2013-02-15 00:00:00')
        res = calendar.schedule_hours(hours, start_dt)
        self.assertEqual([(start_dt, start_dt.replace(minute=1))], res, 'resource_calendar: wrong schedule_hours computation')
WAR_START = date(1932, 11, 2)
WAR_END = date(1932, 12, 10)

class TestWorkDays(TransactionCase):

    def _make_attendance(self, weekday, **kw):
        if False:
            print('Hello World!')
        data = {'name': babel.dates.get_day_names()[weekday], 'dayofweek': str(weekday), 'hour_from': 9, 'hour_to': 17}
        data.update(kw)
        return data

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestWorkDays, self).setUp()
        self._calendar = self.env['resource.calendar'].create({'name': 'Trivial Calendar', 'attendance_ids': [(0, 0, self._make_attendance(i)) for i in range(5)]})
        self._days = [date.fromordinal(o) for o in xrange(WAR_START.toordinal(), WAR_END.toordinal() + 1)]

    def test_no_calendar(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If a resource has no resource calendar, they don't work\n        "
        r = self.env['resource.resource'].create({'name': 'NoCalendar'})
        self.assertEqual([], list(r._iter_work_days(WAR_START, WAR_END)))

    def test_trivial_calendar_no_leaves(self):
        if False:
            for i in range(10):
                print('nop')
        ' If leaves are not involved, only calendar attendances (basic\n        company configuration) are taken in account\n        '
        r = self.env['resource.resource'].create({'name': 'Trivial Calendar', 'calendar_id': self._calendar.id})
        self.assertEqual([d for d in self._days if d.weekday() not in (5, 6)], list(r._iter_work_days(WAR_START, WAR_END)))

    def test_global_leaves(self):
        if False:
            for i in range(10):
                print('nop')
        self.env['resource.calendar.leaves'].create({'calendar_id': self._calendar.id, 'date_from': '1932-11-09 00:00:00', 'date_to': '1932-11-12 23:59:59'})
        r1 = self.env['resource.resource'].create({'name': 'Resource 1', 'calendar_id': self._calendar.id})
        r2 = self.env['resource.resource'].create({'name': 'Resource 2', 'calendar_id': self._calendar.id})
        days = [d for d in self._days if d.weekday() not in (5, 6) if d < date(1932, 11, 9) or d > date(1932, 11, 12)]
        self.assertEqual(days, list(r1._iter_work_days(WAR_START, WAR_END)))
        self.assertEqual(days, list(r2._iter_work_days(WAR_START, WAR_END)))

    def test_personal_leaves(self):
        if False:
            print('Hello World!')
        ' Leaves with a resource_id apply only to that resource\n        '
        r1 = self.env['resource.resource'].create({'name': 'Resource 1', 'calendar_id': self._calendar.id})
        r2 = self.env['resource.resource'].create({'name': 'Resource 2', 'calendar_id': self._calendar.id})
        self.env['resource.calendar.leaves'].create({'calendar_id': self._calendar.id, 'date_from': '1932-11-09 00:00:00', 'date_to': '1932-11-12 23:59:59', 'resource_id': r2.id})
        weekdays = [d for d in self._days if d.weekday() not in (5, 6)]
        self.assertEqual(weekdays, list(r1._iter_work_days(WAR_START, WAR_END)))
        self.assertEqual([d for d in weekdays if d < date(1932, 11, 9) or d > date(1932, 11, 12)], list(r2._iter_work_days(WAR_START, WAR_END)))

    def test_mixed_leaves(self):
        if False:
            return 10
        r = self.env['resource.resource'].create({'name': 'Resource 1', 'calendar_id': self._calendar.id})
        self.env['resource.calendar.leaves'].create({'calendar_id': self._calendar.id, 'date_from': '1932-11-09 00:00:00', 'date_to': '1932-11-12 23:59:59'})
        self.env['resource.calendar.leaves'].create({'calendar_id': self._calendar.id, 'date_from': '1932-12-02 00:00:00', 'date_to': '1932-12-31 23:59:59', 'resource_id': r.id})
        self.assertEqual([d for d in self._days if d.weekday() not in (5, 6) if d < date(1932, 11, 9) or d > date(1932, 11, 12) if d < date(1932, 12, 2)], list(r._iter_work_days(WAR_START, WAR_END)))
        self.assertTrue(r._is_work_day(date(1932, 11, 8)))
        self.assertTrue(r._is_work_day(date(1932, 11, 14)))
        self.assertTrue(r._is_work_day(date(1932, 12, 1)))
        self.assertFalse(r._is_work_day(date(1932, 11, 11)))
        self.assertFalse(r._is_work_day(date(1932, 11, 13)))
        self.assertFalse(r._is_work_day(date(1932, 11, 19)))
        self.assertFalse(r._is_work_day(date(1932, 11, 20)))
        self.assertFalse(r._is_work_day(date(1932, 12, 6)))

def seconds(td):
    if False:
        while True:
            i = 10
    assert isinstance(td, timedelta)
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10 ** 6) / 10.0 ** 6