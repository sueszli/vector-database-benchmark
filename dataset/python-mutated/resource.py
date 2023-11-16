import datetime
import pytz
from datetime import timedelta
from dateutil import rrule
from dateutil.relativedelta import relativedelta
from operator import itemgetter
from odoo import api, fields, models, _
from odoo.exceptions import ValidationError
from odoo.tools.float_utils import float_compare

class ResourceCalendar(models.Model):
    """ Calendar model for a resource. It has

     - attendance_ids: list of resource.calendar.attendance that are a working
                       interval in a given weekday.
     - leave_ids: list of leaves linked to this calendar. A leave can be general
                  or linked to a specific resource, depending on its resource_id.

    All methods in this class use intervals. An interval is a tuple holding
    (begin_datetime, end_datetime). A list of intervals is therefore a list of
    tuples, holding several intervals of work or leaves. """
    _name = 'resource.calendar'
    _description = 'Resource Calendar'
    name = fields.Char(required=True)
    company_id = fields.Many2one('res.company', string='Company', default=lambda self: self.env['res.company']._company_default_get())
    attendance_ids = fields.One2many('resource.calendar.attendance', 'calendar_id', string='Working Time', copy=True)
    manager = fields.Many2one('res.users', string='Workgroup Manager', default=lambda self: self.env.uid)
    leave_ids = fields.One2many('resource.calendar.leaves', 'calendar_id', string='Leaves')

    def interval_clean(self, intervals):
        if False:
            for i in range(10):
                print('nop')
        ' Utility method that sorts and removes overlapping inside datetime\n        intervals. The intervals are sorted based on increasing starting datetime.\n        Overlapping intervals are merged into a single one.\n\n        :param list intervals: list of intervals; each interval is a tuple\n                               (datetime_from, datetime_to)\n        :return list cleaned: list of sorted intervals without overlap '
        intervals = sorted(intervals, key=itemgetter(0))
        cleaned = []
        working_interval = None
        while intervals:
            current_interval = intervals.pop(0)
            if not working_interval:
                working_interval = [current_interval[0], current_interval[1]]
            elif working_interval[1] < current_interval[0]:
                cleaned.append(tuple(working_interval))
                working_interval = [current_interval[0], current_interval[1]]
            elif working_interval[1] < current_interval[1]:
                working_interval[1] = current_interval[1]
        if working_interval:
            cleaned.append(tuple(working_interval))
        return cleaned

    @api.model
    def interval_remove_leaves(self, interval, leave_intervals):
        if False:
            while True:
                i = 10
        ' Utility method that remove leave intervals from a base interval:\n\n         - clean the leave intervals, to have an ordered list of not-overlapping\n           intervals\n         - initiate the current interval to be the base interval\n         - for each leave interval:\n\n          - finishing before the current interval: skip, go to next\n          - beginning after the current interval: skip and get out of the loop\n            because we are outside range (leaves are ordered)\n          - beginning within the current interval: close the current interval\n            and begin a new current interval that begins at the end of the leave\n            interval\n          - ending within the current interval: update the current interval begin\n            to match the leave interval ending\n\n        :param tuple interval: a tuple (beginning datetime, ending datetime) that\n                               is the base interval from which the leave intervals\n                               will be removed\n        :param list leave_intervals: a list of tuples (beginning datetime, ending datetime)\n                                    that are intervals to remove from the base interval\n        :return list intervals: a list of tuples (begin datetime, end datetime)\n                                that are the remaining valid intervals '
        if not interval:
            return interval
        if leave_intervals is None:
            leave_intervals = []
        intervals = []
        leave_intervals = self.interval_clean(leave_intervals)
        current_interval = [interval[0], interval[1]]
        for leave in leave_intervals:
            if leave[1] <= current_interval[0]:
                continue
            if leave[0] >= current_interval[1]:
                break
            if current_interval[0] < leave[0] < current_interval[1]:
                current_interval[1] = leave[0]
                intervals.append((current_interval[0], current_interval[1]))
                current_interval = [leave[1], interval[1]]
            if current_interval[0] <= leave[1]:
                current_interval[0] = leave[1]
        if current_interval and current_interval[0] < interval[1]:
            intervals.append((current_interval[0], current_interval[1]))
        return intervals

    def interval_schedule_hours(self, intervals, hour, remove_at_end=True):
        if False:
            print('Hello World!')
        ' Schedule hours in intervals. The last matching interval is truncated\n        to match the specified hours.\n\n        It is possible to truncate the last interval at its beginning or ending.\n        However this does nothing on the given interval order that should be\n        submitted accordingly.\n\n        :param list intervals:  a list of tuples (beginning datetime, ending datetime)\n        :param int/float hours: number of hours to schedule. It will be converted\n                                into a timedelta, but should be submitted as an\n                                int or float.\n        :param boolean remove_at_end: remove extra hours at the end of the last\n                                      matching interval. Otherwise, do it at the\n                                      beginning.\n\n        :return list results: a list of intervals. If the number of hours to schedule\n        is greater than the possible scheduling in the intervals, no extra-scheduling\n        is done, and results == intervals. '
        results = []
        res = timedelta()
        limit = timedelta(hours=hour)
        for interval in intervals:
            res += interval[1] - interval[0]
            if res > limit and remove_at_end:
                interval = (interval[0], interval[1] + relativedelta(seconds=seconds(limit - res)))
            elif res > limit:
                interval = (interval[0] + relativedelta(seconds=seconds(res - limit)), interval[1])
            results.append(interval)
            if res > limit:
                break
        return results

    @api.multi
    def get_attendances_for_weekday(self, day_dt):
        if False:
            return 10
        ' Given a day datetime, return matching attendances '
        self.ensure_one()
        weekday = day_dt.weekday()
        attendances = self.env['resource.calendar.attendance']
        for attendance in self.attendance_ids.filtered(lambda att: int(att.dayofweek) == weekday and (not (att.date_from and fields.Date.from_string(att.date_from) > day_dt.date())) and (not (att.date_to and fields.Date.from_string(att.date_to) < day_dt.date()))):
            attendances |= attendance
        return attendances

    @api.multi
    def get_weekdays(self, default_weekdays=None):
        if False:
            return 10
        ' Return the list of weekdays that contain at least one working interval.\n        If no id is given (no calendar), return default weekdays. '
        if not self:
            return default_weekdays if default_weekdays is not None else [0, 1, 2, 3, 4]
        self.ensure_one()
        weekdays = set(map(int, self.attendance_ids.mapped('dayofweek')))
        return list(weekdays)

    @api.multi
    def get_next_day(self, day_date):
        if False:
            for i in range(10):
                print('nop')
        ' Get following date of day_date, based on resource.calendar. If no\n        calendar is provided, just return the next day.\n\n        :param date day_date: current day as a date\n\n        :return date: next day of calendar, or just next day '
        if not self:
            return day_date + relativedelta(days=1)
        self.ensure_one()
        weekdays = self.get_weekdays()
        base_index = -1
        for weekday in weekdays:
            if weekday > day_date.weekday():
                break
            base_index += 1
        new_index = (base_index + 1) % len(weekdays)
        days = weekdays[new_index] - day_date.weekday()
        if days < 0:
            days = 7 + days
        return day_date + relativedelta(days=days)

    @api.multi
    def get_previous_day(self, day_date):
        if False:
            while True:
                i = 10
        ' Get previous date of day_date, based on resource.calendar. If no\n        calendar is provided, just return the previous day.\n\n        :param date day_date: current day as a date\n\n        :return date: previous day of calendar, or just previous day '
        if not self:
            return day_date + relativedelta(days=-1)
        self.ensure_one()
        weekdays = self.get_weekdays()
        weekdays.reverse()
        base_index = -1
        for weekday in weekdays:
            if weekday < day_date.weekday():
                break
            base_index += 1
        new_index = (base_index + 1) % len(weekdays)
        days = weekdays[new_index] - day_date.weekday()
        if days > 0:
            days = days - 7
        return day_date + relativedelta(days=days)

    @api.multi
    def get_leave_intervals(self, resource_id=None, start_datetime=None, end_datetime=None):
        if False:
            return 10
        'Get the leaves of the calendar. Leaves can be filtered on the resource,\n        the start datetime or the end datetime.\n\n        :param int resource_id: the id of the resource to take into account when\n                                computing the leaves. If not set, only general\n                                leaves are computed. If set, generic and\n                                specific leaves are computed.\n        :param datetime start_datetime: if provided, do not take into account leaves\n                                        ending before this date.\n        :param datetime end_datetime: if provided, do not take into account leaves\n                                        beginning after this date.\n\n        :return list leaves: list of tuples (start_datetime, end_datetime) of\n                             leave intervals\n        '
        self.ensure_one()
        leaves = []
        for leave in self.leave_ids:
            if leave.resource_id and (not resource_id == leave.resource_id.id):
                continue
            date_from = fields.Datetime.from_string(leave.date_from)
            if end_datetime and date_from > end_datetime:
                continue
            date_to = fields.Datetime.from_string(leave.date_to)
            if start_datetime and date_to < start_datetime:
                continue
            leaves.append((date_from, date_to))
        return leaves

    @api.multi
    def get_working_intervals_of_day(self, start_dt=None, end_dt=None, leaves=None, compute_leaves=False, resource_id=None, default_interval=None):
        if False:
            print('Hello World!')
        ' Get the working intervals of the day based on calendar. This method\n        handle leaves that come directly from the leaves parameter or can be computed.\n\n        :param datetime start_dt: datetime object that is the beginning hours\n                                  for the working intervals computation; any\n                                  working interval beginning before start_dt\n                                  will be truncated. If not set, set to end_dt\n                                  or today() if no end_dt at 00.00.00.\n        :param datetime end_dt: datetime object that is the ending hour\n                                for the working intervals computation; any\n                                working interval ending after end_dt\n                                will be truncated. If not set, set to start_dt()\n                                at 23.59.59.\n        :param list leaves: a list of tuples(start_datetime, end_datetime) that\n                            represent leaves.\n        :param boolean compute_leaves: if set and if leaves is None, compute the\n                                       leaves based on calendar and resource.\n                                       If leaves is None and compute_leaves false\n                                       no leaves are taken into account.\n        :param int resource_id: the id of the resource to take into account when\n                                computing the leaves. If not set, only general\n                                leaves are computed. If set, generic and\n                                specific leaves are computed.\n        :param tuple default_interval: if no id, try to return a default working\n                                       day using default_interval[0] as beginning\n                                       hour, and default_interval[1] as ending hour.\n                                       Example: default_interval = (8, 16).\n                                       Otherwise, a void list of working intervals\n                                       is returned when id is None.\n\n        :return list intervals: a list of tuples (start_datetime, end_datetime)\n                                of work intervals '
        work_limits = []
        if start_dt is None and end_dt is not None:
            start_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif start_dt is None:
            start_dt = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            force_start_dt = self.env.context.get('force_start_dt')
            if force_start_dt and force_start_dt < start_dt:
                work_limits.append((force_start_dt.replace(hour=0, minute=0, second=0, microsecond=0), force_start_dt))
            work_limits.append((start_dt.replace(hour=0, minute=0, second=0, microsecond=0), start_dt))
        if end_dt is None:
            end_dt = start_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            work_limits.append((end_dt, end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)))
        assert start_dt.date() == end_dt.date(), 'get_working_intervals_of_day is restricted to one day'
        intervals = []
        work_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if not self:
            working_interval = []
            if default_interval:
                working_interval = (start_dt.replace(hour=default_interval[0], minute=0, second=0, microsecond=0), start_dt.replace(hour=default_interval[1], minute=0, second=0, microsecond=0))
            intervals = self.interval_remove_leaves(working_interval, work_limits)
            return intervals
        working_intervals = []
        tz_info = fields.Datetime.context_timestamp(self, work_dt).tzinfo
        for calendar_working_day in self.get_attendances_for_weekday(start_dt):
            dt_f = work_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=calendar_working_day.hour_from * 3600)
            dt_t = work_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=calendar_working_day.hour_to * 3600)
            working_interval = (dt_f.replace(tzinfo=tz_info).astimezone(pytz.UTC).replace(tzinfo=None), dt_t.replace(tzinfo=tz_info).astimezone(pytz.UTC).replace(tzinfo=None), calendar_working_day.id)
            if self.env.context.get('force_start_dt'):
                for wi in self.interval_remove_leaves(working_interval, work_limits):
                    if wi[0] >= self.env.context['force_start_dt']:
                        working_intervals += [wi]
            else:
                working_intervals += self.interval_remove_leaves(working_interval, work_limits)
        if leaves is None and compute_leaves:
            leaves = self.get_leave_intervals(resource_id=resource_id)
        for interval in working_intervals:
            work_intervals = self.interval_remove_leaves(interval, leaves)
            intervals += work_intervals
        return intervals

    @api.multi
    def get_working_hours_of_date(self, start_dt=None, end_dt=None, leaves=None, compute_leaves=False, resource_id=None, default_interval=None):
        if False:
            return 10
        ' Get the working hours of the day based on calendar. This method uses\n        get_working_intervals_of_day to have the work intervals of the day. It\n        then calculates the number of hours contained in those intervals. '
        res = timedelta()
        intervals = self.get_working_intervals_of_day(start_dt, end_dt, leaves, compute_leaves, resource_id, default_interval)
        for interval in intervals:
            res += interval[1] - interval[0]
        return seconds(res) / 3600.0

    @api.multi
    def get_working_hours(self, start_dt, end_dt, compute_leaves=False, resource_id=None, default_interval=None):
        if False:
            print('Hello World!')
        hours = 0.0
        for day in rrule.rrule(rrule.DAILY, dtstart=start_dt, until=end_dt.replace(hour=23, minute=59, second=59, microsecond=999999), byweekday=self.get_weekdays()):
            day_start_dt = day.replace(hour=0, minute=0, second=0, microsecond=0)
            if start_dt and day.date() == start_dt.date():
                day_start_dt = start_dt
            day_end_dt = day.replace(hour=23, minute=59, second=59, microsecond=999999)
            if end_dt and day.date() == end_dt.date():
                day_end_dt = end_dt
            hours += self.get_working_hours_of_date(start_dt=day_start_dt, end_dt=day_end_dt, compute_leaves=compute_leaves, resource_id=resource_id, default_interval=default_interval)
        return hours

    @api.multi
    def _schedule_hours(self, hours, day_dt=None, compute_leaves=False, resource_id=None, default_interval=None):
        if False:
            while True:
                i = 10
        ' Schedule hours of work, using a calendar and an optional resource to\n        compute working and leave days. This method can be used backwards, i.e.\n        scheduling days before a deadline.\n\n        :param int hours: number of hours to schedule. Use a negative number to\n                          compute a backwards scheduling.\n        :param datetime day_dt: reference date to compute working days. If days is\n                                > 0 date is the starting date. If days is < 0\n                                date is the ending date.\n        :param boolean compute_leaves: if set, compute the leaves based on calendar\n                                       and resource. Otherwise no leaves are taken\n                                       into account.\n        :param int resource_id: the id of the resource to take into account when\n                                computing the leaves. If not set, only general\n                                leaves are computed. If set, generic and\n                                specific leaves are computed.\n        :param tuple default_interval: if no id, try to return a default working\n                                       day using default_interval[0] as beginning\n                                       hour, and default_interval[1] as ending hour.\n                                       Example: default_interval = (8, 16).\n                                       Otherwise, a void list of working intervals\n                                       is returned when id is None.\n\n        :return tuple (datetime, intervals): datetime is the beginning/ending date\n                                             of the schedulign; intervals are the\n                                             working intervals of the scheduling.\n\n        Note: Why not using rrule.rrule ? Because rrule does not seem to allow\n        getting back in time.\n        '
        if day_dt is None:
            day_dt = datetime.datetime.now()
        elif day_dt is not None and hours > 0:
            self = self.with_context(force_start_dt=day_dt)
        backwards = hours < 0
        hours = abs(hours)
        intervals = []
        remaining_hours = hours * 1.0
        iterations = 0
        current_datetime = day_dt
        call_args = dict(compute_leaves=compute_leaves, resource_id=resource_id, default_interval=default_interval)
        while float_compare(remaining_hours, 0.0, precision_digits=2) in (1, 0) and iterations < 1000:
            if backwards:
                call_args['end_dt'] = current_datetime
            else:
                call_args['start_dt'] = current_datetime
            working_intervals = self.get_working_intervals_of_day(**call_args)
            if not self and (not working_intervals):
                remaining_hours -= 8.0
            elif working_intervals:
                if backwards:
                    working_intervals.reverse()
                new_working_intervals = self.interval_schedule_hours(working_intervals, remaining_hours, not backwards)
                if backwards:
                    new_working_intervals.reverse()
                res = timedelta()
                for interval in working_intervals:
                    res += interval[1] - interval[0]
                remaining_hours -= seconds(res) / 3600.0
                if backwards:
                    intervals = new_working_intervals + intervals
                else:
                    intervals = intervals + new_working_intervals
            if backwards:
                current_datetime = datetime.datetime.combine(self.get_previous_day(current_datetime), datetime.time(23, 59, 59))
            else:
                current_datetime = datetime.datetime.combine(self.get_next_day(current_datetime), datetime.time())
            iterations += 1
        return intervals

    @api.multi
    def schedule_hours_get_date(self, hours, day_dt=None, compute_leaves=False, resource_id=None, default_interval=None):
        if False:
            for i in range(10):
                print('nop')
        ' Wrapper on _schedule_hours: return the beginning/ending datetime of\n        an hours scheduling. '
        res = self._schedule_hours(hours, day_dt, compute_leaves, resource_id, default_interval)
        if res and hours < 0.0:
            return res[0][0]
        elif res:
            return res[-1][1]
        return False

    @api.multi
    def schedule_hours(self, hours, day_dt=None, compute_leaves=False, resource_id=None, default_interval=None):
        if False:
            while True:
                i = 10
        ' Wrapper on _schedule_hours: return the working intervals of an hours\n        scheduling. '
        return self._schedule_hours(hours, day_dt, compute_leaves, resource_id, default_interval)

    @api.multi
    def _schedule_days(self, days, day_date=None, compute_leaves=False, resource_id=None, default_interval=None):
        if False:
            i = 10
            return i + 15
        'Schedule days of work, using a calendar and an optional resource to\n        compute working and leave days. This method can be used backwards, i.e.\n        scheduling days before a deadline.\n\n        :param int days: number of days to schedule. Use a negative number to\n                         compute a backwards scheduling.\n        :param date day_date: reference date to compute working days. If days is > 0\n                              date is the starting date. If days is < 0 date is the\n                              ending date.\n        :param boolean compute_leaves: if set, compute the leaves based on calendar\n                                       and resource. Otherwise no leaves are taken\n                                       into account.\n        :param int resource_id: the id of the resource to take into account when\n                                computing the leaves. If not set, only general\n                                leaves are computed. If set, generic and\n                                specific leaves are computed.\n        :param tuple default_interval: if no id, try to return a default working\n                                       day using default_interval[0] as beginning\n                                       hour, and default_interval[1] as ending hour.\n                                       Example: default_interval = (8, 16).\n                                       Otherwise, a void list of working intervals\n                                       is returned when id is None.\n\n        :return tuple (datetime, intervals): datetime is the beginning/ending date\n                                             of the schedulign; intervals are the\n                                             working intervals of the scheduling.\n\n        Implementation note: rrule.rrule is not used because rrule it des not seem\n        to allow getting back in time.\n        '
        if day_date is None:
            day_date = datetime.datetime.now()
        backwards = days < 0
        days = abs(days)
        intervals = []
        planned_days = 0
        iterations = 0
        current_datetime = day_date.replace(hour=0, minute=0, second=0, microsecond=0)
        while planned_days < days and iterations < 100:
            working_intervals = self.get_working_intervals_of_day(current_datetime, compute_leaves=compute_leaves, resource_id=resource_id, default_interval=default_interval)
            if not self or working_intervals:
                planned_days += 1
                intervals += working_intervals
            if backwards:
                current_datetime = self.get_previous_day(current_datetime)
            else:
                current_datetime = self.get_next_day(current_datetime)
            iterations += 1
        return intervals

    @api.multi
    def schedule_days_get_date(self, days, day_date=None, compute_leaves=False, resource_id=None, default_interval=None):
        if False:
            i = 10
            return i + 15
        ' Wrapper on _schedule_days: return the beginning/ending datetime of\n        a days scheduling. '
        res = self._schedule_days(days, day_date, compute_leaves, resource_id, default_interval)
        return res and res[-1][1] or False

    @api.multi
    def schedule_days(self, days, day_date=None, compute_leaves=False, resource_id=None, default_interval=None):
        if False:
            print('Hello World!')
        ' Wrapper on _schedule_days: return the working intervals of a days\n        scheduling. '
        return self._schedule_days(days, day_date, compute_leaves, resource_id, default_interval)

    @api.multi
    def working_hours_on_day(self, day):
        if False:
            while True:
                i = 10
        ' Used in hr_payroll/hr_payroll.py\n\n        :deprecated: Odoo saas-3. Use get_working_hours_of_date instead. Note:\n        since saas-3, take hour/minutes into account, not just the whole day.'
        if isinstance(day, datetime.datetime):
            day = day.replace(hour=0, minute=0)
        return self.get_working_hours_of_date(start_dt=day)

    @api.multi
    def interval_min_get(self, dt_from, hours, resource=False):
        if False:
            return 10
        ' Schedule hours backwards. Used in mrp_operations/mrp_operations.py.\n\n        :deprecated: Odoo saas-3. Use schedule_hours instead. Note: since\n        saas-3, counts leave hours instead of all-day leaves.'
        return self.schedule_hours(hours * -1.0, day_dt=dt_from.replace(minute=0, second=0, microsecond=0), compute_leaves=True, resource_id=resource, default_interval=(8, 16))

    @api.model
    def interval_get_multi(self, date_and_hours_by_cal, resource=False, byday=True):
        if False:
            print('Hello World!')
        ' Used in mrp_operations/mrp_operations.py (default parameters) and in\n        interval_get()\n\n        :deprecated: Odoo saas-3. Use schedule_hours instead. Note:\n        Byday was not used. Since saas-3, counts Leave hours instead of all-day leaves.'
        res = {}
        for (dt_str, hours, calendar_id) in date_and_hours_by_cal:
            result = self.browse(calendar_id).schedule_hours(hours, day_dt=fields.Datetime.from_string(dt_str).replace(second=0), compute_leaves=True, resource_id=resource, default_interval=(8, 16))
            res[dt_str, hours, calendar_id] = result
        return res

    @api.multi
    def interval_get(self, dt_from, hours, resource=False, byday=True):
        if False:
            return 10
        ' Unifier of interval_get_multi. Used in: mrp_operations/mrp_operations.py,\n        crm/crm_lead.py (res given).\n\n        :deprecated: Odoo saas-3. Use get_working_hours instead.'
        self.ensure_one()
        res = self.interval_get_multi([(fields.Datetime.to_string(dt_from), hours, self.id)], resource, byday)[fields.Datetime.to_string(dt_from), hours, self.id]
        return res

    @api.multi
    def interval_hours_get(self, dt_from, dt_to, resource=False):
        if False:
            while True:
                i = 10
        ' Unused wrapper.\n\n        :deprecated: Odoo saas-3. Use get_working_hours instead.'
        return self._interval_hours_get(dt_from, dt_to, resource_id=resource)

    @api.multi
    def _interval_hours_get(self, dt_from, dt_to, resource_id=False, timezone_from_uid=None, exclude_leaves=True):
        if False:
            i = 10
            return i + 15
        ' Computes working hours between two dates, taking always same hour/minuts.\n        :deprecated: Odoo saas-3. Use get_working_hours instead. Note: since saas-3,\n        now resets hour/minuts. Now counts leave hours instead of all-day leaves.'
        return self.get_working_hours(dt_from, dt_to, compute_leaves=not exclude_leaves, resource_id=resource_id, default_interval=(8, 16))

class ResourceCalendarAttendance(models.Model):
    _name = 'resource.calendar.attendance'
    _description = 'Work Detail'
    _order = 'dayofweek, hour_from'
    name = fields.Char(required=True)
    dayofweek = fields.Selection([('0', 'Monday'), ('1', 'Tuesday'), ('2', 'Wednesday'), ('3', 'Thursday'), ('4', 'Friday'), ('5', 'Saturday'), ('6', 'Sunday')], 'Day of Week', required=True, index=True, default='0')
    date_from = fields.Date(string='Starting Date')
    date_to = fields.Date(string='End Date')
    hour_from = fields.Float(string='Work from', required=True, index=True, help='Start and End time of working.')
    hour_to = fields.Float(string='Work to', required=True)
    calendar_id = fields.Many2one('resource.calendar', string="Resource's Calendar", required=True, ondelete='cascade')

def hours_time_string(hours):
    if False:
        while True:
            i = 10
    " convert a number of hours (float) into a string with format '%H:%M' "
    minutes = int(round(hours * 60))
    return '%02d:%02d' % divmod(minutes, 60)

class ResourceResource(models.Model):
    _name = 'resource.resource'
    _description = 'Resource Detail'
    name = fields.Char(required=True)
    code = fields.Char(copy=False)
    active = fields.Boolean(track_visibility='onchange', default=True, help='If the active field is set to False, it will allow you to hide the resource record without removing it.')
    company_id = fields.Many2one('res.company', string='Company', default=lambda self: self.env['res.company']._company_default_get())
    resource_type = fields.Selection([('user', 'Human'), ('material', 'Material')], string='Resource Type', required=True, default='user')
    user_id = fields.Many2one('res.users', string='User', help='Related user name for the resource to manage its access.')
    time_efficiency = fields.Float(string='Efficiency Factor', required=True, default=100, help='This field depict the efficiency of the resource to complete tasks. e.g  resource put alone on a phase of 5 days with 5 tasks assigned to him, will show a load of 100% for this phase by default, but if we put a efficiency of 200%, then his load will only be 50%.')
    calendar_id = fields.Many2one('resource.calendar', string='Working Time', help='Define the schedule of resource')

    @api.multi
    def copy(self, default=None):
        if False:
            i = 10
            return i + 15
        self.ensure_one()
        if default is None:
            default = {}
        if not default.get('name'):
            default.update(name=_('%s (copy)') % self.name)
        return super(ResourceResource, self).copy(default)

    def _is_work_day(self, date):
        if False:
            for i in range(10):
                print('nop')
        ' Whether the provided date is a work day for the subject resource.\n\n        :type date: datetime.date\n        :rtype: bool\n        '
        return bool(next(self._iter_work_days(date, date), False))

    def _iter_work_days(self, from_date, to_date):
        if False:
            print('Hello World!')
        " Lists the current resource's work days between the two provided\n        dates (inclusive).\n\n        Work days are the company or service's open days (as defined by the\n        resource.calendar) minus the resource's own leaves.\n\n        :param datetime.date from_date: start of the interval to check for\n                                        work days (inclusive)\n        :param datetime.date to_date: end of the interval to check for work\n                                      days (inclusive)\n        :rtype: list(datetime.date)\n        "
        working_intervals = self.calendar_id.get_working_intervals_of_day
        for dt in rrule.rrule(rrule.DAILY, dtstart=from_date, until=to_date):
            intervals = working_intervals(dt, compute_leaves=True, resource_id=self.id)
            if intervals and intervals[0]:
                yield dt.date()

class ResourceCalendarLeaves(models.Model):
    _name = 'resource.calendar.leaves'
    _description = 'Leave Detail'
    name = fields.Char()
    company_id = fields.Many2one('res.company', related='calendar_id.company_id', string='Company', store=True, readonly=True)
    calendar_id = fields.Many2one('resource.calendar', string='Working Time')
    date_from = fields.Datetime(string='Start Date', required=True)
    date_to = fields.Datetime(string='End Date', required=True)
    resource_id = fields.Many2one('resource.resource', string='Resource', help='If empty, this is a generic holiday for the company. If a resource is set, the holiday/leave is only for this resource')

    @api.constrains('date_from', 'date_to')
    def check_dates(self):
        if False:
            i = 10
            return i + 15
        if self.filtered(lambda leave: leave.date_from > leave.date_to):
            raise ValidationError(_('Error! leave start-date must be lower then leave end-date.'))

    @api.onchange('resource_id')
    def onchange_resource(self):
        if False:
            print('Hello World!')
        self.calendar_id = self.resource_id.calendar_id

def seconds(td):
    if False:
        i = 10
        return i + 15
    assert isinstance(td, timedelta)
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10 ** 6) / 10.0 ** 6