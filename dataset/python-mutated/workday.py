"""Plugin to demonstrate timetable registration and accommodate example DAGs."""
from __future__ import annotations
import logging
from datetime import timedelta
from pendulum import UTC, Date, DateTime, Time
from airflow.plugins_manager import AirflowPlugin
from airflow.timetables.base import DagRunInfo, DataInterval, TimeRestriction, Timetable
log = logging.getLogger(__name__)
try:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    holiday_calendar = USFederalHolidayCalendar()
except ImportError:
    log.warning('Could not import pandas. Holidays will not be considered.')
    holiday_calendar = None

class AfterWorkdayTimetable(Timetable):

    def get_next_workday(self, d: DateTime, incr=1) -> DateTime:
        if False:
            return 10
        next_start = d
        while True:
            if next_start.weekday() in (5, 6):
                next_start = next_start + incr * timedelta(days=1)
                continue
            if holiday_calendar is not None:
                holidays = holiday_calendar.holidays(start=next_start, end=next_start).to_pydatetime()
                if next_start in holidays:
                    next_start = next_start + incr * timedelta(days=1)
                    continue
            break
        return next_start

    def infer_manual_data_interval(self, run_after: DateTime) -> DataInterval:
        if False:
            for i in range(10):
                print('nop')
        start = DateTime.combine((run_after - timedelta(days=1)).date(), Time.min).replace(tzinfo=UTC)
        start = self.get_next_workday(start, incr=-1)
        return DataInterval(start=start, end=start + timedelta(days=1))

    def next_dagrun_info(self, *, last_automated_data_interval: DataInterval | None, restriction: TimeRestriction) -> DagRunInfo | None:
        if False:
            print('Hello World!')
        if last_automated_data_interval is not None:
            last_start = last_automated_data_interval.start
            next_start = DateTime.combine((last_start + timedelta(days=1)).date(), Time.min).replace(tzinfo=UTC)
        else:
            next_start = restriction.earliest
            if next_start is None:
                return None
            if not restriction.catchup:
                next_start = max(next_start, DateTime.combine(Date.today(), Time.min).replace(tzinfo=UTC))
            elif next_start.time() != Time.min:
                next_start = DateTime.combine(next_start.date() + timedelta(days=1), Time.min).replace(tzinfo=UTC)
        next_start = self.get_next_workday(next_start)
        if restriction.latest is not None and next_start > restriction.latest:
            return None
        return DagRunInfo.interval(start=next_start, end=next_start + timedelta(days=1))

class WorkdayTimetablePlugin(AirflowPlugin):
    name = 'workday_timetable_plugin'
    timetables = [AfterWorkdayTimetable]