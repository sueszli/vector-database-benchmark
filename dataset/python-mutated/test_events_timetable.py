from __future__ import annotations
import pendulum
import pytest
from airflow.settings import TIMEZONE
from airflow.timetables.base import DagRunInfo, DataInterval, TimeRestriction, Timetable
from airflow.timetables.events import EventsTimetable
START_DATE = pendulum.DateTime(2021, 9, 4, tzinfo=TIMEZONE)
EVENT_DATES = [pendulum.DateTime(2021, 9, 6, tzinfo=TIMEZONE), pendulum.DateTime(2021, 9, 7, tzinfo=TIMEZONE), pendulum.DateTime(2021, 9, 8, tzinfo=TIMEZONE), pendulum.DateTime(2021, 9, 8, tzinfo=TIMEZONE), pendulum.DateTime(2021, 10, 9, tzinfo=TIMEZONE), pendulum.DateTime(2021, 9, 10, tzinfo=TIMEZONE)]
EVENT_DATES_SORTED = [pendulum.DateTime(2021, 9, 6, tzinfo=TIMEZONE), pendulum.DateTime(2021, 9, 7, tzinfo=TIMEZONE), pendulum.DateTime(2021, 9, 8, tzinfo=TIMEZONE), pendulum.DateTime(2021, 9, 10, tzinfo=TIMEZONE), pendulum.DateTime(2021, 10, 9, tzinfo=TIMEZONE)]
NON_EVENT_DATE = pendulum.DateTime(2021, 10, 1, tzinfo=TIMEZONE)
MOST_RECENT_EVENT = pendulum.DateTime(2021, 9, 10, tzinfo=TIMEZONE)

@pytest.fixture()
def restriction():
    if False:
        return 10
    return TimeRestriction(earliest=START_DATE, latest=None, catchup=True)

@pytest.fixture()
def unrestricted_timetable():
    if False:
        for i in range(10):
            print('nop')
    return EventsTimetable(event_dates=EVENT_DATES)

@pytest.fixture()
def restricted_timetable():
    if False:
        return 10
    return EventsTimetable(event_dates=EVENT_DATES, restrict_to_events=True)

@pytest.mark.parametrize('start, end', list(zip(EVENT_DATES, EVENT_DATES)))
def test_dag_run_info_interval(start: pendulum.DateTime, end: pendulum.DateTime):
    if False:
        print('Hello World!')
    expected_info = DagRunInfo(run_after=end, data_interval=DataInterval(start, end))
    assert DagRunInfo.interval(start, end) == expected_info

def test_manual_with_unrestricted(unrestricted_timetable: Timetable, restriction: TimeRestriction):
    if False:
        return 10
    'When not using strict event dates, manual runs have run_after as the data interval'
    manual_run_data_interval = unrestricted_timetable.infer_manual_data_interval(run_after=NON_EVENT_DATE)
    expected_data_interval = DataInterval.exact(NON_EVENT_DATE)
    assert expected_data_interval == manual_run_data_interval

def test_manual_with_restricted_middle(restricted_timetable: Timetable, restriction: TimeRestriction):
    if False:
        while True:
            i = 10
    "\n    Test that when using strict event dates, manual runs after the first event have the\n    most recent event's date as the start interval\n    "
    manual_run_data_interval = restricted_timetable.infer_manual_data_interval(run_after=NON_EVENT_DATE)
    expected_data_interval = DataInterval.exact(MOST_RECENT_EVENT)
    assert expected_data_interval == manual_run_data_interval

def test_manual_with_restricted_before(restricted_timetable: Timetable, restriction: TimeRestriction):
    if False:
        return 10
    "\n    Test that when using strict event dates, manual runs before the first event have the first event's date\n    as the start interval\n    "
    manual_run_data_interval = restricted_timetable.infer_manual_data_interval(run_after=START_DATE)
    expected_data_interval = DataInterval.exact(EVENT_DATES[0])
    assert expected_data_interval == manual_run_data_interval

@pytest.mark.parametrize('last_automated_data_interval, expected_next_info', [pytest.param(DataInterval(day1, day1), DagRunInfo.interval(day2, day2)) for (day1, day2) in zip(EVENT_DATES_SORTED, EVENT_DATES_SORTED[1:])] + [pytest.param(DataInterval(EVENT_DATES_SORTED[-1], EVENT_DATES_SORTED[-1]), None)])
def test_subsequent_weekday_schedule(unrestricted_timetable: Timetable, restriction: TimeRestriction, last_automated_data_interval: DataInterval, expected_next_info: DagRunInfo):
    if False:
        for i in range(10):
            print('nop')
    'The next four subsequent runs cover the next four weekdays each.'
    next_info = unrestricted_timetable.next_dagrun_info(last_automated_data_interval=last_automated_data_interval, restriction=restriction)
    assert next_info == expected_next_info