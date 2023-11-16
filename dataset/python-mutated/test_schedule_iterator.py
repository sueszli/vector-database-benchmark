import pytest
from dagster._check import CheckError
from dagster._seven.compat.pendulum import create_pendulum_time
from dagster._utils.schedules import schedule_execution_time_iterator

def test_cron_schedule_advances_past_dst():
    if False:
        for i in range(10):
            print('nop')
    start_time = create_pendulum_time(year=2021, month=10, day=3, hour=1, minute=30, second=1, tz='Australia/Sydney')
    time_iter = schedule_execution_time_iterator(start_time.timestamp(), '*/15 * * * *', 'Australia/Sydney')
    for _i in range(6):
        next_time = next(time_iter)
    assert next_time.timestamp() == create_pendulum_time(year=2021, month=10, day=3, hour=4, tz='Australia/Sydney').timestamp()

def test_vixie_cronstring_schedule():
    if False:
        print('Hello World!')
    start_time = create_pendulum_time(year=2022, month=2, day=21, hour=1, minute=30, second=1, tz='US/Pacific')
    time_iter = schedule_execution_time_iterator(start_time.timestamp(), '@hourly', 'US/Pacific')
    for _i in range(6):
        next_time = next(time_iter)
    assert next_time.timestamp() == create_pendulum_time(year=2022, month=2, day=21, hour=7, tz='US/Pacific').timestamp()
    time_iter = schedule_execution_time_iterator(start_time.timestamp(), '@daily', 'US/Pacific')
    for _i in range(6):
        next_time = next(time_iter)
    assert next_time.timestamp() == create_pendulum_time(year=2022, month=2, day=27, tz='US/Pacific').timestamp()
    time_iter = schedule_execution_time_iterator(start_time.timestamp(), '@weekly', 'US/Pacific')
    for _i in range(6):
        next_time = next(time_iter)
    assert next_time.timestamp() == create_pendulum_time(year=2022, month=4, day=3, tz='US/Pacific').timestamp()
    time_iter = schedule_execution_time_iterator(start_time.timestamp(), '@monthly', 'US/Pacific')
    for _i in range(6):
        next_time = next(time_iter)
    assert next_time.timestamp() == create_pendulum_time(year=2022, month=8, day=1, tz='US/Pacific').timestamp()
    time_iter = schedule_execution_time_iterator(start_time.timestamp(), '@yearly', 'US/Pacific')
    for _i in range(6):
        next_time = next(time_iter)
    assert next_time.timestamp() == create_pendulum_time(year=2028, month=1, day=1, tz='US/Pacific').timestamp()

def test_union_of_cron_strings_schedule():
    if False:
        while True:
            i = 10
    start_time = create_pendulum_time(year=2022, month=1, day=1, hour=2, tz='UTC')
    time_iter = schedule_execution_time_iterator(start_time.timestamp(), ['0 2 * * FRI-SAT', '0 2,8 * * MON,FRI', '*/30 9 * * SUN'], 'UTC')
    next_timestamps = [next(time_iter).timestamp() for _ in range(8)]
    expected_next_timestamps = [dt.timestamp() for dt in [create_pendulum_time(year=2022, month=1, day=1, hour=2, tz='UTC'), create_pendulum_time(year=2022, month=1, day=2, hour=9, tz='UTC'), create_pendulum_time(year=2022, month=1, day=2, hour=9, minute=30, tz='UTC'), create_pendulum_time(year=2022, month=1, day=3, hour=2, tz='UTC'), create_pendulum_time(year=2022, month=1, day=3, hour=8, tz='UTC'), create_pendulum_time(year=2022, month=1, day=7, hour=2, tz='UTC'), create_pendulum_time(year=2022, month=1, day=7, hour=8, tz='UTC'), create_pendulum_time(year=2022, month=1, day=8, hour=2, tz='UTC')]]
    assert next_timestamps == expected_next_timestamps

def test_invalid_cron_string():
    if False:
        for i in range(10):
            print('nop')
    start_time = create_pendulum_time(year=2022, month=2, day=21, hour=1, minute=30, second=1, tz='US/Pacific')
    with pytest.raises(CheckError):
        next(schedule_execution_time_iterator(start_time.timestamp(), '* * * * * *', 'US/Pacific'))

def test_empty_cron_string_union():
    if False:
        for i in range(10):
            print('nop')
    start_time = create_pendulum_time(year=2022, month=2, day=21, hour=1, minute=30, second=1, tz='US/Pacific')
    with pytest.raises(CheckError):
        next(schedule_execution_time_iterator(start_time.timestamp(), [], 'US/Pacific'))

def test_first_monday():
    if False:
        for i in range(10):
            print('nop')
    start_time = create_pendulum_time(year=2023, month=1, day=1, tz='US/Pacific')
    iterator = schedule_execution_time_iterator(start_time.timestamp(), '0 0 * * mon#1', 'US/Pacific')
    assert next(iterator) == create_pendulum_time(year=2023, month=1, day=2, tz='US/Pacific')
    assert next(iterator) == create_pendulum_time(year=2023, month=2, day=6, tz='US/Pacific')
    assert next(iterator) == create_pendulum_time(year=2023, month=3, day=6, tz='US/Pacific')

def test_on_tick_boundary_simple():
    if False:
        while True:
            i = 10
    start_time = create_pendulum_time(year=2022, month=1, day=1, hour=2, tz='UTC')
    time_iter = schedule_execution_time_iterator(start_time.timestamp(), ['0 3 * * *'], 'UTC')
    next_timestamps = [next(time_iter).timestamp() for _ in range(8)]
    expected_next_timestamps = [create_pendulum_time(year=2022, month=1, day=i + 1, hour=3, tz='UTC').timestamp() for i in range(8)]
    assert next_timestamps == expected_next_timestamps

def test_on_tick_boundary_complex():
    if False:
        return 10
    start_time = create_pendulum_time(year=2022, month=1, day=1, hour=2, tz='UTC')
    time_iter = schedule_execution_time_iterator(start_time.timestamp(), ['0 3 * * MON-FRI'], 'UTC')
    next_timestamps = [next(time_iter).timestamp() for _ in range(10)]
    expected_next_timestamps = [create_pendulum_time(year=2022, month=1, day=i + 1, hour=3, tz='UTC').timestamp() for i in (*range(2, 7), *range(9, 14))]
    assert next_timestamps == expected_next_timestamps