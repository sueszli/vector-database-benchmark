import pytest
from dagster._check import ParameterCheckError
from dagster._core.definitions.freshness_policy import FreshnessPolicy
from dagster._core.errors import DagsterInvalidDefinitionError
from dagster._seven.compat.pendulum import create_pendulum_time

@pytest.mark.parametrize(['policy', 'used_data_time', 'evaluation_time', 'expected_minutes_overdue', 'expected_minutes_lag'], [(FreshnessPolicy(maximum_lag_minutes=30), create_pendulum_time(2022, 1, 1, 0), create_pendulum_time(2022, 1, 1, 0, 25), 0, 25), (FreshnessPolicy(maximum_lag_minutes=120), create_pendulum_time(2022, 1, 1, 0), create_pendulum_time(2022, 1, 1, 1), 0, 60), (FreshnessPolicy(maximum_lag_minutes=30), create_pendulum_time(2022, 1, 1, 0), create_pendulum_time(2022, 1, 1, 1), 30, 60), (FreshnessPolicy(maximum_lag_minutes=500), None, create_pendulum_time(2022, 1, 1, 0, 25), None, None), (FreshnessPolicy(cron_schedule='@daily', maximum_lag_minutes=15), create_pendulum_time(2022, 1, 1, 23, 55), create_pendulum_time(2022, 1, 2, 0, 10), 0, 5), (FreshnessPolicy(cron_schedule='@daily', maximum_lag_minutes=15), create_pendulum_time(2022, 1, 1, 0, 30), create_pendulum_time(2022, 1, 1, 1, 0), 0, 0), (FreshnessPolicy(cron_schedule='@daily', maximum_lag_minutes=60), create_pendulum_time(2022, 1, 1, 22, 0), create_pendulum_time(2022, 1, 2, 2, 0), 60, 120), (FreshnessPolicy(cron_schedule='@hourly', maximum_lag_minutes=60 * 5), create_pendulum_time(2022, 1, 1, 1, 0), create_pendulum_time(2022, 1, 1, 4, 0), 0, 180), (FreshnessPolicy(cron_schedule='@hourly', maximum_lag_minutes=60 * 5), create_pendulum_time(2022, 1, 1, 1, 15), create_pendulum_time(2022, 1, 1, 7, 45), 45, 45 + 60 * 5), (FreshnessPolicy(cron_schedule='0 3 * * *', cron_schedule_timezone='America/Los_Angeles', maximum_lag_minutes=60), create_pendulum_time(2022, 1, 1, 1, 0, tz='America/Los_Angeles'), create_pendulum_time(2022, 1, 1, 3, 15, tz='America/Los_Angeles'), 60, 120), (FreshnessPolicy(cron_schedule='0 3 * * *', cron_schedule_timezone='America/Los_Angeles', maximum_lag_minutes=60), create_pendulum_time(2022, 1, 1, 1, 0, tz='America/Los_Angeles').in_tz('UTC'), create_pendulum_time(2022, 1, 1, 3, 15, tz='America/Los_Angeles').in_tz('UTC'), 60, 120), (FreshnessPolicy(cron_schedule='0 3 * * *', cron_schedule_timezone='America/Los_Angeles', maximum_lag_minutes=60), create_pendulum_time(2022, 1, 1, 0, 0, tz='America/Los_Angeles'), create_pendulum_time(2022, 1, 1, 2, 15, tz='America/Los_Angeles'), 0, 0)])
def test_policies_available_equals_evaluation_time(policy, used_data_time, evaluation_time, expected_minutes_overdue, expected_minutes_lag):
    if False:
        while True:
            i = 10
    result = policy.minutes_overdue(data_time=used_data_time, evaluation_time=evaluation_time)
    assert getattr(result, 'overdue_minutes', None) == expected_minutes_overdue
    assert getattr(result, 'lag_minutes', None) == expected_minutes_lag

def test_invalid_freshness_policies():
    if False:
        print('Hello World!')
    with pytest.raises(DagsterInvalidDefinitionError, match='Invalid cron schedule'):
        FreshnessPolicy(cron_schedule='xyz-123-bad-schedule', maximum_lag_minutes=60)
    with pytest.raises(DagsterInvalidDefinitionError, match='Invalid cron schedule timezone'):
        FreshnessPolicy(cron_schedule='0 1 * * *', maximum_lag_minutes=60, cron_schedule_timezone='Not/ATimezone')
    with pytest.raises(ParameterCheckError, match='without a cron_schedule'):
        FreshnessPolicy(maximum_lag_minutes=0, cron_schedule_timezone='America/Los_Angeles')