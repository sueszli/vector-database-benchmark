import time
from datetime import datetime, timedelta
from dagster import Definitions, In, Nothing, OpExecutionContext, RetryPolicy, ScheduleDefinition, job, op, schedule

@op
def print_date(context: OpExecutionContext) -> datetime:
    if False:
        return 10
    ds = datetime.now()
    context.log.info(ds)
    return ds

@op(retry_policy=RetryPolicy(max_retries=3), ins={'start': In(Nothing)})
def sleep():
    if False:
        i = 10
        return i + 15
    time.sleep(5)

@op
def templated(context: OpExecutionContext, ds: datetime):
    if False:
        for i in range(10):
            print('nop')
    for _i in range(5):
        context.log.info(ds)
        context.log.info(ds - timedelta(days=7))

@job(tags={'dagster/max_retries': 1, 'dag_name': 'example'})
def tutorial_job():
    if False:
        while True:
            i = 10
    ds = print_date()
    sleep(ds)
    templated(ds)
schedule = ScheduleDefinition(job=tutorial_job, cron_schedule='@daily')
defs = Definitions(jobs=[tutorial_job], schedules=[schedule])