from dagster import OpExecutionContext, RunRequest, ScheduleEvaluationContext, job, op, schedule

@op(config_schema={'activity_selection': str})
def configurable_op(context: OpExecutionContext):
    if False:
        while True:
            i = 10
    pass

@job
def configurable_job():
    if False:
        print('Hello World!')
    configurable_op()

@schedule(job=configurable_job, cron_schedule='0 9 * * *')
def configurable_job_schedule(context: ScheduleEvaluationContext):
    if False:
        return 10
    if context.scheduled_execution_time.weekday() < 5:
        activity_selection = 'grind'
    else:
        activity_selection = 'party'
    return RunRequest(run_config={'ops': {'configurable_op': {'config': {'activity': activity_selection}}}})