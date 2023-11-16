from dagster import DagsterRunStatus, RunRequest, SkipReason, run_failure_sensor, run_status_sensor
status_reporting_job = None

@run_status_sensor(run_status=DagsterRunStatus.SUCCESS, request_job=status_reporting_job)
def report_status_sensor(context):
    if False:
        i = 10
        return i + 15
    if context.dagster_run.job_name != status_reporting_job.name:
        run_config = {'ops': {'status_report': {'config': {'job_name': context.dagster_run.job_name}}}}
        return RunRequest(run_key=None, run_config=run_config)
    else:
        return SkipReason("Don't report status of status_reporting_job")

@run_failure_sensor(request_job=status_reporting_job)
def report_failure_sensor(context):
    if False:
        while True:
            i = 10
    run_config = {'ops': {'status_report': {'config': {'job_name': context.dagster_run.job_name}}}}
    return RunRequest(run_key=None, run_config=run_config)