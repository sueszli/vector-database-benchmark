from dagster import ScheduleDefinition

# Definition that will fire an error when it is imported
ScheduleDefinition(cron_schedule="* * * * * * * * * *", job_name="foo")
