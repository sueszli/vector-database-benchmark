import flytekit
from flytekit import CronSchedule, LaunchPlan, Secret, task, workflow
SECRET_NAME = 'user_secret'
SECRET_GROUP = 'user-info'

@task(secret_requests=[Secret(group=SECRET_GROUP, key=SECRET_NAME)])
def secret_task() -> str:
    if False:
        while True:
            i = 10
    secret_val = flytekit.current_context().secrets.get(SECRET_GROUP, SECRET_NAME)
    print(secret_val)
    return secret_val

@workflow
def wf() -> str:
    if False:
        i = 10
        return i + 15
    x = secret_task()
    return x
sslp = LaunchPlan.get_or_create(name='scheduled_secrets', workflow=wf, schedule=CronSchedule(schedule='0/1 * * * *'))