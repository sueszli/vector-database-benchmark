from dagster import job

@job
def a_job():
    if False:
        print('Hello World!')
    pass