from dagster import job, repository

@job
def job_one():
    if False:
        return 10
    pass

@job
def job_two():
    if False:
        i = 10
        return i + 15
    pass

@repository
def multi_job():
    if False:
        while True:
            i = 10
    return [job_one, job_two]