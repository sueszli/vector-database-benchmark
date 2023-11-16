from dagster import job

@job(tags={'dagster/max_retries': 3})
def sample_job():
    if False:
        i = 10
        return i + 15
    pass

@job(tags={'dagster/max_retries': 3, 'dagster/retry_strategy': 'ALL_STEPS'})
def other_sample_sample_job():
    if False:
        while True:
            i = 10
    pass