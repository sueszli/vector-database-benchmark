from dagster import SourceHashVersionStrategy, job

@job(version_strategy=SourceHashVersionStrategy())
def the_job():
    if False:
        for i in range(10):
            print('nop')
    ...