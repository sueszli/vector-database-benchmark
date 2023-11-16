from dagster import Definitions, job

@job
def do_it_all():
    if False:
        i = 10
        return i + 15
    ...
defs = Definitions(jobs=[do_it_all])