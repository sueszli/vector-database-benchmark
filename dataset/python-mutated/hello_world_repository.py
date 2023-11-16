from dagster._core.definitions import job, op, repository

@op
def hello_world(_):
    if False:
        i = 10
        return i + 15
    pass

@job
def hello_world_job():
    if False:
        return 10
    hello_world()

@repository
def hello_world_repository():
    if False:
        i = 10
        return i + 15
    return [hello_world_job]