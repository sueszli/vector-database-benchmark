from dagster._core.definitions import job, op, repository

@op
def hello_world(_):
    if False:
        return 10
    pass

@job
def hello_world_job():
    if False:
        for i in range(10):
            print('nop')
    hello_world()

@repository(name='hello_world_repository_name')
def named_hello_world_repository():
    if False:
        i = 10
        return i + 15
    return [hello_world_job]