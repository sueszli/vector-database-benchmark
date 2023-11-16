from dagster import FilesystemIOManager, job, op

@op
def op_1():
    if False:
        return 10
    return 1

@op
def op_2(a):
    if False:
        return 10
    return a + 1

@job(resource_defs={'io_manager': FilesystemIOManager()})
def my_job():
    if False:
        print('Hello World!')
    op_2(op_1())