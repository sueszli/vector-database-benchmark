from dagster import job, op, static_partitioned_config

@static_partitioned_config(['a', 'b', 'c'])
def partconf(partition):
    if False:
        return 10
    return {'ops': {'op1': {'letter': partition}}}

@op(config_schema={'letter': str})
def op1():
    if False:
        for i in range(10):
            print('nop')
    ...

@job(config=partconf)
def job_with_partition_config():
    if False:
        while True:
            i = 10
    op1()