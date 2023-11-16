from dagster import job, op

@op
def my_op():
    if False:
        print('Hello World!')
    print('foo')

@job(tags={'dagster-k8s/config': {'job_spec_config': {'ttl_seconds_after_finished': 7200}}})
def my_job():
    if False:
        for i in range(10):
            print('nop')
    my_op()