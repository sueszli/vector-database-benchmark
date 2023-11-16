from dagster import Definitions
from dagster import job, op, OpExecutionContext

@op
def hello_logs(context: OpExecutionContext):
    if False:
        for i in range(10):
            print('nop')
    context.log.info('Hello, world!')

@job
def demo_job():
    if False:
        for i in range(10):
            print('nop')
    hello_logs()
from dagster import job, op, OpExecutionContext

@op
def hello_logs_error(context: OpExecutionContext):
    if False:
        for i in range(10):
            print('nop')
    raise Exception('Somebody set up us the bomb')

@job
def demo_job_error():
    if False:
        while True:
            i = 10
    hello_logs_error()
defs = Definitions(jobs=[demo_job, demo_job_error])