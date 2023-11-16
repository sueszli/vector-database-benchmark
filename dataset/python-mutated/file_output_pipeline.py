from dagster import job, op, OpExecutionContext

@op
def file_log_op(context: OpExecutionContext):
    if False:
        for i in range(10):
            print('nop')
    context.log.info('Hello world!')

@job
def file_log_job():
    if False:
        while True:
            i = 10
    file_log_op()