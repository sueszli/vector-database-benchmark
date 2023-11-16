from dagster_aws.cloudwatch.loggers import cloudwatch_logger
from dagster import OpExecutionContext, colored_console_logger, graph, op, repository

@op
def log_op(context: OpExecutionContext):
    if False:
        for i in range(10):
            print('nop')
    context.log.info('Hello, world!')

@graph
def hello_logs():
    if False:
        i = 10
        return i + 15
    log_op()
local_logs = hello_logs.to_job(name='local_logs', logger_defs={'console': colored_console_logger})
prod_logs = hello_logs.to_job(name='prod_logs', logger_defs={'cloudwatch': cloudwatch_logger})

@repository
def logs_repo():
    if False:
        while True:
            i = 10
    return [local_logs, prod_logs]