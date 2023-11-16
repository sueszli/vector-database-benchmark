from dagster import OpExecutionContext, job, op

@op
def my_op(context: OpExecutionContext, input_string: str):
    if False:
        while True:
            i = 10
    context.log.info(f'input string: {input_string}')

@job
def my_job():
    if False:
        for i in range(10):
            print('nop')
    my_op()

def execute_with_config():
    if False:
        i = 10
        return i + 15
    my_job.execute_in_process(run_config={'ops': {'my_op': {'inputs': {'input_string': {'value': 'marmot'}}}}})