from dagster import In, job, op
from dagster._core.storage.input_manager import input_manager

def read_dataframe_from_table(**_kwargs):
    if False:
        while True:
            i = 10
    pass

@op(ins={'dataframe': In(input_manager_key='my_input_manager')})
def my_op(dataframe):
    if False:
        return 10
    'Do some stuff.'

@input_manager(input_config_schema={'table_name': str})
def table_loader(context):
    if False:
        for i in range(10):
            print('nop')
    return read_dataframe_from_table(name=context.config['table_name'])

def execute_with_config():
    if False:
        i = 10
        return i + 15

    @job(resource_defs={'my_input_manager': table_loader})
    def my_job():
        if False:
            for i in range(10):
                print('nop')
        my_op()
    my_job.execute_in_process(run_config={'ops': {'my_op': {'inputs': {'dataframe': {'table_name': 'table1'}}}}})