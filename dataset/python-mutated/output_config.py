from dagster import InputContext, IOManager, OutputContext, io_manager, job, op

def connect():
    if False:
        while True:
            i = 10
    pass

def write_dataframe_to_table(**_kwargs):
    if False:
        print('Hello World!')
    pass

def read_dataframe_from_table(**_kwargs):
    if False:
        return 10
    pass

@op
def op_1():
    if False:
        return 10
    'Return a Pandas DataFrame.'

@op
def op_2(_input_dataframe):
    if False:
        print('Hello World!')
    'Return a Pandas DataFrame.'

class MyIOManager(IOManager):

    def handle_output(self, context: OutputContext, obj):
        if False:
            return 10
        table_name = context.config['table']
        write_dataframe_to_table(name=table_name, dataframe=obj)

    def load_input(self, context: InputContext):
        if False:
            for i in range(10):
                print('nop')
        if context.upstream_output:
            table_name = context.upstream_output.config['table']
            return read_dataframe_from_table(name=table_name)

@io_manager(output_config_schema={'table': str})
def my_io_manager(_):
    if False:
        i = 10
        return i + 15
    return MyIOManager()

def execute_my_job_with_config():
    if False:
        i = 10
        return i + 15

    @job(resource_defs={'io_manager': my_io_manager})
    def my_job():
        if False:
            i = 10
            return i + 15
        op_2(op_1())
    my_job.execute_in_process(run_config={'ops': {'op_1': {'outputs': {'result': {'table': 'table1'}}}, 'op_2': {'outputs': {'result': {'table': 'table2'}}}}})