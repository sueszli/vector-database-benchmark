from dagster import Field, In, Out, String, job, op, repository
from dagster._core.storage.memoizable_io_manager import versioned_filesystem_io_manager

@op(version='create_string_version', config_schema={'input_str': Field(String)}, out={'created_string': Out(io_manager_key='io_manager', metadata={})})
def create_string_1_asset(context):
    if False:
        while True:
            i = 10
    return context.op_config['input_str']

@op(ins={'_string_input': In(String)}, version='take_string_version', config_schema={'input_str': Field(String)}, out={'taken_string': Out(io_manager_key='io_manager', metadata={})})
def take_string_1_asset(context, _string_input):
    if False:
        print('Hello World!')
    return context.op_config['input_str'] + _string_input

@job(resource_defs={'io_manager': versioned_filesystem_io_manager})
def asset_job():
    if False:
        for i in range(10):
            print('nop')
    take_string_1_asset(create_string_1_asset())

@op(version='create_string_version', config_schema={'input_str': Field(String)})
def create_string_1_asset_op(context):
    if False:
        return 10
    return context.op_config['input_str']

@op(version='take_string_version', config_schema={'input_str': Field(String)})
def take_string_1_asset_op(context, _string_input):
    if False:
        i = 10
        return i + 15
    return context.op_config['input_str'] + _string_input

@job
def op_job():
    if False:
        return 10
    take_string_1_asset_op(create_string_1_asset_op())

@repository
def memoized_dev_repo():
    if False:
        print('Hello World!')
    return [op_job, asset_job]