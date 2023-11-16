import os
from dagster import OpExecutionContext, job, make_values_resource, op

@op(required_resource_keys={'file_dir'})
def add_file(context: OpExecutionContext):
    if False:
        return 10
    filename = f'{context.resources.file_dir}/new_file.txt'
    open(filename, 'x', encoding='utf8').close()
    context.log.info(f'Created file: {filename}')

@op(required_resource_keys={'file_dir'})
def total_num_files(context: OpExecutionContext):
    if False:
        return 10
    files_in_dir = os.listdir(context.resources.file_dir)
    context.log.info(f'Total number of files: {len(files_in_dir)}')

@job(resource_defs={'file_dir': make_values_resource()})
def file_dir_job():
    if False:
        print('Hello World!')
    add_file()
    total_num_files()