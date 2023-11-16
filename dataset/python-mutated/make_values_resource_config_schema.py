import os
from dagster import OpExecutionContext, job, make_values_resource, op

@op(required_resource_keys={'file_dirs'})
def write_file(context: OpExecutionContext):
    if False:
        while True:
            i = 10
    filename = f"{context.resources.file_dirs['write_file_dir']}/new_file.txt"
    open(filename, 'x', encoding='utf8').close()
    context.log.info(f'Created file: {filename}')

@op(required_resource_keys={'file_dirs'})
def total_num_files(context: OpExecutionContext):
    if False:
        print('Hello World!')
    files_in_dir = os.listdir(context.resources.file_dirs['count_file_dir'])
    context.log.info(f'Total number of files: {len(files_in_dir)}')

@job(resource_defs={'file_dirs': make_values_resource(write_file_dir=str, count_file_dir=str)})
def file_dirs_job():
    if False:
        i = 10
        return i + 15
    write_file()
    total_num_files()