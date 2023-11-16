import os
from dagster import get_dagster_logger, job, op

@op
def get_file_sizes():
    if False:
        while True:
            i = 10
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    return {f: os.path.getsize(f) for f in files}

@op
def report_total_size(file_sizes):
    if False:
        print('Hello World!')
    total_size = sum(file_sizes.values())
    get_dagster_logger().info(f'Total size: {total_size}')

@job
def serial():
    if False:
        while True:
            i = 10
    report_total_size(get_file_sizes())