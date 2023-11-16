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
def get_total_size(file_sizes):
    if False:
        i = 10
        return i + 15
    return sum(file_sizes.values())

@op
def get_largest_size(file_sizes):
    if False:
        print('Hello World!')
    return max(file_sizes.values())

@op
def report_file_stats(total_size, largest_size):
    if False:
        return 10
    get_dagster_logger().info(f'Total size: {total_size}, largest size: {largest_size}')

@job
def diamond():
    if False:
        i = 10
        return i + 15
    file_sizes = get_file_sizes()
    report_file_stats(total_size=get_total_size(file_sizes), largest_size=get_largest_size(file_sizes))