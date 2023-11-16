from dagster import job, op

@op
def return_one():
    if False:
        while True:
            i = 10
    return 1

@op
def add_two(i: int):
    if False:
        i = 10
        return i + 15
    return i + 2

@op
def multi_three(i: int):
    if False:
        while True:
            i = 10
    return i * 3

@job
def my_job():
    if False:
        while True:
            i = 10
    multi_three(add_two(return_one()))
if __name__ == '__main__':
    result = my_job.execute_in_process()

def execute_subset():
    if False:
        while True:
            i = 10
    my_job.execute_in_process(op_selection=['*add_two'])

@op
def total(in_1: int, in_2: int, in_3: int, in_4: int):
    if False:
        return 10
    return in_1 + in_2 + in_3 + in_4
ip_yaml = '\n# start_ip_yaml\n\nexecution:\n  config:\n    in_process:\n\n# end_ip_yaml\n'

@job(config={'execution': {'config': {'multiprocess': {'start_method': {'forkserver': {}}, 'max_concurrent': 4}}}})
def forkserver_job():
    if False:
        i = 10
        return i + 15
    multi_three(add_two(return_one()))

@job(config={'execution': {'config': {'multiprocess': {'max_concurrent': 4, 'tag_concurrency_limits': [{'key': 'database', 'value': 'redshift', 'limit': 2}]}}}})
def tag_concurrency_job():
    if False:
        i = 10
        return i + 15
    ...