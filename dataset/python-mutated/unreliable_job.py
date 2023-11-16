RESULTS = [1, 1, 0.4]

def some_random_result():
    if False:
        return 10
    return RESULTS.pop()
from dagster import in_process_executor, job, op

@op
def start():
    if False:
        for i in range(10):
            print('nop')
    return 1

@op
def unreliable(num: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    failure_rate = 0.5
    if some_random_result() < failure_rate:
        raise Exception('blah')
    return num

@op
def end(_num: int):
    if False:
        i = 10
        return i + 15
    pass

@job(executor_def=in_process_executor)
def unreliable_job():
    if False:
        print('Hello World!')
    end(unreliable(start()))