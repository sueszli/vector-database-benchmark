from dagster import OpExecutionContext, job, op

@op
def return_fifty():
    if False:
        print('Hello World!')
    return 50.0

@op
def add_thirty_two(number):
    if False:
        return 10
    return number + 32.0

@op
def multiply_by_one_point_eight(number):
    if False:
        i = 10
        return i + 15
    return number * 1.8

@op
def log_number(context: OpExecutionContext, number):
    if False:
        i = 10
        return i + 15
    context.log.info(f'number: {number}')

@job
def all_together_unnested():
    if False:
        print('Hello World!')
    log_number(add_thirty_two(multiply_by_one_point_eight(return_fifty())))