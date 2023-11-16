from dagster import OpExecutionContext, job, op

@op
def return_one(context: OpExecutionContext) -> int:
    if False:
        return 10
    return 1

@op
def add_one(context: OpExecutionContext, number: int) -> int:
    if False:
        print('Hello World!')
    return number + 1

@job
def linear():
    if False:
        print('Hello World!')
    add_one(add_one(add_one(return_one())))