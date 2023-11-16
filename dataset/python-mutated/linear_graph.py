from dagster import OpExecutionContext, graph, op

@op
def return_one(context: OpExecutionContext) -> int:
    if False:
        while True:
            i = 10
    return 1

@op
def add_one(context: OpExecutionContext, number: int) -> int:
    if False:
        print('Hello World!')
    return number + 1

@graph
def linear():
    if False:
        for i in range(10):
            print('nop')
    add_one(add_one(add_one(return_one())))