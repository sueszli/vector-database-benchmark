from dagster import OpExecutionContext, job, op

@op
def return_one(context: OpExecutionContext) -> int:
    if False:
        print('Hello World!')
    return 1

@op
def add_one(context: OpExecutionContext, number: int):
    if False:
        i = 10
        return i + 15
    return number + 1

@op
def adder(context: OpExecutionContext, a: int, b: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return a + b

@job
def inputs_and_outputs():
    if False:
        return 10
    value = return_one()
    a = add_one(value)
    b = add_one(value)
    adder(a, b)