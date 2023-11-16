from dagster import In, Nothing, job, op

@op
def op_with_nothing_output() -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

@op(ins={'in1': In(Nothing)})
def op_with_nothing_input() -> None:
    if False:
        while True:
            i = 10
    ...

@job
def nothing_job():
    if False:
        while True:
            i = 10
    op_with_nothing_input(op_with_nothing_output())