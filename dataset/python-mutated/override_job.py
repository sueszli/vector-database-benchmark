from dagster import In, Int, Out, job, op, repository

@op(ins={'num': In(Int)}, out=Out(Int))
def add_one(num):
    if False:
        return 10
    return num + 1

@op(ins={'num': In(Int)}, out=Out(Int))
def mult_two(num):
    if False:
        for i in range(10):
            print('nop')
    return num * 2

@job
def math():
    if False:
        return 10
    mult_two(num=add_one())

@repository
def test_override_repository():
    if False:
        print('Hello World!')
    return [math]