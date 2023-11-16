from dagster import job, op

@op
def return_five():
    if False:
        print('Hello World!')
    return 5

@op
def add_one(arg):
    if False:
        for i in range(10):
            print('nop')
    return arg + 1

@job
def do_stuff():
    if False:
        i = 10
        return i + 15
    add_one(return_five())