from dagster import job

@job
def pipe_one():
    if False:
        for i in range(10):
            print('nop')
    pass

@job
def pipe_two():
    if False:
        while True:
            i = 10
    pass