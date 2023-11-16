from dagster import repository

@repository(name='repo_one')
def repo_one_symbol():
    if False:
        while True:
            i = 10
    return []

@repository
def repo_two():
    if False:
        for i in range(10):
            print('nop')
    return []