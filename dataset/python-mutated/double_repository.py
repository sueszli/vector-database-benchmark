from dagster import repository

@repository
def repo_one():
    if False:
        for i in range(10):
            print('nop')
    return []

@repository
def repo_two():
    if False:
        while True:
            i = 10
    return []