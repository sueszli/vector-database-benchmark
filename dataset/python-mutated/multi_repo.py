from dagster import repository

@repository
def repo_one():
    if False:
        while True:
            i = 10
    return []

@repository
def repo_two():
    if False:
        return 10
    return []