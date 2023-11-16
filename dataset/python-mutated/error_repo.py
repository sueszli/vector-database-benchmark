from dagster import repository

@repository
def error_repo():
    if False:
        i = 10
        return i + 15
    a = None
    a()