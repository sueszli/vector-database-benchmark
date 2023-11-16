from dagster import repository

@repository
def single_repository():
    if False:
        i = 10
        return i + 15
    return []