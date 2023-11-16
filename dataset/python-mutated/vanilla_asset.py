from dagster import asset

@asset
def a_number():
    if False:
        return 10
    return 1