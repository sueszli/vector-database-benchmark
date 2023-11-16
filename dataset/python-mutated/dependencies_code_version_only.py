from dagster import asset

@asset(code_version='v2')
def versioned_number():
    if False:
        while True:
            i = 10
    return 11

@asset(code_version='v1')
def multiplied_number(versioned_number):
    if False:
        return 10
    return versioned_number * 2