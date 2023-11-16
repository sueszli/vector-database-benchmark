from dagster import asset

@asset(code_version='v3')
def versioned_number():
    if False:
        i = 10
        return i + 15
    return 15

@asset(code_version='v1')
def multiplied_number(versioned_number):
    if False:
        while True:
            i = 10
    return versioned_number * 2