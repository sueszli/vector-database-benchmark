from dagster import asset

@asset(code_version='v1')
def versioned_number():
    if False:
        for i in range(10):
            print('nop')
    return 1