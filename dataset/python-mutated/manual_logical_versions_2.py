from dagster import DataVersion, Output, asset

@asset(code_version='v5')
def versioned_number():
    if False:
        for i in range(10):
            print('nop')
    value = 10 + 10
    return Output(value, logical_version=DataVersion(str(value)))

@asset(code_version='v1')
def multiplied_number(versioned_number):
    if False:
        for i in range(10):
            print('nop')
    return versioned_number * 2