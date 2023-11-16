from dagster import asset

@asset
def my_asset():
    if False:
        for i in range(10):
            print('nop')
    return [1, 2, 3]