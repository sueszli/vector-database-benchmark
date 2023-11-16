import random
from dagster import Output, asset

@asset(output_required=False)
def may_not_materialize():
    if False:
        return 10
    if random.randint(1, 10) < 5:
        yield Output([1, 2, 3, 4])

@asset
def downstream(may_not_materialize):
    if False:
        for i in range(10):
            print('nop')
    return may_not_materialize + [5]