from typing import List
from dagster import job, op

@op
def return_one() -> int:
    if False:
        while True:
            i = 10
    return 1

@op
def sum_fan_in(nums: List[int]) -> int:
    if False:
        return 10
    return sum(nums)

@job
def fan_in():
    if False:
        print('Hello World!')
    fan_outs = []
    for i in range(0, 10):
        fan_outs.append(return_one.alias(f'return_one_{i}')())
    sum_fan_in(fan_outs)