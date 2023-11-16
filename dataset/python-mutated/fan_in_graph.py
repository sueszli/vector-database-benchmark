from typing import List
from dagster import graph, op

@op
def return_one() -> int:
    if False:
        while True:
            i = 10
    return 1

@op
def sum_fan_in(nums: List[int]) -> int:
    if False:
        print('Hello World!')
    return sum(nums)

@graph
def fan_in():
    if False:
        for i in range(10):
            print('nop')
    fan_outs = []
    for i in range(0, 10):
        fan_outs.append(return_one.alias(f'return_one_{i}')())
    sum_fan_in(fan_outs)