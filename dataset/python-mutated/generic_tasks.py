import time
from prefect import task

@task(name=f'{__name__}.noop')
def noop():
    if False:
        i = 10
        return i + 15
    return

@task(name=f'{__name__}.add_one')
def add_one(x: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return x + 1

@task(name=f'{__name__}.subtract_ten')
def subtract_ten(x: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return x - 10

@task(name=f'{__name__}.sleep')
def sleep(x: float):
    if False:
        while True:
            i = 10
    time.sleep(x)

@task(name=f'{__name__}-async_multiply_by_two')
async def async_multiply_by_two(x: int) -> int:
    return x * 2