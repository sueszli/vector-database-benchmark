import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from prefect import flow, task

def noop_function():
    if False:
        print('Hello World!')
    pass

def bench_task_decorator(benchmark: BenchmarkFixture):
    if False:
        while True:
            i = 10
    benchmark(task, noop_function)

def bench_task_call(benchmark: BenchmarkFixture):
    if False:
        for i in range(10):
            print('nop')
    noop_task = task(noop_function)

    @flow
    def benchmark_flow():
        if False:
            print('Hello World!')
        benchmark(noop_task)
    benchmark_flow()

@pytest.mark.parametrize('num_task_runs', [100, 250])
def bench_task_submit(benchmark: BenchmarkFixture, num_task_runs: int):
    if False:
        return 10
    noop_task = task(noop_function)

    @flow
    def benchmark_flow():
        if False:
            for i in range(10):
                print('nop')
        benchmark.pedantic(noop_task.submit, rounds=num_task_runs)
    benchmark_flow()