"""
TODO: Add benches for higher number of tasks; blocked by engine deadlocks in CI.
"""
import anyio
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from prefect import flow, task

def noop_function():
    if False:
        print('Hello World!')
    pass

async def anoop_function():
    pass

def bench_flow_decorator(benchmark: BenchmarkFixture):
    if False:
        print('Hello World!')
    benchmark(flow, noop_function)

@pytest.mark.parametrize('options', [{}, {'timeout_seconds': 10}])
def bench_flow_call(benchmark: BenchmarkFixture, options):
    if False:
        print('Hello World!')
    noop_flow = flow(**options)(noop_function)
    benchmark(noop_flow)

@pytest.mark.parametrize('num_tasks', [10, 50, 100])
def bench_flow_with_submitted_tasks(benchmark: BenchmarkFixture, num_tasks: int):
    if False:
        for i in range(10):
            print('nop')
    test_task = task(noop_function)

    @flow
    def benchmark_flow():
        if False:
            i = 10
            return i + 15
        for _ in range(num_tasks):
            test_task.submit()
    if num_tasks < 100:
        benchmark(benchmark_flow)
    else:
        benchmark.pedantic(benchmark_flow)

@pytest.mark.parametrize('num_tasks', [10, 50, 100, 250])
def bench_flow_with_called_tasks(benchmark: BenchmarkFixture, num_tasks: int):
    if False:
        while True:
            i = 10
    test_task = task(noop_function)

    @flow
    def benchmark_flow():
        if False:
            while True:
                i = 10
        for _ in range(num_tasks):
            test_task()
    if num_tasks < 100:
        benchmark(benchmark_flow)
    else:
        benchmark.pedantic(benchmark_flow)

@pytest.mark.parametrize('num_tasks', [10, 50, 100, 250])
def bench_async_flow_with_async_tasks(benchmark: BenchmarkFixture, num_tasks: int):
    if False:
        for i in range(10):
            print('nop')
    test_task = task(anoop_function)

    @flow
    async def benchmark_flow():
        async with anyio.create_task_group() as tg:
            for _ in range(num_tasks):
                tg.start_soon(test_task)
    if num_tasks < 100:
        benchmark(anyio.run, benchmark_flow)
    else:
        benchmark.pedantic(anyio.run, args=(benchmark_flow,))

@pytest.mark.parametrize('num_tasks', [10, 50, 100])
def bench_async_flow_with_submitted_sync_tasks(benchmark: BenchmarkFixture, num_tasks: int):
    if False:
        for i in range(10):
            print('nop')
    test_task = task(noop_function)

    @flow
    async def benchmark_flow():
        for _ in range(num_tasks):
            test_task.submit()
    if num_tasks < 100:
        benchmark(anyio.run, benchmark_flow)
    else:
        benchmark.pedantic(anyio.run, args=(benchmark_flow,))

@pytest.mark.parametrize('num_flows', [5, 10, 20])
def bench_flow_with_subflows(benchmark: BenchmarkFixture, num_flows: int):
    if False:
        return 10
    test_flow = flow(noop_function)

    @flow
    def benchmark_flow():
        if False:
            for i in range(10):
                print('nop')
        for _ in range(num_flows):
            test_flow()
    benchmark(benchmark_flow)

@pytest.mark.parametrize('num_flows', [5, 10, 20])
def bench_async_flow_with_sequential_subflows(benchmark: BenchmarkFixture, num_flows: int):
    if False:
        print('Hello World!')
    test_flow = flow(anoop_function)

    @flow
    async def benchmark_flow():
        for _ in range(num_flows):
            await test_flow()
    benchmark(anyio.run, benchmark_flow)

@pytest.mark.parametrize('num_flows', [5, 10, 20])
def bench_async_flow_with_concurrent_subflows(benchmark: BenchmarkFixture, num_flows: int):
    if False:
        for i in range(10):
            print('nop')
    test_flow = flow(anoop_function)

    @flow
    async def benchmark_flow():
        async with anyio.create_task_group() as tg:
            for _ in range(num_flows):
                tg.start_soon(test_flow)
    benchmark(anyio.run, benchmark_flow)