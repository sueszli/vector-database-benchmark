import random
import time
import pytest
import dramatiq
from dramatiq.brokers.rabbitmq import RabbitmqBroker
from ..common import RABBITMQ_CREDENTIALS
broker = RabbitmqBroker(host='127.0.0.1', credentials=RABBITMQ_CREDENTIALS)

@dramatiq.actor(queue_name='benchmark-throughput', broker=broker)
def throughput():
    if False:
        print('Hello World!')
    pass

@dramatiq.actor(queue_name='benchmark-fib', broker=broker)
def fib(n):
    if False:
        i = 10
        return i + 15
    (x, y) = (1, 1)
    while n > 2:
        (x, y) = (x + y, x)
        n -= 1
    return x

@dramatiq.actor(queue_name='benchmark-latency', broker=broker)
def latency():
    if False:
        for i in range(10):
            print('nop')
    p = random.randint(1, 100)
    if p == 1:
        durations = [3, 3, 3, 1]
    elif p <= 10:
        durations = [2, 3]
    elif p <= 40:
        durations = [1, 2]
    else:
        durations = [1]
    for duration in durations:
        time.sleep(duration)

@pytest.mark.benchmark(group='rabbitmq-100k-throughput')
def test_rabbitmq_process_100k_messages_with_cli(benchmark, info_logging, start_cli):
    if False:
        return 10

    def setup():
        if False:
            for i in range(10):
                print('nop')
        del broker.channel
        del broker.connection
        for _ in range(100000):
            throughput.send()
        start_cli('tests.benchmarks.test_rabbitmq_cli:broker')
    benchmark.pedantic(broker.join, args=(throughput.queue_name,), setup=setup)

@pytest.mark.benchmark(group='rabbitmq-10k-fib')
def test_rabbitmq_process_10k_fib_with_cli(benchmark, info_logging, start_cli):
    if False:
        return 10

    def setup():
        if False:
            print('Hello World!')
        del broker.channel
        del broker.connection
        for _ in range(10000):
            fib.send(random.choice([1, 512, 1024, 2048, 4096, 8192]))
        start_cli('tests.benchmarks.test_rabbitmq_cli:broker')
    benchmark.pedantic(broker.join, args=(fib.queue_name,), setup=setup)

@pytest.mark.benchmark(group='rabbitmq-1k-latency')
def test_rabbitmq_process_1k_latency_with_cli(benchmark, info_logging, start_cli):
    if False:
        return 10

    def setup():
        if False:
            for i in range(10):
                print('nop')
        del broker.channel
        del broker.connection
        for _ in range(1000):
            latency.send()
        start_cli('tests.benchmarks.test_rabbitmq_cli:broker')
    benchmark.pedantic(broker.join, args=(latency.queue_name,), setup=setup)