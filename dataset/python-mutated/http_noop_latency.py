import asyncio
import logging
import click
import pandas as pd
import requests
from ray import serve
from ray.serve._private.benchmarks.common import run_latency_benchmark

@serve.deployment
class Noop:

    def __init__(self):
        if False:
            return 10
        logging.getLogger('ray.serve').setLevel(logging.WARNING)

    def __call__(self, _):
        if False:
            return 10
        return b''

@click.command(help='Benchmark no-op HTTP latency.')
@click.option('--num-replicas', type=int, default=1)
@click.option('--num-requests', type=int, default=100)
def main(num_replicas: int, num_requests: int):
    if False:
        while True:
            i = 10
    serve.run(Noop.options(num_replicas=num_replicas).bind())
    latencies: pd.Series = asyncio.new_event_loop().run_until_complete(run_latency_benchmark(lambda : requests.get('http://localhost:8000'), num_requests=num_requests))
    print(f'Latency (ms) for noop HTTP requests (num_replicas={num_replicas},num_requests={num_requests}):')
    print(latencies.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
if __name__ == '__main__':
    main()