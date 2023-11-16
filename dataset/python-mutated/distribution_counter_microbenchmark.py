"""A microbenchmark for measuring DistributionAccumulator performance

This runs a sequence of distribution.update for random input value to calculate
average update time per input.
A typical update operation should run into 0.6 microseconds

Run as
  python -m apache_beam.tools.distribution_counter_microbenchmark
"""
import logging
import random
import sys
import time
from apache_beam.tools import utils

def generate_input_values(num_input, lower_bound, upper_bound):
    if False:
        return 10
    values = []
    for i in range(num_input):
        values.append(random.randint(lower_bound, upper_bound))
    return values

def run_benchmark(num_runs=100, num_input=10000, seed=time.time()):
    if False:
        return 10
    total_time = 0
    random.seed(seed)
    lower_bound = 0
    upper_bound = sys.maxsize
    inputs = generate_input_values(num_input, lower_bound, upper_bound)
    from apache_beam.transforms import DataflowDistributionCounter
    print('Number of runs:', num_runs)
    print('Input size:', num_input)
    print('Input sequence from %d to %d' % (lower_bound, upper_bound))
    print('Random seed:', seed)
    for i in range(num_runs):
        counter = DataflowDistributionCounter()
        start = time.time()
        counter.add_inputs_for_test(inputs)
        time_cost = time.time() - start
        print('Run %d: Total time cost %g sec' % (i + 1, time_cost))
        total_time += time_cost / num_input
    print('Per element update time cost:', total_time / num_runs)
if __name__ == '__main__':
    logging.basicConfig()
    utils.check_compiled('apache_beam.transforms.cy_dataflow_distribution_counter')
    run_benchmark()