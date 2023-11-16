"""A microbenchmark for measuring changes in overhead for critical code paths.

This runs a sequence of trivial Maps over a variable number of inputs to
estimate the per-element processing time.  It can be useful to run this
benchmark before and after a proposed set of changes.  A typical per-element
cost should be 1-2 microseconds.

This executes the same codepaths that are run on the Fn API (and Dataflow)
workers, but is generally easier to run (locally) and more stable.  It does
not, on the other hand, excercise any non-trivial amount of IO (e.g. shuffle).

Run as

   python -m apache_beam.tools.map_fn_microbenchmark
"""
import logging
import time
from scipy import stats
import apache_beam as beam
from apache_beam.tools import utils

def run_benchmark(num_maps=100, num_runs=10, num_elements_step=1000):
    if False:
        while True:
            i = 10
    timings = {}
    for run in range(num_runs):
        num_elements = num_elements_step * run + 1
        start = time.time()
        with beam.Pipeline() as p:
            pc = p | beam.Create(list(range(num_elements)))
            for ix in range(num_maps):
                pc = pc | 'Map%d' % ix >> beam.FlatMap(lambda x: (None,))
        timings[num_elements] = time.time() - start
        print('%6d element%s %g sec' % (num_elements, ' ' if num_elements == 1 else 's', timings[num_elements]))
    print()
    (gradient, intercept, r_value, p_value, std_err) = stats.linregress(*list(zip(*list(timings.items()))))
    print('Fixed cost  ', intercept)
    print('Per-element ', gradient / num_maps)
    print('R^2         ', r_value ** 2)
if __name__ == '__main__':
    logging.basicConfig()
    utils.check_compiled('apache_beam.runners.common')
    run_benchmark()