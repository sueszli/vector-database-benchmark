"""A microbenchmark for measuring the overhead of runtime_type_check vs
performance_runtime_type_check vs a pipeline with no runtime type check.

This runs a sequence of trivial DoFn's over a set of inputs to simulate
a real-world pipeline that processes lots of data.

Run as

   python -m apache_beam.tools.runtime_type_check_microbenchmark
"""
import logging
from collections import defaultdict
from time import time
from typing import Iterable
from typing import Tuple
from typing import Union
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.tools import utils

@beam.typehints.with_input_types(Tuple[int, ...])
class SimpleInput(beam.DoFn):

    def process(self, element, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        yield element

@beam.typehints.with_output_types(Tuple[int, ...])
class SimpleOutput(beam.DoFn):

    def process(self, element, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        yield element

@beam.typehints.with_input_types(Tuple[int, str, Tuple[float, ...], Iterable[int], Union[str, int]])
class NestedInput(beam.DoFn):

    def process(self, element, *args, **kwargs):
        if False:
            while True:
                i = 10
        yield element

@beam.typehints.with_output_types(Tuple[int, str, Tuple[float, ...], Iterable[int], Union[str, int]])
class NestedOutput(beam.DoFn):

    def process(self, element, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        yield element

def run_benchmark(num_dofns=100, num_runs=10, num_elements_step=2000, num_for_averaging=4):
    if False:
        return 10
    options_map = {'No Type Check': PipelineOptions(), 'Runtime Type Check': PipelineOptions(runtime_type_check=True), 'Performance Runtime Type Check': PipelineOptions(performance_runtime_type_check=True)}
    for run in range(num_runs):
        num_elements = num_elements_step * run + 1
        simple_elements = [tuple((i for i in range(200))) for _ in range(num_elements)]
        nested_elements = [(1, '2', tuple((float(i) for i in range(100))), [i for i in range(100)], '5') for _ in range(num_elements)]
        timings = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
        for _ in range(num_for_averaging):
            for (option_name, options) in options_map.items():
                start = time()
                with beam.Pipeline(options=options) as p:
                    pc = p | beam.Create(simple_elements)
                    for ix in range(num_dofns):
                        pc = pc | 'SimpleOutput %i' % ix >> beam.ParDo(SimpleOutput()) | 'SimpleInput %i' % ix >> beam.ParDo(SimpleInput())
                timings[num_elements]['Simple Types'][option_name] += time() - start
                start = time()
                with beam.Pipeline(options=options) as p:
                    pc = p | beam.Create(nested_elements)
                    for ix in range(num_dofns):
                        pc = pc | 'NestedOutput %i' % ix >> beam.ParDo(NestedOutput()) | 'NestedInput %i' % ix >> beam.ParDo(NestedInput())
                timings[num_elements]['Nested Types'][option_name] += time() - start
        for (num_elements, element_type_map) in timings.items():
            print('%d Element%s' % (num_elements, ' ' if num_elements == 1 else 's'))
            for (element_type, option_name_map) in element_type_map.items():
                print('-- %s' % element_type)
                for (option_name, time_elapsed) in option_name_map.items():
                    print('---- %.2f sec (%s)' % (time_elapsed / num_for_averaging, option_name))
        print('\n')
if __name__ == '__main__':
    logging.basicConfig()
    utils.check_compiled('apache_beam.runners.common')
    run_benchmark()