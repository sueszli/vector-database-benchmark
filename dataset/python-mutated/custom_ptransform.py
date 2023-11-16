"""Various implementations of a Count custom PTransform.

These example show the different ways you can write custom PTransforms.
"""
import argparse
import logging
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions

class Count1(beam.PTransform):
    """Count as a subclass of PTransform, with an apply method."""

    def expand(self, pcoll):
        if False:
            print('Hello World!')
        return pcoll | 'ParWithOne' >> beam.Map(lambda v: (v, 1)) | beam.CombinePerKey(sum)

def run_count1(known_args, options):
    if False:
        return 10
    'Runs the first example pipeline.'
    logging.info('Running first pipeline')
    with beam.Pipeline(options=options) as p:
        p | beam.io.ReadFromText(known_args.input) | Count1() | beam.io.WriteToText(known_args.output)

@beam.ptransform_fn
def Count2(pcoll):
    if False:
        while True:
            i = 10
    'Count as a decorated function.'
    return pcoll | 'PairWithOne' >> beam.Map(lambda v: (v, 1)) | beam.CombinePerKey(sum)

def run_count2(known_args, options):
    if False:
        i = 10
        return i + 15
    'Runs the second example pipeline.'
    logging.info('Running second pipeline')
    with beam.Pipeline(options=options) as p:
        p | ReadFromText(known_args.input) | Count2() | WriteToText(known_args.output)

@beam.ptransform_fn
def Count3(pcoll, factor=1):
    if False:
        print('Hello World!')
    'Count as a decorated function with a side input.\n\n  Args:\n    pcoll: the PCollection passed in from the previous transform\n    factor: the amount by which to count\n\n  Returns:\n    A PCollection counting the number of times each unique element occurs.\n  '
    return pcoll | 'PairWithOne' >> beam.Map(lambda v: (v, factor)) | beam.CombinePerKey(sum)

def run_count3(known_args, options):
    if False:
        for i in range(10):
            print('nop')
    'Runs the third example pipeline.'
    logging.info('Running third pipeline')
    with beam.Pipeline(options=options) as p:
        p | ReadFromText(known_args.input) | Count3(2) | WriteToText(known_args.output)

def get_args(argv):
    if False:
        print('Hello World!')
    'Determines user specified arguments from the given list of arguments.\n\n  Args:\n    argv: all arguments.\n\n  Returns:\n    A pair of argument lists containing known and remaining arguments.\n  '
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input file to process.')
    parser.add_argument('--output', required=True, help='Output file to write results to.')
    return parser.parse_known_args(argv)

def run(argv=None):
    if False:
        for i in range(10):
            print('nop')
    (known_args, pipeline_args) = get_args(argv)
    run_count1(known_args, PipelineOptions(pipeline_args))
    run_count2(known_args, PipelineOptions(pipeline_args))
    run_count3(known_args, PipelineOptions(pipeline_args))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()