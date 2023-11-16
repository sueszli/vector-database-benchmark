"""A word-counting workflow that uses the SQL transform.

A Java version supported by Beam must be installed locally to run this pipeline.
Additionally, Docker must also be available to run this pipeline locally.
"""
import argparse
import logging
import re
import typing
import apache_beam as beam
from apache_beam import coders
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.portability import portable_runner
from apache_beam.transforms.sql import SqlTransform
MyRow = typing.NamedTuple('MyRow', [('word', str)])
coders.registry.register_coder(MyRow, coders.RowCoder)

def run(p, input_file, output_file):
    if False:
        print('Hello World!')
    p | 'Read' >> ReadFromText(input_file) | 'Split' >> beam.FlatMap(lambda line: re.split('\\W+', line)) | 'ToRow' >> beam.Map(MyRow).with_output_types(MyRow) | 'Sql!!' >> SqlTransform('\n                   SELECT\n                     word as key,\n                     COUNT(*) as `count`\n                   FROM PCOLLECTION\n                   GROUP BY word') | 'Format' >> beam.Map(lambda row: '{}: {}'.format(row.key, row.count)) | 'Write' >> WriteToText(output_file)

def main():
    if False:
        return 10
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', default='gs://dataflow-samples/shakespeare/kinglear.txt', help='Input file to process.')
    parser.add_argument('--output', dest='output', required=True, help='Output file to write results to.')
    (known_args, pipeline_args) = parser.parse_known_args()
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    with beam.Pipeline(options=pipeline_options) as p:
        if isinstance(p.runner, portable_runner.PortableRunner):
            p.runner.create_job_service(pipeline_options)
        run(p, known_args.input, known_args.output)
if __name__ == '__main__':
    main()