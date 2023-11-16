"""A cross-language word-counting workflow."""
import argparse
import logging
import re
import subprocess
import grpc
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
EXPANSION_SERVICE_PORT = '8096'
EXPANSION_SERVICE_ADDR = 'localhost:%s' % EXPANSION_SERVICE_PORT

class WordExtractingDoFn(beam.DoFn):
    """Parse each line of input text into words."""

    def process(self, element):
        if False:
            i = 10
            return i + 15
        'Returns an iterator over the words of this element.\n\n    The element is a line of text.  If the line is blank, note that, too.\n\n    Args:\n      element: the element being processed\n\n    Returns:\n      The processed element.\n    '
        text_line = element.strip()
        return re.findall("[\\w\\']+", text_line)

def build_pipeline(p, input_file, output_file):
    if False:
        while True:
            i = 10
    lines = p | 'read' >> ReadFromText(input_file)
    counts = lines | 'split' >> beam.ParDo(WordExtractingDoFn()).with_output_types(str) | 'count' >> beam.ExternalTransform('beam:transforms:xlang:count', None, EXPANSION_SERVICE_ADDR)

    def format_result(word_count):
        if False:
            i = 10
            return i + 15
        (word, count) = word_count
        return '%s: %d' % (word, count)
    output = counts | 'format' >> beam.Map(format_result)
    output | 'write' >> WriteToText(output_file)

def main():
    if False:
        return 10
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', default='gs://dataflow-samples/shakespeare/kinglear.txt', help='Input file to process.')
    parser.add_argument('--output', dest='output', required=True, help='Output file to write results to.')
    parser.add_argument('--expansion_service_jar', dest='expansion_service_jar', required=True, help='Jar file for expansion service')
    (known_args, pipeline_args) = parser.parse_known_args()
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    try:
        server = subprocess.Popen(['java', '-jar', known_args.expansion_service_jar, EXPANSION_SERVICE_PORT])
        with grpc.insecure_channel(EXPANSION_SERVICE_ADDR) as channel:
            grpc.channel_ready_future(channel).result()
        with beam.Pipeline(options=pipeline_options) as p:
            p.runner.create_job_service(pipeline_options)
            build_pipeline(p, known_args.input, known_args.output)
    finally:
        server.kill()
if __name__ == '__main__':
    main()