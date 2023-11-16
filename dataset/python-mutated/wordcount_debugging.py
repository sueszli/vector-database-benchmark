"""An example that verifies the counts and includes best practices.

On top of the basic concepts in the wordcount example, this workflow introduces
logging to Cloud Logging, and using assertions in a Dataflow pipeline.

To execute this pipeline locally, specify a local output file or output prefix
on GCS::

  --output [YOUR_LOCAL_FILE | gs://YOUR_OUTPUT_PREFIX]

To execute this pipeline using the Google Cloud Dataflow service, specify
pipeline configuration::

  --project YOUR_PROJECT_ID
  --staging_location gs://YOUR_STAGING_DIRECTORY
  --temp_location gs://YOUR_TEMP_DIRECTORY
  --region GCE_REGION
  --job_name YOUR_JOB_NAME
  --runner DataflowRunner

and an output prefix on GCS::

  --output gs://YOUR_OUTPUT_PREFIX
"""
import argparse
import logging
import re
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.metrics import Metrics
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class FilterTextFn(beam.DoFn):
    """A DoFn that filters for a specific key based on a regular expression."""

    def __init__(self, pattern):
        if False:
            print('Hello World!')
        beam.DoFn.__init__(self)
        self.pattern = pattern
        self.matched_words = Metrics.counter(self.__class__, 'matched_words')
        self.umatched_words = Metrics.counter(self.__class__, 'umatched_words')

    def process(self, element):
        if False:
            print('Hello World!')
        (word, _) = element
        if re.match(self.pattern, word):
            logging.info('Matched %s', word)
            self.matched_words.inc()
            yield element
        else:
            logging.debug('Did not match %s', word)
            self.umatched_words.inc()

class CountWords(beam.PTransform):
    """A transform to count the occurrences of each word.

  A PTransform that converts a PCollection containing lines of text into a
  PCollection of (word, count) tuples.
  """

    def expand(self, pcoll):
        if False:
            return 10

        def count_ones(word_ones):
            if False:
                return 10
            (word, ones) = word_ones
            return (word, sum(ones))
        return pcoll | 'split' >> beam.FlatMap(lambda x: re.findall("[A-Za-z\\']+", x)).with_output_types(str) | 'pair_with_one' >> beam.Map(lambda x: (x, 1)) | 'group' >> beam.GroupByKey() | 'count' >> beam.Map(count_ones)

def run(argv=None, save_main_session=True):
    if False:
        i = 10
        return i + 15
    'Runs the debugging wordcount pipeline.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', default='gs://dataflow-samples/shakespeare/kinglear.txt', help='Input file to process.')
    parser.add_argument('--output', dest='output', required=True, help='Output file to write results to.')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    with beam.Pipeline(options=pipeline_options) as p:
        filtered_words = p | 'read' >> ReadFromText(known_args.input) | CountWords() | 'FilterText' >> beam.ParDo(FilterTextFn('Flourish|stomach'))
        assert_that(filtered_words, equal_to([('Flourish', 3), ('stomach', 1)]))

        def format_result(word_count):
            if False:
                return 10
            (word, count) = word_count
            return '%s: %s' % (word, count)
        output = filtered_words | 'format' >> beam.Map(format_result) | 'write' >> WriteToText(known_args.output)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()