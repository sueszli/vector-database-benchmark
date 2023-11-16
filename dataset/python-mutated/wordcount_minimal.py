"""A minimalist word-counting workflow that counts words in Shakespeare.

This is the first in a series of successively more detailed 'word count'
examples.

Next, see the wordcount pipeline, then the wordcount_debugging pipeline, for
more detailed examples that introduce additional concepts.

Concepts:

1. Reading data from text files
2. Specifying 'inline' transforms
3. Counting a PCollection
4. Writing data to Cloud Storage as text files

To execute this pipeline locally, first edit the code to specify the output
location. Output location could be a local file path or an output prefix
on GCS. (Only update the output location marked with the first CHANGE comment.)

To execute this pipeline remotely, first edit the code to set your project ID,
runner type, the staging location, the temp location, and the output location.
The specified GCS bucket(s) must already exist. (Update all the places marked
with a CHANGE comment.)

Then, run the pipeline as described in the README. It will be deployed and run
using the Google Cloud Dataflow Service. No args are required to run the
pipeline. You can see the results in your output bucket in the GCS browser.
"""
import argparse
import logging
import re
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

def main(argv=None, save_main_session=True):
    if False:
        return 10
    'Main entry point; defines and runs the wordcount pipeline.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', default='gs://dataflow-samples/shakespeare/kinglear.txt', help='Input file to process.')
    parser.add_argument('--output', dest='output', default='gs://YOUR_OUTPUT_BUCKET/AND_OUTPUT_PREFIX', help='Output file to write results to.')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    with beam.Pipeline(options=pipeline_options) as p:
        lines = p | ReadFromText(known_args.input)
        counts = lines | 'Split' >> beam.FlatMap(lambda x: re.findall("[A-Za-z\\']+", x)).with_output_types(str) | 'PairWithOne' >> beam.Map(lambda x: (x, 1)) | 'GroupAndSum' >> beam.CombinePerKey(sum)

        def format_result(word_count):
            if False:
                return 10
            (word, count) = word_count
            return '%s: %s' % (word, count)
        output = counts | 'Format' >> beam.Map(format_result)
        output | WriteToText(known_args.output)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()